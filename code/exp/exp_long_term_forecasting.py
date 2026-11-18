from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

warnings.filterwarnings('ignore')

class TemporalDecayHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(TemporalDecayHuberLoss, self).__init__()
        # 基础 Huber Loss, reduction='none' 保留时空矩阵的原始形状 [B, L, D]
        self.huber = nn.HuberLoss(reduction='none', delta=delta)

    def forward(self, y_pred, y_true):
        # y_pred, y_true 形状通常为 [Batch, Seq_Len/Pred_Len, Channels]
        # 获取当前的时间步长度 T (比如 96, 192)
        T = y_pred.size(1)
        
        # 1. 动态生成 CARD 衰减权重: 1 / sqrt(i + 1)
        # 注意：在 GPU 上直接生成，提升速度
        steps = torch.arange(1, T + 1, dtype=torch.float32, device=y_pred.device)
        decay_weights = 1.0 / torch.sqrt(steps)
        
        # 2. 权重归一化（极其重要！）
        # 如果不归一化，整体 Loss 的绝对值会变小，导致模型收敛极慢。
        # 归一化保证了加权后的 Loss 期望值与普通 Huber Loss 在同一量级。
        decay_weights = decay_weights / decay_weights.mean()
        
        # 3. 维度对齐，用于广播机制 (Broadcasting)
        # 将 [T] 变形为 [1, T, 1] 以匹配 [Batch, Time, Channels]
        decay_weights = decay_weights.unsqueeze(0).unsqueeze(-1)
        
        # 4. 计算基础 Huber 误差
        base_loss = self.huber(y_pred, y_true)
        
        # 5. 时序加权
        weighted_loss = base_loss * decay_weights
        
        # 返回全局平均值供反向传播
        return weighted_loss.mean()

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        if self.args.use_multi_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
            model = DDP(model.cuda(), device_ids=[self.args.local_rank],
                        find_unused_parameters=True)
        else:
            self.device = self.args.gpu
            model = model.to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print(n, p.dtype, p.shape)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _loss_window(self, outputs):
        if self.args.model == 'KIBR_PatchTST':
            return outputs[:, -self.args.token_len:, :]
        return outputs

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        self.model.eval()
        eval_model = self.model.module if self.args.use_multi_gpu else self.model
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = eval_model(batch_x, batch_x_mark, None, batch_y_mark)
                else:
                        outputs = eval_model(batch_x, batch_x_mark, None, batch_y_mark)
                if is_test:
                    outputs = outputs[:, -self.args.token_len:, :]
                    batch_y = batch_y[:, -self.args.token_len:, :].to(self.device)
                else:
                    outputs = self._loss_window(outputs)
                    batch_y = batch_y[:, -outputs.shape[1]:, :].to(self.device)

                loss = criterion(outputs, batch_y[:, -outputs.shape[1]:, :])

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
        if self.args.use_multi_gpu:
            weighted = torch.tensor([sum(float(loss) * count for loss, count in zip(total_loss, total_count)),
                                     sum(total_count)], dtype=torch.float64, device=self.device)
            dist.all_reduce(weighted, op=dist.ReduceOp.SUM)
            total_loss = (weighted[0] / weighted[1]).item()
        else:
            total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss

    def train(self, setting):
        self.total_train_start = time.time()
        self.total_params = sum(p.numel() for p in self.model.parameters())
        is_main_process = (not self.args.use_multi_gpu) or (self.args.use_multi_gpu and self.args.local_rank == 0)
        
        if is_main_process:
            print(f"\n[Model Stats] Total Parameters: {self.total_params}")
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        #criterion = GMMWeightedHuberLoss(gmm_priors_path, total_epochs=self.args.train_epochs)
        #criterion = TemporalDecayHuberLoss(delta=1.0)
        criterion = torch.nn.HuberLoss(delta=1.0)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            if self.args.lradj == 'sigmoid' and not self.args.cosine:
                # Apply xPatch's warm-up/decay to the epoch about to train.
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            iter_count = 0

            loss_val = torch.tensor(0., device=self.device)
            count = torch.tensor(0., device=self.device)
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if i == 0 and epoch == 0:
                    try:
                        raise ImportError('Skip optional FLOP profiling')
                        from thop import profile
                        if is_main_process:
                            print("\n[Model Stats] Calculating FLOPs with a dummy CPU sandbox...")
                        
                        dummy_model = self.model_dict[self.args.model].Model(self.args).float().cpu()
                        dummy_model.eval()

                        dummy_batch_x = torch.zeros_like(batch_x).cpu()
                        dummy_batch_x_mark = torch.zeros_like(batch_x_mark).cpu()
                        dummy_batch_y_mark = torch.zeros_like(batch_y_mark).cpu()

                        flops, macs = profile(dummy_model, inputs=(dummy_batch_x, dummy_batch_x_mark, None, dummy_batch_y_mark), verbose=False)
                        
                        self.flops = flops
                        if is_main_process:
                            print(f"[Model Stats] FLOPs: {flops}, MACs: {macs}\n")
                
                        del dummy_model, dummy_batch_x, dummy_batch_x_mark, dummy_batch_y_mark
                        
                    except ImportError:
                        self.flops = "thop_not_installed"
                        if is_main_process:
                            print("[Warning] thop is not installed. Please `pip install thop` to calculate FLOPs.")
                    except Exception as e:
                        self.flops = "calculation_failed"
                        if is_main_process:
                            print(f"[Warning] FLOPs calculation failed: {e}")
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                        loss_outputs = self._loss_window(outputs)
                        loss = criterion(loss_outputs, batch_y[:, -loss_outputs.shape[1]:, :])                        
                        #loss, w_mean, alpha = criterion(outputs, batch_y, epoch) 
                        loss_val += loss.item()
                        count += 1
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    loss_outputs = self._loss_window(outputs)
                    loss = criterion(loss_outputs, batch_y[:, -loss_outputs.shape[1]:, :])
                    #loss, w_mean, alpha = criterion(outputs, batch_y, epoch)
                    loss_val += loss.item()
                    count += 1
                
                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        #print("\titers: {0}, epoch: {1} | loss: {2:.7f} | Alpha: {3:.4f} | W_mean: {4:.4f}".format(i + 1, epoch + 1, loss.item(), alpha, w_mean.item()))
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))   
            if self.args.use_multi_gpu:
                dist.barrier()   
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)      
            train_loss = loss_val.item() / count.item()
            vail_criterion = self._select_criterion()
            vali_loss = self.vali(vali_data, vali_loader, vail_criterion)
            test_loss = self.vali(test_data, test_loader, vail_criterion, is_test=True)
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            elif self.args.lradj != 'sigmoid':
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.use_multi_gpu:
                train_loader.sampler.set_epoch(epoch + 1)
                
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
            
        self.total_train_time = time.time() - self.total_train_start
        if is_main_process:
            print(f"\n[Model Stats] Total Training Time: {self.total_train_time:.4f} seconds")
            
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.test_label_len, self.args.token_len, self.args.test_pred_len)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name

            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            load_item = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()
        eval_model = self.model.module if self.args.use_multi_gpu else self.model
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                inference_steps = self.args.test_pred_len // self.args.token_len
                dis = self.args.test_pred_len - inference_steps * self.args.token_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.token_len:, :], pred_y[-1]], dim=1)
                        if batch_x_mark.ndim == 3:
                            tmp = batch_y_mark[:, j-1:j, :]
                            batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)
                        # Four-dimensional prompts stay tied to the original observed window.
                        
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = eval_model(batch_x, batch_x_mark, None, batch_y_mark)
                    else:
                        outputs = eval_model(batch_x, batch_x_mark, None, batch_y_mark)
                    pred_y.append(outputs[:, -self.args.token_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-(self.args.token_len - dis), :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)
                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    
                if self.args.visualize and i == 0:
                    gt = np.array(true[0, :, -1])
                    pd = np.array(pred[0, :, -1])
                    lookback = batch_x[0, :, -1].detach().cpu().numpy()
                    gt = np.concatenate([lookback, gt], axis=0)
                    pd = np.concatenate([lookback, pd], axis=0)
                    dir_path = folder_path + f'{self.args.test_pred_len}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    visual(gt, pd, os.path.join(dir_path, f'{i}.png'))
                    np.savetxt(os.path.join(folder_path, str(i) + '.txt'), pd)
                    np.savetxt(os.path.join(folder_path, str(i) + 'true.txt'), gt)
        
        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        if self.args.use_multi_gpu:
            indices = np.arange(self.args.local_rank, len(test_data), dist.get_world_size())
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, (indices, preds, trues))
            if self.args.local_rank != 0:
                return
            order = np.argsort(np.concatenate([part[0] for part in gathered]))
            preds = np.concatenate([part[1] for part in gathered], axis=0)[order]
            trues = np.concatenate([part[2] for part in gathered], axis=0)[order]
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}\n'.format(mse, mae))
        t_params = getattr(self, "total_params", "N/A")
        t_flops = getattr(self, "flops", "N/A")
        t_time = getattr(self, "total_train_time", "N/A")
        if isinstance(t_time, float):
            t_time = f"{t_time:.4f}s"
        f.write(f'Total Parameters: {t_params}\n')
        f.write(f'FLOPs: {t_flops}\n')
        f.write(f'Total Training Time: {t_time}\n')
        f.write('\n\n')  # 多加一个换行，让不同实验的日志区分更明显
        f.close()
        
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        
        return
