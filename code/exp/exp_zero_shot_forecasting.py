from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.losses import zero_shot_smape_loss
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np


warnings.filterwarnings('ignore')

def SMAPE(pred, true):
    return np.mean(2 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))
def MAPE(pred, true):
    return np.mean(np.abs(1 * (pred - true) / (true +1e-8)))

class Exp_Zero_Shot_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Zero_Shot_Forecast, self).__init__(args)

    def _build_model(self):
        self.device = self.args.gpu
        model = self.model_dict[self.args.model].Model(self.args).to(self.device)
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
                print(n, p.dtype, p.shape)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'SMAPE':
            return zero_shot_smape_loss()

    def train(self, setting):
        self.total_train_start = time.time()
        self.total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n[Model Stats] Total Parameters: {self.total_params}")

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        self.args.data_path = self.args.test_data_path
        test_data2, test_loader2 = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion(self.args.loss)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if i == 0 and epoch == 0:
                    try:
                        from thop import profile
                        print("\n[Model Stats] Calculating FLOPs with a dummy CPU sandbox...")
                        
                        dummy_model = self.model_dict[self.args.model].Model(self.args).float().cpu()
                        dummy_model.eval()

                        dummy_batch_x = torch.zeros_like(batch_x).cpu()

                        flops, macs = profile(dummy_model, inputs=(dummy_batch_x, None, None, None), verbose=False)
                        
                        self.flops = flops
                        print(f"[Model Stats] FLOPs: {flops}, MACs: {macs}\n")
                        
                        del dummy_model, dummy_batch_x
                        
                    except ImportError:
                        self.flops = "thop_not_installed"
                        print("[Warning] thop is not installed. Please `pip install thop` to calculate FLOPs.")
                    except Exception as e:
                        self.flops = "calculation_failed"
                        print(f"[Warning] FLOPs calculation failed: {e}")

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, None, None, None)
                else:
                    outputs = self.model(batch_x, None, None, None)
                    
                pred = outputs
                true = batch_y
                
                pred = pred[:, -self.args.token_len:, :]
                true = true[:, -self.args.token_len:, :]
                
                align_len = min(pred.shape[1], true.shape[1])
                
                pred = pred[:, :align_len, :]
                true = true[:, :align_len, :]
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)  
            test_loss = self.vali2(test_data, test_loader, criterion)    # test_loss indicates the result on the source datasets, 
            test_loss2 = self.vali2(test_data2, test_loader2, criterion) # test_loss2 indicates the result on the taregt datasets. The latter is what we concerned in zero-shot forecasting​.
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} Test Loss2: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss, test_loss2))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + f'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path), strict=False)

        self.total_train_time = time.time() - self.total_train_start
        print(f"\n[Model Stats] Total Training Time: {self.total_train_time:.4f} seconds")

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, None, None, None)
                else:
                    outputs = self.model(batch_x, None, None, None)
                    
                pred = outputs
                true = batch_y
                
                pred = pred[:, -self.args.token_len:, :]
                true = true[:, -self.args.token_len:, :]
                
                align_len = min(pred.shape[1], true.shape[1])
                
                pred = pred[:, :align_len, :].detach().cpu()
                true = true[:, :align_len, :].detach().cpu()
    
                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        
        return total_loss

    def vali2(self, vali_data, vali_loader, criterion):
        total_loss = []
        count= []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                
                inference_steps = self.args.test_pred_len // self.args.token_len
                dis = self.args.test_pred_len - inference_steps * self.args.token_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []

                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.token_len:, :], pred_y[-1]], dim=1)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, None, None, None)
                    else:
                        outputs = self.model(batch_x, None, None, None)
                    pred_y.append(outputs[:, -self.args.token_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-dis, :]
                
                outputs = pred_y
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
                count.append(batch_x.shape[0])
        total_loss = np.average(total_loss, weights=count)
        self.model.train()
        
        return total_loss    

    def test(self, setting, test=0):
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, best_model_path)), strict=False)
        
        self.args.data_path = self.args.test_data_path
        test_data, test_loader = self._get_data('test')

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                inference_steps = self.args.test_pred_len // self.args.token_len
                dis = self.args.test_pred_len - inference_steps * self.args.token_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []

                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.token_len:, :], pred_y[-1]], dim=1)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, None, None, None)
                    else:
                        outputs = self.model(batch_x, None, None, None)
                    pred_y.append(outputs[:, -self.args.token_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-dis, :]
                
                outputs = pred_y
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        smape = SMAPE(preds, trues)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{:.4f}, mae:{:.4f}, mape:{:.4f}, smape:{:.4f}'.format(mse, mae, mape, smape))

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
             os.makedirs(folder_path)

        f = open("result_zero_shot_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{:.4f}, mae:{:.4f}\n'.format(mse, mae))
        f.write('mape:{:.4f}, smape:{:.4f}\n'.format(mape, smape))
        
        t_params = getattr(self, "total_params", "N/A")
        t_flops = getattr(self, "flops", "N/A")
        t_time = getattr(self, "total_train_time", "N/A")
        if isinstance(t_time, float):
            t_time = f"{t_time:.4f}s"
            
        f.write(f'Total Parameters: {t_params}\n')
        f.write(f'FLOPs: {t_flops}\n')
        f.write(f'Total Training Time: {t_time}\n')
        f.write('\n\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mse, mae, mape, smape]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return