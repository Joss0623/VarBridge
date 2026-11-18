import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe
import warnings

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, seasonal_patterns=None, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
            

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        data_name = self.data_path.split('.')[0]
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}.pt'))
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_DYG(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='DYG_source1.csv',
                 scale=True, seasonal_patterns=None, drop_short=False):
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.token_len = self.seq_len - self.label_len
        
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        
        self.__read_data__()
        
      
        self.enc_in = self.data_list[0].shape[-1]
        self.tot_samples_per_feat = len(self.valid_indices)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        data_name = self.data_path.split('.')[0]
        pt_file_path = os.path.join(self.root_path, f'{data_name}.pt')
        if os.path.exists(pt_file_path):
            full_stamp_vecs = torch.load(pt_file_path)
        else:
            full_stamp_vecs = None 
            
        df_raw['tm'] = pd.to_datetime(df_raw['tm'])
        diff_hours = df_raw['tm'].diff().dt.total_seconds() / 3600.0
        
        reset_indices = df_raw[diff_hours > 1.5].index.tolist()
        raw_batch_starts = [0] + reset_indices
        raw_batch_ends = reset_indices + [len(df_raw)]
        
        window_size = self.seq_len + self.pred_len
        
        batch_starts = []
        batch_ends = []
        for s, e in zip(raw_batch_starts, raw_batch_ends):
            if (e - s) >= window_size:
                batch_starts.append(s)
                batch_ends.append(e)
                
        num_batches = len(batch_starts)
        if num_batches == 0:
            raise ValueError(f"致命错误：整个数据集中没有哪怕一罐的发酵时间大于你的窗口要求 ({window_size}小时)！请减小 seq_len 或 pred_len。")

  
        num_train = int(num_batches * 0.7)
        num_test = int(num_batches * 0.2)
       
        if num_test == 0 and num_batches >= 2: num_test = 1
        num_vali = num_batches - num_train - num_test
        border_batches = [0, num_train, num_train + num_vali, num_batches]
        
        if self.scale:
            train_df_list = []
            for i in range(border_batches[0], border_batches[1]):
                train_df_list.append(df_raw.iloc[batch_starts[i]:batch_ends[i]])
            train_combined = pd.concat(train_df_list).drop(columns=['tm']) 
            self.scaler.fit(train_combined.values)
        
        self.data_list = []      
        self.stamp_list = []     
        self.valid_indices = []  
        
        for b_idx in range(border_batches[self.set_type], border_batches[self.set_type+1]):
            s, e = batch_starts[b_idx], batch_ends[b_idx]
            
            batch_data = df_raw.iloc[s:e].drop(columns=['tm']).values
            if self.scale:
                batch_data = self.scaler.transform(batch_data)
            
            self.data_list.append(batch_data)
            
            if full_stamp_vecs is not None:
                self.stamp_list.append(full_stamp_vecs[s:e]) 
            else:
                self.stamp_list.append([None] * (e - s))
            
            current_batch_pos = len(self.data_list) - 1
            for start_pos in range(len(batch_data) - window_size + 1):
                self.valid_indices.append((current_batch_pos, start_pos))

    def __getitem__(self, index):
        sample_idx = index 
        list_batch_idx, s_begin = self.valid_indices[sample_idx]
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_list[list_batch_idx][s_begin:s_end, :] 
        seq_y = self.data_list[list_batch_idx][r_begin:r_end, :] 
        
        if self.stamp_list[list_batch_idx][0] is not None:
            seq_x_mark = self.stamp_list[list_batch_idx][s_begin:s_end:self.token_len]
            seq_y_mark = self.stamp_list[list_batch_idx][s_end:r_end:self.token_len]
        else:
            seq_x_mark = torch.zeros((self.seq_len // self.token_len, 1))
            seq_y_mark = torch.zeros(((self.label_len + self.pred_len) // self.token_len, 1))
        
        return torch.tensor(seq_x).float(), torch.tensor(seq_y).float(), seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.valid_indices)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 seasonal_patterns=None, scale=True, drop_short=False):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None, data_path='ETTh1.csv',
                 scale=False, inverse=False, seasonal_patterns='Yearly', drop_short=False):
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask

class Dataset_TSF(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=False):
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.pred_len
        self.context_len = 4 * self.token_len
        print(self.seq_len, self.label_len, self.pred_len)
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.drop_short = drop_short
        self.timeseries = self.__read_data__()


    def __read_data__(self):
        df, _, _, _, _ = convert_tsf_to_dataframe(os.path.join(self.root_path, self.data_path))
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        if self.drop_short:
            timeseries = [ts for ts in timeseries if ts.shape[0] > self.context_len]
        self.tot_len = 0
        self.len_seq = []
        self.seq_id = []
        for i in range(len(timeseries)):
            res_len = max(self.pred_len + self.seq_len - timeseries[i].shape[0], 0)
            pad_zeros = np.zeros(res_len)
            timeseries[i] = np.hstack([pad_zeros, timeseries[i]])

            _len = timeseries[i].shape[0]
            train_len = _len-self.pred_len
            border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]
            border2s = [train_len - self.pred_len,  train_len,                                _len]
            
            curr_len = border2s[self.set_type] - max(border1s[self.set_type], 0) - self.pred_len - self.seq_len + 1
            curr_len = max(0, curr_len)
            
            self.len_seq.append(np.zeros(curr_len) + self.tot_len)
            self.seq_id.append(np.zeros(curr_len) + i)
            self.tot_len += curr_len
            
        self.len_seq = np.hstack(self.len_seq)
        self.seq_id = np.hstack(self.seq_id)

        return timeseries

    def __getitem__(self, index):
        len_seq = self.len_seq[index]
        seq_id = int(self.seq_id[index])
        index = index - int(len_seq)

        _len = self.timeseries[seq_id].shape[0]
        train_len = _len - self.pred_len
        border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]

        s_begin = index + border1s[self.set_type]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        data_x = self.timeseries[seq_id][s_begin:s_end]
        data_y = self.timeseries[seq_id][r_begin:r_end]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)

        return data_x, data_y, data_x, data_y

    def __len__(self):
        return self.tot_len

class Dataset_TSF_ICL(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=True):
        
        self.pred_len = size[2]
        self.token_len = self.pred_len
        self.context_len = 4 * self.token_len

        self.root_path = root_path
        self.data_path = data_path
        self.timeseries = self.__read_data__()

    def __read_data__(self):
        df, _, _, _, _ = convert_tsf_to_dataframe(os.path.join(self.root_path, self.data_path))
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        timeseries = [ts for ts in timeseries if ts.shape[0] > self.context_len]
        return timeseries

    # we uniformly adopting the first time points of the time series as the corresponding prompt.
    def __getitem__(self, index):        
        data_x1 = self.timeseries[index][:2*self.token_len]
        data_x2 = self.timeseries[index][-2*self.token_len:-1*self.token_len]
        data_x = np.concatenate((data_x1, data_x2))
        data_y = self.timeseries[index][-1*self.token_len:]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)
        return data_x, data_y, data_x, data_y

    def __len__(self):
        return len(self.timeseries)

class Dataset_Preprocess(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTh1.csv', scale=True, seasonal_patterns=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        self.data_set_type = data_path.split('.')[0]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.tot_len = len(self.data_stamp)

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        if self.data_set_type in ['DYG_source1', 'DYG_target1']:
            df_stamp = df_raw[['tm']].copy()
            df_stamp['tm'] = pd.to_datetime(df_stamp.tm).apply(str)
            self.data_stamp = df_stamp['tm'].values
            
        else:
            df_stamp = df_raw[['date']]
            df_stamp['date'] = pd.to_datetime(df_stamp.date).apply(str)
            self.data_stamp = df_stamp['date'].values
        
        self.data_stamp = [str(x) for x in self.data_stamp]
        

    def __getitem__(self, index):
        s_begin = index % self.tot_len
    
        s_end = (s_begin + self.token_len - 1) % self.tot_len
        
        if self.data_set_type in ['DYG_new_sub1', 'DYG_new_sub2']:
            start_time = self.data_stamp[s_begin]
            end_time = self.data_stamp[s_end]

            if self.data_set_type == 'DYG_new_sub1':
                data_description = (
                    "Industrial erythromycin fermentation; this physicochemical group "
                    "contains total sugar, reducing sugar, and amino nitrogen."
                )
                process_knowledge = (
                    "Hydrolysis of complex sugars can replenish reducing sugar. "
                    "Microbial growth and product formation consume available carbon "
                    "and amino nitrogen; changing sugar-consumption rates may signal "
                    "feeding demand or substrate depletion."
                )
            else:
                data_description = (
                    "Industrial erythromycin fermentation; this biochemical group "
                    "contains biomass concentration, broth viscosity, and chemical efficiency."
                )
                process_knowledge = (
                    "Biomass accumulation can increase broth viscosity. Higher viscosity "
                    "may impede oxygen transfer and influence metabolic conversion; "
                    "these tendencies depend on the operating state."
                )

            seq_x_mark = (
                f"<|start_prompt|> Data description: {data_description} "
                f"Task description: Forecast future process values from the preceding "
                f"{self.seq_len} observations; the evaluation horizon is specified separately. "
                f"Input context: This numerical input patch runs from {start_time} "
                f"to {end_time}. "
                f"Process knowledge: {process_knowledge} <|end_prompt|>"
            )

        elif self.data_set_type in ['DYG_source1', 'DYG_target1']:
            start_time = self.data_stamp[s_begin]
            end_time = self.data_stamp[s_end]
            
            domain_desc = "source" if self.data_set_type == 'DYG_source1' else "target"
            
            seq_x_mark = (
                f"Industrial fermentation bioprocess ({domain_desc} domain) dynamics from {start_time} to {end_time}. "
                f"This segment captures the multivariate metabolic perturbations "
                f"and operational shifts (such as OUR, CER, RQ, and kLa) over this specific period."
            )
            
        else:
            start_time = self.data_stamp[s_begin]
            end_time = self.data_stamp[s_end]
            seq_x_mark = f"This is Time Series from {start_time} to {end_time}"
        
        return seq_x_mark

    def __len__(self):
        return len(self.data_stamp)
class Dataset_BatchMTSF(Dataset):
    """Batch-disjoint multivariate windows; one sample contains every process variable."""
    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=False,
                 prompt_encoder='gpt2'):
        import json
        assert flag in ('train', 'val', 'test')
        self.seq_len, self.label_len, self.pred_len = map(int, size)
        self.token_len = self.seq_len - self.label_len
        if self.token_len <= 0 or self.seq_len % self.token_len:
            raise ValueError('seq_len must be divisible by positive token_len')
        self.root_path, self.data_path = root_path, data_path
        raw = pd.read_csv(os.path.join(root_path, data_path))
        name = os.path.splitext(data_path)[0]
        if name in ('EFP_long_sub1','EFP_long_sub2','EFP_long_sub3'):
            if raw['batch_id'].isna().any():
                raise ValueError('Missing EFP batch_id')
            ordered = raw['batch_id'].drop_duplicates().tolist()
            ranges = {}
            for bid in ordered:
                positions = np.flatnonzero(raw['batch_id'].to_numpy() == bid)
                if positions[-1]-positions[0]+1 != len(positions):
                    raise ValueError(f'Noncontiguous batch {bid}')
                ranges[bid] = (int(positions[0]),int(positions[-1])+1)
            # Common eligibility for both 48->24 and 48->48.
            eligible = [bid for bid in ordered if ranges[bid][1]-ranges[bid][0] >= 96]
            n_train = int(len(eligible)*0.7)
            n_val = int(len(eligible)*0.1)
            split_ids = {'train':eligible[:n_train], 'val':eligible[n_train:n_train+n_val],
                         'test':eligible[n_train+n_val:]}
            self.feature_names = [c for c in raw.columns if c not in ('date','batch_id','hh')]
        else:
            raise ValueError(f'Unsupported batch dataset: {name}')
        train_rows = np.concatenate([np.arange(*ranges[bid]) for bid in split_ids['train']])
        data = raw[self.feature_names].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
        if not np.isfinite(data).all():
            raise ValueError('Nonfinite numerical features in data')
        self.scaler = StandardScaler() if scale else None
        if scale:
            self.scaler.fit(data[train_rows])
            data = self.scaler.transform(data)
        self.data = torch.tensor(data, dtype=torch.float32)
        text_encoder, marker, window_mode = prompt_encoder.partition('_window_')
        self.window_mode = bool(marker and window_mode)
        if self.window_mode:
            structured = window_mode.startswith('structured_horizon_')
            self.horizon_semantic = window_mode.startswith('horizon_') or structured
            semantic = window_mode.startswith('semantic_') or self.horizon_semantic
            prefix = ('structured_horizon_' if structured else
                      'horizon_' if self.horizon_semantic else
                      'semantic_' if semantic else '')
            variant = window_mode[len(prefix):]
            cache_name = text_encoder + ('_window_structured' if structured else
                                         '_window_semantic' if semantic else '_window')
            if semantic and self.pred_len in (6, 12):
                cache_name += f'_h{self.pred_len}'
            path = os.path.join(root_path, f'{name}__{cache_name}.pt')
            cached = torch.load(path, map_location='cpu')
            if cached['features'] != self.feature_names or variant not in cached['variants']:
                raise ValueError(f'Window prompt cache mismatch: {path}')
            self.window_lookup = {int(start):i for i,start in enumerate(cached['starts'].tolist())}
            self.window_embeddings = cached['window_embeddings']
            self.rule_embeddings = cached['rule_embeddings'][cached['variants'].index(variant)]
            if self.horizon_semantic:
                if 'knowledge_embeddings' not in cached:
                    raise ValueError(f'Missing separate knowledge tokens in {path}; rerun preprocessing')
                self.knowledge_embeddings = cached['knowledge_embeddings'][cached['variants'].index(variant)]
        else:
            path = os.path.join(root_path, f'{name}__{prompt_encoder}.pt')
            stamps = torch.load(path, map_location='cpu')
            if stamps.ndim != 2 or len(stamps) != len(raw):
                raise ValueError(f'Prompt embedding shape mismatch: {path}')
            self.stamps = stamps.float()
        self.valid_indices = []
        for bid in split_ids[flag]:
            s,e = ranges[bid]
            for start in range(s, e-self.seq_len-self.pred_len+1):
                self.valid_indices.append(start)
        self.enc_in = len(self.feature_names)
        self.batch_ids = split_ids
        self.selected_batches = split_ids[flag]
        self.name = name

    def __getitem__(self, index):
        s = self.valid_indices[index]
        x_end = s+self.seq_len
        y_begin = x_end-self.label_len
        y_end = x_end+self.pred_len
        x = self.data[s:x_end]
        y = self.data[y_begin:y_end]
        if self.window_mode:
            pos = self.window_lookup[s]
            if self.horizon_semantic:
                bank = self.knowledge_embeddings[:, 0, :].unsqueeze(0).expand(self.enc_in, -1, -1)
                xm = torch.cat((self.window_embeddings[pos, :, None, :], bank,
                                self.knowledge_embeddings[:, 1:, :]), dim=1).float()
            else:
                xm = torch.stack((self.rule_embeddings, self.window_embeddings[pos]), dim=1).float()
            ym = xm
        else:
            xm = self.stamps[s:x_end:self.token_len]
            ym = self.stamps[x_end:y_end:self.token_len]
        return x,y,xm,ym

    def __len__(self):
        return len(self.valid_indices)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data) if self.scaler is not None else data
