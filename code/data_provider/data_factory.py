from data_provider.data_loader import Dataset_Custom, Dataset_DYG, Dataset_BatchMTSF, Dataset_M4, Dataset_Solar, Dataset_TSF, Dataset_TSF_ICL
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

class ExactDistributedEvalSampler(Sampler):
    """Shard validation/test rows across ranks without padding duplicates."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return len(range(self.rank, len(self.dataset), self.world_size))

data_dict = {
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'Solar': Dataset_Solar,
    'DYG': Dataset_DYG,
    'EFP_long': Dataset_BatchMTSF,
    'BatchMTSF': Dataset_BatchMTSF,
    'tsf': Dataset_TSF,
    'tsf_icl': Dataset_TSF_ICL
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size 
    elif flag == 'val':
        shuffle_flag = args.val_set_shuffle
        drop_last = False
        batch_size = args.batch_size 
    else:
        shuffle_flag = True
        drop_last = args.drop_last
        batch_size = args.batch_size

    extra = {'prompt_encoder': args.prompt_encoder} if Data is Dataset_BatchMTSF else {}

    if flag in ['train', 'val']:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.token_len],
            seasonal_patterns=args.seasonal_patterns,
            drop_short=args.drop_short,
            **extra,
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.test_seq_len, args.test_label_len, args.test_pred_len],
            seasonal_patterns=args.seasonal_patterns,
            drop_short=args.drop_short,
            **extra,
        )
    if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
        print(flag, len(data_set))
    if args.use_multi_gpu:
        train_datasampler = (DistributedSampler(data_set, shuffle=shuffle_flag)
                             if flag == 'train' else ExactDistributedEvalSampler(data_set))
        data_loader = DataLoader(data_set, 
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
            )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    return data_set, data_loader
