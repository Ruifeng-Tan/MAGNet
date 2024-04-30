from data_provider.data_loader import Dataset_Pred, Dataset_Battery_cycle_ShortLongMove, Dataset_Battery_cycle_ShortLongMove_DA
from torch.utils.data import DataLoader

data_dict = {
    'Batteries_cycle_SLMove': Dataset_Battery_cycle_ShortLongMove,
    'DA': Dataset_Battery_cycle_ShortLongMove_DA
    
}


def data_provider_meta(args, flag, set_data):
    Data = data_dict[args.data] if not set_data else data_dict[set_data]
    timeenc = 0 if args.embed != 'timeF' else 1
    freq = args.freq
    shuffle_flag = False
    if flag == 'test':
        batch_size = args.batch_size
        drop_last = False
    elif flag == 'pred' or flag == 'set_files':
        batch_size = 1
        drop_last = False

    data_set = Data(
        args=args,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        scale=args.scale
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader


def data_provider(args, flag, set_data):
    Data = data_dict[args.data] if not set_data else data_dict[set_data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred' or flag == 'set_files':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        if 'Batteries' not in args.data:
            Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    if args.FT and flag != 'set_files':
        batch_size = 16
    if 'set_files' in args.__dict__:
        data_set = Data(
            args=args,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            scale=args.scale,
            set_files=args.set_files
        )
    else:
        data_set = Data(
            args=args,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            scale=args.scale
        )
    # print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
