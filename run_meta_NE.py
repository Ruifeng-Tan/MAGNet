import random
import numpy as np
import os
import torch
import datetime
import json
# fix_seed = 2021
# random.seed(fix_seed)
# np.random.seed(fix_seed)
# torch.manual_seed(fix_seed)
# torch.cuda.manual_seed(fix_seed)
# torch.cuda.manual_seed_all(fix_seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)


set_seed(2021)

import argparse
from exp.exp_main import Exp_Main


def main():
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='NE', help='model id')
    parser.add_argument('--model', type=str, default='Informer',
                        help='model name, options: [Autoformer, Informer, Transformer, PIInformer, OneShotLSTM, OSLSTMv2]')

    # data loader
    parser.add_argument('--data', type=str, default='Batteries_cycle_SLMove', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/NatureEnergy_cycle_data/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--scale', action='store_true', default=True, help='scale the input')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=20, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=20, help='start token length')
    parser.add_argument('--pred_len', type=int, default=500, help='prediction sequence length')

    # model define
    parser.add_argument('--bucket_size', type=int, default=48, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=2, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=2, help='output size')
    parser.add_argument('--d_model', type=int, default=12, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=3, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=4, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=15, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=5, help='attn factor')
    parser.add_argument('--factor2', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, setting False means not using distilling',
                        default=False)
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--embed', type=str, default='Cycle',
                        help='time features encoding, options:[timeF, fixed, learned, SOC_Cycle, Cycle]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', default=False,
                        help='whether to predict unseen future data')

    # optimization
    
    parser.add_argument('--FT', action='store_true', default=False, help='Set True to start fine-tune')
    parser.add_argument('--val_ratio', type=float, default=0.5, help='the ratio of validation data in the training set')
    parser.add_argument('--iterations', type=float, default=10000, help='The maximum of total training iterations')
    parser.add_argument('--test_every', type=float, default=500, help='validate the model on validation set every xxx iterations')
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--wd', type=float, default=0, help='weight decay in Adam')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='optimizer learning rate') # meta
    parser.add_argument('--meta_learning_rate', type=float, default=0.0075, help='meta optimizer learning rate') # meta
    parser.add_argument('--meta_beta', type=float, default=2, help='weight of the test domain loss')
    parser.add_argument('--auxiliary_gamma', type=float, default=0.2, help='weight of the test domain loss')
    parser.add_argument('--meta_train', action='store_true', default=True, help='set True to use meta learning') # meta
    parser.add_argument('--lr_align', action='store_true', default=False, help='set True to align the lrs of meta and clone')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='awmse', help='loss function options:[mse,wmse,awmse]') # meta
    parser.add_argument('--vali_loss', type=str, default='nw', help='loss function options:[w,nw]')
    parser.add_argument('--gamma1', type=float, default=0.0, help='weight for proportion loss')
    parser.add_argument('--gamma2', type=float, default=0.0, help='weight for voltage limitation loss')
    parser.add_argument('--lradj', type=str, default='type4', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='4,5,6,7', help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if 'NC' in args.root_path:
        args.nominal_capacity = 3.5
    else:
        args.nominal_capacity = 1.1
    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    if args.meta_train:
        args.model_id += '_meta'
        if args.lr_align:
            args.model_id += '_metaAlign'
    print('Args in experiment:')
    print(args)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    Exp = Exp_Main  
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_lr{}_metalr{}_mavg{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_fc2{}_eb{}_dt{}_valratio{}_{}_loss{}_valloss{}_dp{}_bs{}_wd{}_mb{}_agamma{}_lradj{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.learning_rate,
                args.meta_learning_rate,
                args.moving_avg,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.factor2,
                args.embed,
                args.distil,
                args.val_ratio,
                args.des, args.loss, args.vali_loss,args.dropout, args.batch_size, args.wd, args.meta_beta, args.auxiliary_gamma,args.lradj, ii)
            os.makedirs(f'./results/{setting}', exist_ok=True)
            with open(f'./results/{setting}/args.txt', 'w') as f:
                json.dump(args.__dict__, f)
                
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            if not args.meta_train:
                exp.train(setting)
            else:
                exp.train_meta_pre_clustering_parallel_new_robust(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test(setting, test=True)
            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
