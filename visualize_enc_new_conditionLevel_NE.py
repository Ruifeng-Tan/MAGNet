import json
import os.path
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from data_provider.data_loader import fixed_files
from matplotlib import cm
from sklearn.decomposition import PCA
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
fix_seed = 2021
random.seed(fix_seed)
np.random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import fitlog
import argparse

from sklearn.manifold import TSNE
from exp.exp_main import Exp_Main

def set_ax_linewidth(ax, bw=1.5):
    ax.spines['bottom'].set_linewidth(bw)
    ax.spines['left'].set_linewidth(bw)
    ax.spines['top'].set_linewidth(bw)
    ax.spines['right'].set_linewidth(bw)

def check_policy_avg_life(policy_of_interest, name_policy, return_cell_number=False):
    cycle_life = []
    for name, policy in name_policy.items():
        if policy == policy_of_interest:
            df = pd.read_csv(f'./dataset/NatureEnergy_cycle_data/{name}.csv')
            cycle_life.append(len(df))
    if return_cell_number:
        return np.mean(cycle_life), len(cycle_life)
    return np.mean(cycle_life)

def main():
    fitlog.set_log_dir("./logs/")
    seen_threshold = 520
    seen_threshold_upper = 600
    contour_levels = [0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0]
    parser = argparse.ArgumentParser()
    # basic config
    parser.add_argument('--args_path', type=str,
                        default='./results/NE_meta_Informer_Batteries_cycle_SLMove_lr1e-06_metalr0.0075_mavg15_ftM_sl20_ll20_pl500_dm12_nh4_el2_dl2_df4_fc5_fc21_ebCycle_dtFalse_valratio0.5_test_lossawmse_vallossnw_dp0.0_bs128_wd0_mb2_agamma0.2_lradjtype4_0',
                        help='status')
    parser.add_argument('--Meta', action='store_true',
                        default=True,
                        help='status')
    tmp_args = parser.parse_args()
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(f'{tmp_args.args_path}/args.txt', 'r') as f:
        args.__dict__ = json.load(f)
        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        if args.use_gpu and args.use_multi_gpu:
            args.dvices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        if 'NC' in args.root_path:
            args.nominal_capacity = 3.5
        else:
            args.nominal_capacity = 1.1
        if args.root_path == './dataset/NC_NCM_autoformer_cycle_data/':
            args.set_files = [i for i in fixed_files.NC_test_files if 'NCM' in i]
        elif args.root_path == './dataset/NC_NCA_autoformer_cycle_data/':
            args.set_files = [i for i in fixed_files.NC_test_files if 'NCA' in i]
        else:
            args.set_files = fixed_files.NE_test_files
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
            args.des, args.loss, args.vali_loss, args.dropout, args.batch_size, args.wd, args.meta_beta,
            args.auxiliary_gamma, args.lradj, 0)

    if not tmp_args.Meta:
        Exp = Exp_Main
    else:
        # Exp = Exp_Meta_Main
        Exp = Exp_Main
    exp = Exp(args)  # set experiments
    # exp.predict_specific_cells(setting,load=True,set_dataset='Batteries_cycle_containShort')
    set_files = args.set_files
    root_path = args.root_path
    seen_files = fixed_files.NE_train_files
    total_enc_outs = []
    total_unseen_enc_outs = {}
    total_SOHs = []
    max_color = 1
    min_color = 0.8
    for file in set_files:
        # unseen
        condition = file.split('.')[0]  # condition is the file name for nature energy dataset

        args.set_files = [file]
        exp.args = args
        enc_outs, ruls, labels = exp.visualize_enc_out_new(setting, load=True,
                                                   save_path='', return_labels=True)
        SOHs = np.array(labels) / args.nominal_capacity
        total_SOHs = total_SOHs + list(SOHs)
        enc_outs = np.concatenate(enc_outs, axis=0)
        total_enc_outs = total_enc_outs + [enc_outs]
        total_unseen_enc_outs[condition] = [enc_outs, ruls, SOHs]



    total_seen_enc_outs = {}
    for file in seen_files:
        # seen
        condition = file.split('.')[0]  # condition is the file name for nature energy dataset
        args.set_files = [file]
        exp.args = args
        enc_outs, ruls, labels = exp.visualize_enc_out_new(setting, load=True,
                                                   save_path='', return_labels=True)
        SOHs = np.array(labels) / args.nominal_capacity
        total_SOHs = total_SOHs + list(SOHs)
        enc_outs = np.concatenate(enc_outs, axis=0)
        total_enc_outs = total_enc_outs + [enc_outs]
        total_seen_enc_outs[condition] = [enc_outs, ruls, SOHs]

    condition_colormaps = [cm.get_cmap(name='Oranges'), cm.get_cmap(name='Blues'),
                            cm.get_cmap(name='Purples'), cm.get_cmap(name='Reds'), 
                            cm.get_cmap(name='Greens'),cm.get_cmap(name='copper'),
                            cm.get_cmap(name='Greys')]

    total_enc_outs = np.concatenate(total_enc_outs, axis=0)
    pca = PCA(n_components=2, random_state=2021)
    # tsne = TSNE(n_components=2, init='pca', verbose=1, perplexity=5, random_state=2021)
    # embedd = tsne.fit_transform(total_enc_outs)  # decrease the dimension
    embedd = pca.fit_transform(total_enc_outs)  # decrease the dimension
    interp = LinearNDInterpolator(embedd, np.array(total_SOHs))
    print('Get the interpolation of SOH')
    min_embedd = np.min(embedd, axis=0)
    max_embedd = np.max(embedd, axis=0)
    # generate the meshgrid to plot the contour line
    x = np.arange(min_embedd[0], max_embedd[0], 1)
    y = np.arange(min_embedd[1], max_embedd[1], 1)
    X,Y = np.meshgrid(x,y)
    Z = interp(X, Y)
    if np.any(np.isnan(Z)):
        print('MAGNet: There is nan in Z!')
    fig = plt.figure(figsize=(10, 4)) 
    # We plot the visulization for MAGNet
    # Firstly, we plot the seen and unseen background
    name_policy = json.load(open('./dataset/name_policy.json'))
    plt.subplot(121)
    Grey_data = {'scaled_SOH': [],
                 'First principal component': [],
                 'Second principal component': []}
    for condition, value in total_seen_enc_outs.items():
        policy = name_policy[condition]
        if check_policy_avg_life(policy, name_policy) >= 1000:
            embedd = pca.transform(value[0])
            tmp_cm = condition_colormaps[6] # Grey color
            SOHs = value[2]
            max_SOH = max(SOHs)
            min_SOH = min(SOHs)
            # # max_color = max_SOH * 0.8
            # min_color = 0.2
            SOH_std = np.array([(i-min_SOH) / (max_SOH - min_SOH) for i in SOHs])
            SOH_scaled = list(SOH_std * (max_color-min_color) + min_color)
            colors = [tmp_cm(i) for i in SOH_scaled]
            Grey_data['scaled_SOH'] = Grey_data['scaled_SOH'] + SOH_scaled
            Grey_data['First principal component'] = Grey_data['First principal component'] + list(embedd[:, 0])
            Grey_data['Second principal component'] = Grey_data['Second principal component'] + list(embedd[:, 1])
            plt.scatter(embedd[:, 0], embedd[:, 1], c=colors)
    Grey_df = pd.DataFrame(Grey_data)
    Grey_df.to_csv('./fig4/LFP_scatter_data/MAGNet_grey.csv', index=False)
    
    Green_data = {'scaled_SOH': [],
                 'First principal component': [],
                 'Second principal component': []}
    for condition, value in total_seen_enc_outs.items():
        policy = name_policy[condition]
        if check_policy_avg_life(policy, name_policy) <= seen_threshold:
            embedd = pca.transform(value[0])
            tmp_cm = condition_colormaps[4] # Green color
            SOHs = value[2]
            max_SOH = max(SOHs)
            min_SOH = min(SOHs)
            # # max_color = max_SOH * 0.8
            # min_color = 0.2
            SOH_std = np.array([(i-min_SOH) / (max_SOH - min_SOH) for i in SOHs])
            SOH_scaled = list(SOH_std * (max_color-min_color) + min_color)
            colors = [tmp_cm(i) for i in SOH_scaled]
            Green_data['scaled_SOH'] = Green_data['scaled_SOH'] + SOH_scaled
            Green_data['First principal component'] = Green_data['First principal component'] + list(embedd[:, 0])
            Green_data['Second principal component'] = Green_data['Second principal component'] + list(embedd[:, 1])
            plt.scatter(embedd[:, 0], embedd[:, 1], c=colors)
    Green_df = pd.DataFrame(Green_data)
    Green_df.to_csv('./fig4/LFP_scatter_data/MAGNet_green.csv', index=False)
    
    Blue_data = {'scaled_SOH': [],
                 'First principal component': [],
                 'Second principal component': []}
    for condition, value in total_unseen_enc_outs.items():
        policy = name_policy[condition]
        if check_policy_avg_life(policy, name_policy) > 332:
            embedd = pca.transform(value[0])
            tmp_cm = condition_colormaps[1] # Blue color
            SOHs = value[2]
            max_SOH = max(SOHs)
            min_SOH = min(SOHs)
            # # max_color = max_SOH * 0.8
            # min_color = 0.2
            SOH_std = np.array([(i-min_SOH) / (max_SOH - min_SOH) for i in SOHs])
            SOH_scaled = list(SOH_std * (max_color-min_color) + min_color)
            colors = [tmp_cm(i) for i in SOH_scaled]
            Blue_data['scaled_SOH'] = Blue_data['scaled_SOH'] + SOH_scaled
            Blue_data['First principal component'] = Blue_data['First principal component'] + list(embedd[:, 0])
            Blue_data['Second principal component'] = Blue_data['Second principal component'] + list(embedd[:, 1])
            plt.scatter(embedd[:, 0], embedd[:, 1], c=colors)
    Blue_df = pd.DataFrame(Blue_data)
    Blue_df.to_csv('./fig4/LFP_scatter_data/MAGNet_blue.csv', index=False)
    
    
    red_data = {'scaled_SOH': [],
                 'First principal component': [],
                 'Second principal component': []}
    for condition, value in total_unseen_enc_outs.items():
        policy = name_policy[condition]
        if check_policy_avg_life(policy, name_policy) <= 332:
            # 'b2c1.csv', 'b2c0.csv', 'b2c3.csv'
            print(condition)
            embedd = pca.transform(value[0])
            tmp_cm = condition_colormaps[3] # Red color
            SOHs = value[2]
            max_SOH = max(SOHs)
            min_SOH = min(SOHs)
            # # max_color = max_SOH * 0.8
            # min_color = 0.2
            SOH_std = np.array([(i-min_SOH) / (max_SOH - min_SOH) for i in SOHs])
            SOH_scaled = list(SOH_std * (max_color-min_color) + min_color)
            colors = [tmp_cm(i) for i in SOH_scaled]
            red_data['scaled_SOH'] = red_data['scaled_SOH'] + SOH_scaled
            red_data['First principal component'] = red_data['First principal component'] + list(embedd[:, 0])
            red_data['Second principal component'] = red_data['Second principal component'] + list(embedd[:, 1])
            plt.scatter(embedd[:, 0], embedd[:, 1], c=colors)
    red_df = pd.DataFrame(red_data)
    red_df.to_csv('./fig4/LFP_scatter_data/MAGNet_red.csv', index=False)
    # contour = plt.contour(X,Y,Z, contour_levels)
    # plt.clabel(contour,fontsize=10,colors='black')
    plt.xlabel('First principal component', fontsize=10)
    plt.ylabel('Second principal component', fontsize=10)
    plt.title('(LFP) MAGNet encoder embeddings', fontsize=12)
    set_ax_linewidth(plt.gca())

    # for Informer
    parser = argparse.ArgumentParser()
    # basic config
    parser.add_argument('--args_path', type=str,
                        default='./results/NE_Informer_Batteries_cycle_SLMove_lr0.0075_metalr0.0075_mavg15_ftM_sl20_ll20_pl500_dm8_nh4_el2_dl2_df4_fc5_fc21_ebCycle_dtFalse_valratio0.5_test_lossmse_vallossnw_dp0.0_bs128_wd0_mb2_agamma0.2_lradjtype4_0',
                        help='status')
    parser.add_argument('--Meta', action='store_true',
                        default=True,
                        help='status')
    tmp_args = parser.parse_args()
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(f'{tmp_args.args_path}/args.txt', 'r') as f:
        args.__dict__ = json.load(f)
        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        if args.use_gpu and args.use_multi_gpu:
            args.dvices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        if 'NC' in args.root_path:
            args.nominal_capacity = 3.5
        else:
            args.nominal_capacity = 1.1
        if args.root_path == './dataset/NC_NCM_autoformer_cycle_data/':
            args.set_files = [i for i in fixed_files.NC_test_files if 'NCM' in i]
        elif args.root_path == './dataset/NC_NCA_autoformer_cycle_data/':
            args.set_files = [i for i in fixed_files.NC_test_files if 'NCA' in i]
        else:
            args.set_files = fixed_files.NE_test_files
            highlight_conditions = ['b2c1.csv', 'b2c0.csv', 'b2c3.csv']
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
            args.des, args.loss, args.vali_loss, args.dropout, args.batch_size, args.wd, args.meta_beta,
            args.auxiliary_gamma, args.lradj, 0)

    if not tmp_args.Meta:
        Exp = Exp_Main
    else:
        # Exp = Exp_Meta_Main
        Exp = Exp_Main
    exp = Exp(args)  # set experiments
    # exp.predict_specific_cells(setting,load=True,set_dataset='Batteries_cycle_containShort')
    set_files = args.set_files
    root_path = args.root_path
    seen_files = fixed_files.NE_train_files
    total_enc_outs = []
    total_unseen_enc_outs = {}
    total_SOHs = []
    max_color = 1 #0.8
    min_color = 0.8 #0.2
    for file in set_files:
        # unseen
        condition = file.split('.')[0]  # condition is the file name for nature energy dataset

        args.set_files = [file]
        exp.args = args
        enc_outs, ruls, labels = exp.visualize_enc_out_new(setting, load=True,
                                                   save_path='', return_labels=True)
        SOHs = np.array(labels) / args.nominal_capacity
        total_SOHs = total_SOHs + list(SOHs)
        enc_outs = np.concatenate(enc_outs, axis=0)
        total_enc_outs = total_enc_outs + [enc_outs]
        total_unseen_enc_outs[condition] = [enc_outs, ruls, SOHs]



    total_seen_enc_outs = {}
    for file in seen_files:
        # seen
        condition = file.split('.')[0]  # condition is the file name for nature energy dataset
        args.set_files = [file]
        exp.args = args
        enc_outs, ruls, labels = exp.visualize_enc_out_new(setting, load=True,
                                                   save_path='', return_labels=True)
        SOHs = np.array(labels) / args.nominal_capacity
        total_SOHs = total_SOHs + list(SOHs)
        enc_outs = np.concatenate(enc_outs, axis=0)
        total_enc_outs = total_enc_outs + [enc_outs]
        total_seen_enc_outs[condition] = [enc_outs, ruls, SOHs]

    condition_colormaps = [cm.get_cmap(name='Oranges'), cm.get_cmap(name='Blues'),
                            cm.get_cmap(name='Purples'), cm.get_cmap(name='Reds'), 
                            cm.get_cmap(name='Greens'),cm.get_cmap(name='copper'),
                            cm.get_cmap(name='Greys')]


    total_enc_outs = np.concatenate(total_enc_outs, axis=0)
    pca = PCA(n_components=2, random_state=2021)
    # tsne = TSNE(n_components=2, init='pca', verbose=1, perplexity=5, random_state=2021)
    # embedd = tsne.fit_transform(total_enc_outs)  # decrease the dimension
    embedd = pca.fit_transform(total_enc_outs)  # decrease the dimension
    interp = LinearNDInterpolator(embedd, np.array(total_SOHs))
    print('Get the interpolation of SOH')
    min_embedd = np.min(embedd, axis=0)
    max_embedd = np.max(embedd, axis=0)
    # generate the meshgrid to plot the contour line
    x = np.arange(min_embedd[0], max_embedd[0], 1)
    y = np.arange(min_embedd[1], max_embedd[1], 1)
    X,Y = np.meshgrid(x,y)
    Z = interp(X, Y)
    if np.any(np.isnan(Z)):
        print('Informer: There is nan in Z!')
    # We plot the visulization for MAGNet
    # Firstly, we plot the seen and unseen background
    name_policy = json.load(open('./dataset/name_policy.json'))
    plt.subplot(122)
    Grey_data = {'scaled_SOH': [],
                 'First principal component': [],
                 'Second principal component': []}
    for condition, value in total_seen_enc_outs.items():
        policy = name_policy[condition]
        if check_policy_avg_life(policy, name_policy) >= 1000:
            embedd = pca.transform(value[0])
            tmp_cm = condition_colormaps[6] # Grey color
            SOHs = value[2]
            max_SOH = max(SOHs)
            min_SOH = min(SOHs)
            # # max_color = max_SOH * 0.8
            # min_color = 0.2
            SOH_std = np.array([(i-min_SOH) / (max_SOH - min_SOH) for i in SOHs])
            SOH_scaled = list(SOH_std * (max_color-min_color) + min_color)
            colors = [tmp_cm(i) for i in SOH_scaled]
            Grey_data['scaled_SOH'] = Grey_data['scaled_SOH'] + SOH_scaled
            Grey_data['First principal component'] = Grey_data['First principal component'] + list(embedd[:, 0])
            Grey_data['Second principal component'] = Grey_data['Second principal component'] + list(embedd[:, 1])
            plt.scatter(embedd[:, 0], embedd[:, 1], c=colors)
    Grey_df = pd.DataFrame(Grey_data)
    Grey_df.to_csv('./fig4/LFP_scatter_data/Informer_grey.csv', index=False)
    
    Green_data = {'scaled_SOH': [],
                 'First principal component': [],
                 'Second principal component': []}
    for condition, value in total_seen_enc_outs.items():
        policy = name_policy[condition]
        if check_policy_avg_life(policy, name_policy) <= seen_threshold:
            embedd = pca.transform(value[0])
            tmp_cm = condition_colormaps[4] # Green color
            SOHs = value[2]
            max_SOH = max(SOHs)
            min_SOH = min(SOHs)
            # # max_color = max_SOH * 0.8
            # min_color = 0.2
            SOH_std = np.array([(i-min_SOH) / (max_SOH - min_SOH) for i in SOHs])
            SOH_scaled = list(SOH_std * (max_color-min_color) + min_color)
            colors = [tmp_cm(i) for i in SOH_scaled]
            Green_data['scaled_SOH'] = Green_data['scaled_SOH'] + SOH_scaled
            Green_data['First principal component'] = Green_data['First principal component'] + list(embedd[:, 0])
            Green_data['Second principal component'] = Green_data['Second principal component'] + list(embedd[:, 1])
            plt.scatter(embedd[:, 0], embedd[:, 1], c=colors)
    Green_df = pd.DataFrame(Green_data)
    Green_df.to_csv('./fig4/LFP_scatter_data/Informer_green.csv', index=False)
    
    Blue_data = {'scaled_SOH': [],
                 'First principal component': [],
                 'Second principal component': []}
    for condition, value in total_unseen_enc_outs.items():
        policy = name_policy[condition]
        if check_policy_avg_life(policy, name_policy) > 332:
            embedd = pca.transform(value[0])
            tmp_cm = condition_colormaps[1] # Blue color
            SOHs = value[2]
            max_SOH = max(SOHs)
            min_SOH = min(SOHs)
            # # max_color = max_SOH * 0.8
            # min_color = 0.2
            SOH_std = np.array([(i-min_SOH) / (max_SOH - min_SOH) for i in SOHs])
            SOH_scaled = list(SOH_std * (max_color-min_color) + min_color)
            colors = [tmp_cm(i) for i in SOH_scaled]
            Blue_data['scaled_SOH'] = Blue_data['scaled_SOH'] + SOH_scaled
            Blue_data['First principal component'] = Blue_data['First principal component'] + list(embedd[:, 0])
            Blue_data['Second principal component'] = Blue_data['Second principal component'] + list(embedd[:, 1])
            plt.scatter(embedd[:, 0], embedd[:, 1], c=colors)
    Blue_df = pd.DataFrame(Blue_data)
    Blue_df.to_csv('./fig4/LFP_scatter_data/Informer_blue.csv', index=False)
    
    
    red_data = {'scaled_SOH': [],
                 'First principal component': [],
                 'Second principal component': []}
    for condition, value in total_unseen_enc_outs.items():
        policy = name_policy[condition]
        if check_policy_avg_life(policy, name_policy) <= 332:
            # 'b2c1.csv', 'b2c0.csv', 'b2c3.csv'
            print(condition)
            embedd = pca.transform(value[0])
            tmp_cm = condition_colormaps[3] # Red color
            SOHs = value[2]
            max_SOH = max(SOHs)
            min_SOH = min(SOHs)
            # # max_color = max_SOH * 0.8
            # min_color = 0.2
            SOH_std = np.array([(i-min_SOH) / (max_SOH - min_SOH) for i in SOHs])
            SOH_scaled = list(SOH_std * (max_color-min_color) + min_color)
            colors = [tmp_cm(i) for i in SOH_scaled]
            red_data['scaled_SOH'] = red_data['scaled_SOH'] + SOH_scaled
            red_data['First principal component'] = red_data['First principal component'] + list(embedd[:, 0])
            red_data['Second principal component'] = red_data['Second principal component'] + list(embedd[:, 1])
            plt.scatter(embedd[:, 0], embedd[:, 1], c=colors)
    red_df = pd.DataFrame(red_data)
    red_df.to_csv('./fig4/LFP_scatter_data/Informer_red.csv', index=False)

    # contour = plt.contour(X,Y,Z, contour_levels)
    # plt.clabel(contour,fontsize=10,colors='black')
    plt.xlabel('First principal component', fontsize=10)
    plt.ylabel('Second principal component', fontsize=10)
    plt.title('(LFP) Informer encoder embeddings', fontsize=12)
    fig.tight_layout()
    set_ax_linewidth(plt.gca())
    plt.subplots_adjust(wspace=0.2, hspace=0)
    plt.savefig('LFP.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
