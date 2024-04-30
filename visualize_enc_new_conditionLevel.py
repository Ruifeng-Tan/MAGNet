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
from scipy.interpolate import CloughTocher2DInterpolator
import joblib
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

def main():
    # NCA_Informer_Batteries_cycle_SLMove_lr0.01_metalr0.005_mavg15_ftM_sl10_ll10_pl150_dm8_nh4_el1_dl2_df2_fc4_fc21_ebCycle_dtFalse_valratio0.33_test_lossmse_vallossnw_dp0.0_bs32_wd0_mb2_agamma0.25_lradjtype6_0
    # NCA_meta_Informer_Batteries_cycle_SLMove_lr1e-06_metalr0.005_mavg15_ftM_sl10_ll10_pl150_dm8_nh4_el1_dl2_df4_fc4_fc21_ebCycle_dtFalse_valratio0.33_test_lossawmse_vallossw_dp0.0_bs32_wd0_mb2_agamma0.25_lradjtype6_0
    # NCM_Informer_Batteries_cycle_SLMove_lr0.005_metalr0.005_mavg15_ftM_sl10_ll10_pl250_dm8_nh4_el2_dl2_df2_fc4_fc21_ebCycle_dtFalse_valratio0.33_test_lossmse_vallossnw_dp0.0_bs32_wd0_mb2_agamma0.25_lradjtype6_0
    # NCM_meta_Informer_Batteries_cycle_SLMove_lr1e-06_metalr0.005_mavg15_ftM_sl10_ll10_pl250_dm8_nh4_el2_dl1_df2_fc4_fc21_ebCycle_dtFalse_valratio0.33_test_lossawmse_vallossw_dp0.0_bs32_wd0_mb2_agamma0.25_lradjtype6_0
    # NE_meta_Informer_Batteries_cycle_SLMove_lr1e-06_metalr0.005_mavg15_ftM_sl20_ll20_pl500_dm12_nh4_el1_dl1_df4_fc5_fc21_ebCycle_dtFalse_valratio0.33_test_lossawmse_vallossw_dp0.0_bs128_wd0_mb2_agamma0.2_lradjtype6_0
    # w/o TTL: NCA_meta_Informer_Batteries_cycle_SLMove_lr1e-06_metalr0.005_mavg15_ftM_sl10_ll10_pl150_dm8_nh4_el1_dl2_df2_fc4_fc21_ebCycle_dtFalse_valratio0.33_test_losswmse_vallossnw_dp0.0_bs32_wd0_mb0.25_agamma0.0_lradjtype4_0
    # w/o DG: NCA_Informer_Batteries_cycle_SLMove_lr0.005_metalr0.005_mavg15_ftM_sl10_ll10_pl150_dm8_nh4_el1_dl2_df4_fc4_fc21_ebCycle_dtFalse_valratio0.33_test_lossamse_vallossnw_dp0.0_bs32_wd0_mb0.1_agamma0.05_lradjtype4_0
    # Informer: NCA_Informer_Batteries_cycle_SLMove_lr0.005_metalr0.005_mavg15_ftM_sl10_ll10_pl150_dm8_nh4_el1_dl2_df4_fc4_fc21_ebCycle_dtFalse_valratio0.33_test_lossmse_vallossnw_dp0.0_bs32_wd0_mb2_agamma0.25_lradjtype4_0
    pca_seed = 2021 # 2021
    fitlog.set_log_dir("./logs/")
    parser = argparse.ArgumentParser()
    # basic config
    parser.add_argument('--args_path', type=str,
                        default='./results/NCA_meta_Informer_Batteries_cycle_SLMove_lr1e-06_metalr0.005_mavg15_ftM_sl10_ll10_pl150_dm8_nh4_el1_dl2_df2_fc4_fc21_ebCycle_dtFalse_valratio0.33_test_lossawmse_vallossnw_dp0.0_bs32_wd0_mb0.25_agamma0.25_lradjtype4_0',
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
            # args.set_files = [i for i in fixed_files.NC_test_files if ('NCA' in i and not i.startswith('NCA_CY25-1_1'))]
        else:
            # args.set_files = fixed_files.NE_test_files
            args.set_files = ['b2c3.csv', 'b2c22.csv', 'b3c8.csv', 'b3c30.csv', 'b3c16.csv']
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
    total_enc_outs = []
    total_colors = []
    total_normalized_cycles = []
    total_SOHs = []
    total_scaled_SOHs = []
    conditions = []
    file_names = []
    condition_colormaps_dict = {'NCA_CY25-1_1-':cm.get_cmap(name='Reds'),
                                'NCA_CY25-05_1-':cm.get_cmap(name='Blues'),
                                'NCA_CY25-025_1-':cm.get_cmap(name='Purples'),
                                'NCA_CY35-05_1-':cm.get_cmap(name='Greys'),
                                'NCA_CY45-05_1-':cm.get_cmap(name='Greens')}
    condition_zorder_dict = {'NCA_CY25-1_1-':3,
                                'NCA_CY25-05_1-':2,
                                'NCA_CY25-025_1-':2,
                                'NCA_CY35-05_1-':2,
                                'NCA_CY45-05_1-':2}
    condition_markers_dict = {'NCA_CY25-1_1-':'o',
                                'NCA_CY25-05_1-':'o',
                                'NCA_CY25-025_1-':'o',
                                'NCA_CY35-05_1-':'o',
                                'NCA_CY45-05_1-':'o'}
    NCA_representative_files_embedds = {'NCA_CY45-05_1-#25':[],
                                        'NCA_CY35-05_1-#1':[],
                                        'NCA_CY25-05_1-#3':[],
                                        'NCA_CY25-025_1-#7':[],
                                        'NCA_CY25-1_1-#4':[]}
    for file in set_files:
        condition = file.split('#')[0]
        args.set_files = [file]
        enc_outs, ruls, labels = exp.visualize_enc_out_new(setting, load=True,
                                                   save_path='', return_labels=True)
        SOHs = np.array(labels) / 3.5
        conditions += [condition for _ in range(len(ruls))]
        file_name = file.split('.')[0]
        file_names += [file_name for _ in range(len(ruls))]
        enc_outs = np.concatenate(enc_outs, axis=0)
        total_enc_outs = total_enc_outs + [enc_outs]
        total_SOHs = total_SOHs + list(SOHs)
        tmp_cm = condition_colormaps_dict[condition]
        # max_rul = max(ruls)
        # min_rul = min(ruls)
        # rul_std = np.array([(i-min_rul) / (max_rul - min_rul) for i in ruls])
        # rul_scaled = list(rul_std * (max_color-min_color) + min_color)
        
        max_SOH = max(SOHs)
        min_SOH = min(SOHs)
        # max_color = max_SOH * 0.8
        max_color = 1
        min_color = 0.8
        SOH_std = np.array([(i-min_SOH) / (max_SOH - min_SOH) for i in SOHs])
        SOH_scaled = list(SOH_std * (max_color-min_color) + min_color)
        total_scaled_SOHs = total_scaled_SOHs + SOH_scaled
        total_colors.append([tmp_cm(i) for i in SOH_scaled])
        
        cycles = np.flip(ruls)
        normalized_cycles = np.array([(i - np.min(cycles)) / (np.max(cycles) - np.min(cycles)) for i in cycles]).reshape(-1)
        total_normalized_cycles += list(normalized_cycles)

    total_enc_outs = np.concatenate(total_enc_outs, axis=0)
    total_colors = np.concatenate(total_colors, axis=0)
    pca = PCA(n_components=2, random_state=pca_seed)
    embedd = pca.fit_transform(total_enc_outs)  # decrease the dimension
    # save the plotting data
    saved_data = {
        'red': {},
        'blue': {},
        'purple': {},
        'grey': {},
        'green': {}
    }
    for index, condition in enumerate(conditions):
        if condition == 'NCA_CY25-1_1-':
            saved_data['red']['scaled_SOH'] = saved_data['red'].get('scaled_SOH', []) + [total_scaled_SOHs[index]]
            saved_data['red']['First principal component'] = saved_data['red'].get('First principal component', []) + [embedd[index,0]]
            saved_data['red']['Second principal component'] = saved_data['red'].get('Second principal component', []) + [embedd[index,1]]
        elif condition == 'NCA_CY25-05_1-':
            saved_data['blue']['scaled_SOH'] = saved_data['blue'].get('scaled_SOH', []) + [total_scaled_SOHs[index]]
            saved_data['blue']['First principal component'] = saved_data['blue'].get('First principal component', []) + [embedd[index,0]]
            saved_data['blue']['Second principal component'] = saved_data['blue'].get('Second principal component', []) + [embedd[index,1]]
        elif condition == 'NCA_CY25-025_1-':
            saved_data['purple']['scaled_SOH'] = saved_data['purple'].get('scaled_SOH', []) + [total_scaled_SOHs[index]]
            saved_data['purple']['First principal component'] = saved_data['purple'].get('First principal component', []) + [embedd[index,0]]
            saved_data['purple']['Second principal component'] = saved_data['purple'].get('Second principal component', []) + [embedd[index,1]]
        elif condition == 'NCA_CY35-05_1-':
            saved_data['grey']['scaled_SOH'] = saved_data['grey'].get('scaled_SOH', []) + [total_scaled_SOHs[index]]
            saved_data['grey']['First principal component'] = saved_data['grey'].get('First principal component', []) + [embedd[index,0]]
            saved_data['grey']['Second principal component'] = saved_data['grey'].get('Second principal component', []) + [embedd[index,1]]
        elif condition == 'NCA_CY45-05_1-':
            saved_data['green']['scaled_SOH'] = saved_data['green'].get('scaled_SOH', []) + [total_scaled_SOHs[index]]
            saved_data['green']['First principal component'] = saved_data['green'].get('First principal component', []) + [embedd[index,0]]
            saved_data['green']['Second principal component'] = saved_data['green'].get('Second principal component', []) + [embedd[index,1]]
        else:
            raise Exception('Not implemented!')
    interp = CloughTocher2DInterpolator(embedd, np.array(total_SOHs))
    print('Get the interpolation of SOH')
    min_embedd = np.min(embedd, axis=0)
    max_embedd = np.max(embedd, axis=0)
    # generate the meshgrid to plot the contour line
    x = np.arange(min_embedd[0], max_embedd[0], 0.1)
    y = np.arange(min_embedd[1], max_embedd[1], 0.1)
    X,Y = np.meshgrid(x,y)
    Z = interp(X, Y)
    # tsne = TSNE(n_components=2, init='pca', verbose=1, perplexity=5, random_state=pca_seed)
    # embedd = tsne.fit_transform(total_enc_outs)  # decrease the dimension
    for index, emb in enumerate(embedd):
        file_name = file_names[index]
        if file_name in NCA_representative_files_embedds:
            timeE = np.array(total_normalized_cycles[index]).reshape(1)
            emb_timeE = np.concatenate([emb, timeE], axis=0).reshape(1,-1)
            NCA_representative_files_embedds[file_name].append(emb_timeE)
    for key, data in NCA_representative_files_embedds.items():
        data = np.concatenate(data,axis=0) 
        np.save(f"./fig5/PCA_results/{key}_MAGNet.npy", data)
    
    
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    condition_split_index = [0]
    last_condition = conditions[0]
    for index, condition in enumerate(conditions):
        if condition != last_condition:
            condition_split_index.append(index)
            last_condition = condition
    print(condition_split_index)
    for index, _ in enumerate(condition_split_index):
        if index == (len(condition_split_index) - 1):
            condition = conditions[condition_split_index[index]]
            plt.scatter(embedd[condition_split_index[index]:, 0],
                        embedd[condition_split_index[index]:, 1],
                        c=total_colors[condition_split_index[index]:], 
                        marker=condition_markers_dict[condition],
                        zorder=condition_zorder_dict[condition])
            break
        condition = conditions[condition_split_index[index]]
        plt.scatter(embedd[condition_split_index[index]:condition_split_index[index + 1], 0],
                    embedd[condition_split_index[index]:condition_split_index[index + 1], 1],
                    c=total_colors[condition_split_index[index]:condition_split_index[index + 1]],
                    marker=condition_markers_dict[condition],
                    zorder=condition_zorder_dict[condition])
    # plt.contour(X,Y,Z, [0.8, 0.85, 0.9, 0.95])
    # plt.xticks([])
    # plt.yticks([])

    plt.xlabel('First principal component', fontsize=10)
    plt.ylabel('Second principal component', fontsize=10)
    plt.title('(NCA) MAGNet encoder embeddings', fontsize=12)
    set_ax_linewidth(plt.gca())

    # plot Informer
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_path', type=str,
                        default='./results/NCA_Informer_Batteries_cycle_SLMove_lr0.005_metalr0.005_mavg15_ftM_sl10_ll10_pl150_dm8_nh4_el1_dl2_df4_fc4_fc21_ebCycle_dtFalse_valratio0.33_test_lossmse_vallossnw_dp0.0_bs32_wd0_mb2_agamma0.25_lradjtype4_0',
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
            # args.set_files = fixed_files.NE_test_files
            args.set_files = ['b2c3.csv', 'b2c22.csv', 'b3c8.csv', 'b3c30.csv', 'b3c16.csv']
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
    total_enc_outs = []
    total_colors = []
    total_normalized_cycles = []
    total_SOHs = []
    conditions = []
    file_names = []
    condition_colormaps_dict = {'NCA_CY25-1_1-':cm.get_cmap(name='Reds'),
                                'NCA_CY25-05_1-':cm.get_cmap(name='Blues'),
                                'NCA_CY25-025_1-':cm.get_cmap(name='Purples'),
                                'NCA_CY35-05_1-':cm.get_cmap(name='Greys'),
                                'NCA_CY45-05_1-':cm.get_cmap(name='Greens')}
    condition_zorder_dict = {'NCA_CY25-1_1-':3,
                                'NCA_CY25-05_1-':2,
                                'NCA_CY25-025_1-':2,
                                'NCA_CY35-05_1-':2,
                                'NCA_CY45-05_1-':2}
    condition_markers_dict = {'NCA_CY25-1_1-':'o',
                                'NCA_CY25-05_1-':'o',
                                'NCA_CY25-025_1-':'o',
                                'NCA_CY35-05_1-':'o',
                                'NCA_CY45-05_1-':'o'}
    NCA_representative_files_embedds = {'NCA_CY45-05_1-#25':[],
                                        'NCA_CY35-05_1-#1':[],
                                        'NCA_CY25-05_1-#3':[],
                                        'NCA_CY25-025_1-#7':[],
                                        'NCA_CY25-1_1-#4':[]}
    for file in set_files:
        condition = file.split('#')[0]
        args.set_files = [file]
        enc_outs, ruls, labels = exp.visualize_enc_out_new(setting, load=True,
                                                   save_path='', return_labels=True)
        SOHs = np.array(labels) / 3.5
        conditions += [condition for _ in range(len(ruls))]
        file_name = file.split('.')[0]
        file_names += [file_name for _ in range(len(ruls))]
        enc_outs = np.concatenate(enc_outs, axis=0)
        total_enc_outs = total_enc_outs + [enc_outs]
        total_SOHs = total_SOHs + list(SOHs)
        tmp_cm = condition_colormaps_dict[condition]
        # max_rul = max(ruls)
        # min_rul = min(ruls)
        # rul_std = np.array([(i-min_rul) / (max_rul - min_rul) for i in ruls])
        # rul_scaled = list(rul_std * (max_color-min_color) + min_color)
        
        max_SOH = max(SOHs)
        min_SOH = min(SOHs)
        max_color = 1
        min_color = 0.8
        SOH_std = np.array([(i-min_SOH) / (max_SOH - min_SOH) for i in SOHs])
        SOH_scaled = list(SOH_std * (max_color-min_color) + min_color)
        total_colors.append([tmp_cm(i) for i in SOH_scaled])
        
        cycles = np.flip(ruls)
        normalized_cycles = np.array([(i - np.min(cycles)) / (np.max(cycles) - np.min(cycles)) for i in cycles]).reshape(-1)
        total_normalized_cycles += list(normalized_cycles)

    total_enc_outs = np.concatenate(total_enc_outs, axis=0)
    total_colors = np.concatenate(total_colors, axis=0)
    pca = PCA(n_components=2, random_state=pca_seed)
    embedd = pca.fit_transform(total_enc_outs)  # decrease the dimension
    # save the plotting data
    saved_data = {
        'red': {},
        'blue': {},
        'purple': {},
        'grey': {},
        'green': {}
    }
    for index, condition in enumerate(conditions):
        if condition == 'NCA_CY25-1_1-':
            saved_data['red']['scaled_SOH'] = saved_data['red'].get('scaled_SOH', []) + [total_scaled_SOHs[index]]
            saved_data['red']['First principal component'] = saved_data['red'].get('First principal component', []) + [embedd[index,0]]
            saved_data['red']['Second principal component'] = saved_data['red'].get('Second principal component', []) + [embedd[index,1]]
        elif condition == 'NCA_CY25-05_1-':
            saved_data['blue']['scaled_SOH'] = saved_data['blue'].get('scaled_SOH', []) + [total_scaled_SOHs[index]]
            saved_data['blue']['First principal component'] = saved_data['blue'].get('First principal component', []) + [embedd[index,0]]
            saved_data['blue']['Second principal component'] = saved_data['blue'].get('Second principal component', []) + [embedd[index,1]]
        elif condition == 'NCA_CY25-025_1-':
            saved_data['purple']['scaled_SOH'] = saved_data['purple'].get('scaled_SOH', []) + [total_scaled_SOHs[index]]
            saved_data['purple']['First principal component'] = saved_data['purple'].get('First principal component', []) + [embedd[index,0]]
            saved_data['purple']['Second principal component'] = saved_data['purple'].get('Second principal component', []) + [embedd[index,1]]
        elif condition == 'NCA_CY35-05_1-':
            saved_data['grey']['scaled_SOH'] = saved_data['grey'].get('scaled_SOH', []) + [total_scaled_SOHs[index]]
            saved_data['grey']['First principal component'] = saved_data['grey'].get('First principal component', []) + [embedd[index,0]]
            saved_data['grey']['Second principal component'] = saved_data['grey'].get('Second principal component', []) + [embedd[index,1]]
        elif condition == 'NCA_CY45-05_1-':
            saved_data['green']['scaled_SOH'] = saved_data['green'].get('scaled_SOH', []) + [total_scaled_SOHs[index]]
            saved_data['green']['First principal component'] = saved_data['green'].get('First principal component', []) + [embedd[index,0]]
            saved_data['green']['Second principal component'] = saved_data['green'].get('Second principal component', []) + [embedd[index,1]]
        else:
            raise Exception('Not implemented!')
    
    # tsne = TSNE(n_components=2, init='pca', verbose=1, perplexity=5, random_state=pca_seed)
    # embedd = tsne.fit_transform(total_enc_outs)  # decrease the dimension
    for index, emb in enumerate(embedd):
        file_name = file_names[index]
        if file_name in NCA_representative_files_embedds:
            timeE = np.array(total_normalized_cycles[index]).reshape(1)
            emb_timeE = np.concatenate([emb, timeE], axis=0).reshape(1,-1)
            NCA_representative_files_embedds[file_name].append(emb_timeE)
    for key, data in NCA_representative_files_embedds.items():
        data = np.concatenate(data,axis=0) 
        np.save(f"./fig5/PCA_results/{key}_Informer.npy", data)

    condition_split_index = [0]
    last_condition = conditions[0]
    for index, condition in enumerate(conditions):
        if condition != last_condition:
            condition_split_index.append(index)
            last_condition = condition
    print(condition_split_index)
    plt.subplot(122)
    for index, _ in enumerate(condition_split_index):
        if index == (len(condition_split_index) - 1):
            condition = conditions[condition_split_index[index]]
            plt.scatter(embedd[condition_split_index[index]:, 0],
                        embedd[condition_split_index[index]:, 1],
                        c=total_colors[condition_split_index[index]:], 
                        marker=condition_markers_dict[condition],
                        zorder=condition_zorder_dict[condition])
            break
        condition = conditions[condition_split_index[index]]
        plt.scatter(embedd[condition_split_index[index]:condition_split_index[index + 1], 0],
                    embedd[condition_split_index[index]:condition_split_index[index + 1], 1],
                    c=total_colors[condition_split_index[index]:condition_split_index[index + 1]],
                    marker=condition_markers_dict[condition],
                    zorder=condition_zorder_dict[condition])

    plt.xlabel('First principal component', fontsize=10)
    plt.ylabel('Second principal component', fontsize=10)
    plt.title('(NCA) Informer encoder embeddings', fontsize=12)
    # plt.savefig('Ours.pdf', bbox_inches='tight')
    # plt.show()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0)
    set_ax_linewidth(plt.gca())
    plt.savefig('NCA.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
