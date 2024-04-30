'''
Evaluate the model performance
'''
import json
import os.path
import random
import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
from data_provider.data_loader import fixed_files
font = {'family' : 'Arial'}
matplotlib.rcParams['mathtext.fontset'] = 'custom'

matplotlib.rcParams['mathtext.rm'] = 'Arial'

matplotlib.rcParams['mathtext.it'] = 'Arial'

matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42 # make the text editable for Adobe Illustrator
matplotlib.rcParams['ps.fonttype'] = 42

fix_seed = 2021
random.seed(fix_seed)
np.random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import argparse
from exp.exp_main import Exp_Main

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--args_path', type=str,
                        default='./results/NE_meta_Informer_Batteries_cycle_SLMove_lr1e-06_metalr0.0075_mavg15_ftM_sl20_ll20_pl500_dm14_nh4_el2_dl2_df4_fc5_fc21_ebCycle_dtFalse_valratio0.5_test_lossawmse_vallossnw_dp0.0_bs128_wd0_mb2_agamma0.2_lradjtype4_0',
                        help='just copy the path to the ./results/xxx of the trained model here')
    
    
    parser.add_argument('--save', action='store_true',
                        default=True,
                        help='set True to save the results')
    parser.add_argument('--alpha', type=float,
                        default=2,
                        help='the alpha for alpha accuracy')
    tmp_args = parser.parse_args()
    tmp_args.alpha = 4 if tmp_args.args_path.startswith('./results/NE') else 2
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
            short_lived_threshold = 200
        else:
            args.nominal_capacity = 1.1
            short_lived_threshold = float('inf')  # this is not the judgement for NE
        if args.root_path == './dataset/NC_NCM_autoformer_cycle_data/':
            args.set_files = [i for i in fixed_files.NC_test_files if 'NCM' in i]
        elif args.root_path == './dataset/NC_NCA_autoformer_cycle_data/':
            args.set_files = [i for i in fixed_files.NC_test_files if 'NCA' in i]
            # args.set_files = ['NCA_CY25-1_1-#5.csv']
        else:
            args.set_files = fixed_files.NE_test_files
        if 'FT' in args.__dict__ and args.FT:
            setting = 'FT_{}_{}_{}_lr{}_metalr{}_mavg{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_fc2{}_eb{}_dt{}_valratio{}_{}_loss{}_valloss{}_dp{}_bs{}_wd{}_mb{}_agamma{}_lradj{}_{}'.format(
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
        elif 'DA' in args.__dict__ and args.DA:
            setting = '{}_{}_{}_lr{}_metalr{}_mavg{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_fc2{}_eb{}_dt{}_valratio{}_{}_loss{}_valloss{}_dp{}_bs{}_wd{}_mb{}_agamma{}_lradj{}_{}_DA'.format(
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
        else:
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

    Exp = Exp_Main
    exp = Exp(args)  # set experiments
    # exp.predict_specific_cells(setting,load=True,set_dataset='Batteries_cycle_containShort')
    set_files = args.set_files
    fig_path = f'./visual_figs/{args.model_id}_{args.model}/'
    
    root_path = args.root_path
    file_Data = {}

    cell_Qd_maes, cell_Qd_mapes, cell_Qd_mae_stds, cell_Qd_mape_stds, cell_Ed_maes, cell_Ed_mapes, cell_Ed_mae_stds, cell_Ed_mape_stds = {}, {}, {}, {}, {}, {}, {}, {}
    cell_Qd_rmses, cell_Ed_rmses = {}, {}
    cell_Qd_rmses_detailed, cell_Ed_rmses_detailed = {}, {}
    cell_Qd_maes_detailed, cell_Ed_maes_detailed = {}, {}
    condition_files = {}
    r2_Qds, r2_Eds = {}, {}

    alpha_Qds = {}
    alpha_Eds = {}
    # if not os.path.exists(f'{fig_path}data.json'):
    gt_trajectories = {}
    pred_trajectories = {}
    file_detailed_alphas = {}
    for file in set_files:
        args.set_files = [file]
        df = pd.read_csv(f'{root_path}{file}')
        if 'NC' in root_path:
            condition = file.split('#')[0] 
        else:
            cell_name = file.split('.')[0]
            condition = fixed_files.NE_name_policy[cell_name]
        # if not len(df) >= args.seq_len + args.pred_len:
        #     continue
        name = file.split('.')[0] + f'_{len(df)}.pdf'
        # cell_Qd_mae, cell_Qd_mape, cell_Qd_mae_std, cell_Qd_mape_std, cell_Ed_mae, cell_Ed_mape, cell_Ed_mae_std, cell_Ed_mape_std, gt_trajectory, pred_trajectory, r2_Qd, r2_Ed = exp.predict_specific_cell_new(
        #     setting, load=True,
        #     save_path='')
        cell_Qd_mae, cell_Qd_mape, cell_Qd_mae_std, cell_Qd_mape_std, cell_Ed_mae, cell_Ed_mape, cell_Ed_mae_std, cell_Ed_mape_std, gt_trajectory, pred_trajectory, r2_Qd, r2_Ed, alpha_Qd, alpha_Ed, detailed_alphas, cell_Qd_rmse, cell_Ed_rmse, total_RMSE_Qd, total_RMSE_Ed, total_mae_Qd, total_mae_Ed = exp.predict_specific_cell_new_robust(
            setting, load=True,
            save_path='', robust_threshold=tmp_args.alpha)
        file_Data[file] = [cell_Qd_mae, cell_Qd_mape, cell_Qd_mae_std, cell_Qd_mape_std, cell_Ed_mae, cell_Ed_mape,
                           cell_Ed_mae_std, cell_Ed_mape_std, len(df)]
        alpha_Qds[condition] = alpha_Qds.get(condition, []) + [alpha_Qd]
        alpha_Eds[condition] = alpha_Eds.get(condition, []) + [alpha_Ed]

        condition_files[condition] = condition_files.get(condition, []) + [file]

        cell_Qd_maes[condition] = cell_Qd_maes.get(condition, []) + [cell_Qd_mae]
        cell_Qd_mapes[condition] = cell_Qd_mapes.get(condition, []) + [cell_Qd_mape]
        cell_Qd_mae_stds[condition] = cell_Qd_mae_stds.get(condition, []) + [cell_Qd_mae_std]
        cell_Qd_mape_stds[condition] = cell_Qd_mape_stds.get(condition, []) + [cell_Qd_mape_std]
        cell_Ed_maes[condition] = cell_Ed_maes.get(condition, []) + [cell_Ed_mae]
        cell_Ed_mapes[condition] = cell_Ed_mapes.get(condition, []) + [cell_Ed_mape]
        cell_Ed_mae_stds[condition] = cell_Ed_mae_stds.get(condition, []) + [cell_Ed_mae_std]
        cell_Ed_mape_stds[condition] = cell_Ed_mape_stds.get(condition, []) + [cell_Ed_mape_std]
        cell_Qd_rmses_detailed[condition] = cell_Qd_rmses_detailed.get(condition, []) + total_RMSE_Qd
        cell_Ed_rmses_detailed[condition] = cell_Ed_rmses_detailed.get(condition, []) + total_RMSE_Ed
        cell_Qd_maes_detailed[condition] = cell_Qd_maes_detailed.get(condition, []) + total_mae_Qd
        cell_Ed_maes_detailed[condition] = cell_Ed_maes_detailed.get(condition, []) + total_mae_Ed
        
        
        cell_Qd_rmses[condition] = cell_Qd_rmses.get(condition, []) + [cell_Qd_rmse]
        cell_Ed_rmses[condition] = cell_Ed_rmses.get(condition, []) + [cell_Ed_rmse]

        r2_Qds[condition] = r2_Qds.get(condition, []) + [r2_Qd]
        r2_Eds[condition] = r2_Eds.get(condition, []) + [r2_Ed]

        gt_trajectories[file] = gt_trajectory
        pred_trajectories[file] = pred_trajectory
        file_detailed_alphas[file] = detailed_alphas

    # compute from each prediction
    # Qd
    print('RMSE')
    mean_cell_Qd_rmses = 0
    mean_cell_Qd_rmse_stds = []
    for key, value in cell_Qd_rmses.items():
        mean_cell_Qd_rmses += np.mean(value)
        mean_cell_Qd_rmse_stds += [np.mean(value)]
    mean_cell_Qd_rmses = mean_cell_Qd_rmses / len(cell_Qd_rmses)
    mean_cell_Qd_rmse_stds = np.std(mean_cell_Qd_rmse_stds)
    
    print('MAE')
    mean_cell_Qd_maes = 0
    mean_cell_Qd_mae_stds = []
    for key, value in cell_Qd_maes.items():
        mean_cell_Qd_maes += np.mean(value)
        mean_cell_Qd_mae_stds += [np.mean(value)]
    mean_cell_Qd_maes = mean_cell_Qd_maes / len(cell_Qd_maes)
    mean_cell_Qd_mae_stds = np.std(mean_cell_Qd_mae_stds)

    print('MAPE')
    mean_cell_Qd_mapes = 0
    mean_cell_Qd_mape_stds = []
    for key, value in cell_Qd_mapes.items():
        print(key, np.mean(value))
        mean_cell_Qd_mapes += np.mean(value)
        mean_cell_Qd_mape_stds += [np.mean(value)]
    mean_cell_Qd_mapes = mean_cell_Qd_mapes / len(cell_Qd_mapes)
    mean_cell_Qd_mape_stds = np.std(mean_cell_Qd_mape_stds)

    mean_r2_Qds = 0
    mean_r2_Qds_stds = []
    for key, value in r2_Qds.items():
        mean_r2_Qds += np.mean(value)
        mean_r2_Qds_stds += [np.mean(value)]
    mean_r2_Qds = mean_r2_Qds / len(r2_Qds)
    mean_r2_Qds_stds = np.std(mean_r2_Qds_stds)

    mean_alpha_Qds = 0
    mean_alpha_Qds_stds = []
    print('Alpha-acc')
    for key, value in alpha_Qds.items():
        print(key, np.mean(value))
        mean_alpha_Qds += np.mean(value)
        mean_alpha_Qds_stds += [np.mean(value)]
    mean_alpha_Qds = mean_alpha_Qds / len(alpha_Qds)
    mean_alpha_Qds_stds = np.std(mean_alpha_Qds_stds)

    # compute Ed
    print('RMSE')
    mean_cell_Ed_rmses = 0
    mean_cell_Ed_rmse_stds = []
    for key, value in cell_Ed_rmses.items():
        mean_cell_Ed_rmses += np.mean(value)
        mean_cell_Ed_rmse_stds += [np.mean(value)]
    mean_cell_Ed_rmses = mean_cell_Ed_rmses / len(cell_Ed_rmses)
    mean_cell_Ed_rmse_stds = np.std(mean_cell_Ed_rmse_stds)
    
    mean_cell_Ed_maes = 0
    mean_cell_Ed_mae_stds = []
    for key, value in cell_Ed_maes.items():
        mean_cell_Ed_maes += np.mean(value)
        mean_cell_Ed_mae_stds += [np.mean(value)]
    mean_cell_Ed_maes = mean_cell_Ed_maes / len(cell_Ed_maes)
    mean_cell_Ed_mae_stds = np.std(mean_cell_Ed_mae_stds)

    mean_cell_Ed_mapes = 0
    mean_cell_Ed_mape_stds = []
    for key, value in cell_Ed_mapes.items():
        mean_cell_Ed_mapes += np.mean(value)
        mean_cell_Ed_mape_stds += [np.mean(value)]
    mean_cell_Ed_mapes = mean_cell_Ed_mapes / len(cell_Ed_mapes)
    mean_cell_Ed_mape_stds = np.std(mean_cell_Ed_mape_stds)

    mean_r2_Eds = 0
    mean_r2_Eds_stds = []
    for key, value in r2_Eds.items():
        mean_r2_Eds += np.mean(value)
        mean_r2_Eds_stds += [np.mean(value)]
    mean_r2_Eds = mean_r2_Eds / len(r2_Eds)
    mean_r2_Eds_stds = np.std(mean_r2_Eds_stds)

    mean_alpha_Eds = 0
    mean_alpha_Eds_stds = []
    for key, value in alpha_Eds.items():
        mean_alpha_Eds += np.mean(value)
        mean_alpha_Eds_stds += [np.mean(value)]
    mean_alpha_Eds = mean_alpha_Eds / len(alpha_Eds)
    mean_alpha_Eds_stds = np.std(mean_alpha_Eds_stds)

    print(
        f'Qd MAE:{mean_cell_Qd_maes}±{mean_cell_Qd_mae_stds} || Qd MAPE:{mean_cell_Qd_mapes}±{mean_cell_Qd_mape_stds} || RMSE:{mean_cell_Qd_rmses}±{mean_cell_Qd_rmse_stds} || α_score:{mean_alpha_Qds}±{mean_alpha_Qds_stds} \n'
        f'Ed MAE:{mean_cell_Ed_maes}±{mean_cell_Ed_mae_stds} || Ed MAPE:{mean_cell_Ed_mapes}±{mean_cell_Ed_mape_stds} || RMSE:{mean_cell_Ed_rmses}±{mean_cell_Ed_rmse_stds} || α_score:{mean_alpha_Eds}±{mean_alpha_Eds_stds} \n')
    print(r'Averaged $\alpha$-accuracy',(mean_alpha_Qds+mean_alpha_Eds)/2)
    print('Averaged RMSE',(mean_cell_Qd_rmses+mean_cell_Ed_rmses)/2)
    print('Averaged MAE',(mean_cell_Qd_maes+mean_cell_Ed_maes)/2)
    plt.hist(list(cell_Qd_maes.values()), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel('Mean absolute error (Ah)')
    plt.ylabel('The number of cells')
    plt.show()

    plt.hist(np.array(list(cell_Qd_mapes.values())), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel('Mean absolute percentage error (%)')
    plt.ylabel('The number of cells')
    plt.show()

    Ys = []
    conditions = []
    for key, value in alpha_Qds.items():
        Ys += [np.mean(value)]
        conditions += [key]
    plt.bar(conditions, Ys)
    plt.show()

    if tmp_args.save:
        folder_path = './detailed_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        with open(f'{folder_path}pred_trajectories.json', 'w') as f:
            json.dump(pred_trajectories, f)
        with open(f'{folder_path}gt_trajectories.json', 'w') as f:
            json.dump(gt_trajectories, f)
        with open(f'{folder_path}file_detailed_alphas.json', 'w') as f:
            json.dump(file_detailed_alphas, f)
        with open(f'{folder_path}cell_maes.json', 'w') as f:
            json.dump(cell_Qd_maes, f)
        with open(f'{folder_path}Ed_cell_maes.json', 'w') as f:
            json.dump(cell_Ed_maes, f)
        with open(f'{folder_path}cell_rmses.json', 'w') as f:
            json.dump(cell_Qd_rmses, f)
        with open(f'{folder_path}Ed_cell_rmses.json', 'w') as f:
            json.dump(cell_Ed_rmses, f)
            
        with open(f'{folder_path}cell_maes_detailed.json', 'w') as f:
            json.dump(cell_Qd_maes_detailed, f)
        with open(f'{folder_path}Ed_cell_maes_detailed.json', 'w') as f:
            json.dump(cell_Ed_maes_detailed, f)
        with open(f'{folder_path}cell_rmses_detailed.json', 'w') as f:
            json.dump(cell_Qd_rmses_detailed, f)
        with open(f'{folder_path}Ed_cell_rmses_detailed.json', 'w') as f:
            json.dump(cell_Ed_rmses_detailed, f)
            
        with open(f'{folder_path}alpha_scores_Qd.json', 'w') as f:
            json.dump(alpha_Qds, f)
        with open(f'{folder_path}alpha_scores_Ed.json', 'w') as f:
            json.dump(alpha_Eds, f)
        with open(f'{folder_path}Qd_R2s.json', 'w') as f:
            json.dump(r2_Qds, f)
        with open(f'{folder_path}Ed_R2s.json', 'w') as f:
            json.dump(r2_Eds, f)
        with open(f'{folder_path}condition_files.json', 'w') as f:
            json.dump(condition_files, f)


if __name__ == "__main__":
    main()
