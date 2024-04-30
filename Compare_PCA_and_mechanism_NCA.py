import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import wasserstein
from dtaidistance import dtw_ndim
import torch
from torch.nn.functional import softmax
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, wilcoxon, spearmanr


if __name__ == '__main__':
    emd = wasserstein.EMD()  # create the object for computing wasserstein distance
    matrix_order = ['CY45-05_1#25','CY35-05_1#1','CY25-05_1#3','CY25-025_1#7','CY25-1_1#4']  # the meaning of the index order of matrix
    # 加载PCA数据
    further_mechanism_scale = True
    PCA_results_path = './fig5/PCA_results/'
    files = os.listdir(PCA_results_path)
    MAGNet_condition_embedded = {}  # 不含timestamp的embedding
    Informer_condition_embedded = {}
    MAGNet_condition_Tembedded = {}  # 含timestamp的embedding
    Informer_condition_Tembedded = {}
    MAGNet_scaler = StandardScaler()
    Informer_scaler = StandardScaler()
    MAGNet_total_embedded = []
    Informer_total_embedded = []
    for file in files:
        key = file.replace('-#', '#')[4:]
        condition = '_'.join(key.split('_')[:-1])
        embedded = np.load(f'{PCA_results_path}{file}')
        if file.endswith('MAGNet.npy'):
            MAGNet_condition_embedded[condition] = embedded[:, :-1]
            MAGNet_condition_Tembedded[condition] = embedded[:, -1:] # cycle number
            MAGNet_total_embedded.append(embedded[:, :-1])
        else:
            Informer_condition_embedded[condition] = embedded[:, :-1]
            Informer_condition_Tembedded[condition] = embedded[:, -1:]
            Informer_total_embedded.append(embedded[:, :-1])
    MAGNet_total_embedded = np.concatenate(MAGNet_total_embedded, axis=0)
    MAGNet_scaler.fit(MAGNet_total_embedded)  # fit the sacler
    Informer_total_embedded = np.concatenate(Informer_total_embedded, axis=0)
    Informer_scaler.fit(Informer_total_embedded)
    # 形成含Timestamp的scaled embedding
    for condition, embedded in MAGNet_condition_embedded.items():
        scaled_embedded = MAGNet_scaler.transform(embedded)
        MAGNet_condition_Tembedded[condition] = np.concatenate(
            [scaled_embedded, MAGNet_condition_Tembedded[condition]], axis=1)

    for condition, embedded in Informer_condition_embedded.items():
        scaled_embedded = Informer_scaler.transform(embedded)
        Informer_condition_Tembedded[condition] = np.concatenate(
            [scaled_embedded, Informer_condition_Tembedded[condition]], axis=1)
    Mechanism_condition_embedded = {}
    Mechanism_condition_Tembedded = {}
    Mechanism_results_path = './fig5/plt_data/'
    Mechanism_total_embedded = []
    Mechanism_scaler = StandardScaler()
    files = os.listdir(Mechanism_results_path)
    files = [i for i in files if i.endswith('.csv')]
    for file in files:
        key = file.split('_forPlot')[0]
        # condition = '_'.join(key)
        condition = key
        #matrix_order.append(condition)  # 按照机理的condition读取顺序，形成距离矩阵用于比较两个空间
        df = pd.read_csv(f'{Mechanism_results_path}{file}')
        df.loc[df['LAM_PE'] == '--'] = np.nan
        df.loc[df['LLI_Cap'] == '--'] = np.nan
        df = df.astype(float)
        df.dropna()
        LLI_Cap = df['LLI_Cap'].values
        LAM_PE = df['LAM_PE'].values
        cycles = df['Cycle'].values

        mean_cycles = np.nanmean(cycles)
        std_cycles = np.nanstd(cycles)
        max_cycles = np.nanmax(cycles)
        min_cycles = np.nanmin(cycles)
        cycles = cycles[~np.isnan(LAM_PE)]
        LLI_Cap = LLI_Cap[~np.isnan(LAM_PE)]
        LAM_PE = LAM_PE[~np.isnan(LAM_PE)]

        cycles = cycles[~np.isnan(LLI_Cap)]
        LAM_PE = LAM_PE[~np.isnan(LLI_Cap)]
        LLI_Cap = LLI_Cap[~np.isnan(LLI_Cap)]
        # LLI_Cap = LLI_Cap / LLI_Cap[0]
        # LAM_PE = LAM_PE / LAM_PE[0]
        # LLI_Cap = LLI_Cap / (Qds[0]*1000)
        # LAM_PE = LAM_PE / LAM_PE[0]
        # LAM_PE = 1 - LAM_PE
        LLI_Cap = LLI_Cap.reshape(-1, 1)
        LAM_PE = LAM_PE.reshape(-1, 1)
        cycles = np.array([(i - min_cycles) / (max_cycles - min_cycles) for i in cycles])
        mechanism_embedded = np.concatenate([LLI_Cap, LAM_PE], axis=1)
        Mechanism_condition_embedded[condition] = mechanism_embedded
        Mechanism_condition_Tembedded[condition] = cycles.reshape(-1, 1)
        Mechanism_total_embedded.append(mechanism_embedded)
    Mechanism_total_embedded = np.concatenate(Mechanism_total_embedded, axis=0)
    Mechanism_scaler.fit(Mechanism_total_embedded)
    # 形成含Timestamp的scaled embedding
    for condition, embedded in Mechanism_condition_embedded.items():
        if further_mechanism_scale:
            scaled_embedded = Mechanism_scaler.transform(embedded)
        else:
            scaled_embedded = embedded
        Mechanism_condition_Tembedded[condition] = np.concatenate(
            [scaled_embedded, Mechanism_condition_Tembedded[condition]], axis=1)

    # 形成距离矩阵
    # Form the distance matrix for MAGNet
    MAGNet_dist_matrix = np.array([[0.0 for _ in range(len(matrix_order))] for _ in range(len(matrix_order))])
    for row_index, condition in enumerate(matrix_order):
        anchor_embedded = MAGNet_condition_embedded[condition]
        anchor_embedded = MAGNet_scaler.transform(anchor_embedded)
        anchor_weight = np.ones(len(anchor_embedded)) / len(anchor_embedded)
        anchor_coords = anchor_embedded
        for column_index, condition in enumerate(matrix_order):
            compared_embedded = MAGNet_condition_embedded[condition]
            compared_embedded = MAGNet_scaler.transform(compared_embedded)
            compared_weight = np.ones(len(compared_embedded)) / len(compared_embedded)
            compared_coords = compared_embedded
            dist = emd(anchor_weight, anchor_coords, compared_weight, compared_coords)
            MAGNet_dist_matrix[row_index][column_index] = dist
    # Form the distance matrix for Informer
    Informer_dist_matrix = np.array([[0.0 for _ in range(len(matrix_order))] for _ in range(len(matrix_order))])
    for row_index, condition in enumerate(matrix_order):
        anchor_embedded = Informer_condition_embedded[condition]
        anchor_embedded = Informer_scaler.transform(anchor_embedded)
        anchor_weight = np.ones(len(anchor_embedded)) / len(anchor_embedded)
        anchor_coords = anchor_embedded
        for column_index, condition in enumerate(matrix_order):
            compared_embedded = Informer_condition_embedded[condition]
            compared_embedded = Informer_scaler.transform(compared_embedded)
            compared_weight = np.ones(len(compared_embedded)) / len(compared_embedded)
            compared_coords = compared_embedded
            dist = emd(anchor_weight, anchor_coords, compared_weight, compared_coords)
            Informer_dist_matrix[row_index][column_index] = dist
    # Form the distance matrix for mechanism
    Mechanism_dist_matrix = np.array([[0.0 for _ in range(len(matrix_order))] for _ in range(len(matrix_order))])
    for row_index, condition in enumerate(matrix_order):
        anchor_embedded = Mechanism_condition_embedded[condition]
        if further_mechanism_scale:
            anchor_embedded = Mechanism_scaler.transform(anchor_embedded)
        anchor_weight = np.ones(len(anchor_embedded)) / len(anchor_embedded)
        anchor_coords = anchor_embedded
        for column_index, condition in enumerate(matrix_order):
            compared_embedded = Mechanism_condition_embedded[condition]
            if further_mechanism_scale:
                compared_embedded = Mechanism_scaler.transform(compared_embedded)
            compared_weight = np.ones(len(compared_embedded)) / len(compared_embedded)
            compared_coords = compared_embedded
            dist = emd(anchor_weight, anchor_coords, compared_weight, compared_coords)
            Mechanism_dist_matrix[row_index][column_index] = dist

    print(matrix_order)
    print('\n')
    print('MAGNet dist matrix:')
    print(MAGNet_dist_matrix)
    print('\n')
    print('Informer dist matrix:')
    print(Informer_dist_matrix)
    print('\n')
    print('Mechanism dist matrix:')
    print(Mechanism_dist_matrix)

    print('Start to compute the distance between the statistical space and mechanism space')
    print(matrix_order)
    print('MAGNet')
    MAGNet_diff = np.abs(MAGNet_dist_matrix - Mechanism_dist_matrix)
    print(MAGNet_diff)
    # print(MAGNet_diff)
    MAGNet_mechanism_dist = np.sqrt(np.sum(MAGNet_diff * MAGNet_diff)) # Frobenius norm
    print(f'The overall distance from MAGNet space to Mechanism space:{MAGNet_mechanism_dist}\n')

    print(matrix_order)
    print('Informer')
    Informer_diff = np.abs(Informer_dist_matrix - Mechanism_dist_matrix)
    print(Informer_diff)
    Informer_mechanism_dist = np.sqrt(np.sum(Informer_diff * Informer_diff))
    print(f'The overall distance from Informer space to Mechanism space:{Informer_mechanism_dist}')
    print('\nThe superiority of MAGNet over Informer in each condition：')
    print(matrix_order)
    print(-np.sum(MAGNet_diff-Informer_diff,axis=1))
    print(f'Superiority matrix：\n{Informer_diff-MAGNet_diff}\n')
    superiority_matrix = Informer_diff-MAGNet_diff
    saved_superioirty = {}
    for index, condition_name in enumerate(matrix_order):
        superiority = list(superiority_matrix[index])
        saved_superioirty[condition_name] = superiority
    saved_df = pd.DataFrame(saved_superioirty)
    saved_df.to_csv('./fig5/superiority_matrix/superiority_matrix.csv', index=False)
