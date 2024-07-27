import copy
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from sklearn.linear_model import LinearRegression
import random
from sklearn.metrics import mean_absolute_percentage_error
import json
warnings.filterwarnings('ignore')


class fixed_files:
    NC_train_files = ['NCA_CY25-1_1-#6.csv', 'NCA_CY25-1_1-#1.csv', 'NCA_CY25-1_1-#9.csv', 'NCA_CY25-1_1-#3.csv',
                      'NCA_CY25-1_1-#7.csv',  'NCA_CY25-05_1-#14.csv', 'NCA_CY25-05_1-#11.csv', 'NCA_CY25-05_1-#6.csv',
                      'NCA_CY25-05_1-#1.csv', 'NCA_CY25-05_1-#2.csv', 'NCA_CY25-05_1-#19.csv', 'NCA_CY25-05_1-#13.csv',
                      'NCA_CY25-05_1-#7.csv', 'NCA_CY25-05_1-#17.csv', 'NCA_CY25-05_1-#4.csv', 'NCA_CY45-05_1-#28.csv',
                      'NCA_CY45-05_1-#17.csv', 'NCA_CY45-05_1-#5.csv', 'NCA_CY45-05_1-#2.csv', 'NCA_CY45-05_1-#21.csv',
                      'NCA_CY45-05_1-#7.csv', 'NCA_CY45-05_1-#9.csv', 'NCA_CY45-05_1-#19.csv', 'NCA_CY45-05_1-#22.csv',
                      'NCA_CY45-05_1-#14.csv', 'NCA_CY45-05_1-#16.csv', 'NCA_CY45-05_1-#8.csv', 'NCA_CY45-05_1-#23.csv',
                      'NCA_CY45-05_1-#24.csv', 'NCA_CY25-025_1-#3.csv', 'NCA_CY25-025_1-#5.csv',
                      'NCA_CY25-025_1-#4.csv', 'NCA_CY35-05_1-#2.csv']
    NC_val_files = ['NCA_CY25-1_1-#8.csv', 'NCA_CY25-1_1-#2.csv',  'NCA_CY25-05_1-#16.csv',
                    'NCA_CY25-05_1-#10.csv', 'NCA_CY25-05_1-#15.csv', 'NCA_CY45-05_1-#12.csv',
                    'NCA_CY45-05_1-#10.csv', 'NCA_CY45-05_1-#20.csv', 'NCA_CY45-05_1-#26.csv', 'NCA_CY45-05_1-#13.csv',
                    'NCA_CY45-05_1-#27.csv', 'NCA_CY25-025_1-#6.csv']
    NC_test_files = ['NCA_CY25-1_1-#5.csv', 'NCA_CY25-1_1-#4.csv',  'NCA_CY25-05_1-#12.csv',
                     'NCA_CY25-05_1-#5.csv', 'NCA_CY25-05_1-#3.csv', 'NCA_CY25-05_1-#18.csv', 'NCA_CY45-05_1-#11.csv', 
                     'NCA_CY45-05_1-#15.csv', 'NCA_CY45-05_1-#1.csv', 'NCA_CY45-05_1-#18.csv', 'NCA_CY45-05_1-#25.csv', 'NCA_CY25-025_1-#7.csv',
                     'NCA_CY35-05_1-#1.csv']
    
    NE_train_files = ['b1c45.csv', 'b1c44.csv', 'b1c42.csv', 'b1c43.csv', 'b1c41.csv', 'b1c40.csv', 'b1c36.csv', 'b1c37.csv', 'b1c38.csv', 'b1c39.csv', 'b1c35.csv', 
                      'b1c34.csv', 'b1c31.csv', 'b1c30.csv', 'b1c24.csv', 'b1c25.csv', 'b1c26.csv', 'b1c27.csv', 'b2c45.csv', 'b1c23.csv', 'b3c33.csv', 'b3c20.csv', 
                      'b3c0.csv', 'b3c8.csv', 'b3c24.csv', 'b3c38.csv', 'b3c31.csv', 'b3c22.csv', 'b3c15.csv', 'b3c29.csv', 'b2c40.csv', 'b2c37.csv', 'b3c5.csv', 'b3c12.csv', 
                      'b3c36.csv', 'b3c3.csv', 'b3c14.csv', 'b3c19.csv', 'b3c41.csv', 'b3c27.csv', 'b1c18.csv', 'b1c19.csv', 'b1c15.csv', 'b1c14.csv', 'b1c11.csv', 'b1c9.csv', 
                      'b3c9.csv', 'b3c17.csv', 'b3c1.csv', 'b3c44.csv', 'b3c25.csv', 'b3c39.csv', 'b3c34.csv', 'b3c30.csv', 'b2c33.csv', 'b2c32.csv', 'b2c31.csv', 'b2c30.csv',
                        'b1c3.csv', 'b1c4.csv', 'b2c13.csv', 'b2c14.csv', 'b2c29.csv', 'b2c28.csv', 'b2c22.csv', 'b2c21.csv', 'b1c5.csv', 'b2c18.csv', 'b2c17.csv', 'b3c21.csv', 
                        'b3c6.csv', 'b3c28.csv', 'b2c10.csv', 'b1c2.csv', 'b1c0.csv', 'b1c1.csv', 'b2c6.csv']
    NE_val_files = ['b1c32.csv', 'b1c33.csv', 'b2c47.csv', 'b1c28.csv', 'b1c29.csv', 'b2c36.csv', 'b3c18.csv', 'b3c11.csv', 'b3c4.csv', 'b3c35.csv', 'b3c40.csv', 'b3c26.csv', 
                    'b3c13.csv', 'b1c21.csv', 'b1c20.csv', 'b1c17.csv', 'b1c16.csv', 'b2c34.csv', 'b2c11.csv', 'b1c7.csv', 'b2c24.csv', 'b1c6.csv', 'b3c10.csv', 'b2c26.csv',
                      'b3c16.csv', 'b3c7.csv', 'b3c45.csv', 'b2c25.csv', 'b2c23.csv', 'b2c20.csv', 'b2c5.csv']
    NE_test_files = ['b2c46.csv', 'b2c44.csv', 'b2c43.csv', 'b2c42.csv', 'b2c41.csv', 'b2c39.csv', 'b2c38.csv', 'b2c35.csv', 'b2c12.csv', 'b2c27.csv', 'b2c19.csv', 'b2c4.csv', 
                     'b2c3.csv', 'b2c2.csv', 'b2c1.csv', 'b2c0.csv']
    
    EES_train_files = ['tagged_2-6.csv', 'tagged_2-3.csv', 'tagged_3-4.csv', 'tagged_3-2.csv', 'tagged_7-7.csv', 'tagged_5-4.csv', 'tagged_1-3.csv', 'tagged_6-1.csv', 'tagged_3-3.csv', 'tagged_8-8.csv', 'tagged_5-3.csv', 'tagged_6-8.csv', 'tagged_3-1.csv', 'tagged_5-2.csv', 'tagged_9-3.csv', 'tagged_2-2.csv', 'tagged_10-2.csv', 'tagged_4-5.csv', 'tagged_4-7.csv', 'tagged_4-2.csv', 'tagged_9-6.csv', 'tagged_8-4.csv', 'tagged_10-6.csv', 'tagged_3-8.csv', 'tagged_5-6.csv', 'tagged_6-2.csv', 'tagged_10-5.csv', 'tagged_8-6.csv', 'tagged_3-7.csv', 'tagged_6-3.csv', 'tagged_3-6.csv', 'tagged_8-7.csv', 'tagged_9-1.csv', 'tagged_10-7.csv', 'tagged_4-8.csv', 'tagged_5-1.csv', 'tagged_4-1.csv', 'tagged_9-4.csv', 'tagged_9-5.csv', 'tagged_10-1.csv', 'tagged_6-6.csv', 'tagged_5-5.csv', 'tagged_6-4.csv', 'tagged_3-5.csv', 'tagged_7-1.csv', 'tagged_6-5.csv', 'tagged_1-7.csv', 'tagged_2-7.csv', 'tagged_7-8.csv', 'tagged_10-3.csv', 'tagged_1-5.csv']
    EES_val_files = ['tagged_9-7.csv', 'tagged_10-4.csv', 'tagged_8-3.csv', 'tagged_9-2.csv', 'tagged_1-8.csv', 'tagged_7-5.csv', 'tagged_9-8.csv', 'tagged_8-2.csv', 'tagged_1-2.csv', 'tagged_7-2.csv']
    EES_test_files = ['tagged_8-1.csv', 'tagged_10-8.csv', 'tagged_4-4.csv', 'tagged_1-6.csv', 'tagged_4-3.csv', 'tagged_7-6.csv', 'tagged_2-4.csv', 'tagged_4-6.csv', 'tagged_8-5.csv', 'tagged_1-1.csv', 'tagged_5-7.csv', 'tagged_2-5.csv', 'tagged_7-3.csv', 'tagged_7-4.csv', 'tagged_2-8.csv', 'tagged_1-4.csv']

    # domain id for Nature energy dataset
    NE_unseen_test_files = NE_test_files
    NE_name_policy = json.load(open('./dataset/name_policy.json'))
    policies = []
    for name, policy in NE_name_policy.items():
        if policy not in policies:
            policies.append(policy)
    policy2id = dict(zip(policies,[i for i in range(len(policies))]))
    
    # domain id for EES dataset
    EES_unseen_test_files = EES_test_files
    EES_name_policy = json.load(open('./dataset/HUST_name_policy.json'))
    EES_policies = []
    for name, policy in EES_name_policy.items():
        if policy not in EES_policies:
            EES_policies.append(policy)
    EES_policy2id = dict(zip(EES_policies,[i for i in range(len(EES_policies))]))


class Dataset_Battery_cycle_ShortLongMove(Dataset):
    # I implement a Dataset that can use meta learning for all three datasets in this version.
    # In addition, the cycle life distance is included.
    # This version is developed based on Dataset_Battery_cycle_ContainShort with file id.
    # This version contains the file id, which is used to discriminate the cells.
    # In addition, since we focus on the reliable full-life forecasting. The forecasting will not end until
    # the user has seen the end of a cell.
    def __init__(self, args, flag='train', size=None,
                 features='S', data_path='',
                 target='OT', scale=True, timeenc=0, freq='t', set_files=''):
        root_path = args.root_path
        if root_path == './dataset/HUST_cycle_data/':
            self.train_files = fixed_files.EES_train_files
            self.val_files = fixed_files.EES_val_files
            self.test_files = fixed_files.EES_test_files
        elif root_path == './dataset/NatureEnergy_cycle_data/':
            self.train_files = fixed_files.NE_train_files
            self.val_files = fixed_files.NE_val_files
            self.test_files = fixed_files.NE_test_files
        elif root_path == './dataset/NC_NCA_autoformer_cycle_data/':
            if not args.FT:
                self.train_files = [i for i in fixed_files.NC_train_files if i.startswith('NCA')]
                self.val_files = [i for i in fixed_files.NC_val_files if i.startswith('NCA')]
                self.test_files = [i for i in fixed_files.NC_test_files if i.startswith('NCA')]
            else:
                # fine-tune
                self.train_files = [i for i in fixed_files.NC_train_files if i.startswith('NCA_CY25-1_1')]
                self.val_files = [i for i in fixed_files.NC_val_files if i.startswith('NCA_CY25-1_1')]
                self.test_files = [i for i in fixed_files.NC_test_files if i.startswith('NCA_CY25-1_1')]
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'pred', 'set_files']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'pred': 3, 'set_files': 4}
        self.set_type = type_map[flag]
        self.set_files = set_files

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.cell_len = []
        self.cell_data_num = []
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        if not self.set_type == 4:
            print(f'training cells are:{self.train_files}\n'
                  f'validation files are:{self.val_files}\n'
                  f'testing files are:{self.test_files}\n')
        total_dfs = []
        self.cycle_life_scaler = StandardScaler()
        seen_conditions = []
        domain_id = 0
        if self.set_type == 0:
            # train
            for file in self.train_files:
                df = pd.read_csv(self.root_path + f'{file}')
                # The condition name of NCA dataset can be directly obtained from the file name. The condition name of the LFP dataset 
                # is obtained from the record
                if 'NC' in self.root_path:
                    condition = file.split('#')[0] 
                    df['file'] = domain_id
                elif 'NatureEnergy' in self.root_path:
                    cell_name = file.split('.')[0]
                    policy = fixed_files.NE_name_policy[cell_name]
                    domain_id = fixed_files.policy2id[policy]
                    df['file'] = domain_id
                elif 'EES' in self.root_path:
                    cell_name = file
                    policy = fixed_files.EES_name_policy[cell_name]
                    domain_id = fixed_files.EES_policy2id[policy]
                    df['file'] = domain_id
                if 'NC' in self.root_path:
                    first_life_df = df.loc[df['Qd'] >= 0.8 * 3.5]
                    df['cycle distance'] = first_life_df['cycle number'].max() - df['cycle number']
                    if condition not in seen_conditions:
                        # The `seen_conditions` here is just used for recording what conditions have appeared
                        seen_conditions.append(condition)
                        domain_id += 1
                else:
                    df['cycle distance'] = df['cycle number'].max() - df['cycle number']
                time = df['cost_time'].values
                df['cost_time'] = [sum(time[:index + 1]) for index, _ in enumerate(time)]
                self.cell_len.append(len(df))
                data_num = len(df) - self.seq_len
                self.cell_data_num.append(data_num)
                total_dfs.append(df)
        elif self.set_type == 1:
            # val
            for file in self.val_files:
                df = pd.read_csv(self.root_path + f'{file}')
                # The condition name of NCA dataset can be directly obtained from the file name. The condition name of the LFP dataset 
                # is obtained from the record
                if 'NC' in self.root_path:
                    condition = file.split('#')[0] 
                    df['file'] = domain_id
                elif 'NatureEnergy' in self.root_path:
                    cell_name = file.split('.')[0]
                    policy = fixed_files.NE_name_policy[cell_name]
                    domain_id = fixed_files.policy2id[policy]
                    df['file'] = domain_id
                elif 'EES' in self.root_path:
                    cell_name = file
                    policy = fixed_files.EES_name_policy[cell_name]
                    domain_id = fixed_files.EES_policy2id[policy]
                    df['file'] = domain_id
                if 'NC' in self.root_path:
                    first_life_df = df.loc[df['Qd'] >= 0.8 * 3.5]
                    df['cycle distance'] = first_life_df['cycle number'].max() - df['cycle number']
                    if condition not in seen_conditions:
                        seen_conditions.append(condition)
                        domain_id += 1
                else:
                    df['cycle distance'] = df['cycle number'].max() - df['cycle number']
                time = df['cost_time'].values
                df['cost_time'] = [sum(time[:index + 1]) for index, _ in enumerate(time)]
                self.cell_len.append(len(df))
                data_num = len(df) - self.seq_len
                self.cell_data_num.append(data_num)
                total_dfs.append(df)
        elif self.set_type == 2 or self.set_type == 3:
            # test or predict
            for file in self.test_files:
                df = pd.read_csv(self.root_path + f'{file}')
                # The condition name of NCA dataset can be directly obtained from the file name. The condition name of the LFP dataset 
                # is obtained from the record
                if 'NC' in self.root_path:
                    condition = file.split('#')[0] 
                    df['file'] = domain_id
                elif 'NatureEnergy' in self.root_path:
                    cell_name = file.split('.')[0]
                    policy = fixed_files.NE_name_policy[cell_name]
                    domain_id = fixed_files.policy2id[policy]
                    df['file'] = domain_id
                elif 'EES' in self.root_path:
                    cell_name = file
                    policy = fixed_files.EES_name_policy[cell_name]
                    domain_id = fixed_files.EES_policy2id[policy]
                    df['file'] = domain_id
                if 'NC' in self.root_path:
                    first_life_df = df.loc[df['Qd'] >= 0.8 * 3.5]
                    df['cycle distance'] = first_life_df['cycle number'].max() - df['cycle number']
                    if condition not in seen_conditions:
                        seen_conditions.append(condition)
                        domain_id += 1
                else:
                    df['cycle distance'] = df['cycle number'].max() - df['cycle number']

                time = df['cost_time'].values
                df['cost_time'] = [sum(time[:index + 1]) for index, _ in enumerate(time)]
                self.cell_len.append(len(df))
                data_num = len(df) - self.seq_len
                self.cell_data_num.append(data_num)
                total_dfs.append(df)
        else:
            # set_files
            for file in self.set_files:
                df = pd.read_csv(self.root_path + f'{file}')
                # The condition name of NCA dataset can be directly obtained from the file name. The condition name of the LFP dataset 
                # is obtained from the record
                if 'NC' in self.root_path:
                    condition = file.split('#')[0] 
                    df['file'] = domain_id
                elif 'NatureEnergy' in self.root_path:
                    cell_name = file.split('.')[0]
                    policy = fixed_files.NE_name_policy[cell_name]
                    domain_id = fixed_files.policy2id[policy]
                    df['file'] = domain_id
                elif 'EES' in self.root_path:
                    cell_name = file
                    policy = fixed_files.EES_name_policy[cell_name]
                    domain_id = fixed_files.EES_policy2id[policy]
                    df['file'] = domain_id
                if 'NC' in self.root_path:
                    first_life_df = df.loc[df['Qd'] >= 0.8 * 3.5]
                    df['cycle distance'] = first_life_df['cycle number'].max() - df['cycle number']
                    if condition not in seen_conditions:
                        seen_conditions.append(condition)
                        domain_id += 1
                else:
                    df['cycle distance'] = df['cycle number'].max() - df['cycle number']
                time = df['cost_time'].values
                df['cost_time'] = [sum(time[:index + 1]) for index, _ in enumerate(time)]
                self.cell_len.append(len(df))
                data_num = len(df) - self.seq_len
                self.cell_data_num.append(data_num)
                total_dfs.append(df)
        if not self.set_type == 4:
            print(f"Using historical {self.seq_len} cycles to predict {self.pred_len} in the future."
                  f" Under this setting, there are {np.sum(np.array(self.cell_data_num) != 0)} cells left")
        # self.original_cell_data_num = copy.deepcopy(self.cell_data_num)
        # if self.set_type == 0:
        #     self.cell_data_num = [self.pred_len if i < self.pred_len else i for i in self.cell_data_num] # upsampling
        total_df = pd.concat(total_dfs)
        self.max_Ed_in_train = np.max(total_df[['Ed']].values)
        if self.scale:
            scale_df = []  # only use the training data
            for file in self.train_files:
                df = pd.read_csv(self.root_path + f'{file}')
                if 'NC' in self.root_path:
                    first_life_df = df.loc[df['Qd'] >= 0.8 * 3.5]
                    df['cycle distance'] = first_life_df['cycle number'].max() - df['cycle number']
                else:
                    df['cycle distance'] = df['cycle number'].max() - df['cycle number']
                scale_df.append(df)
            scale_df = pd.concat(scale_df)
            feat_df = scale_df[['Qd', 'Ed']].values
            cycle_life_df = scale_df['cycle distance'].values.reshape(-1, 1)
            self.cycle_life_scaler.fit(cycle_life_df)
            self.scaler.fit(feat_df)
            data = self.scaler.transform(total_df[['Qd', 'Ed']].values)
            cycle_distance_label = self.cycle_life_scaler.transform(total_df['cycle distance'].values.reshape(-1, 1))
        else:
            data = total_df[['Qd', 'Ed']].values
            cycle_distance_label = total_df['cycle distance'].values.reshape(-1, 1)
        cost_times = total_df['cost_time'].values.reshape(-1, 1)
        data = np.concatenate([data, cost_times], axis=1)
        data_stamp = total_df['cycle number'].values.reshape(-1, 1)
        self.data_x = data
        self.data_y = data
        self.data_file_id = total_df['file'].values.reshape(-1, 1)

        self.data_stamp = data_stamp
        self.files_record = total_df['file'].values
        self.cycle_distance_label = cycle_distance_label

    def __getitem__(self, index):
        # get a sample accoding to the given index
        base_begin = 0
        for cell_index, cell_bound in enumerate(self.cell_data_num):
            data_before = sum(self.cell_data_num[:cell_index + 1])
            if index < data_before:
                # base_begin = self.cell_data_num[cell_index - 1] if cell_index != 0 else 0
                # 需要考虑有一个电池提供的数据因为不足self.pred_len+self.seq_len而为0时的特殊情况。
                # base_begin = cell_index * (self.pred_len + self.seq_len) - cell_index
                if cell_index != 0:
                    # factor = np.sum((np.array(self.cell_data_num[:cell_index]) != 0))
                    # base_begin = factor * (self.pred_len + self.seq_len) - factor
                    base_begin = np.sum(self.cell_len[:cell_index]) - np.sum(self.cell_data_num[:cell_index])
                else:
                    base_begin = 0
                break
        s_begin = index + base_begin
        s_end = s_begin + self.seq_len
        # if s_end >= np.sum(self.cell_len[:cell_index+1]):
        #     # This is a short cell that needs upsampling.
        #     s_begin = np.sum(self.cell_len[:cell_index]) + np.random.randint(0, self.original_cell_data_num[cell_index], 1)[0]
        #     s_end = s_begin + self.seq_len
        #     s_begin = int(s_begin)
        #     s_end = int(s_end)
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        if r_end > np.sum(self.cell_len[:cell_index + 1]):
            r_end = int(np.sum(self.cell_len[:cell_index + 1]))
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end].astype(np.float64)
        seq_y_mark = self.data_stamp[r_begin:r_end].astype(np.float64)
        seq_y_file_id = self.data_file_id[r_begin:r_end].astype(np.float64)
        seq_y_cycle_distance = self.cycle_distance_label[r_begin:r_end].astype(np.float64)
        masks = np.ones_like(seq_y)[:, :2]
        if seq_y.shape[0] != (self.label_len + self.pred_len):
            filled_data = np.zeros((self.label_len + self.pred_len - seq_y.shape[0], seq_y.shape[1]))
            masks = np.concatenate([np.ones_like(seq_y), np.zeros_like(filled_data)], axis=0)[:, :2]
            seq_y = np.concatenate([seq_y, filled_data], axis=0)
            seq_y_file_id = np.concatenate([seq_y_file_id, np.ones_like(filled_data[:, :1]) * seq_y_file_id[0][0]])
            seq_y_mark_max = int(np.max(seq_y_mark.reshape(-1))) + 1
            filled_seq_y_mark = np.array([float(i) for i in range(seq_y_mark_max,
                                                                  seq_y_mark_max + self.label_len + self.pred_len -
                                                                  seq_y_mark.shape[0])]).reshape(-1, 1)
            seq_y_mark = np.concatenate([seq_y_mark, filled_seq_y_mark], axis=0)
            filled_seq_y_cycle_distance = np.ones_like(filled_data[:, :1])
            seq_y_cycle_distance = np.concatenate([seq_y_cycle_distance, filled_seq_y_cycle_distance], axis=0)
        seq_y = np.concatenate([seq_y, masks, seq_y_file_id, seq_y_cycle_distance], axis=1)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return sum(self.cell_data_num)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

if __name__ == '__main__':
    pass
