import copy
import math
import random
from data_provider.data_loader import fixed_files
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, Reformer, PIInformer, PIInfoEncoder, OneShotLSTM, vLSTM, vCNN, \
    OSLSTM
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_QE_vs_cycle, visual_one_cell, clone_module, \
    maml_update, HiddenPrints, get_parameter_number
from utils.custom_loss import PILoss
import torch.autograd as autograd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from utils.kmeans_pytorch import kmeans
import torch
import torch.nn as nn
from torch import optim
import pickle
import os
import time
import fitlog
import warnings
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')
def set_ax_font_size(ax, fontsize=10):
    ax.tick_params(axis='y',
                 labelsize=fontsize # y轴字体大小设置
                  )
    ax.tick_params(axis='x',
                 labelsize=fontsize # x轴字体大小设置
                  )

def set_ax_linewidth(ax, bw=1.5):
    ax.spines['bottom'].set_linewidth(bw)
    ax.spines['left'].set_linewidth(bw)
    ax.spines['top'].set_linewidth(bw)
    ax.spines['right'].set_linewidth(bw)

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'PIInformer': PIInformer,
            'Reformer': Reformer,
            'OSLSTMv2': OSLSTM,
            'vLSTM': vLSTM,
            'vCNN': vCNN
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        self.model = model.to(self.device)
        return

    def _get_data(self, flag, set_data=''):
        data_set, data_loader = data_provider(self.args, flag, set_data)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.meta_train:
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.meta_learning_rate,
                                      weight_decay=self.args.wd)
        elif self.args.FT:
            model_optim = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                      lr=self.args.FT_learning_rate,
                                      weight_decay=self.args.wd)
        else:
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.wd)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss(reduction='none')
        # if self.args.loss == 'PI':
        #     criterion = PI_loss
        return criterion

    def vali(self, vali_data, vali_loader, criterion, epoch):
        total_raw_loss = []
        total_proportion_loss = []
        total_voltage_limitation_loss = []
        preds = []
        trues = []
        cell_file_ids = []
        total_masks = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                masks = batch_y.float()[:, :, 3:5]
                cell_file_id = batch_y.float()[:, :, 5][:, :]  # [B,L]
                batch_x = batch_x.float().to(self.device)[:, :, :2]
                batch_y = batch_y.float()[:, :, :2]

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs, _, cycle_distance_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                masks = masks[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                masks = masks.detach().cpu()
                cell_file_id = cell_file_id.detach().cpu()[:, -self.args.pred_len:]

                preds += [pred]
                trues += [true]
                cell_file_ids += [cell_file_id]
                total_masks += [masks]

        preds = np.concatenate(preds, axis=0).reshape(-1, 2)
        trues = np.concatenate(trues, axis=0).reshape(-1, 2)
        total_masks = np.concatenate(total_masks, axis=0).reshape(-1, 2)
        total_loss = np.sum((total_masks * (preds - trues)) ** 2) / np.sum(total_masks != 0)
        self.model.train()
        return total_loss, 0, 0, 0

    def vali_new(self, set_files, setting):
        ''' This validation considers the loss on the cell-level '''
        cell_Qd_maes = []
        cell_Ed_maes = []
        cell_Qd_mapes = []
        cell_Ed_mapes = []
        self.model.eval()
        with torch.no_grad():
            for file in set_files:
                self.args.set_files = [file]

                cell_Qd_mae, cell_Qd_mape, cell_Qd_mae_std, cell_Qd_mape_std, cell_Ed_mae, cell_Ed_mape, cell_Ed_mae_std, cell_Ed_mape_std, gt_trajectory, pred_trajectory, r2_Qd, r2_Ed = self.predict_specific_cell_new(
                    setting, load=False,
                    save_path='')
                cell_Qd_maes.append(cell_Qd_mae)
                cell_Ed_maes.append(cell_Ed_mae)
                cell_Qd_mapes.append(cell_Qd_mape)
                cell_Ed_mapes.append(cell_Ed_mape)
        # total_loss = (np.mean(cell_Qd_maes) + np.mean(cell_Ed_maes)) / 2
        total_loss = (np.mean(cell_Qd_mapes) + np.mean(cell_Ed_mapes)) / 2
        self.model.train()
        return total_loss, 0, 0, 0

    def vali_new_robust(self, set_files, setting, load=False):
        ''' This validation considers the loss on the condition-level '''
        cell_Qd_maes = {}
        cell_Ed_maes = {}
        cell_Qd_mapes = {}
        cell_Ed_mapes = {}
        self.model.eval()
        with torch.no_grad():
            for file in set_files:
                self.args.set_files = [file]
                if 'NC' in self.args.root_path:
                    condition = file.split('#')[0] 
                else:
                    cell_name = file.split('.')[0]
                    condition = fixed_files.NE_name_policy[cell_name]
                cell_Qd_mae, cell_Qd_mape, cell_Qd_mae_std, cell_Qd_mape_std, cell_Ed_mae, cell_Ed_mape, cell_Ed_mae_std, cell_Ed_mape_std, gt_trajectory, pred_trajectory, r2_Qd, r2_Ed, alpha_Qd, alpha_Ed, _ = self.predict_use_during_training(
                    setting, load=load,
                    save_path='')
                cell_Qd_mapes[condition] = cell_Qd_mapes.get(condition, []) + [cell_Qd_mape]
                cell_Ed_mapes[condition] = cell_Ed_mapes.get(condition, []) + [cell_Ed_mape]
        mean_cell_Qd_mapes = 0
        mean_cell_Qd_mape_stds = []
        mean_cell_Ed_mapes = 0
        mean_cell_Ed_mape_stds = []
        for key, value in cell_Qd_mapes.items():
            mean_cell_Qd_mapes += np.mean(value)
            mean_cell_Qd_mape_stds += [np.mean(value)]
            mean_cell_Ed_mapes += np.mean(cell_Ed_mapes[key])
            mean_cell_Ed_mape_stds += [np.mean(cell_Ed_mapes[key])]
        mean_cell_Qd_mapes = mean_cell_Qd_mapes / len(cell_Qd_mapes)
        mean_cell_Ed_mapes = mean_cell_Ed_mapes / len(cell_Ed_mapes)
        total_loss = (mean_cell_Qd_mapes + mean_cell_Ed_mapes) / 2
        self.model.train()
        return total_loss, 0, 0, 0

    def vali_new_weighted(self, set_files, setting):
        ''' This validation considers the loss on the cell-level '''
        cell_Qd_maes = {}
        cell_Ed_maes = {}
        cell_Qd_mapes = {}
        cell_Ed_mapes = {}
        self.model.eval()
        with torch.no_grad():
            for file in set_files:
                self.args.set_files = [file]
                condition = file.split('#')[0]
                cell_Qd_mae, cell_Qd_mape, cell_Qd_mae_std, cell_Qd_mape_std, cell_Ed_mae, cell_Ed_mape, cell_Ed_mae_std, cell_Ed_mape_std, gt_trajectory, pred_trajectory, r2_Qd, r2_Ed = self.predict_specific_cell_new(
                    setting, load=False,
                    save_path='')
                cell_Qd_maes[condition] = cell_Qd_maes.get(condition, []) + [cell_Qd_mae]
                cell_Qd_mapes[condition] = cell_Qd_mapes.get(condition, []) + [cell_Qd_mape]
                cell_Ed_maes[condition] = cell_Ed_maes.get(condition, []) + [cell_Ed_mae]
                cell_Ed_mapes[condition] = cell_Ed_mapes.get(condition, []) + [cell_Ed_mape]
        mean_cell_Qd_mapes = 0
        mean_cell_Qd_mape_stds = []
        for key, value in cell_Qd_mapes.items():
            mean_cell_Qd_mapes += np.mean(value)
            mean_cell_Qd_mape_stds += [np.mean(value)]
        mean_cell_Qd_mapes = mean_cell_Qd_mapes / len(cell_Qd_mapes)

        mean_cell_Ed_mapes = 0
        mean_cell_Ed_mape_stds = []
        for key, value in cell_Ed_mapes.items():
            mean_cell_Ed_mapes += np.mean(value)
            mean_cell_Ed_mape_stds += [np.mean(value)]
        mean_cell_Ed_mapes = mean_cell_Ed_mapes / len(cell_Ed_mapes)
        total_loss = (mean_cell_Qd_mapes + mean_cell_Ed_mapes) / 2
        self.model.train()
        return total_loss, 0, 0, 0

    def train_meta_pre_clustering_parallel_new_robust(self, setting):
        '''

        Args:
            setting:
            train_stage: 'train' by default. You can also use fine-tune

        Returns:

        '''
        train_data, train_loader = self._get_data(flag='train')
        self.args.std = train_data.scaler.scale_
        self.args.mean = train_data.scaler.mean_
        self.args.max_Ed_in_train = train_data.max_Ed_in_train
        self._build_model()  # build the model
        path = os.path.join(self.args.checkpoints, setting)
        print(f'Save name:{path}')
        if self.args.FT:
            Exception('A FT based on meta learning is not implemented!')
        if not os.path.exists(path):
            os.makedirs(path)
        get_parameter_number(self.model)
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        best_score = float('-inf')
        if not self.args.FT:
            lr = self.args.learning_rate if not self.args.lr_align else self.args.meta_learning_rate
        else:
            lr = self.args.FT_learning_rate
        iter_count = 0
        for epoch in range(self.args.train_epochs):
            train_loss = []
            train_raw_loss = []
            train_p_loss = []
            train_v_loss = []

            self.model.train()

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                self.clone_model = clone_module(self.model)
                self.clone_model.train()
                iter_count += 1
                model_optim.zero_grad()
                masks = batch_y.float()[:, :, 3:5]
                cell_file_id = batch_y.float()[:, :, 5]
                cycle_distance_label = batch_y.float().to(self.device)[:, :, 6][:, self.args.seq_len - 1].unsqueeze(-1)
                cell_file_id_array = cell_file_id[:, 0].detach().cpu().numpy()
                total_different_files = list(
                    set(list(cell_file_id_array)))  # the different files contained in this batch
                test_domain_num = int(len(total_different_files) * self.args.val_ratio) if int(
                    len(total_different_files) * self.args.val_ratio) >= 1 else 1
                test_domain_id = random.sample(total_different_files, test_domain_num)
                train_domain_id = [i for i in total_different_files if i not in test_domain_id]
                batch_x = batch_x.float().to(self.device)[:, :, :2]
                batch_y = batch_y.float().to(self.device)[:, :, :2]

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # split the batch into support set and query set
                train_domain_mask = torch.zeros_like(cell_file_id[:, 0])
                for id in train_domain_id:
                    train_domain_mask += cell_file_id[:, 0] == id
                train_domain_mask = train_domain_mask == 1
                test_domain_mask = torch.zeros_like(cell_file_id[:, 0])
                for id in test_domain_id:
                    test_domain_mask += cell_file_id[:, 0] == id
                test_domain_mask = test_domain_mask == 1
                # get the meta-train data
                train_domain_cluster_batch_x = batch_x[train_domain_mask]
                train_domain_cluster_batch_y = batch_y[train_domain_mask]
                train_domain_cluster_batch_x_mark = batch_x_mark[train_domain_mask]
                train_domain_cluster_batch_y_mark = batch_y_mark[train_domain_mask]
                train_domain_cluster_masks = masks[train_domain_mask]
                train_domain_cluster_file_id = cell_file_id[train_domain_mask][:, 0]
                train_domain_cluster_cycle_distance_label = cycle_distance_label[train_domain_mask]
                # get the meta-test data
                test_domain_cluster_batch_x = batch_x[test_domain_mask]
                test_domain_cluster_batch_y = batch_y[test_domain_mask]
                test_domain_cluster_batch_x_mark = batch_x_mark[test_domain_mask]
                test_domain_cluster_batch_y_mark = batch_y_mark[test_domain_mask]
                test_domain_cluster_masks = masks[test_domain_mask]
                test_domain_cluster_file_id = cell_file_id[test_domain_mask][:, 0]
                test_domain_cluster_cycle_distance_label =  cycle_distance_label[test_domain_mask]

                train_cell_file_id_array = train_domain_cluster_file_id.detach().cpu().numpy()
                total_Metatrain_different_files = list(
                    set(list(train_cell_file_id_array)))  # the different files contained in meta-training domains
                total_loss = 0.0
                total_trn_loss = 0.0
                total_test_loss = 0.0
                # fast adapt
                batch_x, batch_y, batch_x_mark, batch_y_mark, masks, cycle_distance_label = train_domain_cluster_batch_x, train_domain_cluster_batch_y, train_domain_cluster_batch_x_mark, train_domain_cluster_batch_y_mark, train_domain_cluster_masks, train_domain_cluster_cycle_distance_label
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs, _, _, cycle_distance_outputs = self.clone_model(batch_x, batch_x_mark, dec_inp,
                                                                             batch_y_mark)
                else:
                    outputs, _, cycle_distance_outputs = self.clone_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                masks = masks[:, -self.args.pred_len:, :].to(self.device)
                if self.args.loss == 'mse':
                    loss = criterion(outputs, batch_y)
                    loss = torch.sum(loss * masks) / torch.sum(masks != 0)
                    raw_loss, proportion_loss, voltage_limitation_loss = loss - loss, loss - loss, loss - loss
                elif self.args.loss == 'wmse':
                    # weighted mse
                    loss = criterion(outputs, batch_y)
                    tmp_loss = 0
                    for file_i in total_Metatrain_different_files:
                        meta_train_domain_mask = train_domain_cluster_file_id == file_i
                        meta_train_domain_loss_mask = masks[meta_train_domain_mask]
                        meta_train_domain_loss = loss[meta_train_domain_mask]
                        tmp_loss += torch.sum(meta_train_domain_loss * meta_train_domain_loss_mask) / torch.sum(
                            meta_train_domain_loss_mask != 0)
                    loss = tmp_loss / len(total_Metatrain_different_files)
                    raw_loss, proportion_loss, voltage_limitation_loss = loss - loss, loss - loss, loss - loss
                elif self.args.loss == 'amse':
                    loss1 = criterion(outputs, batch_y)
                    loss1 = torch.sum(loss1 * masks) / torch.sum(masks != 0)
                    loss2 = criterion(cycle_distance_outputs, cycle_distance_label)
                    loss2 = torch.sum(loss2) / loss2.shape[0]
                    loss = loss1 + self.args.auxiliary_gamma * loss2
                    raw_loss, proportion_loss, voltage_limitation_loss = loss - loss, loss - loss, loss - loss
                elif self.args.loss == 'awmse':
                    loss1 = criterion(outputs, batch_y)
                    loss2 = criterion(cycle_distance_outputs, cycle_distance_label)
                    loss = 0
                    for file_i in total_Metatrain_different_files:
                        meta_train_domain_mask = train_domain_cluster_file_id == file_i
                        meta_train_domain_loss_mask = masks[meta_train_domain_mask]
                        meta_train_domain_loss = loss1[meta_train_domain_mask]
                        # MSE Loss
                        loss += torch.sum(meta_train_domain_loss * meta_train_domain_loss_mask) / torch.sum(
                            meta_train_domain_loss_mask != 0) / len(total_Metatrain_different_files)
                        
                        # TDDG Loss
                        meta_train_domain_loss = loss2[meta_train_domain_mask]
                        loss += self.args.auxiliary_gamma * torch.sum(meta_train_domain_loss) / meta_train_domain_loss.shape[0] / len(
                            total_Metatrain_different_files)

                    raw_loss, proportion_loss, voltage_limitation_loss = loss - loss, loss - loss, loss - loss
                train_loss.append(loss.item())
                train_raw_loss.append(raw_loss.item())
                train_p_loss.append(proportion_loss.item())
                train_v_loss.append(voltage_limitation_loss.item())

                total_trn_loss += loss

                # update the clone_model
                # we get the gradients of model parameters using autograd.grad
                diff_params = [p for p in self.clone_model.parameters() if p.requires_grad]
                grad_params = autograd.grad(loss,
                                            diff_params,
                                            retain_graph=True,
                                            create_graph=True, allow_unused=True)
                gradients = []
                grad_counter = 0

                # Handles gradients for non-differentiable parameters
                for param in self.clone_model.parameters():
                    if param.requires_grad:
                        gradient = grad_params[grad_counter]
                        grad_counter += 1
                    else:
                        gradient = None
                    gradients.append(gradient)
                self.clone_model = maml_update(self.clone_model, lr, gradients)
                # self.clone_model.eval()

                test_cell_file_id_array = test_domain_cluster_file_id.detach().cpu().numpy()
                total_Metatest_different_files = list(
                    set(list(test_cell_file_id_array)))  # the different files contained in meta-training domains
                batch_x, batch_y, batch_x_mark, batch_y_mark, masks, cycle_distance_label = test_domain_cluster_batch_x, test_domain_cluster_batch_y, test_domain_cluster_batch_x_mark, test_domain_cluster_batch_y_mark, test_domain_cluster_masks, test_domain_cluster_cycle_distance_label
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs, _, _cycle_distance_outputs = self.clone_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs, _, cycle_distance_outputs = self.clone_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                masks = masks[:, -self.args.pred_len:, :].to(self.device)
                if self.args.loss == 'mse':
                    loss = criterion(outputs, batch_y)
                    loss = torch.sum(loss * masks) / torch.sum(masks != 0)
                    raw_loss, proportion_loss, voltage_limitation_loss = loss - loss, loss - loss, loss - loss
                elif self.args.loss == 'wmse':
                    # weighted mse
                    loss = criterion(outputs, batch_y)
                    tmp_loss = 0
                    for file_i in total_Metatest_different_files:
                        meta_test_domain_mask = test_domain_cluster_file_id == file_i
                        meta_test_domain_loss_mask = masks[meta_test_domain_mask]
                        meta_test_domain_loss = loss[meta_test_domain_mask]
                        tmp_loss += torch.sum(meta_test_domain_loss * meta_test_domain_loss_mask) / torch.sum(
                            meta_test_domain_loss_mask != 0)
                    loss = tmp_loss / len(total_Metatest_different_files)
                    raw_loss, proportion_loss, voltage_limitation_loss = loss - loss, loss - loss, loss - loss
                elif self.args.loss == 'amse':
                    loss1 = criterion(outputs, batch_y)
                    loss1 = torch.sum(loss1 * masks) / torch.sum(masks != 0)
                    loss2 = criterion(cycle_distance_outputs, cycle_distance_label)
                    loss2 = torch.sum(loss2) / loss2.shape[0]
                    loss = loss1 + self.args.auxiliary_gamma * loss2
                    raw_loss, proportion_loss, voltage_limitation_loss = loss - loss, loss - loss, loss - loss
                elif self.args.loss == 'awmse':
                    loss1 = criterion(outputs, batch_y)
                    loss2 = criterion(cycle_distance_outputs, cycle_distance_label)
                    loss = 0
                    for file_i in total_Metatest_different_files:
                        meta_test_domain_mask = test_domain_cluster_file_id == file_i
                        meta_test_domain_loss_mask = masks[meta_test_domain_mask]
                        meta_test_domain_loss = loss1[meta_test_domain_mask]
                        # MSE Loss
                        loss += torch.sum(meta_test_domain_loss * meta_test_domain_loss_mask) / torch.sum(
                            meta_test_domain_loss_mask != 0) / len(total_Metatest_different_files)

                        # TDDG Loss
                        meta_test_domain_loss = loss2[meta_test_domain_mask]
                        loss += self.args.auxiliary_gamma * torch.sum(meta_test_domain_loss) / meta_test_domain_loss.shape[0] / len(
                            total_Metatest_different_files)

                    raw_loss, proportion_loss, voltage_limitation_loss = loss - loss, loss - loss, loss - loss
                train_loss.append(loss.item())
                train_raw_loss.append(raw_loss.item())
                train_p_loss.append(proportion_loss.item())
                train_v_loss.append(voltage_limitation_loss.item())

                total_test_loss += loss

                total_loss = total_trn_loss + self.args.meta_beta * total_test_loss
                model_optim.zero_grad()
                total_loss.backward()
                model_optim.step()
                if (i + 1) % 20 == 0:
                    print(
                        f'\titers: {i + 1}, epoch: {epoch + 1} | loss: {total_loss.item():.4f}, raw_loss: {raw_loss.item():.4f}'
                        f' proportion_loss: {proportion_loss.item():.4f}, voltage_limitation_loss: {voltage_limitation_loss.item():.4f}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    # iter_count = 0
                    time_now = time.time()

                # adjust_learning_rate(model_optim, epoch + 1, self.args)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_raw_loss = np.average(train_raw_loss)
            train_p_loss = np.average(train_p_loss)
            train_v_loss = np.average(train_v_loss)
            vali_set_files = train_data.val_files
            test_set_files = train_data.test_files
            vali_loss, vali_total_raw_loss, vali_total_proportion_loss, vali_total_voltage_limitation_loss = self.vali_new_robust(
                vali_set_files, setting)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} | Train raw loss: {train_raw_loss:.7f} | "
                f"Train_p_loss: {train_p_loss:.7f} | Train_v_loss: {train_v_loss:.7f}\n"
                f"Vali Loss: {vali_loss:.7f} | Vali raw loss: {vali_total_raw_loss:.7f} | Vali_p_loss: {vali_total_proportion_loss:.7f} | Vali_v_loss: {vali_total_voltage_limitation_loss:.7f}\n")
            if -vali_loss > best_score:
                best_score = -vali_loss  # update the score
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # if self.args.lr_align:
            #     lr = adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        vali_set_files = train_data.val_files
        test_set_files = train_data.test_files
        vali_loss, vali_total_raw_loss, vali_total_proportion_loss, vali_total_voltage_limitation_loss = self.vali_new_robust(
                vali_set_files, setting)
        test_loss, test_total_raw_loss, test_total_proportion_loss, test_total_voltage_limitation_loss = self.vali_new_robust(
                test_set_files, setting,load=True)
        print(path)
        return self.model

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        self.args.std = train_data.scaler.scale_
        self.args.mean = train_data.scaler.mean_
        self.args.max_Ed_in_train = train_data.max_Ed_in_train
        self._build_model()  # build the model
        path = os.path.join(self.args.checkpoints, setting)
        if self.args.FT:
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            frozen_names = ['enc_embedding', 'encoder', 'linear']
            for name, parameter in self.model.named_parameters():
                for frozen_name in frozen_names:
                    if frozen_name in name:
                        parameter.requires_grad = False
            path = os.path.join(self.args.checkpoints, 'FT_' + setting)

        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        best_score = float('-inf')
        iter_count = 0
        for epoch in range(self.args.train_epochs):
            train_loss = []
            train_raw_loss = []
            train_p_loss = []
            train_v_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                masks = batch_y.float()[:, :, 3:5]
                cell_file_id = batch_y.float()[:, :, 5]
                cell_file_id_array = cell_file_id[:, 0].detach().cpu().numpy()
                cycle_distance_label = batch_y.float().to(self.device)[:, :, 6][:, self.args.seq_len - 1].unsqueeze(-1)
                total_different_files = list(
                    set(list(cell_file_id_array)))  # the different files contained in this batch
                cost_time = batch_y.float().to(self.device)[:, :, 2]
                batch_x = batch_x.float().to(self.device)[:, :, :2]
                batch_y = batch_y.float().to(self.device)[:, :, :2]

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs, _, _, cycle_distance_outputs = self.model(batch_x, batch_x_mark, dec_inp,
                                                                       batch_y_mark)
                else:
                    outputs, _, cycle_distance_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                masks = masks[:, -self.args.pred_len:, :].to(self.device)
                if self.args.loss == 'mse':
                    loss = criterion(outputs, batch_y)
                    loss = torch.sum(loss * masks) / torch.sum(masks != 0)
                elif self.args.loss == 'wmse':
                    # weighted mse
                    loss = criterion(outputs, batch_y)
                    tmp_loss = 0
                    for file_i in total_different_files:
                        domain_mask = (cell_file_id[:, 0] == file_i)
                        domain_loss_mask = masks[domain_mask]
                        domain_loss = loss[domain_mask]
                        tmp_loss += torch.sum(domain_loss * domain_loss_mask) / torch.sum(domain_loss_mask != 0)
                    loss = tmp_loss / len(total_different_files)
                elif self.args.loss == 'amse':
                    loss1 = criterion(outputs, batch_y)
                    loss1 = torch.sum(loss1 * masks) / torch.sum(masks != 0)
                    loss2 = criterion(cycle_distance_outputs, cycle_distance_label)
                    loss2 = torch.sum(loss2) / loss2.shape[0]
                    loss = loss1 + self.args.auxiliary_gamma * loss2
                elif self.args.loss == 'awmse':
                    loss1 = criterion(outputs, batch_y)
                    tmp_loss = 0
                    for file_i in total_different_files:
                        domain_mask = (cell_file_id[:, 0] == file_i)
                        domain_loss_mask = masks[domain_mask]
                        domain_loss = loss1[domain_mask]
                        tmp_loss += torch.sum(domain_loss * domain_loss_mask) / torch.sum(domain_loss_mask != 0) / len(
                            total_different_files)
                    loss1 = tmp_loss
                    loss2 = criterion(cycle_distance_outputs, cycle_distance_label)
                    tmp_loss = 0
                    for file_i in total_different_files:
                        domain_mask = (cell_file_id[:, 0] == file_i)
                        domain_loss = loss2[domain_mask]
                        tmp_loss += torch.sum(domain_loss) / domain_loss.shape[0] / len(total_different_files)
                    loss2 = tmp_loss
                    loss = loss1 + self.args.auxiliary_gamma * loss2
                raw_loss, proportion_loss, voltage_limitation_loss = loss - loss, loss - loss, loss - loss
                train_loss.append(loss.item())
                train_raw_loss.append(raw_loss.item())
                train_p_loss.append(proportion_loss.item())
                train_v_loss.append(voltage_limitation_loss.item())

                if (i + 1) % 50 == 0:
                    print(
                        f'\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.4f}, raw_loss: {raw_loss.item():.4f}'
                        f' proportion_loss: {proportion_loss.item():.4f}, voltage_limitation_loss: {voltage_limitation_loss.item():.4f}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    # iter_count = 0
                    time_now = time.time()

                loss.backward()
                # if epoch>=3 and self.args.gra_clip != 0:
                #     nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.gra_clip, norm_type=2)
                model_optim.step()

                # print('a')

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_raw_loss = np.average(train_raw_loss)
            train_p_loss = np.average(train_p_loss)
            train_v_loss = np.average(train_v_loss)
            if self.args.vali_loss == 'nw':
                vali_loss, vali_total_raw_loss, vali_total_proportion_loss, vali_total_voltage_limitation_loss = self.vali(
                    vali_data, vali_loader, criterion, epoch)
            elif self.args.vali_loss == 'w':
                vali_set_files = train_data.val_files
                test_set_files = train_data.test_files
                vali_loss, vali_total_raw_loss, vali_total_proportion_loss, vali_total_voltage_limitation_loss = self.vali_new(
                    vali_set_files, setting)
            elif self.args.vali_loss == 'wr':
                vali_set_files = train_data.val_files
                test_set_files = train_data.test_files
                vali_loss, vali_total_raw_loss, vali_total_proportion_loss, vali_total_voltage_limitation_loss = self.vali_new_robust(
                    vali_set_files, setting)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} | Train raw loss: {train_raw_loss:.7f} | "
                f"Train_p_loss: {train_p_loss:.7f} | Train_v_loss: {train_v_loss:.7f}\n"
                f"Vali Loss: {vali_loss:.7f} | Vali raw loss: {vali_total_raw_loss:.7f} | Vali_p_loss: {vali_total_proportion_loss:.7f} | Vali_v_loss: {vali_total_voltage_limitation_loss:.7f}\n")
            if -vali_loss > best_score:
                best_score = -vali_loss  # update the score
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if self.args.vali_loss == 'nw':
            vali_loss, vali_total_raw_loss, vali_total_proportion_loss, vali_total_voltage_limitation_loss = self.vali(
                    vali_data, vali_loader, criterion, epoch)
            test_loss, test_total_raw_loss, test_total_proportion_loss, test_total_voltage_limitation_loss = self.vali(
                    test_data, test_loader, criterion, epoch)
        elif self.args.vali_loss == 'w':
            vali_set_files = train_data.val_files
            test_set_files = train_data.test_files
            vali_loss, vali_total_raw_loss, vali_total_proportion_loss, vali_total_voltage_limitation_loss = self.vali_new(
                    vali_set_files, setting)
            test_loss, test_total_raw_loss, test_total_proportion_loss, test_total_voltage_limitation_loss = self.vali_new(
                    test_set_files, setting)
        elif self.args.vali_loss == 'wr':
            vali_set_files = train_data.val_files
            test_set_files = train_data.test_files
            vali_loss, vali_total_raw_loss, vali_total_proportion_loss, vali_total_voltage_limitation_loss = self.vali_new_robust(
                    vali_set_files, setting)
            test_loss, test_total_raw_loss, test_total_proportion_loss, test_total_voltage_limitation_loss = self.vali_new_robust(
                    test_set_files, setting)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        total_masks = []
        folder_path = './test_results/' + setting + '/'
        if self.args.FT:
            folder_path = './test_results/' + 'FT_' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                masks = batch_y.float()[:, :, 3:5]
                cost_time = batch_y.float().to(self.device)[:, :, 2]
                batch_x = batch_x.float().to(self.device)[:, :, :2]
                batch_y = batch_y.float().to(self.device)[:, :, :2]

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model != 'PIInformer':
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs, _, cycle_distance_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, attns, As, BS = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cost_time)[0]
                    else:
                        outputs, As, BS = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cost_time)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                masks = masks[:, -self.args.pred_len:, :].detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                total_masks.append(masks)
                if i % 20 == 0:
                    batch_size = true.shape[0]
                    input = test_data.inverse_transform(
                        batch_x.detach().cpu().numpy().reshape(-1, true.shape[-1])).reshape(batch_size, -1,
                                                                                            true.shape[-1])
                    true = test_data.inverse_transform(true.reshape(-1, true.shape[-1])).reshape(batch_size, -1,
                                                                                                 true.shape[-1])
                    pred = test_data.inverse_transform(pred.reshape(-1, true.shape[-1])).reshape(batch_size, -1,
                                                                                                 true.shape[-1])
                    gt = np.concatenate((input[0, :, 0], true[0, :, 0]), axis=0)
                    pd = np.concatenate((input[0, :, 0], pred[0, :, 0]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        total_masks = np.concatenate(total_masks, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-1])
        total_masks = total_masks.reshape(-1, total_masks.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        # currentDateAndTime = datetime.now()
        folder_path = './results/' + setting + '/'
        if self.args.FT:
            folder_path = './results/' + 'FT_' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(f'{folder_path}args.txt', 'w') as f:
            self.args.std = list(self.args.std)
            self.args.mean = list(self.args.mean)
            json.dump(self.args.__dict__, f, indent=2)
        pickle.dump(test_data.scaler, open(f'{folder_path}scaler.pkl', 'wb'))
        mse = np.sum((total_masks * (preds - trues)) ** 2) / np.sum(total_masks != 0)
        mae = np.sum((total_masks * (np.abs(preds - trues)))) / np.sum(total_masks != 0)
        print(f'mse:{mse} mae:{mae}')

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                cost_time = batch_y.float().to(self.device)[:, :, 2]
                batch_x = batch_x.float().to(self.device)[:, :, :2]
                batch_y = batch_y.float()[:, :, :2]

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model != 'PIInformer':
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, _, cycle_distance_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, attns, As, BS = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cost_time)[0]
                    else:
                        outputs, As, BS = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cost_time)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def analyze(self, setting, load=True, set_dataset=''):
        def analyze_bad_case(preds, trues, masks, bad_threshold=0.1):
            mae = np.abs(preds - trues) * masks
            print(f'The worst mae is {np.max(mae)}')
            return np.sum(mae > bad_threshold)

        pred_data, pred_loader = self._get_data(flag='pred', set_data=set_dataset)
        self._build_model()
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        if 'NC' in self.args.root_path:
            bad_threshold = 0.175
        else:
            bad_threshold = 0.055
        preds = []
        trues = []
        total_masks = []
        self.model.eval()
        best_score = float('-inf')
        best_gt = 0
        best_pd = 0
        best_marks = 0
        best_time_x = 0
        best_masks = 0

        worst_score = float('inf')
        worst_gt = 0
        worst_pd = 0
        worst_marks = 0
        worst_time_x = 0
        worst_masks = 0

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                masks = batch_y.float()[:, :, 3:5]
                time_x = batch_x[:, :, 2].detach().cpu().numpy()[0]
                cost_time = batch_y.float().to(self.device)[:, :, 2]
                batch_x = batch_x.float().to(self.device)[:, :, :2]
                batch_y = batch_y.float()[:, :, :2]

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model != 'PIInformer':
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, _, cycle_distance_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, attns, As, BS = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cost_time)[0]
                    else:
                        outputs, As, BS = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cost_time)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[0, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[0, -self.args.pred_len:, f_dim:]
                masks = masks[0, -self.args.pred_len:, :].detach().cpu().numpy()
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y.detach().cpu().numpy()
                neg_mae_score = - np.sum((masks * (np.abs(pred - masks)))) / np.sum(masks)
                original_truth = batch_x.detach().cpu().numpy()
                marks = batch_y_mark.detach().cpu().numpy()
                gt = np.concatenate((original_truth[0, :, :], true[:, :]), axis=0)
                pd = np.concatenate((original_truth[0, :, :], pred[:, :]), axis=0)
                gt = pred_data.inverse_transform(gt)
                pd = pred_data.inverse_transform(pd)
                preds.append(pred)
                trues.append(true)
                total_masks.append(masks)
                if i % 30 == 0:
                    tmp_masks = masks.reshape(1, -1, masks.shape[-1])
                    tmp_masks = np.concatenate(
                        [np.ones_like(original_truth[:, :, :]), tmp_masks], axis=1)[0]
                    plt_masks = np.where(tmp_masks != 0)
                    tmp_masks = masks.reshape(1, -1, masks.shape[-1])
                    plt_mark_masks = np.concatenate(
                        [np.ones_like(original_truth[:, :, :]), tmp_masks], axis=1)
                    plt_mark_masks = np.where(plt_mark_masks[:, :, :1] != 0)
                    plt_gt = gt[plt_masks].reshape(-1, 2)
                    plt_pd = pd[plt_masks].reshape(-1, 2)
                    plt_marks = marks[plt_mark_masks].reshape(1, -1, 1)
                    visual_QE_vs_cycle(plt_gt, plt_pd, plt_marks, self.args.seq_len, time_x,
                                       nominal_capacity=self.args.nominal_capacity)
                    pass
                if neg_mae_score > best_score:
                    best_score = neg_mae_score
                    tmp_masks = masks.reshape(1, -1, masks.shape[-1])
                    tmp_masks = np.concatenate(
                        [np.ones_like(original_truth[:, :, :]), tmp_masks], axis=1)[0]
                    best_masks = np.where(tmp_masks != 0)
                    tmp_masks = masks.reshape(1, -1, masks.shape[-1])
                    best_mark_masks = np.concatenate(
                        [np.ones_like(original_truth[:, :, :]), tmp_masks], axis=1)
                    best_mark_masks = np.where(best_mark_masks[:, :, :1] != 0)
                    best_gt = gt[best_masks].reshape(-1, 2)
                    best_pd = pd[best_masks].reshape(-1, 2)
                    best_marks = marks[best_mark_masks].reshape(1, -1, 1)
                    best_time_x = time_x

                if neg_mae_score < worst_score:
                    worst_score = neg_mae_score
                    tmp_masks = masks.reshape(1, -1, masks.shape[-1])
                    tmp_masks = np.concatenate(
                        [np.ones_like(original_truth[:, :, :]), tmp_masks], axis=1)[0]
                    worst_masks = np.where(tmp_masks != 0)
                    tmp_masks = masks.reshape(1, -1, masks.shape[-1])
                    worst_mark_masks = np.concatenate(
                        [np.ones_like(original_truth[:, :, :]), tmp_masks], axis=1)
                    worst_mark_masks = np.where(worst_mark_masks[:, :, :1] != 0)
                    worst_gt = gt[worst_masks].reshape(-1, 2)
                    worst_pd = pd[worst_masks].reshape(-1, 2)
                    worst_marks = marks[worst_mark_masks].reshape(1, -1, 1)
                    worst_time_x = time_x

        visual_QE_vs_cycle(best_gt, best_pd, best_marks, self.args.seq_len, best_time_x, title='Best case',
                           nominal_capacity=self.args.nominal_capacity)
        visual_QE_vs_cycle(worst_gt, worst_pd, worst_marks, self.args.seq_len, worst_time_x, title='Worst case',
                           nominal_capacity=self.args.nominal_capacity)
        preds = np.array(preds).reshape(-1, pred.shape[-1])
        trues = np.array(trues).reshape(-1, true.shape[-1])
        total_masks = np.array(total_masks)
        preds = pred_data.inverse_transform(preds)
        trues = pred_data.inverse_transform(trues)
        total_masks = total_masks.reshape(-1, total_masks.shape[-1])

        total_rmse = np.sqrt(np.sum((total_masks * (preds - trues)) ** 2) / np.sum(total_masks != 0))
        Qd_rmse = np.sqrt(np.sum((total_masks[:, 0] * (preds[:, 0] - trues[:, 0])) ** 2) / np.sum(total_masks[:, 0]))
        Ed_rmse = np.sqrt(np.sum((total_masks[:, 1] * (preds[:, 1] - trues[:, 1])) ** 2) / np.sum(total_masks[:, 1]))
        print(f'total RMSE:{total_rmse}')
        print(f'Qd RMSE:{Qd_rmse}')
        print(f'Ed RMSE:{Ed_rmse}\n')

        total_mae = np.sum((total_masks * (np.abs(preds - trues)))) / np.sum(total_masks != 0)
        Qd_mae = np.sum((total_masks[:, 0] * (np.abs(preds[:, 0] - trues[:, 0])))) / np.sum(total_masks[:, 0])
        Ed_mae = np.sum((total_masks[:, 1] * (np.abs(preds[:, 1] - trues[:, 1])))) / np.sum(total_masks[:, 1])
        print(f'total MAE:{total_mae}')
        print(f'Qd MAE:{Qd_mae}')
        print(f'Ed MAE:{Ed_mae}\n')

        total_mse = np.sum((total_masks * (preds - trues)) ** 2) / np.sum(total_masks != 0)
        Qd_mse = np.sum((total_masks[:, 0] * (preds[:, 0] - trues[:, 0])) ** 2) / np.sum(total_masks[:, 0])
        Ed_mse = np.sum((total_masks[:, 1] * (preds[:, 1] - trues[:, 1])) ** 2) / np.sum(total_masks[:, 1])
        print(f'total MSE:{total_mse}')
        print(f'Qd MSE:{Qd_mse}')
        print(f'Ed MSE:{Ed_mse}\n')

        bad_Qd_count = analyze_bad_case(preds[:, 0], trues[:, 0], total_masks[:, 0], bad_threshold=bad_threshold)
        print(f'Bad Qd prediction num is: {bad_Qd_count} among {np.sum(total_masks[:, 0])} predictions\n'
              f'The ratio is: {bad_Qd_count / np.sum(total_masks[:, 0])}')

        return

    def predict_specific_cells(self, setting, load=True, set_dataset='', save_path=''):
        def analyze_bad_case(preds, trues, masks, bad_threshold=0.1):
            mae = np.abs(preds - trues) * (masks != 0)
            print(f'The worst mae is {np.max(mae)}')
            return np.sum(mae > bad_threshold)

        pred_data, pred_loader = self._get_data(flag='set_files', set_data=set_dataset)
        self._build_model()
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        if 'NC' in self.args.root_path:
            bad_threshold = 0.175
        else:
            bad_threshold = 0.055
        preds = []
        trues = []
        self.model.eval()

        # record the trajectory data
        gt_trajectory = {}
        pred_trajectory = {}
        total_masks = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                masks = batch_y.float()[:, :, 3:5]
                cost_time = batch_y.float().to(self.device)[:, :, 2]
                batch_x = batch_x.float().to(self.device)[:, :, :2]
                batch_y = batch_y.float()[:, :, :2]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model != 'PIInformer':
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, _, cycle_distance_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, attns, As, BS = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cost_time)[0]
                    else:
                        outputs, As, BS = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cost_time)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[0, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[0, -self.args.pred_len:, f_dim:]
                masks = masks[:, -self.args.pred_len:, :].detach().cpu().numpy()
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y.detach().cpu().numpy()
                transformed_true = pred_data.inverse_transform(true)
                transformed_pred = pred_data.inverse_transform(pred)

                marks = batch_y_mark.detach().cpu().numpy()
                cycles = list(marks.reshape(-1)[-self.args.pred_len:])

                for index, cycle_number in enumerate(cycles):
                    if masks[0, index, 0] != 0:
                        gt_trajectory[cycle_number] = gt_trajectory.get(cycle_number, []) + [transformed_true[index]]
                        pred_trajectory[cycle_number] = pred_trajectory.get(cycle_number, []) + [
                            transformed_pred[index]]

                preds.append(pred)
                trues.append(true)
                total_masks.append(masks)

        visual_one_cell(gt_trajectory, pred_trajectory, save_path=save_path)
        preds = np.array(preds).reshape(-1, pred.shape[-1])
        trues = np.array(trues).reshape(-1, true.shape[-1])
        total_masks = np.array(total_masks)
        preds = pred_data.inverse_transform(preds)
        trues = pred_data.inverse_transform(trues)
        total_masks = total_masks.reshape(-1, total_masks.shape[-1])

        total_rmse = np.sqrt(np.sum((total_masks * (preds - trues)) ** 2) / np.sum(total_masks != 0))
        Qd_rmse = np.sqrt(np.sum((total_masks[:, 0] * (preds[:, 0] - trues[:, 0])) ** 2) / np.sum(total_masks[:, 0]))
        Ed_rmse = np.sqrt(np.sum((total_masks[:, 1] * (preds[:, 1] - trues[:, 1])) ** 2) / np.sum(total_masks[:, 1]))
        print(f'total RMSE:{total_rmse}')
        print(f'Qd RMSE:{Qd_rmse}')
        print(f'Ed RMSE:{Ed_rmse}\n')

        total_mae = np.sum((total_masks * (np.abs(preds - trues)))) / np.sum(total_masks != 0)
        Qd_mae = np.sum((total_masks[:, 0] * (np.abs(preds[:, 0] - trues[:, 0])))) / np.sum(total_masks[:, 0])
        Ed_mae = np.sum((total_masks[:, 1] * (np.abs(preds[:, 1] - trues[:, 1])))) / np.sum(total_masks[:, 1])
        print(f'total MAE:{total_mae}')
        print(f'Qd MAE:{Qd_mae}')
        print(f'Ed MAE:{Ed_mae}\n')

        total_mse = np.sum((total_masks * (preds - trues)) ** 2) / np.sum(total_masks != 0)
        Qd_mse = np.sum((total_masks[:, 0] * (preds[:, 0] - trues[:, 0])) ** 2) / np.sum(total_masks[:, 0])
        Ed_mse = np.sum((total_masks[:, 1] * (preds[:, 1] - trues[:, 1])) ** 2) / np.sum(total_masks[:, 1])
        print(f'total MSE:{total_mse}')
        print(f'Qd MSE:{Qd_mse}')
        print(f'Ed MSE:{Ed_mse}\n')

        bad_Qd_count = analyze_bad_case(preds[:, 0], trues[:, 0], total_masks[:, 0], bad_threshold=bad_threshold)
        print(f'Bad Qd prediction num is: {bad_Qd_count} among {np.sum(total_masks[:, 0])} predictions\n'
              f'The ratio is: {bad_Qd_count / np.sum(total_masks[:, 0])}')

        return bad_Qd_count / np.sum(total_masks[:, 0]), total_rmse

    def predict_specific_cell_new(self, setting, load=True, set_dataset='', save_path=''):
        def analyze_bad_case(preds, trues, masks, bad_threshold=0.1):
            mae = np.abs(preds - trues) * (masks != 0)
            print(f'The worst mae is {np.max(mae)}')
            return np.sum(mae > bad_threshold)

        pred_data, pred_loader = self._get_data(flag='set_files', set_data=set_dataset)
        if load:
            self._build_model()
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        if 'NC' in self.args.root_path:
            bad_threshold = 0.175
        else:
            bad_threshold = 0.055
        preds = []
        trues = []
        self.model.eval()

        # record the trajectory data
        gt_trajectory = {}
        pred_trajectory = {}
        total_masks = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                # temp_mark = batch_x_mark[0, 0, 0]
                # temp_mark -= 1
                # if not temp_mark % self.args.pred_len == 0:
                #     continue
                masks = batch_y.float()[:, :, 3:5]
                cost_time = batch_y.float().to(self.device)[:, :, 2]
                batch_x = batch_x.float().to(self.device)[:, :, :2]
                batch_y = batch_y.float()[:, :, :2]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs, *useless = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[0, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[0, -self.args.pred_len:, f_dim:]
                masks = masks[:, -self.args.pred_len:, :].detach().cpu().numpy()
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y.detach().cpu().numpy()
                transformed_true = pred_data.inverse_transform(true)
                transformed_pred = pred_data.inverse_transform(pred)

                marks = batch_y_mark.detach().cpu().numpy()
                cycles = list(marks.reshape(-1)[-self.args.pred_len:])

                for index, cycle_number in enumerate(cycles):
                    if masks[0, index, 0] != 0:
                        gt_trajectory[float(cycle_number)] = gt_trajectory.get(float(cycle_number), []) + [
                            transformed_true[index].tolist()]
                        pred_trajectory[float(cycle_number)] = pred_trajectory.get(float(cycle_number), []) + [
                            transformed_pred[index].tolist()]

                preds.append(pred)
                trues.append(true)
                total_masks.append(masks)

        cell_Qd_mae, cell_Qd_mape, cell_Qd_mae_std, cell_Qd_mape_std, cell_Ed_mae, cell_Ed_mape, cell_Ed_mae_std, cell_Ed_mape_std, r2_Qd, r2_Ed = visual_one_cell(
            gt_trajectory, pred_trajectory, save_path=save_path)

        return float(cell_Qd_mae), float(cell_Qd_mape), float(cell_Qd_mae_std), float(cell_Qd_mape_std), float(
            cell_Ed_mae), float(cell_Ed_mape), float(cell_Ed_mae_std), float(
            cell_Ed_mape_std), gt_trajectory, pred_trajectory, r2_Qd, r2_Ed

    def predict_use_during_training(self, setting, load=True, set_dataset='', save_path='', robust_threshold=2):
        pred_data, pred_loader = self._get_data(flag='set_files', set_data=set_dataset)
        if load:
            self._build_model()
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        # record the trajectory data
        gt_trajectory = {}
        pred_trajectory = {}
        total_count = 0
        bad_count = 0
        bad_count_Ed = 0
        total_mape_Qd = 0
        total_mape_Ed = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                # temp_mark = batch_x_mark[0, 0, 0]
                # temp_mark -= 1
                # if not temp_mark % self.args.pred_len == 0:
                #     continue
                masks = batch_y.float()[:, :, 3:5]
                cost_time = batch_y.float().to(self.device)[:, :, 2]
                batch_x = batch_x.float().to(self.device)[:, :, :2]
                batch_y = batch_y.float()[:, :, :2]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs, *useless = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[0, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[0, -self.args.pred_len:, f_dim:]
                masks = masks[:, -self.args.pred_len:, :].detach().cpu().numpy()
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y.detach().cpu().numpy()
                transformed_true = pred_data.inverse_transform(true)
                transformed_pred = pred_data.inverse_transform(pred)

                transformed_pred = transformed_pred[masks[0, :, 0] == 1]
                transformed_true = transformed_true[masks[0, :, 0] == 1]

                mapes = (np.abs(transformed_true - transformed_pred) / transformed_true)
                mapes_Qd = mapes[:, 0] * 100
                mapes_Ed = mapes[:, 1] * 100
                total_mape_Qd += np.sum(mapes_Qd)
                total_mape_Ed += np.sum(mapes_Ed)
                total_count += transformed_true.shape[0]

        new_cell_mapes_Qd = total_mape_Qd / total_count
        new_cell_mapes_Ed = total_mape_Ed / total_count

        return -1, new_cell_mapes_Qd, 0, 0, -1, new_cell_mapes_Ed, 0, 0, gt_trajectory, pred_trajectory, -1, -1, 1 - bad_count / total_count, 1 - bad_count_Ed / total_count, -1

    def predict_specific_cell_new_robust(self, setting, load=True, set_dataset='', save_path='', robust_threshold=2):
        '''
        Give the robust score instead of vanilla R2
        Args:
            setting:
            load:
            set_dataset:
            save_path:

        Returns:

        '''
        pred_data, pred_loader = self._get_data(flag='set_files', set_data=set_dataset)
        if load:
            self._build_model()
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        if 'NC' in self.args.root_path:
            bad_threshold = 0.175
        else:
            bad_threshold = 0.055
        preds = []
        trues = []
        self.model.eval()
        # record the trajectory data
        gt_trajectory = {}
        pred_trajectory = {}
        total_masks = []
        detailed_alphas = []
        total_count = 0
        r2_count = 0
        bad_count = 0
        bad_count_Ed = 0
        total_r2_Qd = 0
        total_r2_Ed = 0
        total_mae_Qd = 0
        total_mae_Ed = 0
        total_mape_Qd = 0
        total_mape_Ed = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                # temp_mark = batch_x_mark[0, 0, 0]
                # temp_mark -= 1
                # if not temp_mark % self.args.pred_len == 0:
                #     continue
                masks = batch_y.float()[:, :, 3:5]
                cost_time = batch_y.float().to(self.device)[:, :, 2]
                batch_x = batch_x.float().to(self.device)[:, :, :2]
                batch_y = batch_y.float()[:, :, :2]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs, *useless = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[0, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[0, -self.args.pred_len:, f_dim:]
                masks = masks[:, -self.args.pred_len:, :].detach().cpu().numpy()
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y.detach().cpu().numpy()
                transformed_true = pred_data.inverse_transform(true)
                transformed_pred = pred_data.inverse_transform(pred)

                marks = batch_y_mark.detach().cpu().numpy()
                cycles = list(marks.reshape(-1)[-self.args.pred_len:])

                for index, cycle_number in enumerate(cycles):
                    if masks[0, index, 0] != 0:
                        gt_trajectory[float(cycle_number)] = gt_trajectory.get(float(cycle_number), []) + [
                            transformed_true[index].tolist()]
                        pred_trajectory[float(cycle_number)] = pred_trajectory.get(float(cycle_number), []) + [
                            transformed_pred[index].tolist()]

                transformed_pred = transformed_pred[masks[0, :, 0] == 1]
                transformed_true = transformed_true[masks[0, :, 0] == 1]
                # fig = plt.figure(figsize=(7, 3))
                # plt_cycles = cycles[:len(transformed_pred)]
                # plt.subplot(1, 2, 1)
                # plt.xlabel('Cycle number')
                # plt.ylabel('Discharging capacity (Ah)')
                # plt.plot(plt_cycles, transformed_pred[:,0], label='predicted trajectory')
                # plt.plot(plt_cycles, transformed_true[:,0], label='measured trajectory')
                # set_ax_linewidth(plt.gca())
                # set_ax_font_size(plt.gca())
                # plt.subplot(1, 2, 2)
                # plt.xlabel('Cycle number')
                # plt.ylabel('Discharging energy (Wh)')
                # plt.plot(plt_cycles, transformed_pred[:,1], label='predicted trajectory')
                # plt.plot(plt_cycles, transformed_true[:,1], label='measured trajectory')
                # fig.tight_layout()  # 调整整体空白
                # plt.subplots_adjust(wspace=0.25, hspace=0)  # 调整子图间距
                # plt.legend()
                # set_ax_linewidth(plt.gca())
                # set_ax_font_size(plt.gca())
                # plt.savefig(f'./visual_figs/SI_figures/RBDPNet_{plt_cycles[0]}.pdf', bbox_inches='tight')
                # plt.show()

                mapes = (np.abs(transformed_true - transformed_pred) / transformed_true)
                maes = np.abs(transformed_true - transformed_pred)
                r2_Qd = r2_score(transformed_true[:, 0], transformed_pred[:, 0])
                r2_Ed = r2_score(transformed_true[:, 1], transformed_pred[:, 1])
                r2_count += 1
                maes_Qd = maes[:, 0]
                maes_Ed = maes[:, 1]
                mapes_Qd = mapes[:, 0] * 100
                mapes_Ed = mapes[:, 1] * 100

                total_mae_Qd += np.sum(maes_Qd)
                total_mae_Ed += np.sum(maes_Ed)
                total_mape_Qd += np.sum(mapes_Qd)
                total_mape_Ed += np.sum(mapes_Ed)
                total_r2_Qd += r2_Qd
                total_r2_Ed += r2_Ed
                bad_count += np.sum(mapes_Qd > robust_threshold)
                bad_count_Ed += np.sum(mapes_Ed > robust_threshold)
                total_count += transformed_true.shape[0]
                tmp_alpha_Qd, tmp_alpha_Ed = 1 - bad_count / total_count, 1 - bad_count_Ed / total_count
                detailed_alphas += [(tmp_alpha_Ed + tmp_alpha_Qd) / 2]
                preds.append(pred)
                trues.append(true)
                total_masks.append(masks)

        cell_Qd_mae, cell_Qd_mape, cell_Qd_mae_std, cell_Qd_mape_std, cell_Ed_mae, cell_Ed_mape, cell_Ed_mae_std, cell_Ed_mape_std, r2_Qd, r2_Ed = visual_one_cell(
            gt_trajectory, pred_trajectory, save_path=save_path)
        new_cell_maes_Qd = total_mae_Qd / total_count
        new_cell_maes_Ed = total_mae_Ed / total_count
        new_cell_mapes_Qd = total_mape_Qd / total_count
        new_cell_mapes_Ed = total_mape_Ed / total_count

        return new_cell_maes_Qd, new_cell_mapes_Qd, 0, 0, new_cell_maes_Ed, new_cell_mapes_Ed, 0, 0, gt_trajectory, pred_trajectory, r2_Qd, r2_Ed, 1 - bad_count / total_count, 1 - bad_count_Ed / total_count, detailed_alphas

    def visualize_enc_out(self, setting, load=True, set_dataset='', save_path=''):
        pred_data, pred_loader = self._get_data(flag='set_files', set_data=set_dataset)
        self._build_model()
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        if 'NC' in self.args.root_path:
            bad_threshold = 0.175
        else:
            bad_threshold = 0.055
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                masks = batch_y.float()[:, :, 3:5]
                cost_time = batch_y.float().to(self.device)[:, :, 2]
                batch_x = batch_x.float().to(self.device)[:, :, :2]
                batch_y = batch_y.float()[:, :, :2]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs, _, enc_out, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs, enc_out, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                return enc_out

    def visualize_enc_out_new(self, setting, load=True, set_dataset='', save_path='', return_labels=False):
        pred_data, pred_loader = self._get_data(flag='set_files', set_data=set_dataset)
        self._build_model()
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        if 'NC' in self.args.root_path:
            bad_threshold = 0.175
        else:
            bad_threshold = 0.055
        self.model.eval()
        enc_outs = []
        ruls = []
        labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                masks = batch_y.float()[:, :, 3:5]
                cost_time = batch_y.float().to(self.device)[:, :, 2]
                cycle_life_label = pred_data.cycle_life_scaler.inverse_transform(
                    batch_y.float().to(self.device)[:, :, 6][:, self.args.seq_len - 1].unsqueeze(-1).cpu().numpy())
                batch_x = batch_x.float().to(self.device)[:, :, :2]
                batch_y = batch_y.float()[:, :, :2]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs, _, enc_out, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs, enc_out, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                true = batch_y[:, self.args.seq_len-1, f_dim:].detach().cpu()
                transformed_true = pred_data.inverse_transform(true)
                label = transformed_true[0, 0]
                labels.append(label)
                enc_outs.append(enc_out.cpu().numpy().reshape(1, -1))
                ruls.append(round(cycle_life_label[0][0]))
        if return_labels:
            return enc_outs, ruls, labels
        return enc_outs, ruls

    def evalute_short_cells(self, setting, load=True, set_dataset=''):
        pred_data, pred_loader = self._get_data(flag='set_files', set_data=set_dataset)
        self._build_model()
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        if 'NC' in self.args.root_path:
            first_life_end = 3.5 * 0.8
        else:
            first_life_end = 1.1 * 0.8
        self.model.eval()

        # record the trajectory data
        gt_trajectory = {}
        pred_trajectory = {}
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                masks = batch_y.float()[:, :, 3:5]
                batch_x = batch_x.float().to(self.device)[:, :, :2]
                batch_y = batch_y.float()[:, :, :2]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs, *useless = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[0, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[0, -self.args.pred_len:, f_dim:]
                masks = masks[:, -self.args.pred_len:, :].detach().cpu().numpy()
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y.detach().cpu().numpy()
                transformed_true = pred_data.inverse_transform(true)
                transformed_pred = pred_data.inverse_transform(pred)

                marks = batch_y_mark.detach().cpu().numpy()
                cycles = [i + 1 for i in range(transformed_pred.shape[0])]

                for index, cycle_number in enumerate(cycles):
                    gt_trajectory[cycle_number] = gt_trajectory.get(cycle_number, []) + [transformed_true[index]]
                    pred_trajectory[cycle_number] = pred_trajectory.get(cycle_number, []) + [
                        transformed_pred[index]]

                # end until the prediction shows the EOL
                pred_Qds = np.array(list(pred_trajectory.values()))[:, 0, 0]
                if np.min(pred_Qds) > first_life_end:
                    continue
                else:
                    break
        # pred_Qds = np.array(list(pred_trajectory.values()))[:,0,0]
        # if np.min(pred_Qds) > first_life_end:
        #     return None, None
        true_cycle_life = 0
        pred_cycle_life = 0
        for cycle_number, data in gt_trajectory.items():
            true_Qd = data[0][0]
            if true_Qd <= first_life_end:
                true_cycle_life = cycle_number
                break
        for cycle_number, data in pred_trajectory.items():
            pred_Qd = data[0][0]
            if pred_Qd <= first_life_end:
                pred_cycle_life = cycle_number
                break

        return true_cycle_life, pred_cycle_life
