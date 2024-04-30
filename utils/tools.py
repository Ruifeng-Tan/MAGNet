import numpy as np
import torch
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import os, sys
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
matplotlib.rcParams['pdf.fonttype'] = 42 # make the text editable for Adobe Illustrator
matplotlib.rcParams['ps.fonttype'] = 42
#plt.switch_backend('agg') # no figure will be shown
from matplotlib.font_manager import FontProperties

font = {'family' : 'Arial'}


matplotlib.rcParams['mathtext.fontset'] = 'custom'

matplotlib.rcParams['mathtext.rm'] = 'Arial'

matplotlib.rcParams['mathtext.it'] = 'Arial'

matplotlib.rc('font', **font)

#matplotlib.rc('font', **font0)

matplotlib.rc('text', usetex=False)
x_y_label_size = 12


def set_ax_linewidth(the_plt, bw=1.5):
    ax = the_plt.gca()
    ax.spines['bottom'].set_linewidth(bw)
    ax.spines['left'].set_linewidth(bw)
    ax.spines['top'].set_linewidth(bw)
    ax.spines['right'].set_linewidth(bw)


def set_ax_font_size(the_plt, fontsize=10):
    ax = the_plt.gca()
    ax.tick_params(axis='y',
                   labelsize=fontsize  # y轴字体大小设置
                   )
    ax.tick_params(axis='x',
                   labelsize=fontsize  # x轴字体大小设置
                   )


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def clone_module(module, memo=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    **Description**
    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().
    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.
    **Arguments**
    * **module** (Module) - Module to be cloned.
    **Return**
    * (Module) - The cloned module.
    **Example**
    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[buff_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone


def update_module(module, updates=None, memo=None):
    r"""
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    **Description**
    Updates the parameters of a module in-place, in a way that preserves differentiability.
    The parameters of the module are swapped with their update values, according to:
    \[
    p \gets p + u,
    \]
    where \(p\) is the parameter, and \(u\) is its corresponding update.
    **Arguments**
    * **module** (Module) - The module to update.
    * **updates** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the tensors in .update attributes.
    **Example**
    ~~~python
    error = loss(model(X), y)
    grads = torch.autograd.grad(
        error,
        model.parameters(),
        create_graph=True,
    )
    updates = [-lr * g for g in grads]
    l2l.update_module(model, updates=updates)
    ~~~
    """
    if memo is None:
        memo = {}
    if updates is not None:
        params = list(module.parameters())
        if not len(updates) == len(list(params)):
            msg = 'WARNING:update_module(): Parameters and updates have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(updates)) + ')'
            print(msg)
        for p, g in zip(params, updates):
            p.update = g

    # Update the params
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p in memo:
            module._parameters[param_key] = memo[p]
        else:
            if p is not None and hasattr(p, 'update') and p.update is not None:
                updated = p + p.update
                p.update = None
                memo[p] = updated
                module._parameters[param_key] = updated

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff in memo:
            module._buffers[buffer_key] = memo[buff]
        else:
            if buff is not None and hasattr(buff, 'update') and buff.update is not None:
                updated = buff + buff.update
                buff.update = None
                memo[buff] = updated
                module._buffers[buffer_key] = updated

    # Then, recurse for each submodule
    for module_key in module._modules:
        module._modules[module_key] = update_module(
            module._modules[module_key],
            updates=None,
            memo=memo,
        )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(module, 'flatten_parameters'):
        module._apply(lambda x: x)
    return module

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})
    return {'Total': total_num, 'Trainable': trainable_num}

def maml_update(model, lr, grads=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)
    **Description**
    Performs a MAML update on model using grads and lr.
    The function re-routes the Python object, thus avoiding in-place
    operations.
    NOTE: The model itself is updated in-place (no deepcopy), but the
          parameters' tensors are not.
    **Arguments**
    * **model** (Module) - The model to update.
    * **lr** (float) - The learning rate used to update the model.
    * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the gradients in .grad attributes.
    **Example**
    ~~~python
    maml = l2l.algorithms.MAML(Model(), lr=0.1)
    model = maml.clone() # The next two lines essentially implement model.adapt(loss)
    grads = autograd.grad(loss, model.parameters(), create_graph=True)
    maml_update(model, lr=0.1, grads)
    ~~~
    """
    if grads is not None:
        params = list(model.parameters())
        if not len(grads) == len(list(params)):
            msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
            print(msg)
        for p, g in zip(params, grads):
            if g is not None:
                p.update = - lr * g
    return update_module(model)


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 3) // 1))} if epoch >= 3 else {
            epoch: args.learning_rate}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type4':
        # constant learning rate
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'type5':
        base = epoch // 3
        lr_adjust = {epoch: args.learning_rate * (0.5 ** (base))} if epoch >= 3 else {
            epoch: args.learning_rate}
    elif args.lradj == 'type6':
        base = epoch // 2
        lr_adjust = {epoch: args.learning_rate * (0.5 ** (base))} if epoch >= 2 else {
            epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    return lr


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def empirical_equation(x, A, B):
    y = 1 - A * (x ** (B))
    return y


def predict_linearRegression(true, mark, true_index, history_len=50):
    linear_model = LinearRegression()
    baseline_train_x = mark.reshape(-1, 1)[:history_len, :]
    baseline_train_y = true[:history_len, true_index]
    baseline_test_x = mark.reshape(-1, 1)[history_len:, :]
    linear_model.fit(baseline_train_x, baseline_train_y)
    linear_preds = linear_model.predict(baseline_test_x)
    linear_preds = np.concatenate([true[:history_len, true_index], linear_preds], axis=0)
    return linear_preds


def predict_empirical_equation(true, mark, history_len, time_x, nominal_capacity):
    time_linear_reg = LinearRegression()
    time_linear_reg.fit(mark.reshape(-1, 1)[:history_len], time_x)
    pred_time_y = time_linear_reg.predict(mark.reshape(-1, 1))
    pred_time_y[:history_len] = time_x
    try:
        parameters, _ = curve_fit(empirical_equation2, time_x, true[:history_len, 0] / nominal_capacity, maxfev=10000)
        empirical_pred = empirical_equation2(pred_time_y, parameters[0], parameters[1]) * nominal_capacity
    except RuntimeError:
        parameters, _ = curve_fit(empirical_equation1, time_x, true[:history_len, 0] / nominal_capacity)
        empirical_pred = empirical_equation1(pred_time_y, parameters[0]) * nominal_capacity
    empirical_pred[:history_len] = true[:history_len, 0]
    return empirical_pred


def empirical_equation2(x, A, B):
    y = 1 - A * (x ** (B))
    return y


def empirical_equation1(x, A):
    y = 1 - A * (x ** (0.5))
    return y


def visual_QE_vs_cycle(true, preds, mark, history_len, time_x, nominal_capacity=3.5, plot_baseline=True, title=''):
    mark = mark.reshape(-1)
    # l = list(mark)
    # if not all(x<y for x, y in zip(l, l[1:])):
    #     print('a')
    plt.plot(mark, true[:, 0], linewidth=2, color='black', label='ground_truth')  # plot true Qd
    plt.plot(mark, preds[:, 0], linewidth=1, color='red', label='pred_Informer')  # plot pred Qd
    if plot_baseline:
        empirical_pred = predict_empirical_equation(true, mark, history_len, time_x, nominal_capacity)
        linear_preds = predict_linearRegression(true, mark, 0, history_len)
        plt.plot(mark, empirical_pred, linewidth=1, color='blue', label='pred_empirical')
        plt.plot(mark, linear_preds, linewidth=1, color='green', label='pred_linear')
    plt.ylabel('Discharging capacity (Ah)')
    if not title:
        plt.title("Discharging capacity v.s. cycle number")
    else:
        plt.title(title)
    # ax1.set_ylabel('Discharging energy (Wh)')
    # ax1.set_title("Discharging energy v.s. cycle number")
    ymin = min([np.min(preds[:, 0], axis=0), np.min(true[:, 0], axis=0), np.min(linear_preds, axis=0),
                np.min(empirical_pred, axis=0)]) - 0.1
    ymax = max([np.max(preds[:, 0], axis=0), np.max(true[:, 0], axis=0), np.max(linear_preds, axis=0),
                np.max(empirical_pred, axis=0)]) + 0.1
    plt.ylim((ymin, ymax))
    plt.legend(loc='lower left')
    plt.show()

    # plt.plot(mark, true[:, 1], linewidth=2, color='black', label='ground_truth_Ed')  # plot true Qd
    # plt.plot(mark, preds[:, 1], linewidth=1, color='red', label='pred_Informer_Ed')  # plot pred Qd
    # if plot_baseline:
    #     linear_preds = predict_linearRegression(true, mark, 1, history_len)
    #     plt.plot(mark, linear_preds, linewidth=1, color='green', label='pred_linear')
    # plt.ylabel('Discharging energy (Wh)')
    # if not title:
    #     plt.title("Discharging energy v.s. cycle number")
    # else:
    #     plt.title(title)
    # # ax1.set_ylabel('Discharging energy (Wh)')
    # # ax1.set_title("Discharging energy v.s. cycle number")
    # ymin = min([np.min(preds[:, 1],axis=0),np.min(true[:, 1],axis=0),np.min(linear_preds,axis=0)]) - 0.1
    # ymax = max([np.max(preds[:, 1],axis=0),np.max(true[:, 1],axis=0),np.max(linear_preds,axis=0)]) + 0.1
    # plt.ylim((ymin, ymax))
    # plt.legend(loc='lower left')
    # plt.show()


def cp_visual_one_cell(gt_trajectory, pred_trajectory, save_path):
    cycles = []
    Qd_stds, Ed_stds = [], []
    Qd_gts, Ed_gts = [], []
    Qd_preds, Ed_preds = [], []

    for cycle_number, _ in gt_trajectory.items():
        cycles.append(cycle_number)
        gt_data = gt_trajectory[cycle_number]
        pred_data = pred_trajectory[cycle_number]

        Qd_gt, Ed_gt = np.mean(gt_data, axis=0)[0], np.mean(gt_data, axis=0)[1]
        Qd_pred, Ed_pred = np.mean(pred_data, axis=0)[0], np.mean(pred_data, axis=0)[1]
        Qd_std, Ed_std = np.std(pred_data, axis=0)[0], np.std(pred_data, axis=0)[1]

        Qd_gts.append(Qd_gt)
        Ed_gts.append(Ed_gt)
        Qd_preds.append(Qd_pred)
        Ed_preds.append(Ed_pred)
        Qd_stds.append(Qd_std)
        Ed_stds.append(Ed_std)
    plt.plot(cycles, Qd_gts, label='Qd_gt', color='black', linewidth=2)
    plt.plot(cycles, Qd_preds, label='Qd_pred', color='blue', linewidth=2)
    plt.fill_between(cycles, np.array(Qd_preds) - np.array(Qd_stds), np.array(Qd_preds) + np.array(Qd_stds),
                     facecolor='green',
                     alpha=0.3)
    plt.xlabel('Cycle number')
    plt.ylabel('Discharging capacity(Ah)')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    # plt.plot(cycles, Ed_gts, label='Ed_gt', color='black', linewidth=2)
    # plt.plot(cycles, Ed_preds, label='Ed_pred', color='blue', linewidth=2)
    # plt.fill_between(cycles, np.array(Ed_preds) - np.array(Ed_stds), np.array(Ed_preds) + np.array(Ed_stds),
    #                  facecolor='green',
    #                  alpha=0.3)
    # plt.xlabel('Cycle number')
    # plt.ylabel('Discharging energy(Wh)')
    # plt.legend()
    # plt.show()


def visual_one_cell(gt_trajectory, pred_trajectory, save_path):
    cycles = []
    Qd_stds, Ed_stds = [], []
    Qd_gts, Ed_gts = [], []
    Qd_preds, Ed_preds = [], []

    # Qd_maes = []
    # Qd_mapes = []
    #
    # Ed_maes = []
    # Ed_mapes = []

    for cycle_number, _ in gt_trajectory.items():
        cycles.append(cycle_number)
        gt_data = gt_trajectory[cycle_number]
        pred_data = pred_trajectory[cycle_number]

        Qd_gt, Ed_gt = np.mean(gt_data, axis=0)[:1], np.mean(gt_data, axis=0)[1:]
        Qd_pred, Ed_pred = np.mean(pred_data, axis=0)[:1], np.mean(pred_data, axis=0)[1:]
        Qd_std, Ed_std = np.std(pred_data, axis=0)[:1], np.std(pred_data, axis=0)[1:]

        # Qd_mae = mean_absolute_error(Qd_gt, Qd_pred)
        # Qd_mape = mean_absolute_percentage_error(Qd_gt, Qd_pred)
        # Qd_maes.append(Qd_mae)
        # Qd_mapes.append(Qd_mape)
        #
        # Ed_mae = mean_absolute_error(Ed_gt, Ed_pred)
        # Ed_mape = mean_absolute_percentage_error(Ed_gt, Ed_pred)
        # Ed_maes.append(Ed_mae)
        # Ed_mapes.append(Ed_mape)

        Qd_gts.append(Qd_gt[0])
        Ed_gts.append(Ed_gt[0])
        Qd_preds.append(Qd_pred[0])
        Ed_preds.append(Ed_pred[0])
        Qd_stds.append(Qd_std[0])
        Ed_stds.append(Ed_std[0])

    Qd_gts = np.array(Qd_gts)
    Qd_preds = np.array(Qd_preds)
    Qd_maes = np.abs(Qd_gts - Qd_preds)
    Qd_mapes = np.abs(Qd_gts - Qd_preds) / Qd_gts
    cell_Qd_mae, cell_Qd_mape, cell_Qd_mae_std, cell_Qd_mape_std = np.mean(Qd_maes), np.mean(Qd_mapes), np.std(
        Qd_maes), np.std(Qd_mapes)
    r2_Qd = r2_score(Qd_gts, Qd_preds)

    Ed_gts = np.array(Ed_gts)
    Ed_preds = np.array(Ed_preds)
    Ed_maes = np.abs(Ed_gts - Ed_preds)
    Ed_mapes = np.abs(Ed_gts - Ed_preds) / Ed_gts
    cell_Ed_mae, cell_Ed_mape, cell_Ed_mae_std, cell_Ed_mape_std = np.mean(Ed_maes), np.mean(Ed_mapes), np.std(
        Ed_maes), np.std(Ed_mapes)
    r2_Ed = r2_score(Ed_gts, Ed_preds)
    return cell_Qd_mae, cell_Qd_mape, cell_Qd_mae_std, cell_Qd_mape_std, cell_Ed_mae, cell_Ed_mape, cell_Ed_mae_std, cell_Ed_mape_std, r2_Qd, r2_Ed

def visual_one_cell_robust(gt_trajectory, pred_trajectory, save_path):
    cycles = []
    Qd_stds, Ed_stds = [], []
    Qd_gts, Ed_gts = [], []
    Qd_preds, Ed_preds = [], []

    # Qd_maes = []
    # Qd_mapes = []
    #
    # Ed_maes = []
    # Ed_mapes = []

    for cycle_number, _ in gt_trajectory.items():
        cycles.append(cycle_number)
        gt_data = gt_trajectory[cycle_number]
        pred_data = pred_trajectory[cycle_number]

        Qd_gt, Ed_gt = np.mean(gt_data, axis=0)[:1], np.mean(gt_data, axis=0)[1:]
        Qd_pred, Ed_pred = np.mean(pred_data, axis=0)[:1], np.mean(pred_data, axis=0)[1:]
        Qd_std, Ed_std = np.std(pred_data, axis=0)[:1], np.std(pred_data, axis=0)[1:]

        # Qd_mae = mean_absolute_error(Qd_gt, Qd_pred)
        # Qd_mape = mean_absolute_percentage_error(Qd_gt, Qd_pred)
        # Qd_maes.append(Qd_mae)
        # Qd_mapes.append(Qd_mape)
        #
        # Ed_mae = mean_absolute_error(Ed_gt, Ed_pred)
        # Ed_mape = mean_absolute_percentage_error(Ed_gt, Ed_pred)
        # Ed_maes.append(Ed_mae)
        # Ed_mapes.append(Ed_mape)

        Qd_gts.append(Qd_gt[0])
        Ed_gts.append(Ed_gt[0])
        Qd_preds.append(Qd_pred[0])
        Ed_preds.append(Ed_pred[0])
        Qd_stds.append(Qd_std[0])
        Ed_stds.append(Ed_std[0])
    plt.plot(cycles, Qd_gts, label='Capacity_truth', color='black', linewidth=2)
    plt.plot(cycles, Qd_preds, label='Capacity_prediction', color='blue', linewidth=2)
    # plt.fill_between(cycles, np.array(Qd_gts) - 0.03*np.array(Qd_gts),  np.array(Qd_gts) + 0.03*np.array(Qd_gts),
    #                  facecolor='green',
    #                  alpha=0.3,label=r'Accuracy zone [-$\alpha$,$\alpha$]')
    plt.fill_between(cycles, np.array(Qd_preds) - np.array(Qd_stds), np.array(Qd_preds) + np.array(Qd_stds),
                     facecolor='green',
                     alpha=0.4,label=r'$1\sigma$ bound')

    plt.xlabel('Cycle number', fontsize=x_y_label_size)
    plt.ylabel('Discharging capacity(Ah)', fontsize=x_y_label_size)
    plt.legend(prop={'size': 10},loc=3)
    set_ax_font_size(plt)
    set_ax_linewidth(plt)
    if save_path:
        plt.savefig(save_path, dpi=1200)
    plt.show()

    plt.plot(cycles, Ed_gts, label='Energy_truth', color='black', linewidth=2)
    plt.plot(cycles, Ed_preds, label='Energy_prediction', color='blue', linewidth=2)
    # plt.fill_between(cycles, np.array(Ed_gts) - 0.03*np.array(Ed_gts),  np.array(Ed_gts) + 0.03*np.array(Ed_gts),
    #                  facecolor='green',
    #                  alpha=0.3,label=r'Accuracy zone [-$\alpha$,$\alpha$]')
    plt.fill_between(cycles, np.array(Ed_preds) - np.array(Ed_stds), np.array(Ed_preds) + np.array(Ed_stds),
                     facecolor='green',
                     alpha=0.4,label=r'$1\sigma$ bound')

    plt.xlabel('Cycle number', fontsize=x_y_label_size)
    plt.ylabel('Discharging energy(Wh)', fontsize=x_y_label_size)
    plt.legend(prop={'size': 10},loc=3)
    set_ax_font_size(plt)
    set_ax_linewidth(plt)
    if save_path:
        new_save_path = save_path.split('.pdf')[0] + '_Ed.pdf'
        plt.savefig(new_save_path, dpi=1200)
    plt.show()
    Qd_gts = np.array(Qd_gts)
    Qd_preds = np.array(Qd_preds)
    Qd_maes = np.abs(Qd_gts - Qd_preds)
    Qd_mapes = np.abs(Qd_gts - Qd_preds) / Qd_gts
    cell_Qd_mae, cell_Qd_mape, cell_Qd_mae_std, cell_Qd_mape_std = np.mean(Qd_maes), np.mean(Qd_mapes), np.std(
        Qd_maes), np.std(Qd_mapes)
    r2_Qd = r2_score(Qd_gts, Qd_preds)

    Ed_gts = np.array(Ed_gts)
    Ed_preds = np.array(Ed_preds)
    Ed_maes = np.abs(Ed_gts - Ed_preds)
    Ed_mapes = np.abs(Ed_gts - Ed_preds) / Ed_gts
    cell_Ed_mae, cell_Ed_mape, cell_Ed_mae_std, cell_Ed_mape_std = np.mean(Ed_maes), np.mean(Ed_mapes), np.std(
        Ed_maes), np.std(Ed_mapes)
    r2_Ed = r2_score(Ed_gts, Ed_preds)
    return cell_Qd_mae, cell_Qd_mape, cell_Qd_mae_std, cell_Qd_mape_std, cell_Ed_mae, cell_Ed_mape, cell_Ed_mae_std, cell_Ed_mape_std, r2_Qd, r2_Ed

