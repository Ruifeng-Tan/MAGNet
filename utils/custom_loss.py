import numpy as np
from scipy.optimize import curve_fit
from torch import nn
import torch
from torch.autograd import Function

def empirical_equation_basic(x, A):
    y = 1 - A * (x ** (0.5))
    return y

def empirical_equation(x, A, B):
    y = 1 - A * (x ** (B))
    return y


def PILoss(transformed_gt, As, Bs, cost_time, nominal_capacity, max_Ed_in_train, device):  # As: [B, 1, 2]
    # As = As.cpu().numpy()
    # Bs = Bs.cpu().numpy()

    ideal_As = []
    ideal_Bs = []
    flag = True
    for i in range(transformed_gt.shape[0]):
        # for Qd
        try:
            parameters_Qd, _ = curve_fit(empirical_equation, cost_time[i, :].cpu().numpy(),
                                         transformed_gt[i, :, 0] / nominal_capacity, maxfev=10000)
            # for Ed
            parameters_Ed, _ = curve_fit(empirical_equation, cost_time[i, :].cpu().numpy(),
                                         transformed_gt[i, :, 1] / max_Ed_in_train, maxfev=10000)
            tmp_As = np.array([parameters_Qd[0], parameters_Ed[0]]).reshape(1, -1)
            tmp_Bs = np.array([parameters_Qd[1], parameters_Ed[1]]).reshape(1, -1)
            ideal_As.append(tmp_As)
            ideal_Bs.append(tmp_Bs)
        except RuntimeError:
            flag = False
            parameters_Qd, _ = curve_fit(empirical_equation_basic, cost_time[i, :].cpu().numpy(),
                                         transformed_gt[i, :, 0] / nominal_capacity, maxfev=10000)
            # for Ed
            parameters_Ed, _ = curve_fit(empirical_equation_basic, cost_time[i, :].cpu().numpy(),
                                         transformed_gt[i, :, 1] / max_Ed_in_train, maxfev=10000)
            tmp_As = np.array([parameters_Qd[0], parameters_Ed[0]]).reshape(1, -1)
            tmp_Bs = np.array([0.5, 0.5]).reshape(1, -1)
            ideal_As.append(tmp_As)
            ideal_Bs.append(tmp_Bs)
    ideal_As = np.concatenate(ideal_As, axis=0)
    ideal_Bs = np.concatenate(ideal_Bs, axis=0)
    ideal_As = torch.ones_like(As) * torch.Tensor(ideal_As).to(device)
    ideal_Bs = torch.ones_like(Bs) * torch.Tensor(ideal_Bs).to(device)
    loss = (torch.sum(torch.pow((ideal_As - As), 2))) / ideal_As.numel() + (
            torch.sum(torch.pow((ideal_Bs - Bs), 2))) / ideal_Bs.numel()


    return loss, flag
