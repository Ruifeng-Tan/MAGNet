import numpy as np
from scipy.optimize import curve_fit
from torch import nn
import torch
from torch.autograd import Function

class RBF(nn.Module):

    def __init__(self, device, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        #self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2).to(device)
        self.bandwidth_multipliers = torch.FloatTensor([0.1, 0.5, 1.0, 1.5, 2.0]).to(device)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):
    # https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py
    def __init__(self, device, kernel=None):
        super().__init__()
        self.kernel = RBF(device=device) if kernel is None else kernel
        self.device = device

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
    
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
