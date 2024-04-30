import numpy as np
import torch
from torch import nn


def PI_loss(pred, true, true_mark, args, data_set, device):
    '''
    Compute the physics-informed loss. Specifically, we add two loss terms:
    1. The proportion loss: The proportion ratio of the predicted capacity should align with the ratio of querying SOC
    2. Voltage limitation loss: There is a cutoff voltage, and the predicted voltage should not be obviously lower than it.
    '''
    criterion = nn.MSELoss()
    raw_loss = criterion(pred, true)
    if data_set.scale:
        inverse_pred = torch.FloatTensor(data_set.inverse_transform(pred.reshape(-1, 2).detach().cpu())).to(device)
        inverse_pred = inverse_pred.reshape(args.batch_size,-1,2)
    else:
        inverse_pred = pred
    pred_Qd = inverse_pred[:, :, 0]
    max_pred_Qd, _ = torch.max(pred_Qd, dim=1)
    pred_Qd_proportion = pred_Qd / max_pred_Qd.unsqueeze(
        -1)  # the loss is suspended for implementation. to compute the ratio, we need inverse transformation first.
    query_SOCs = 100 - true_mark[:, :, 0]
    query_SOC_max, _ = torch.max(query_SOCs, dim=1)
    SOC_ratio = query_SOCs / query_SOC_max.unsqueeze(-1)
    proportion_loss = criterion(pred_Qd_proportion, SOC_ratio)

    pred_V = inverse_pred[:, :, 1]
    pred_min_V, _ = torch.min(pred_V, dim=1)
    cutoff_V = torch.ones_like(pred_min_V) * args.end_V
    voltage_limitation_loss = criterion(pred_min_V, cutoff_V)

    # sum the loss
    loss = raw_loss + args.gamma1 * proportion_loss + args.gamma2 * voltage_limitation_loss
    return loss, raw_loss, proportion_loss, voltage_limitation_loss


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
