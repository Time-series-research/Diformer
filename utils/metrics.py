import numpy as np
import torch


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


def TRADE(args, pred, true):
    # MAE
    # return torch.mean(torch.mean(torch.abs(pred - true), 1))
    pre_length = args.pred_len

    # Trade loss
    y_mid = pred.clone()
    y_mid = y_mid[:, torch.arange(y_mid.size(1)) != pre_length - 1, :]
    y_step_1 = torch.cat((torch.zeros(32, 1, 1).cuda(), y_mid), 1)

    x_mid = true.clone()
    x_mid = x_mid[:, torch.arange(x_mid.size(1)) != pre_length - 1, :]
    x_step_1 = torch.cat((torch.zeros(32, 1, 1).cuda(), x_mid), 1)

    zero_matrix = torch.zeros(32, pre_length, 1).cuda()

    first_order_pre = pred - y_step_1
    first_order_true = true - x_step_1

    res = torch.where((first_order_pre * first_order_true) < 0, (first_order_pre - first_order_true),
                      zero_matrix)

    loss_1 = torch.mean(torch.mean(torch.abs(pred - true), 1))
    # loss_2 = torch.mean(torch.mean(torch.abs((pred - y_step_1) - (true - x_step_1)), 1))
    loss_2 = torch.mean(torch.mean(torch.abs(res), 1))

    return 1 * loss_1 + 0 * loss_2
