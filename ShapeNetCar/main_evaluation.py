import os
import torch
import argparse
import yaml
import numpy as np
import time
from torch import nn
from torch_geometric.loader import DataLoader
from utils.drag_coefficient import cal_coefficient
from dataset.load_dataset import load_train_val_fold_file
from dataset.dataset import GraphDataset
import scipy as sc

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/home/qikai/mlcfd_data/training_data_2')
parser.add_argument('--save_dir', default='/home/qikai/mlcfd_data/preprocessed_data_2')
parser.add_argument('--fold_id', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--cfd_model', default='Transolver')
parser.add_argument('--cfd_mesh', action='store_true')
parser.add_argument('--r', default=0.2, type=float)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--nb_epochs', default=200, type=float)
args = parser.parse_args()
print(args)


n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')

# path = f'/data/qikai/metrics/{args.cfd_model}/{args.fold_id}/{args.nb_epochs}_{args.weight}'
model = torch.load('/data/qikai/metrics/None/0/200_0.5/model_200.pth').to(device)



train_data, val_data, coef_norm, vallst = load_train_val_fold_file(args, preprocessed=True)
val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)


test_loader = DataLoader(val_ds, batch_size=1)

if not os.path.exists('/data/qikai/Transolver-main/Car-Design-ShapeNetCar/results/' + args.cfd_model + '/'):
    os.makedirs('/data/qikai/Transolver-main/Car-Design-ShapeNetCar/results/' + args.cfd_model + '/')

with torch.no_grad():
    model.eval()
    criterion_func = nn.MSELoss(reduction='none')
    l2errs_press = []
    l2errs_velo = []
    mses_press = []
    mses_velo_var = []
    times = []
    gt_coef_list = []
    pred_coef_list = []
    coef_error = 0
    index = 0
    for cfd_data, geom in test_loader:
        print(vallst[index])
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        tic = time.time()
        out = model((cfd_data, geom))
        toc = time.time()
        targets = cfd_data.y

        if coef_norm is not None:
            mean = torch.tensor(coef_norm[2]).to(device)
            std = torch.tensor(coef_norm[3]).to(device)
            pred_press = out[cfd_data.surf, -1] * std[-1] + mean[-1]
            gt_press = targets[cfd_data.surf, -1] * std[-1] + mean[-1]
            pred_velo = out[~cfd_data.surf, :-1] * std[:-1] + mean[:-1]
            gt_velo = targets[~cfd_data.surf, :-1] * std[:-1] + mean[:-1]
            out_denorm = out * std + mean
            y_denorm = targets * std + mean

        # np.save('/data/qikai/Transolver-main/Car-Design-ShapeNetCar/results/' + args.cfd_model + '/' + str(index) + '_pred.npy', out_denorm.detach().cpu().numpy())
        # np.save('/data/qikai/Transolver-main/Car-Design-ShapeNetCar/results/' + args.cfd_model + '/' + str(index) + '_gt.npy', y_denorm.detach().cpu().numpy())

        pred_coef = cal_coefficient(vallst[index].split('/')[1], pred_press[:, None].detach().cpu().numpy(),
                                    pred_velo.detach().cpu().numpy())
        gt_coef = cal_coefficient(vallst[index].split('/')[1], gt_press[:, None].detach().cpu().numpy(),
                                  gt_velo.detach().cpu().numpy())

        gt_coef_list.append(gt_coef)
        pred_coef_list.append(pred_coef)
        coef_error += (abs(pred_coef - gt_coef) / gt_coef)
        print(coef_error / (index + 1))

        l2err_press = torch.norm(pred_press - gt_press) / torch.norm(gt_press)
        l2err_velo = torch.norm(pred_velo - gt_velo) / torch.norm(gt_velo)

        mse_press = criterion_func(out[cfd_data.surf, -1], targets[cfd_data.surf, -1]).mean(dim=0)
        mse_velo_var = criterion_func(out[~cfd_data.surf, :-1], targets[~cfd_data.surf, :-1]).mean(dim=0)

        l2errs_press.append(l2err_press.cpu().numpy())
        l2errs_velo.append(l2err_velo.cpu().numpy())
        mses_press.append(mse_press.cpu().numpy())
        mses_velo_var.append(mse_velo_var.cpu().numpy())
        times.append(toc - tic)
        index += 1

    gt_coef_list = np.array(gt_coef_list)
    pred_coef_list = np.array(pred_coef_list)
    spear = sc.stats.spearmanr(gt_coef_list, pred_coef_list)[0]
    print("rho_d: ", spear)
    print("c_d: ", coef_error / index)
    l2err_press = np.mean(l2errs_press)
    l2err_velo = np.mean(l2errs_velo)
    rmse_press = np.sqrt(np.mean(mses_press))
    rmse_velo_var = np.sqrt(np.mean(mses_velo_var, axis=0))
    if coef_norm is not None:
        rmse_press *= coef_norm[3][-1]
        rmse_velo_var *= coef_norm[3][:-1]
    print('relative l2 error press:', l2err_press)
    print('relative l2 error velo:', l2err_velo)
    print('press:', rmse_press)
    print('velo:', rmse_velo_var, np.sqrt(np.mean(np.square(rmse_velo_var))))
    print('time:', np.mean(times))
