
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
torch.set_num_threads(8) 
import argparse
import operator
from torch_geometric.loader import DataLoader
import yaml
import numpy as np
import time
from torch import nn
from torch_geometric.loader import DataLoader
import scipy as sc
import datetime
from functools import reduce

import train_SpiderSolver
from utils.drag_coefficient import cal_coefficient
from dataset.load_dataset import load_train_val_fold_file
from models.SpiderSolver import SpiderSolver
from dataset.load_dataset_SpiderSolver import load_train_val_fold
from dataset.dataset_SpiderSolver import GraphDataset

from utils.SpiderSolver_OT_alignment import (
    compute_template,
    perform_spectral_clustering,
    save_template_and_plot,
    process_dataset
)

# compute the total number of model parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c

def get_samples(root):
    folds = [f'param{i}' for i in range(9)]
    samples = []
    for fold in folds:
        fold_samples = []
        files = os.listdir(os.path.join(root, fold))
        for file in files:
            path = os.path.join(root, os.path.join(fold, file))
            if os.path.isdir(path):
                fold_samples.append(os.path.join(fold, file))
        samples.append(fold_samples)
    return samples  # 100 + 99 + 97 + 100 + 100 + 96 + 100 + 98 + 99 = 889 samples


global time_str
time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
print(time_str)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/home/qikai/mlcfd_data/training_data')
parser.add_argument('--save_dir', default=None)

parser.add_argument('--fold_id', default=0, type=int)
parser.add_argument('--val_iter', default=1, type=int)
parser.add_argument('--cfd_config_dir', default='cfd/cfd_params.yaml')
parser.add_argument('--cfd_model')
parser.add_argument('--cfd_mesh', action='store_true')
parser.add_argument('--r', default=0.2, type=float)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--nb_epochs', default=20, type=int)
parser.add_argument('--preprocessed', default=0, type=int)
parser.add_argument('--save_name', default='SpideSolver', type=str)
parser.add_argument('--n_head', default=8, type=int)

parser.add_argument('--model', default='SpideSolver', type=str)
parser.add_argument('--n_clusters', default=4, type=int)
parser.add_argument('--onion_num', default=5, type=int)
args = parser.parse_args()
print(args)


model = SpiderSolver(n_hidden=256, n_layers=8, space_dim=7,
                fun_dim=0,
                n_head=args.n_head,
                mlp_ratio=2, out_dim=4,
                unified_pos=0,
                onion_num = args.onion_num,
                surf_clustering_num = args.n_clusters,
                ).cuda()  
print(count_params(model))


hparams = {'lr': args.lr, 'batch_size': args.batch_size, 'nb_epochs': args.nb_epochs}
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start = time.time()

train_data, val_data, coef_norm = load_train_val_fold(args, preprocessed=args.preprocessed, onion_num = args.onion_num, surf_clustering_num = args.n_clusters)

end = time.time()
time_elapsed = end - start
print('Data loading Time elapsed: {0:.2f} seconds'.format(time_elapsed))

current_file_path = os.path.dirname(__file__)
path = current_file_path + '/model_save/' + str(time_str) + '/'
if not os.path.exists(path):
    os.makedirs(path)
    
    
# # # # # # # # # Optimal transport-based alignment for spectral clustering # # # # # # # # #
# Compute template from training data
template = compute_template(train_data)
# Perform spectral clustering
labels_SpectralClustering = perform_spectral_clustering(template, args.n_clusters)
# Save template and create visualization
save_template_and_plot(template, labels_SpectralClustering, path, coef_norm)

# Process training and validation datasets
process_dataset(train_data, template, labels_SpectralClustering, coef_norm, path)
process_dataset(val_data, template, labels_SpectralClustering, coef_norm, path)

##############################################################################################

train_ds = GraphDataset(train_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)

model = train_SpiderSolver.main(device, train_ds, val_ds, model, hparams, path, val_iter=args.val_iter, reg=args.weight, coef_norm=coef_norm, save_name = args.save_name, labels_SpectralClustering = labels_SpectralClustering)


samples = get_samples(args.data_dir)
trainlst = []
for i in range(len(samples)):
    if i == args.fold_id:
        continue
    trainlst += samples[i]
vallst = samples[args.fold_id] if 0 <= args.fold_id < len(samples) else None

test_loader = DataLoader(val_ds, batch_size=1)
labels_SpectralClustering = torch.from_numpy(labels_SpectralClustering).cuda()

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
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        tic = time.time()
        if labels_SpectralClustering.all() == None:
            out = model((cfd_data, geom))
        else:
            out = model((cfd_data, geom, labels_SpectralClustering) )
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

        pred_coef = cal_coefficient(vallst[index].split('/')[1], pred_press[:, None].detach().cpu().numpy(),
                                    pred_velo.detach().cpu().numpy())
        gt_coef = cal_coefficient(vallst[index].split('/')[1], gt_press[:, None].detach().cpu().numpy(),
                                gt_velo.detach().cpu().numpy())

        gt_coef_list.append(gt_coef)
        pred_coef_list.append(pred_coef)
        coef_error += (abs(pred_coef - gt_coef) / gt_coef)

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

print(time_str)