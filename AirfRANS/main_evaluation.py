
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '3' 

import argparse, json
import torch
torch.set_num_threads(8) 
import numpy as np
import datetime
import os.path as osp
from scipy.spatial.distance import cdist
import scipy.io as io
import utils.metrics_SpiderSolver as metrics
from models.MLP import MLP
from models.SpiderSolver import SpiderSolver
import train_clip
from utils.SpiderSolver_OT_alignment import (
    compute_template,
    perform_spectral_clustering,
    save_template_and_plot,
    process_dataset
)


global time_str
time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
print(time_str)

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nmodel', help='Number of trained models for standard deviation estimation (default: 1)',
                    default=1, type=int)
parser.add_argument('-w', '--weight', help='Weight in front of the surface loss (default: 1)', default=1, type=float)
parser.add_argument('-t', '--task',
                    help='Task to train on. Choose between "full", "scarce", "reynolds" and "aoa" (default: full)',
                    default='full', type=str)
parser.add_argument('-s', '--score',
                    help='If you want to compute the score of the models on the associated test set. (default: 0)',
                    default=1, type=int)
parser.add_argument('--my_path',
                    default='/home/qikai/Dataset', type=str, help='path of Airfoil dataset')

parser.add_argument('--batch_size', type=int, default = 1, help='Batch size for training')
parser.add_argument('--nb_epochs', type=int,  default = 2, help='Number of training epochs')
parser.add_argument('--lr', type=float,  default = 0.001, help='Learning rate')
parser.add_argument('--max_neighbors', type=int,  default = 64, help='Maximum number of neighbors')
parser.add_argument('--subsampling', type=int,  default = 32000, help='Number of subsamples')
parser.add_argument('--r', type=float,  default = 0.05, help='Radius parameter') 
parser.add_argument('--model', default='SpiderSolver', type=str)  

parser.add_argument('--onion_num', type=int,  default = 5, help='number of onion')
parser.add_argument('--n_clusters', type=int,  default = 4, help='number of clusters')
parser.add_argument('--file', type=str,  default = '/manifest', help='data file of Airfoil dataset')
parser.add_argument('--clip', default=10e6, type=int, help='clip_grad_norm, default no grad clip')

args = parser.parse_args()
file = args.file

with open(args.my_path +  file +  '.json', 'r') as f:
    manifest = json.load(f)

manifest_train = manifest[args.task + '_train']
test_dataset = manifest[args.task + '_test'] if args.task != 'scarce' else manifest['full_test']
n = int(.1 * len(manifest_train))
train_dataset = manifest_train[:-n]
val_dataset = manifest_train[-n:]

use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')

hparams = {
    'batch_size': args.batch_size,
    'nb_epochs': args.nb_epochs ,
    'lr': args.lr,
    'max_neighbors': args.max_neighbors,
    'subsampling': args.subsampling,
    'r': args.r}

current_file_path = os.path.dirname(__file__)
path = current_file_path + '/model_save/' + str(time_str) + '/'



model = SpiderSolver(n_hidden=256,
                    n_layers=8,
                    space_dim=7,
                    fun_dim=0,
                    n_head=8,
                    mlp_ratio=2,
                    out_dim=4,
                    unified_pos=1,
                    onion_num = args.onion_num,
                    n_clusters=args.n_clusters).cuda() 

from dataset.dataset_SpiderSolver import Dataset   
print("start load data")
train_dataset, coef_norm = Dataset(train_dataset, norm=True, sample=None, my_path=args.my_path, onion_num = args.onion_num, n_clusters = args.n_clusters)
val_dataset = Dataset(val_dataset, sample=None, coef_norm=coef_norm, my_path=args.my_path, onion_num = args.onion_num, n_clusters = args.n_clusters)
print("load data finish")


model_path = "/home/qikai/SpiderSolver/AirfRANS/model_save"
model = torch.load(model_path + '/2025_12_08_15_10_11/model').to(device)


# # # # # # # # # Optimal transport-based alignment for spectral clustering # # # # # # # # #


save_template_SpectralClustering= io.loadmat(model_path + "/2025_12_08_15_10_11/template.mat")
template = save_template_SpectralClustering["template"]
labels_SpectralClustering = save_template_SpectralClustering["labels_SpectralClustering"][0]


##############################################################################################

models = []
log_path = path


models.append(model)

if bool(args.score):
    print('start score')
    s = args.task + '_test' if args.task != 'scarce' else 'full_test'
    coefs = metrics.Results_test(template, labels_SpectralClustering, device, [models], [hparams], coef_norm, args.my_path, path_out = path, n_test=3,
                                 criterion='MSE', s=s, file = file, onion_num = args.onion_num, n_clusters = args.n_clusters)
    # models can be a stack of the same model (for example MLP) on the task s, if you have another stack of another model (for example GraphSAGE)
    # you can put in model argument [models_MLP, models_GraphSAGE] and it will output the results for both models (mean and std) in an ordered array.
        
    np.save(osp.join(path,  'true_coefs'), coefs[0])
    np.save(osp.join(path,  'pred_coefs_mean'), coefs[1])
    np.save(osp.join(path, 'pred_coefs_std'), coefs[2])
    for n, file in enumerate(coefs[3]):
        np.save(osp.join(path,  'true_surf_coefs_' + str(n)), file)
    for n, file in enumerate(coefs[4]):
        np.save(osp.join(path, 'surf_coefs_' + str(n)), file)
    np.save(osp.join(path,  'true_bls'), coefs[5])
    np.save(osp.join(path, 'bls'), coefs[6])
    print('end score')
    
print('%.4f'% coefs[-4])
print('%.4f'% coefs[-3])
print('%.4f'% coefs[-2])
print('%.4f'% coefs[-1])

print(time_str)

