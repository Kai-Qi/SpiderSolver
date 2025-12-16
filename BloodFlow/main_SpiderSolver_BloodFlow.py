import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
torch.set_num_threads(8) 
import numpy as np
import scipy.io as sio
import time
import pandas as pd
from utils import count_params, LpLoss, GaussianNormalizer
import os
from Adam import Adam
import datetime
from matplotlib import pylab as plt
from SpiderSolver_BloodFlow_DataProcess import SpiderSolver_BloodFlow_DataProcess
from torch_geometric.data import Data, Dataset


global time_str
time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
print(time_str)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(args):  
    print("\n=============================")
    print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
    print("=============================\n")
    
    PATH = args.data_dir
    ntrain = args.num_train  
    ntest = args.num_test  
    batch_size = args.batch_size 
    learning_rate = args.lr    
    epochs = args.epochs    
    step_size = 100
    gamma = 0.1
    
    ################################################################
    # reading data reading LBO basis
    ################################################################  
    current_file_path = os.path.dirname(__file__)
    data = sio.loadmat(PATH) 
    points = pd.read_csv(current_file_path + '/coordinates.csv',header=None).values
    points = torch.from_numpy(points)
    
    x_dataIn = torch.Tensor(data['BC_time'])
    y_dataIn1 = torch.Tensor(data['velocity_x'])
    y_dataIn2 = torch.Tensor(data['velocity_y'])
    y_dataIn3 = torch.Tensor(data['velocity_z'])
    
    x_data = x_dataIn
    y_data = torch.zeros((y_dataIn1.shape[0],y_dataIn1.shape[1],y_dataIn1.shape[2],3))
     
    y_data[:,:,:,0] = y_dataIn1
    y_data[:,:,:,1] = y_dataIn2
    y_data[:,:,:,2] = y_dataIn3
    
    ################################################################
    # normalization
    ################################################################  
    x_train = x_data[:ntrain,:,:]
    y_train = y_data[:ntrain,:,:]
    x_test = x_data[-ntest:,:,:]
    y_test = y_data[-ntest:,:,:]
            
    norm_x1 = GaussianNormalizer(x_train[:,:,0])
    norm_x2 = GaussianNormalizer(x_train[:,:,1:])
    
    x_train[:,:,0] = norm_x1.encode(x_train[:,:,0])
    x_train[:,:,1:] = norm_x2.encode(x_train[:,:,1:])
    x_test[:,:,0] = norm_x1.encode(x_test[:,:,0])
    x_test[:,:,1:] = norm_x2.encode(x_test[:,:,1:])
    
    norm_y  = GaussianNormalizer(y_train)
    y_train = norm_y.encode(y_train)
    y_test  = norm_y.encode(y_test)
       
    points_train = points[None,:,None,:].repeat(ntrain,1,121,1) 
    x_train = x_train[:,None,:,:].repeat(1,1656,1,1)  
    x_train_cat = torch.cat([points_train,x_train], dim = -1)   
    x_train_cat = x_train_cat.reshape(x_train_cat.shape[0], x_train_cat.shape[1], 
                                      x_train_cat.shape[2]*x_train_cat.shape[3])   
    
    points_test = points[None,:,None,:].repeat(ntest,1,121,1) 
    x_test = x_test[:,None,:,:].repeat(1,1656,1,1)  
    x_test_cat = torch.cat([points_test,x_test], dim = -1)   
    x_test_cat = x_test_cat.reshape(x_test_cat.shape[0], x_test_cat.shape[1], 
                                      x_test_cat.shape[2]*x_test_cat.shape[3])        
       
    x_train_cat = x_train_cat.to(torch.float32)
    x_test_cat = x_test_cat.to(torch.float32)

              
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_cat, y_train), 
                                               batch_size=batch_size, shuffle=True, drop_last=True) 
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_cat, y_test), 
                                              batch_size=batch_size, shuffle=False, drop_last=True) 

        
    current_directory = current_file_path + '/model_save/'
    sava_path =  current_directory + str(time_str) + '/'
    if not os.path.exists(sava_path):
        os.makedirs(sava_path)

 
    if args.model == 'Transolver':
        from models.Transolver import Model
        model = Model(n_hidden=args.width, n_layers=8, space_dim=1089,
                        fun_dim=0,
                        n_head=8,
                        mlp_ratio=2, out_dim=363,
                        slice_num=32,
                        unified_pos=0).cuda()
    
    if args.model == 'SpiderSolver':
        from models.SpiderSolver import SpiderSolver
        model = SpiderSolver(n_hidden=args.width, n_layers=args.n_layers, space_dim=1089,
                        fun_dim=0,
                        n_head=args.n_head,
                        mlp_ratio=args.mlp_ratio, out_dim=363,
                        unified_pos=0,
                        n_clusters = args.n_clusters, 
                        onion_num = args.onion_num).cuda()
        surf, onion_index, onion_index_0, velo_cluster_index_one_hot,labels_SpectralClustering = SpiderSolver_BloodFlow_DataProcess(points, args.n_clusters, args.onion_num, sava_path)   

        surf = surf.float()
        onion_index = onion_index.float()
        velo_cluster_index_one_hot = velo_cluster_index_one_hot.float()
        points = points.float()

        cfd_data = Data(points=points, surf=surf.bool(), onion_index=onion_index, onion_index_0 =onion_index_0,
                        velo_cluster_index_one_hot=velo_cluster_index_one_hot)
        cfd_data = cfd_data.cuda()



    ################################################################
    # training and evaluation
    ################################################################

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) 
    myloss = LpLoss(d=3, p=2, size_average  = False)
    time_start = time.perf_counter()
    time_step = time.perf_counter()
    
    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    ET_list = np.zeros((epochs))

    
    for ep in range(epochs):
        model.train() 
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if args.model == 'Transolver':
                out = model(x).reshape(batch_size,1656,121,3)
            else:
                out = model(x, cfd_data).reshape(batch_size,1656,121,3)
                      
            l2 = myloss(out, y) 
            l2 = l2.mean()           
            l2.backward()        
            out_real = norm_y.decode(out.cpu()).contiguous().reshape(batch_size, -1)
            y_real = norm_y.decode(y.cpu()).reshape(batch_size, -1) 
            train_l2 += myloss(out_real, y_real).mean()         
                   
            optimizer.step()
        scheduler.step()
        model.eval() 
        test_l2 = 0.0
        
        with torch.no_grad(): 
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
    
                if args.model == 'Transolver':
                    out = model(x).reshape(batch_size,1656,121,3)
                else:
                    out = model(x, cfd_data).reshape(batch_size,1656,121,3)
                        
                out_real = norm_y.decode(out.cpu()).contiguous().reshape(batch_size, -1)
                y_real = norm_y.decode(y.cpu()).reshape(batch_size, -1)
                test_l2 += myloss(out_real, y_real).mean()           
                loss_max_test= (abs(out_real- y_real)).max(axis=1).values.mean()
    
        train_l2 /= (ntrain/batch_size)
        test_l2 /= (ntest/batch_size)
        train_error[ep] = train_l2
        test_error[ep] = test_l2
        
        ET_list[ep] = loss_max_test
        time_step_end = time.perf_counter()
        T = time_step_end - time_step

        print('Epoch: %d, Train L2: %.5f, Test L2: %.5f, Emax_te: %.5f, Time: %.3fs'%(ep, train_l2, test_l2, loss_max_test, T))
        time_step = time.perf_counter()
               
    print("\n=============================")
    print("Training done...")
    print("=============================\n")
    
    ep = np.linspace(0,args.epochs-1,args.epochs)   
    plt.plot(ep, np.array(train_error) )
    plt.savefig(sava_path + "train_loss.png",pad_inches=0)  
    plt.close()
    plt.plot(ep,np.array(test_error) )
    plt.savefig(sava_path + "val_loss.png",pad_inches=0)      
    plt.close()   
    
    torch.save(model, sava_path + 'SpiderSolver.pth')
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_cat, y_test), 
                                              batch_size=1, shuffle=False, drop_last=True) 
    pre_test = torch.zeros(y_test.shape)     
    y_test   = torch.zeros(y_test.shape)      
    x_test   = torch.zeros(x_test.shape)      
    test_l2 = 0.0

    index = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if args.model == 'Transolver':
                out = model(x).reshape(1,1656,121,3)
            else:
                out = model(x, cfd_data).reshape(1,1656,121,3)              
            out_real = norm_y.decode(out.cpu())
            y_real   = norm_y.decode(y.cpu())
            pre_test[index,:,:,:] = out_real[0,:,:,:]      
            test_l2 +=  torch.norm(out_real - y_real) / torch.norm(y_real)
            index = index + 1
        test_l2 /= ntest  
        print('Finally Test L2: %.5f,'%(test_l2))  
    print(time_str)
            
    # ================ Save Data ====================
    pred_dict = {'pre_test': pre_test.cpu().detach().numpy(),}
    
    sio.savemat(sava_path +'NORM_pre_'  + args.CaseName + '.mat', mdict = pred_dict)
    print('Training time: %.3f'%(time_step_end - time_start))
    print('Num of paras : %d'%(count_params(model)))

if __name__ == "__main__":
    
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    import argparse
    parser = argparse.ArgumentParser(description='Arguments for SpiderSolver')

    parser.add_argument('--model', default='SpiderSolver', type=str, help='Model name')
    parser.add_argument('--modes', default=64, type=int, help='Modes for Fourier layers')
    parser.add_argument('--Fmodes', default=16, type=int, help='Feature modes for Fourier layers')
    parser.add_argument('--width', default=512, type=int, help='Width of the network')
    parser.add_argument('--size_of_nodes', default=1656, type=int, help='Number of nodes in the graph')
    parser.add_argument('--batch_size', default=10, type=int, help='Batch size for training')
    parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--data_dir', default='/home/qikai/Rebuttle_Copy_from_161/BloodFlow.mat', 
                        type=str, help='Path to the dataset')
    
    parser.add_argument('--num_train', default=400, type=int, help='Number of training samples')
    parser.add_argument('--num_test', default=100, type=int, help='Number of testing samples')
    parser.add_argument('--CaseName', default='velocity_xyz_0', type=str, help='Case name for saving results')
    parser.add_argument('--basis', default='LBO', type=str, help='Type of basis used')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--n_head', default=8, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--mlp_ratio', default=2, type=int)

    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--onion_num', default=5, type=int)

    args = parser.parse_args()
    print(args)
    main(args)

