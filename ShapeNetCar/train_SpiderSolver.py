import numpy as np
import time, json, os
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader
from tqdm import tqdm
from timeit import default_timer
import scipy.io as io
from matplotlib import pylab as plt

def get_nb_trainable_params(model):
    '''
    Return the number of trainable parameters
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def train_SpiderSolver(device, model, train_loader, optimizer, scheduler, reg=1, labels_SpectralClustering = None):
    model.train()

    criterion_func = nn.MSELoss(reduction='none')
    losses_press = []
    losses_velo = []
    norm10 = []
    for cfd_data, geom in train_loader:
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        optimizer.zero_grad()
   
   
        out = model((cfd_data, geom, labels_SpectralClustering) )

        targets = cfd_data.y

        loss_press = criterion_func(out[cfd_data.surf, -1], targets[cfd_data.surf, -1]).mean(dim=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
        loss_velo = loss_velo_var.mean()
        total_loss = loss_velo + reg * loss_press

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6, norm_type=2)
        norm10.append( nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6, norm_type=2) )

        optimizer.step()
        scheduler.step()

        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())

    return np.mean(losses_press), np.mean(losses_velo), norm10


@torch.no_grad()
def test(device, model, test_loader, labels_SpectralClustering = None):
    model.eval()

    criterion_func = nn.MSELoss(reduction='none')
    losses_press = []
    losses_velo = []
    for cfd_data, geom in test_loader:
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
 
        out = model((cfd_data, geom, labels_SpectralClustering) )
        targets = cfd_data.y

        loss_press = criterion_func(out[cfd_data.surf, -1], targets[cfd_data.surf, -1]).mean(dim=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
        loss_velo = loss_velo_var.mean()

        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())

    return np.mean(losses_press), np.mean(losses_velo)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(device, train_dataset, val_dataset, Net, hparams, path, reg=1, val_iter=1, coef_norm=[], save_name = [], labels_SpectralClustering = None):
    model = Net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps= int((len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs']),
        final_div_factor=1000.,
    )
    start = time.time()
    

    labels_SpectralClustering = torch.from_numpy(labels_SpectralClustering).cuda()
    train_loss, val_loss = 1e5, 1e5
    # pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    train_l2_record = []
    val_l2_record = []   
    for epoch in range(hparams['nb_epochs']):
        t1 = default_timer()
        train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, drop_last=True)
        loss_velo, loss_press, norm10 = train_SpiderSolver(device, model, train_loader, optimizer, lr_scheduler, reg=reg, labels_SpectralClustering = labels_SpectralClustering)
        train_loss = loss_velo + reg * loss_press
        del (train_loader)

        if val_iter is not None and (epoch == hparams['nb_epochs'] - 1 or epoch % val_iter == 0):
            val_loader = DataLoader(val_dataset, batch_size=1)

            loss_velo, loss_press = test(device, model, val_loader,labels_SpectralClustering = labels_SpectralClustering)
            val_loss = loss_velo + reg * loss_press
            del (val_loader)

        t2 = default_timer()
        print(epoch, '%.2f'% (t2-t1), 'train_loss:', '%.4f'% train_loss, 'val_loss:', '%.4f'% val_loss,\
            'grad_norm_max = ', '%.4f'% max(norm10),   'grad_norm_mean = ', '%.4f'% (sum(norm10)/len(norm10)))
        
        train_l2_record.append(train_loss)
        val_l2_record.append(val_loss)
        
    
    io.savemat(path + 'train_process.mat', 
            {'train_loss': np.array(train_l2_record, dtype=object), 'train_val_loss': np.array(val_l2_record, dtype=object)})   
        
    train_l2_record = np.array(train_l2_record)
    val_l2_record = np.array(val_l2_record)
    ep = np.linspace(0,hparams['nb_epochs']-1,hparams['nb_epochs'])   
        
    plt.plot(ep,train_l2_record )
    plt.savefig(path + "train_loss.png",pad_inches=0)  
    plt.close()
    plt.plot(ep,val_l2_record )
    plt.savefig(path + "val_loss.png",pad_inches=0)      
    plt.close()        
        
        
    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
    torch.save(model, path + 'onion.pth')

    if val_iter is not None:
        with open(path + os.sep + save_name + '.json', 'a') as f:
            json.dump(
                {
                    'nb_parameters': params_model,
                    'time_elapsed': time_elapsed,
                    'hparams': hparams,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'coef_norm': list(coef_norm),
                }, f, indent=12, cls=NumpyEncoder
            )

    return model
