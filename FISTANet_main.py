import datetime
import scipy.io as sio
from loader import DataSplit
import numpy as np
from helpers import show_image_matrix
import torch 
from M1LapReg import callLapReg, MatMask
from M5FISTANet import FISTANet
import argparse
from solver import Solver
import os
from os.path import dirname, join as pjoin
from metric import compute_measure
import hdf5storage

import matplotlib.pyplot as plt

if __name__ == '__main__':
    batch_size = 64
    validation_split = 0.2

    root_dir = './data/'
    train_loader, val_loader, test_loader = DataSplit(root_dir=root_dir,
                                                    batch_size=batch_size,
                                                    validation_split=validation_split)
    for i, (y_v, images_v) in enumerate(test_loader):
        if i==0:
            test_images = images_v
            test_data = y_v
        elif i==2:
            break
        else:
            test_images = torch.cat((test_images, images_v), axis=0)
            test_data = torch.cat((test_data, y_v), axis=0)

    test_images = torch.unsqueeze(test_images, 1)  
    test_images = test_images.reshape(-1,1,56,56)  
    test_data = torch.unsqueeze(test_data, 1)
    test_data = torch.unsqueeze(test_data, 3)      

    print('===========================================')
    print('FISTA-Net')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = './data/'

    Phi_data = hdf5storage.loadmat(pjoin(data_dir, 'J_sen_56.mat'))
    J_eit = Phi_data['J']
    J_eit = torch.from_numpy(J_eit)
    J_eit = torch.tensor(J_eit, dtype=torch.float32, device=device)

    L_data = hdf5storage.loadmat(pjoin(data_dir, 'Lapmat.mat'))
    DM = L_data['Lapmat']

    DMts = torch.from_numpy(DM) 
    mask = MatMask(64)
    mask = torch.from_numpy(mask)
    mask = torch.tensor(mask, dtype=torch.float32, device=device)
    DMts = torch.tensor(DMts, dtype=torch.float32, device=device)

    fista_net_mode = 0    # 0, test mode; 1, train mode.
    fista_net = FISTANet(9, J_eit, DMts, mask)
    fista_net = fista_net.to(device)

    print('Total number of parameters fista net:',
        sum(p.numel() for p in fista_net.parameters()))

    # define arguments of fista_net
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='FISTANet')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--save_path', type=str, default='./models/FISTANet/')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--device', default=device)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--test_epoch', type=int, default=50)
    args = parser.parse_args()

    if args.start_epoch > 0:
        f_trained = pjoin(args.save_path, 'epoch_{}.ckpt'.format(args.start_epoch))
        fista_net.load_state_dict(torch.load(f_trained))

    solver = Solver(fista_net, train_loader, args, test_data,)
    if fista_net_mode == 1:
        solver.train()
        fista_net_test = solver.test()
    else:
        fista_net_test = solver.test()

    fista_net_test = fista_net_test.cpu().double()

    if fista_net_mode == 0:
        Result_test = './data/Reconstruct_layer_fistanet.mat'
        print(len(fista_net_test))
        sio.savemat(Result_test, {'fista_net_test': np.array(fista_net_test)})
     
    fig_name = './figures/fista_net_' + str(args.test_epoch) + 'epoch.png'
    results = [test_images, fista_net_test]
    # Evalute reconstructed images with PSNR, SSIM, RMSE.
    p_fista, s_fista, m_fista = compute_measure(test_images, fista_net_test, 1)
    print('PSNR: {:.5f}\t SSIM: {:.5f} \t RMSE: {:.5f}'.format(p_fista, s_fista, m_fista))
    titles = ['truth', 'fista_net']
    show_image_matrix(fig_name, results, titles=titles, indices=slice(0, 15))