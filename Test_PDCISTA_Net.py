import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
import glob
from time import time
import math
from torch.nn import init
import copy
# import cv2
import hdf5storage
import datetime
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim
from argparse import ArgumentParser

parser = ArgumentParser(description='PDCISTA-Net-plus')

parser.add_argument('--epoch_num', type=int, default=60, help='epoch number of model')
parser.add_argument('--layer_num', type=int, default=4, help='phase number of PDCISTA-Net')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--matrix_dir', type=str, default='sensing_matrix', help='sensing matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')

args = parser.parse_args()

epoch_num = args.epoch_num
learning_rate = args.learning_rate
layer_num = args.layer_num
gpu_list = args.gpu_list
test_name = args.test_name

try:
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
    torch.backends.cuda.matmul.allow_tf32 = False
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = False
except:
    pass

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_input = 120  
n_output = 3136
nrtrain = 78500  
batch_size = 64

Phi_data_Name = './%s/J_sen.mat' % (args.matrix_dir)
Phi_data = hdf5storage.loadmat(Phi_data_Name)
Phi_input = Phi_data['J']

Training_data_Name = 'Training_Data.mat'
Training_data = hdf5storage.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
Training_labels = Training_data['train_data_1']

Qinit_Name = './%s/Qinit_data_g2s_noser_0_1.mat' % (args.matrix_dir)

if os.path.exists(Qinit_Name):
    Qinit_data = hdf5storage.loadmat(Qinit_Name)
    Qinit = Qinit_data['Qinit']
else:
    X_data = Training_labels[:,0:3136].transpose()
    Y_data = np.dot(Phi_input, X_data)
    Y_YT = np.dot(Y_data, Y_data.transpose())
    X_YT = np.dot(X_data, Y_data.transpose())
    Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
    del X_data, Y_data, X_YT, Y_YT
    sio.savemat(Qinit_Name, {'Qinit': Qinit})

# Define PDCISTA-Net Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        self.lambda_step1 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step2 = nn.Parameter(torch.Tensor([0.5]))

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 1, 5, 5)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 5, 5)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 5, 5)))
        self.conv3_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 5, 5)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 5, 5)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 5, 5)))
        self.conv3_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 5, 5)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 64, 5, 5)))

        self.conv_D2 = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 1, 3, 3)))

        self.conv1_forward2 = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
        self.conv2_forward2 = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
        self.conv3_forward2 = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
        self.conv1_backward2 = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
        self.conv2_backward2 = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
        self.conv3_backward2 = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))

        self.conv_G2 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 64, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb  
        x_input = x.view(-1, 1, 56, 56) 

        x_D = F.conv2d(x_input, self.conv_D, padding=2) 
        x = F.conv2d(x_D, self.conv1_forward, padding=2)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=2)
        x = F.relu(x_forward) 
        x_forward = F.conv2d(x, self.conv3_forward, padding=2) 
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=2)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=2)
        x = F.relu(x_backward) 
        x_backward = F.conv2d(x, self.conv3_backward, padding=2) 

        x_G = F.conv2d(x_backward, self.conv_G, padding=2)

        x_D2 = F.conv2d(x_input, self.conv_D2, padding=1) #
        x2 = F.conv2d(x_D2, self.conv1_forward2, padding=1)
        x2 = F.relu(x2)
        x_forward2 = F.conv2d(x2, self.conv2_forward2, padding=1)
        x2 = F.relu(x_forward2) 
        x_forward2 = F.conv2d(x, self.conv3_forward2, padding=1) 
        x2 = torch.mul(torch.sign(x_forward2), F.relu(torch.abs(x_forward2) - self.soft_thr))

        x2 = F.conv2d(x2, self.conv1_backward2, padding=1)
        x2 = F.relu(x2)
        x_backward2 = F.conv2d(x2, self.conv2_backward2, padding=1)
        x2 = F.relu(x_backward2) 
        x_backward2 = F.conv2d(x2, self.conv3_backward2, padding=1) 

        x_G2 = F.conv2d(x_backward2, self.conv_G2, padding=1)

        x_pred = x_input + x_G + x_G2

        x_pred = x_pred.view(-1, 3136)  

        x = F.conv2d(x_forward, self.conv1_backward, padding=2)
        x = F.relu(x)
        x_D_est_1 = F.conv2d(x, self.conv2_backward, padding=2)
        x = F.relu(x_D_est_1) 
        x_D_est = F.conv2d(x, self.conv3_backward, padding=2)  

        x2 = F.conv2d(x_forward2, self.conv1_backward2, padding=1)
        x2 = F.relu(x2)
        x_D_est_2 = F.conv2d(x2, self.conv2_backward2, padding=1)
        x2 = F.relu(x_D_est_2) 
        x_D_est2 = F.conv2d(x2, self.conv3_backward2, padding=1) 


        symloss1 = x_D_est - x_D
        symloss2 = x_D_est2 - x_D2
        symloss = self.lambda_step1 * symloss1 + self.lambda_step2 * symloss2

        return [x_pred, symloss]

# Define ISTA-Net-plus
class PDCISTANet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(PDCISTANet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit):

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi) 
        PhiTb = torch.mm(Phix, Phi) 

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1)) 

        layers_sym = []   

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]


model = PDCISTANet(layer_num)
model = nn.DataParallel(model)
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/PDCISTA_Net_layer_%d_lr_%.4f_rec" % (args.model_dir, layer_num, learning_rate)

# Load pre-trained model with epoch number
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_num)))

# Load testing data 
Testing_data_Name = 'Testing_mixcircule_200'
Testing_data = hdf5storage.loadmat('./%s/%s' % (args.data_dir, Testing_data_Name))
Testing_labels = Testing_data['Testing_mixcircule_200'] 
# 
Result_test = './%s/Reconstruct_%d_layer_pdcista.mat' % (args.result_dir, layer_num)
##################################################Testing_Data_1_object_g2s

Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Phi = Phi.to(device)


Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
Qinit = Qinit.to(device)

print('\n')
print("Reconstruction Start")

output_result = []

starttime = datetime.datetime.now()
with torch.no_grad():
    for img_no in range(len(Testing_labels)):

        imgName_ = Testing_labels[img_no,3136:]
        imgName = imgName_[np.newaxis,:]
        start = time()

        batch_x = torch.from_numpy(imgName)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)

        [x_output, loss_layers_sym] = model(batch_x, Phi, Qinit)

        end = time()

        Prediction_value = x_output.cpu().data.numpy()
        output_result.append(Prediction_value)
        loss_sym = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(layer_num - 1):
            loss_sym += torch.mean(torch.pow(loss_layers_sym[k + 1], 2))       
        loss_sym = loss_sym.cpu().data.numpy()

        output_loss = "loss_sym: %.4f\n" % (loss_sym)

sio.savemat(Result_test, {'output_result': output_result})        
print('\n')
output_data = "Epoch number of model is %d \n" % (epoch_num)
print(output_data)

print("Reconstruction End")
