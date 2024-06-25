import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import tensorflow as tf
print('TF Version : ', tf.__version__)
from network3 import EITNN_Network
from utils import load_dataset, save_history
from preprocess_data_chf import *
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.client import device_lib
import scipy.io as scio
import random

data2 = scio.loadmat('RM.mat')
RM=data2['RM']
## %%%%%%%%%%%%%%
image_size = 96

def pre_processing(input_feature):
    data2 = scio.loadmat('RM.mat')
    RM=data2['RM']
    pre_img = tf.matmul(input_feature, RM.T)
    n_pic, n_con = np.array(pre_img).shape
    features_new = np.flip(np.array(pre_img).reshape(n_pic,image_size,image_size),axis=1) 
    return features_new
def custom_objective(y_true, y_pred):
	"""
	costum objective function
	balanced sum of mse and mae
	"""
	mse_loss= tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1)
	# mae_loss = tf.keras.backend.mean(tf.keras.backend.abs(y_pred - y_true), axis=-1)
	# loss = mse_loss +  1.2 * mae_loss 
	loss = mse_loss
	return loss	

optimizer = tf.keras.optimizers.Adam(lr=0.00006, beta_1=0.9, beta_2=0.999)
model = tf.keras.models.load_model('checkpoints\EIT_NN_model3\\temp\model1.h5', compile=False)
model.compile(optimizer= optimizer , loss=custom_objective, metrics=['mean_absolute_error', 'mean_squared_error'])
data1 = scio.loadmat('experiment\exp_test_data_f.mat')

exp_features=data1['exp_test_data1']

i = random.randint(1,len(exp_features)-1)
print(i)
pos_1 = []
pos_1to6 = []
for pos_i in range(0,len(exp_features)):
    temp_min = []
    for i in range(0,len(exp_features[pos_i])):

        features_test = pre_processing(exp_features[pos_i][i].reshape(-1,120))*10

        plt.figure(1)
        plt.imshow(features_test.reshape(image_size,image_size))
        plt.title('Pre reconstruction',fontsize='large',fontweight='bold') 

        # print('features_train.shape:',features_test.shape)

        plt.figure(2)
        plt.plot(np.reshape(features_test,(-1,1)))
        plt.title('features_test_conduct',fontsize='large',fontweight='bold') 


        y_pred = model.predict(features_test.reshape(1,image_size,image_size,1), batch_size=1, verbose=0, steps=None, callbacks=None)


        plt.figure(3)
        plt.plot(np.reshape(y_pred,(-1,1)))
        plt.title('y_pred',fontsize='large',fontweight='bold') 

        plt.figure(4)
        plt.imshow(y_pred.reshape(image_size,image_size))
        plt.title('Predict',fontsize='large',fontweight='bold') 
        temp_min.append(y_pred.reshape(image_size,image_size).min())
    pos_1to6.append(temp_min)
