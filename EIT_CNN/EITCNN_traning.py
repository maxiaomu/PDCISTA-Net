import os
import argparse
import tensorflow as tf
print('TF Version : ', tf.__version__)
# from keras.callbacks import EarlyStopping

from network3 import EITNN_Network
# from ssamse import SSAMSE
from utils import load_dataset, save_history
import scipy.io as scio
from preprocess_data_chf import *
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.client import device_lib

import scipy.io as scio
data1 = scio.loadmat('datasets_train.mat')
labels=data1['cond_array_rot90']
features=data1['volt_train']

features_volt, test_features, labels_train, test_labels = train_test_split(features, labels, test_size=0.1, random_state=1024)

data2 = scio.loadmat('RM.mat')
RM=data2['RM']
## %%%%%%%%%%%%%%
image_size = 96
def pre_processing(input_feature):
    # Change voltage vector (120,) into pre_img (40,40)
    data2 = scio.loadmat('RM.mat')
    RM=data2['RM']
    # RM_path = 'F:\CHF\eidors-v3.9.1-ng\eidors\persenal\Tactile simulation\Deep learning\Data_generation\EITNN_test\RM.csv'
    # RM = pd.read_csv(RM_path,header=None)
    # RM = np.array(RM)
    # print(RM.T.shape)  #RM.T (120, 1600)
    pre_img = tf.matmul(input_feature, RM.T)
    n_pic, n_con = np.array(pre_img).shape
    features_new = np.flip(np.array(pre_img).reshape(n_pic,image_size,image_size),axis=1) 
    return features_new
    
features_train = pre_processing(features_volt)
##%%%%%%%%%%%%%

features_train=np.expand_dims(features_train,-1)
labels_train=np.expand_dims(labels_train,-1)

def EITNN_Network(image_size, PREIN_EN=True, noise_level=0):
	if PREIN_EN:
		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Conv2D(16,
                                 (3,3),
                                 input_shape=(image_size,image_size,1),
                                 activation='relu',
                                 padding='same'))
		model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Activation('relu'))

		model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Activation('relu'))

		model.add(tf.keras.layers.MaxPool2D())
		model.add(tf.keras.layers.Dropout(0.5))						
		model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Activation('relu'))

		model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Activation('relu'))

		model.add(tf.keras.layers.MaxPool2D())
		model.add(tf.keras.layers.Dropout(0.5))	

		model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Activation('relu'))

		model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Activation('relu'))
		model.add(tf.keras.layers.UpSampling2D((2,2)))

		model.add(tf.keras.layers.Conv2D(16, (3, 3), padding="same", use_bias=False))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Activation('relu'))

		model.add(tf.keras.layers.UpSampling2D((2,2)))
		model.add(tf.keras.layers.Conv2D(16, (3, 3), padding="same", use_bias=False))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Activation('relu'))

		model.add(tf.keras.layers.Conv2DTranspose(1, (3, 3), padding="same", use_bias=False))

	return model
model = EITNN_Network(image_size = 96, PREIN_EN=True, noise_level=0)
model.summary()

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

# 开始训练 epochs=300, batch_size=64,
if __name__ == '__main__':
	
	x_train, x_valid, y_train, y_valid = train_test_split(features_train, labels_train, test_size=0.1, random_state=1024)
	
	model = EITNN_Network(image_size = 96, PREIN_EN=True, noise_level=0)

	optimizer = tf.keras.optimizers.Adam(lr=0.00006, beta_1=0.9, beta_2=0.999)

	model.compile(loss=custom_objective, optimizer=optimizer, metrics=['mse'])

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)
	
	hist = model.fit(x_train, y_train, epochs=300, batch_size=64,
	                 validation_data=(x_valid, y_valid), callbacks=[early_stopping])
	path = os.path.join("checkpoints\EIT_NN_model3", 'temp')
	if not os.path.exists(path):
		os.mkdir(path)

	save_history(os.path.join(path, "train_loss2.txt"), hist.history["loss"])
	save_history(os.path.join(path, "val_loss2.txt"), hist.history["val_loss"])
	model.save(os.path.join(path, "model2.h5"))
