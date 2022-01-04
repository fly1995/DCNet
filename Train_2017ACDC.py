from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import keras.backend as K
from Network import *
import numpy as np
import time
import cv2
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))
K.set_image_data_format('channels_last')
time_start = time.time()


def train():
    #train= np.load('E:\D4_2017ACDC_EDES_Segmentation\\2017ACDC\\train_img.npy')#（13314,160,160,1） 完整的100个病例训练，50个测试，需要在在线平台进行测试，因为没有公布50的标签，ED和ES拼接
    #train_mask = np.load('E:\D4_2017ACDC_EDES_Segmentation\\2017ACDC\\train_mask.npy')#（13314,160,160,4）标签为0：背景， 标签为1：右心室，标签为2：心肌，标签为3：左心室
    train = np.load('E:\D4_2017ACDC_EDES_Segmentation\\2017ACDC\Ablation Study\\train_aug.npy')#(10766, 160, 160, 1) 80个病例for train，增广后的数据  ED和ES拼接，剩下20个做测试 训练：验证=2:1
    train_mask = np.load('E:\D4_2017ACDC_EDES_Segmentation\\2017ACDC\Ablation Study\\train_gt_aug.npy')#(10766, 160, 160, 4) 标签为0：背景， 标签为1：右心室，标签为2：心肌，标签为3：左心室
    earlystop = EarlyStopping(monitor='val_dice_coef', patience=10, verbose=1, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.1, patience=10, mode='max')
    #model = U_Net()
    #model = MSCMR()
    #model = FCN()
    #model = UNet_pp()
    model = DCNet() # Proposed
    csv_logger = CSVLogger('DCNet.csv')
    model_checkpoint = ModelCheckpoint(filepath='DCNet.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')
    model.fit(train, train_mask, batch_size=4, validation_split=0.1, epochs=1000, verbose=1, shuffle=True,callbacks=[model_checkpoint, csv_logger,  earlystop])#完整数据训练


if __name__ == '__main__':
    train()

time_end = time.time()
print('totally cost', time_end-time_start)
