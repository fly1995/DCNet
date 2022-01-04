import os
import nibabel as nb
import SimpleITK as sitk
import numpy as np
import keras.backend as K
import keras.backend.tensorflow_backend as KTF
from Network import *
from keras.preprocessing.image import array_to_img
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))
K.set_image_data_format('channels_last')
import time
time_start = time.time()


def predict_ED(root, predict_save,Net,weights):
    file_name1 = os.listdir(root)
    for file in file_name1:
        file_name2 = os.listdir(os.path.join(root, file))
        ED = nb.load(os.path.join(root, file, str(file_name2[2]))).get_data()#测试50没有标签的数据和测试自己划分的33个数据ED都=2
        ED_CROP = np.zeros((512, 512, ED.shape[2]))
        ED_CROP[int((512 - ED.shape[0]) / 2):int((512 - ED.shape[0]) / 2 + ED.shape[0]),int((512 - ED.shape[1]) / 2):int((512 - ED.shape[1]) / 2 + ED.shape[1]), :] = ED[:, :, :]
        out = ED_CROP[int((ED_CROP.shape[0] - 160) / 2):int((ED_CROP.shape[0] + 160) / 2),int((ED_CROP.shape[1] - 160) / 2):int((ED_CROP.shape[1] + 160) / 2), :]#(160, 160, x)
        ED_CROP = out[:, :, :, np.newaxis].transpose(2, 1, 0, 3)#(x, 160, 160, 1)
        model = Net
        model.load_weights(weights)
        preds = model.predict(ED_CROP) #(x, 160, 160, 4)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        preds = np.argmax(preds, axis=-1)  #(x, 160, 160)
        x = np.zeros((int(ED.shape[2]), 512, 512))  #(x, 512,512)
        x[:, int((x.shape[1] - 160) / 2): int((x.shape[1] + 160) / 2), int((x.shape[2] - 160) / 2): int((x.shape[2] + 160) / 2)] = preds #(x, 512,512)
        y = x[:, int((512-ED.shape[1])/2): int((512+ED.shape[1])/2), int((512-ED.shape[0])/2): int((512+ED.shape[0])/2)]
        for i in range(y.shape[0]):  # out.shape[0]
            img = y[i, :, :, np.newaxis]
            img = array_to_img(img)
            img.save(predict_save+str(file_name2[2][0:10])+'_ED%d.jpg' % i)
        y = y.astype('uint8')
        image = sitk.GetImageFromArray(y)
        sitk.WriteImage(image, predict_save + str(file_name2[2][0:10]) + '_ED.nii.gz')


def predict_ES(root, predict_save, Net, weights):
    file_name1 = os.listdir(root)
    for file in file_name1:
        file_name2 = os.listdir(os.path.join(root, file))
        ED = nb.load(os.path.join(root, file, str(file_name2[4]))).get_data()#测试50没有标签的数据ED=3，测试自己划分的20个数据ED=4
        ED_CROP = np.zeros((512, 512, ED.shape[2]))
        ED_CROP[int((512 - ED.shape[0]) / 2):int((512 - ED.shape[0]) / 2 + ED.shape[0]),int((512 - ED.shape[1]) / 2):int((512 - ED.shape[1]) / 2 + ED.shape[1]), :] = ED[:, :, :]
        out = ED_CROP[int((ED_CROP.shape[0] - 160) / 2):int((ED_CROP.shape[0] + 160) / 2),int((ED_CROP.shape[1] - 160) / 2):int((ED_CROP.shape[1] + 160) / 2), :]#(160, 160, x)
        ED_CROP = out[:, :, :, np.newaxis].transpose(2, 1, 0, 3)#(x, 160, 160, 1)
        model = Net
        model.load_weights(weights)
        preds = model.predict(ED_CROP) #(x, 160, 160, 4)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        preds = np.argmax(preds, axis=-1)  #(x, 160, 160)
        x = np.zeros((int(ED.shape[2]), 512, 512))  #(x, 512,512)
        x[:, int((x.shape[1] - 160) / 2): int((x.shape[1] + 160) / 2), int((x.shape[2] - 160) / 2): int((x.shape[2] + 160) / 2)] = preds #(x, 512,512)
        y = x[:, int((512-ED.shape[1])/2): int((512+ED.shape[1])/2), int((512-ED.shape[0])/2): int((512+ED.shape[0])/2)]
        for i in range(y.shape[0]):  # out.shape[0]
            img = y[i, :, :, np.newaxis]
            img = array_to_img(img)
            img.save(predict_save+str(file_name2[2][0:10])+'_ES%d.jpg' % i)
        y = y.astype('uint8')
        image = sitk.GetImageFromArray(y)
        sitk.WriteImage(image, predict_save + str(file_name2[2][0:10]) + '_ES.nii.gz')


root = r'E:\D4_2017ACDC_EDES_Segmentation\2017ACDC\Ablation Study\test_data'
predict_save = r'E:\D4_2017ACDC_EDES_Segmentation\2017ACDC\Ablation Study\test_results\\Proposed+k=[1-5]\\'
Net = DBDB_Net()
weights = r'E:\D4_2017ACDC_EDES_Segmentation\code\Proposed+k=[1-5].hdf5'
predict_ED(root, predict_save, Net, weights)
predict_ES(root, predict_save, Net, weights)

time_end = time.time()
print('totally cost', time_end-time_start)
