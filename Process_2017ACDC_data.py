import os
import nibabel as nb
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
from keras.utils import to_categorical
from keras.preprocessing.image import array_to_img
import tensorflow as tf

def data_aug(data):
    data_rot90 = tf.image.rot90(data)
    data_rot180 = tf.image.rot90(data_rot90)
    data_rot270 = tf.image.rot90(data_rot180)
    data_u2d = tf.image.flip_up_down(data)
    data_l2r = tf.image.flip_left_right(data)
    data_trans = tf.image.transpose_image(data)
    session = tf.Session()
    data_rot90 = session.run(data_rot90)
    data_rot180 = session.run(data_rot180)
    data_rot270 = session.run(data_rot270)
    data_u2d = session.run(data_u2d)
    data_l2r = session.run(data_l2r)
    data_trans = session.run(data_trans)
    #data_flip_rot_save = np.concatenate([data, data_rot90, data_rot180, data_rot270, data_u2d, data_l2r, data_trans], axis=0)
    data_flip_rot_save = np.concatenate([data, data_rot90, data_rot180, data_rot270, data_u2d, data_l2r, data_trans],axis=0)
    return data_flip_rot_save

def data(root):
    file_name1 = os.listdir(root)
    list1 = []
    list2 = []
    list3 = []
    list4 = []

    for file in file_name1:
        file_name2 = os.listdir(os.path.join(root, file))

        ED = nb.load(os.path.join(root, file, str(file_name2[2]))).get_data()
        ED_GT = nb.load(os.path.join(root, file, str(file_name2[3]))).get_data()
        ES = nb.load(os.path.join(root, file, str(file_name2[4]))).get_data()
        ES_GT = nb.load(os.path.join(root, file, str(file_name2[5]))).get_data()#(216, 256, 10)
        print(file_name2[2])
        print(ED.shape)

        ED_CROP = np.zeros((512, 512, ED.shape[2]))
        ED_CROP[int((512-ED.shape[0]) / 2):int((512-ED.shape[0]) / 2+ED.shape[0]), int((512-ED.shape[1]) / 2):int((512-ED.shape[1]) / 2+ED.shape[1]), :] = ED[:, :, :]
        out1 = ED_CROP[int((ED_CROP.shape[0] - 160) / 2):int((ED_CROP.shape[0] + 160) / 2),int((ED_CROP.shape[1] - 160) / 2):int((ED_CROP.shape[1] + 160) / 2), :]

        ED_GT_CROP = np.zeros((512, 512, ED_GT.shape[2]))
        ED_GT_CROP[int((512 - ED_GT.shape[0]) / 2):int((512 - ED_GT.shape[0]) / 2 + ED_GT.shape[0]),int((512 - ED_GT.shape[1]) / 2):int((512 - ED_GT.shape[1]) / 2 + ED_GT.shape[1]), :] = ED_GT[:, :, :]
        out2 = ED_GT_CROP[int((ED_GT_CROP.shape[0] - 160) / 2):int((ED_GT_CROP.shape[0] + 160) / 2), int((ED_GT_CROP.shape[1] - 160) / 2):int((ED_GT_CROP.shape[1] + 160) / 2), :]

        ES_CROP = np.zeros((512, 512, ES.shape[2]))
        ES_CROP[int((512 - ES.shape[0]) / 2):int((512 - ES.shape[0]) / 2 + ES.shape[0]),int((512 - ES.shape[1]) / 2):int((512 - ES.shape[1]) / 2 + ES.shape[1]), :] = ES[:, :, :]
        out3 = ES_CROP[int((ES_CROP.shape[0] - 160) / 2):int((ES_CROP.shape[0] + 160) / 2), int((ES_CROP.shape[1] - 160) / 2):int((ES_CROP.shape[1] + 160) / 2), :]

        ES_GT_CROP = np.zeros((512, 512, ES_GT.shape[2]))
        ES_GT_CROP[int((512 - ES_GT.shape[0]) / 2):int((512 - ES_GT.shape[0]) / 2 + ES_GT.shape[0]),int((512 - ES_GT.shape[1]) / 2):int((512 - ES_GT.shape[1]) / 2 + ES_GT.shape[1]), :] = ES_GT[:, :, :]
        out4 = ES_GT_CROP[int((ES_GT_CROP.shape[0] - 160) / 2):int((ES_GT_CROP.shape[0] + 160) / 2),int((ES_GT_CROP.shape[1] - 160) / 2):int((ES_GT_CROP.shape[1] + 160) / 2), :]
        
        for i in range(out1.shape[2]):
            img = out1[:, :, i:i+1]
            img = array_to_img(img)
            img.save(r'E:\D4_map_segmentation\data\ED\ED_image\\'+str(file_name2[2][0:10])+'_%d.jpg' % i)
        for i in range(out2.shape[2]):
            img = out2[:, :, i:i+1]
            img = array_to_img(img)
            img.save(r'E:\D4_map_segmentation\data\ED\ED_GT_image\\'+str(file_name2[2][0:10])+'_%d.jpg' % i)
        for i in range(out3.shape[2]):
            img = out3[:, :, i:i + 1]
            img = array_to_img(img)
            img.save(r'E:\D4_map_segmentation\data\ES\ES_image\\' + str(file_name2[2][0:10]) + '_%d.jpg' % i)
        for i in range(out4.shape[2]):
            img = out4[:, :, i:i + 1]
            img = array_to_img(img)
            img.save(r'E:\D4_map_segmentation\data\ES\ES_GT_image\\' + str(file_name2[2][0:10]) + '_%d.jpg' % i)
        
        list1.append(out1)
        list2.append(out2)
        list3.append(out3)
        list4.append(out4)

    r1 = np.concatenate(list1, axis=-1)
    r1 = np.expand_dims(r1, axis=0)
    r1 = np.transpose(r1, axes=(3, 1, 2, 0))
    #np.save('E:\D4_2017ACDC_EDES_Segmentation\\2017ACDC\Ablation Study\\train_ed_img.npy', r1)
    print(r1.shape)

    r2 = np.concatenate(list2, axis=-1)
    r2 = np.expand_dims(r2, axis=0)
    r2 = np.transpose(r2, axes=(3, 1, 2, 0))
    r21 = to_categorical(r2, num_classes=4)
    #np.save('E:\D4_2017ACDC_EDES_Segmentation\\2017ACDC\Ablation Study\\train_ed_gt.npy', r21)
    print(r21.shape)

    r3 = np.concatenate(list3, axis=-1)
    r3 = np.expand_dims(r3, axis=0)
    r3 = np.transpose(r3, axes=(3, 1, 2, 0))
    #np.save('E:\D4_2017ACDC_EDES_Segmentation\\2017ACDC\Ablation Study\\train_es_img.npy', r3)
    print(r3.shape)

    r4 = np.concatenate(list4, axis=-1)
    r4 = np.expand_dims(r4, axis=0)
    r4 = np.transpose(r4, axes=(3, 1, 2, 0))
    r41 = to_categorical(r4, num_classes=4)
    #np.save('E:\D4_2017ACDC_EDES_Segmentation\\2017ACDC\Ablation Study\\train_es_gt.npy', r41)
    print(r41.shape)

    c1 = np.concatenate([r1, r3], axis=0)
    c2 = np.concatenate([r21, r41], axis=0)

    c1 = data_aug(c1)#7倍数据增广(10766，160, 160, 1),80个病例的数据增广
    print(c1.shape)
    np.save('E:\D4_2017ACDC_EDES_Segmentation\\2017ACDC\Ablation Study\\train_aug.npy', c1)

    c2 = data_aug(c2)#14倍数据增广(10766，160, 160, 4)
    print(c2.shape)
    np.save('E:\D4_2017ACDC_EDES_Segmentation\\2017ACDC\Ablation Study\\train_gt_aug.npy', c2)


root = r'E:\D4_2017ACDC_EDES_Segmentation\2017ACDC\Ablation Study\train_data'
data(root)






