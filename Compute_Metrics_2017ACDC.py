import os
import glob
from hausdorff import hausdorff_distance
from Metrics import *
import numpy as np
import tensorflow as tf
import nibabel as nb

def compute_metrics(label_dir, pred_dir, value, position):
    name= []
    Dice = []
    Jaccard = []
    Sensitivity = []
    Specificity = []
    Hausdorff = []
    ASSD = []
    imgname = glob.glob(label_dir + '/*' + 'nii.gz')
    for file_name in imgname:
        midname = file_name[file_name.rindex("\\") + 1:]
        gt = nb.load(label_dir+midname).get_data()#原始标记为0,1,2,3
        label = np.zeros([gt.shape[0], gt.shape[1], gt.shape[2]])#rv:1  #myo:2  #lv:3
        pred = np.zeros([gt.shape[0], gt.shape[1], gt.shape[2]])  # rv:1  #myo:2  #lv:3
        pred0 = nb.load(pred_dir + midname).get_data()

        label[gt == value] = 1#(240, 160, 88) 标签为0：背景， 标签为1：右心室，标签为2：心肌，标签为3：左心室
        pred[pred0 == value] = 1#(240, 160, 88) 标签为0：背景， 标签为1：右心室，标签为2：心肌，标签为3：左心室

        session = tf.compat.v1.Session()
        dice = dice_coef(label, pred.astype('float64'))
        Dice.append(dice)
        jacc = jaccard(label, pred.astype('float64'))
        Jaccard.append(jacc)
        sen = sensitivity(label, pred.astype('float64'))
        Sensitivity.append(sen)
        spe = specificity(label, pred.astype('float64'))
        Specificity.append(spe)

        hau = []
        for i in range(label.shape[2]):
            a = label[:, :, i]
            b = pred[:, :, i]
            x = hausdorff_distance(a.astype('float32'), b, distance='manhattan')  # distance="euclidean",distance="chebyshev",distance="cosine"，distance='manhattan'
            hau.append(x)
        Hausdorff.append(sum(hau)/len(hau))
        surface = Surface(label.astype('float32'), pred, connectivity=2)
        assd = surface.get_average_symmetric_surface_distance()
        ASSD.append(assd)
        name.append(midname[0:13])

    save_dir = pred_dir
    np.savetxt(save_dir + position+ 'dice.csv', session.run(Dice), fmt='%s')
    np.savetxt(save_dir + position+ 'jaccard.csv', session.run(Jaccard), fmt='%s')
    np.savetxt(save_dir + position+ 'sensitivity.csv', session.run(Sensitivity), fmt='%s')
    np.savetxt(save_dir + position+ 'specificity.csv', session.run(Specificity), fmt='%s')
    np.savetxt(save_dir + position+ 'hausdorff.csv', Hausdorff, fmt='%s')
    np.savetxt(save_dir + position+ 'assd.csv', ASSD, fmt='%s')
    np.savetxt(save_dir + position+ 'name.csv', name, fmt='%s')

    Dice = session.run(Dice)
    Jaccard = session.run(Jaccard)
    Sensitivity = session.run(Sensitivity)
    Specificity = session.run(Specificity)

    print("Dice\tJaccard\tSensitivity\tSpecifity\tHausdorff\tASSD\t")
    print("%.2f±%.2f\t" % (np.mean(Dice) * 100, np.std(Dice) * 100), "%.2f±%.2f\t" % (np.mean(Jaccard) * 100, np.std(Jaccard) * 100),"%.3f±%.3f\t" % (np.mean(Sensitivity) * 100, np.std(Sensitivity) * 100),
          "%.3f±%.3f\t" % (np.mean(Specificity) * 100, np.std(Specificity) * 100),"%.3f±%.3f\t" % (np.mean(Hausdorff), np.std(Hausdorff)), "%.3f±%.3f\t" % (np.mean(ASSD), np.std(ASSD)))

value1 = 1 #1：右心室，2：心肌，3：左心室
position1 = 'rv' # 1:rv 2:myo 3:lv
value2 = 2 #1：右心室，2：心肌，3：左心室
position2 = 'myo' # 1:rv 2:myo 3:lv
value3 = 3 #1：右心室，2：心肌，3：左心室
position3 = 'lv' # 1:rv 2:myo 3:lv
label_dir = 'E:\D4_2017ACDC_EDES_Segmentation\\2017ACDC\Ablation Study\\test_results\ground_truth\\'
pred_dir = 'E:\D4_2017ACDC_EDES_Segmentation\\2017ACDC\Ablation Study\\test_results\\Proposed+k=[1-5]\\'
compute_metrics(label_dir, pred_dir, value3, position3)#lv
compute_metrics(label_dir, pred_dir, value1, position1)#rv
compute_metrics(label_dir, pred_dir, value2, position2)#myo
