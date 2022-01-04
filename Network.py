from __future__ import division, print_function
from keras.layers import Dense,  Concatenate, concatenate, MaxPooling2D, ZeroPadding2D, UpSampling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, multiply
from keras.optimizers import Adam, SGD
from Loss_function import *
from keras.layers import Conv2D, Conv2DTranspose, Input, Cropping2D, add, Dropout, Reshape, Activation
from keras import Model, layers
from keras.layers import Input,Conv2D,BatchNormalization,Activation,Reshape
from keras import optimizers
from keras.layers import Dropout, Lambda
from keras import backend as K
from keras.layers import Input, average
import tensorflow as tf
import keras

shape1 = (160, 160, 1)
shape2 = (256, 256, 1)
smooth = 1e-5

def dice_coef(y_true, y_pred):
    sum1 = 2*tf.reduce_sum(y_true*y_pred, axis=(0, 1, 2))
    sum2 = tf.reduce_sum(y_true+y_pred, axis=(0, 1, 2))
    dice = (sum1+smooth)/(sum2+smooth)
    dice = tf.reduce_mean(dice)
    return dice
def dice_coef_loss(y_true, y_pred):
    #return -(dice_coef(y_true, y_pred)) #loss1
    return -(dice_coef(y_true, y_pred)+myo(y_true, y_pred)+lv(y_true, y_pred)+rv(y_true, y_pred))#loss2 2017ACDC
 
def diceCoeff(gt, pred, smooth = 1e-5):
    pred_flat = tf.layers.flatten(pred)
    gt_flat = tf.layers.flatten(gt)
    intersection = K.sum((pred_flat * gt_flat))
    unionset = K.sum(pred_flat) + K.sum(gt_flat)
    score = (2 * intersection + smooth) / (unionset + smooth)
    return score
#2017ACDC 1:rv,2:myo 3:lv  4:la, 5；ra, 6：ao, 7：pa
def rv(y_true,y_pred,):
    class_dice = []
    for i in range(1,2):
        class_dice.append(diceCoeff(y_true[:,:,:,i:i +1 ],y_pred[:,:,:,i:i + 1]))
    mean_dice = sum(class_dice) / len(class_dice)
    return mean_dice
def myo(y_true,y_pred,):
    class_dice = []
    for i in range(2,3):
        class_dice.append(diceCoeff(y_true[:,:,:,i:i +1 ],y_pred[:,:,:,i:i + 1]))
    mean_dice = sum(class_dice) / len(class_dice)
    return mean_dice
def lv(y_true,y_pred,):
    class_dice = []
    for i in range(3,4):
        class_dice.append(diceCoeff(y_true[:,:,:,i:i +1 ],y_pred[:,:,:,i:i + 1]))
    mean_dice = sum(class_dice) / len(class_dice)
    return mean_dice
def la(y_true,y_pred,):
    class_dice = []
    for i in range(4, 5):
        class_dice.append(diceCoeff(y_true[:,:,:,i:i +1],y_pred[:,:,:,i:i + 1]))
    mean_dice = sum(class_dice) / len(class_dice)
    return mean_dice
def ra(y_true,y_pred,):
    class_dice = []
    for i in range(5, 6):
        class_dice.append(diceCoeff(y_true[:,:,:,i:i +1],y_pred[:,:,:,i:i + 1]))
    mean_dice = sum(class_dice) / len(class_dice)
    return mean_dice
def ao(y_true,y_pred,):
    class_dice = []
    for i in range(6, 7):
        class_dice.append(diceCoeff(y_true[:,:,:,i:i +1],y_pred[:,:,:,i:i + 1]))
    mean_dice = sum(class_dice) / len(class_dice)
    return mean_dice
def pa(y_true,y_pred,):
    class_dice = []
    for i in range(7, 8):
        class_dice.append(diceCoeff(y_true[:,:,:,i:i +1],y_pred[:,:,:,i:i + 1]))
    mean_dice = sum(class_dice) / len(class_dice)
    return mean_dice

def conv_bn_relu(input_tensor, flt):
    x = Conv2D(flt, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(flt, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x
def UNet_pp(inputs=Input(shape1)):#Unet
    conv1_1 = conv_bn_relu(inputs, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)

    conv2_1 = conv_bn_relu(pool1, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

    conv3_1 = conv_bn_relu(pool2, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_1)

    conv4_1 = conv_bn_relu(pool3, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_1)

    conv5_1 = conv_bn_relu(pool4, 512)

    up1_2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2_1)
    conv1_2 = concatenate([conv1_1, up1_2], 3)
    conv1_2 = conv_bn_relu(conv1_2, 32)

    up2_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3_1)
    conv2_2 = concatenate([conv2_1, up2_2], 3)
    conv2_2 = conv_bn_relu(conv2_2, 64)

    up3_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4_1)
    conv3_2 = concatenate([conv3_1, up3_2], 3)
    conv3_2 = conv_bn_relu(conv3_2, 128)

    up4_2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5_1)
    conv4_2 = concatenate([conv4_1, up4_2], 3)
    conv4_2 = conv_bn_relu(conv4_2, 256)

    up1_3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2_2)
    conv1_3 = concatenate([conv1_1, conv1_2, up1_3], 3)
    conv1_3 = conv_bn_relu(conv1_3, 32)

    up2_3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3_2)
    conv2_3 = concatenate([conv2_1, conv2_2, up2_3], 3)
    conv2_3 = conv_bn_relu(conv2_3, 64)

    up3_3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4_2)
    conv3_3 = concatenate([conv3_1, conv3_2, up3_3], 3)
    conv3_3 = conv_bn_relu(conv3_3, 128)

    up1_4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2_3)
    conv1_4 = concatenate([conv1_1, conv1_2, conv1_3, up1_4], 3)
    conv1_4 = conv_bn_relu(conv1_4, 32)

    up2_4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3_3)
    conv2_4 = concatenate([conv2_1, conv2_2, conv2_3, up2_4], 3)
    conv2_4 = conv_bn_relu(conv2_4, 64)

    up1_5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv2_4)
    conv1_5 = concatenate([conv1_1, conv1_2, conv1_3, conv1_4, up1_5], 3)
    conv1_5 = conv_bn_relu(conv1_5, 32)

    output = Conv2D(4, (1, 1), activation='softmax',)(conv1_5)
    model = Model(inputs=[inputs], outputs=[output])
    #model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef, myo, lv, rv])
    model.summary()
    return model
def U_Net(input_size = shape1):
    flt = 64
    inputs = Input(input_size)

    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(flt * 16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv5)

    up6 = concatenate([Conv2DTranspose(flt * 8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    #up6 = concatenate([UpSampling2D( (2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv6)

    up7 = concatenate([Conv2DTranspose(flt * 4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    #up7 = concatenate([UpSampling2D( (2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up7)
    conv7 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv7)

    up8 = concatenate([Conv2DTranspose(flt * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    #up8 = concatenate([UpSampling2D((2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up8)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv8)

    up9 = concatenate([Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    #up9 = concatenate([UpSampling2D((2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up9)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(4, (1, 1), activation='softmax')(conv9)
    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef, myo, lv, rv])
    model.summary()

    return model

def IB3(input,flt):
    conv1 = Conv2D(flt, (1,1), activation='relu', padding='same')(input)
    conv3 = Conv2D(flt, (3,3), activation='relu', padding='same')(input)
    conv5 = Conv2D(flt, (5,5), activation='relu', padding='same')(input)
    concate = concatenate([conv3,conv5,conv1],axis=3)
    conv = Conv2D(flt, (1, 1), activation='relu')(concate)
    output = conv
    return output
def convblock(m, dim, layername, res=1, drop=0, **kwargs):
    n = Conv2D(filters=dim, name= layername + '_conv1', **kwargs)(m)
    n = BatchNormalization(momentum=0.95, epsilon=0.001)(n)
    n = Dropout(drop)(n) if drop else n
    n = Conv2D(filters=dim, name= layername + '_conv2', **kwargs)(n)
    n = BatchNormalization(momentum=0.95, epsilon=0.001)(n)
    return Concatenate()([m, n]) if res else n
def MSCMR(input_shape=(shape1), num_classes=4, maxpool=True, weights=None):
    kwargs = dict(kernel_size=3, strides=1,activation='relu',padding='same',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,trainable=True)
    num_classes = num_classes
    data = Input(shape=input_shape, dtype='float', name='data')
    # encoder
    enconv1 = convblock(data, dim=32, layername='block1', **kwargs)
    pool1 = MaxPooling2D(pool_size=3, strides=2,padding='same',name='pool1')(enconv1) if maxpool \
        else Conv2D(filters=32, strides=2, name='pool1')(enconv1)

    enconv2 = convblock(pool1, dim=64, layername='block2', **kwargs)
    pool2 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool2')(enconv2) if maxpool \
        else Conv2D(filters=64, strides=2, name='pool2')(enconv2)

    enconv3 = convblock(pool2, dim=128, layername='block3', **kwargs)
    pool3 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool3')(enconv3) if maxpool \
        else Conv2D( filters=128, strides=2, name='pool3')(enconv3)

    enconv4 = convblock(pool3, dim=256, layername='block4', **kwargs)
    pool4 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool4')(enconv4) if maxpool \
        else Conv2D(filters=256, strides=2, name='pool4')(enconv4)

    enconv5 = convblock(pool4, dim=512, layername='block5notl', **kwargs)
    # decoder
    up1 = Conv2D(filters=256, kernel_size=1, padding='same', activation='relu',
                 name='up1')(UpSampling2D(size=[2, 2])(enconv5))
    merge1 = Concatenate()([up1,enconv4])
    deconv6 = convblock(merge1, dim=256, layername='deconv6', **kwargs)

    up2 = Conv2D(filters=128, kernel_size=1, padding='same', activation='relu',
                 name='up2')(UpSampling2D(size=[2,2])(deconv6))
    merge2 = Concatenate()([up2,enconv3])
    deconv7 = convblock(merge2, dim=128, layername='deconv7', **kwargs)

    up3 = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu',name='up3')(UpSampling2D(size=[2, 2])(deconv7))
    merge3 = Concatenate()([up3, enconv2])
    deconv8 = convblock(merge3, dim=64, layername='deconv8', **kwargs)

    up4 = Conv2D(filters=32, kernel_size=1, padding='same', activation='relu',name='up4')(UpSampling2D(size=[2, 2])(deconv8))
    merge4 = Concatenate()([up4, enconv1])
    deconv9 = convblock(merge4, dim=32, drop=0.5, layername='deconv9', **kwargs)
    conv10 = Conv2D(filters=num_classes, kernel_size=1, padding='same', activation='relu',name='conv10')(deconv9)
    predictions = Conv2D(filters=num_classes, kernel_size=1, activation='softmax',padding='same', name='predictions')(conv10)
    model = Model(inputs=data, outputs=predictions)
    model.compile(optimizer=Adam(lr=0.001), loss=[dice_coef_loss], metrics=[dice_coef, myo, lv, rv])
    model.summary()
    return model

def FCN(input_size=shape1):
    flt = 64
    inputs = Input(input_size)
    conv1 = Conv2D(flt,(3,3),activation='relu',padding='same')(inputs)
    conv1 = Conv2D(flt,(3,3),activation='relu',padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(flt * 16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(flt * 16, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(flt * 16, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv6 = Conv2D(flt, (5, 5), activation='relu', padding='same')(pool5)
    conv7 = Conv2D(flt, (5, 5), activation='relu', padding='same')(conv6)
    up1 = UpSampling2D(size=(32, 32))(conv7)
    up2 =UpSampling2D(size=(16, 16))(pool4)
    conv_up2 =Conv2D(flt,(1,1))(up2)
    up3 =UpSampling2D(size=(8, 8))(pool3)
    conv_up3=Conv2D(flt,(1,1))(up3)
    add =layers.add([up1,conv_up2,conv_up3])
    output = Conv2D(4, (1, 1),activation='softmax')(add)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(lr=0.00001), loss=[dice_coef_loss], metrics=[dice_coef,myo,lv,rv])
    model.summary()
    return model

def CA(inputs):
    inputs_channels = int(inputs.shape[-1])
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(int(inputs_channels / 4))(x)#(?, 8)
    x = Activation('relu')(x)
    x = Dense(int(inputs_channels))(x)
    x = Activation('sigmoid')(x)
    x = Reshape((1, 1, inputs_channels))(x)#(?, 1, 1, 32)
    x = keras.layers.Multiply()([inputs, x])
    return x
def DCA(inputs):
    flt = inputs.get_shape().as_list()[-1]
    avg_pool = GlobalAveragePooling2D()(inputs)
    avg_pool = Reshape((1, 1, flt))(avg_pool)
    avg_pool = Conv2D(flt, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(avg_pool)
    max_pool = GlobalMaxPooling2D()(inputs)
    max_pool = Reshape((1, 1, flt))(max_pool)
    max_pool = Conv2D(flt, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(max_pool)
    feature = Add()([avg_pool, max_pool])
    feature = Activation('sigmoid')(feature)
    return multiply([feature, inputs])
def DBDB(inputs, rate, flt):
    conv1 = Conv2D(flt, (1, 1),  activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv2D(flt, (1, 1),  activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(flt, (3, 3),  dilation_rate=rate, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv2D(flt, (1, 1),  activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv3 = BatchNormalization()(conv3)
    conv3 = DCA(conv3)

    conv4 = Conv2D(flt, (3, 3),  dilation_rate=rate, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv4 = BatchNormalization()(conv4)

    output = concatenate([conv1, conv2, conv3, conv4], axis=-1)
    output = Conv2D(flt, (1, 1),  activation='relu', padding='same', kernel_initializer='he_normal')(output)
    return output
def DCNet(input_size=(160, 160, 1)):#2017ACDC(160, 160, 1)
    flt = 64
    inputs = Input(input_size)
    conv1 = DBDB(inputs, rate=1, flt=flt)
    conv2 = DBDB(inputs, rate=2, flt=flt)
    conv3 = DBDB(inputs, rate=3, flt=flt)
    conv4 = DBDB(inputs, rate=4, flt=flt)
    conv5 = DBDB(inputs, rate=5, flt=flt)
    #conv6 = DBDB(inputs, rate=6, flt=flt)
    concate1 = concatenate([conv1, conv2, conv3, conv4, conv5], axis=-1)#这里是通过消融实验确定的，最后rate=1,2，也就是conv1 conv2
    concate1 = Conv2D(flt*2, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(concate1)
    f1 = DCA(concate1)
    conv6 = DBDB(f1, rate=1, flt=flt*2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv6)#80
    conv7 = DBDB(pool1, rate=1, flt=flt*4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv7)#40
    conv8 = DBDB(pool2, rate=1, flt=flt*6)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv8)#20
    conv9 = DBDB(pool3, rate=1, flt=flt*8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv9)#10
    conv10 = DBDB(pool4, rate=1, flt=flt*16)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv10)#5

    conv11 = DBDB(pool5, rate=1, flt=flt*8)
    up1 = concatenate([Conv2DTranspose(flt * 8, (2, 2), strides=(2, 2), padding='same')(conv11), pool4], axis=-1) # 10
    conv12 = DBDB(up1, rate=1, flt=flt * 6)
    up2 = concatenate([Conv2DTranspose(flt * 6, (2, 2), strides=(2, 2), padding='same')(conv12), pool3], axis=-1)  # 20
    conv13 = DBDB(up2, rate=1, flt=flt * 4)
    up3 = concatenate([Conv2DTranspose(flt * 4, (2, 2), strides=(2, 2), padding='same')(conv13), pool2], axis=-1)  # 40
    conv14 = DBDB(up3, rate=1, flt=flt * 2)
    up4 = concatenate([Conv2DTranspose(flt * 2, (2, 2), strides=(2, 2), padding='same')(conv14), pool1], axis=-1)  # 80
    conv15 = DBDB(up4, rate=1, flt=flt)
    up5 = concatenate([Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv15), conv6], axis=-1)  # 160

    output = Conv2D(4, (1, 1), activation='softmax')(up5)#2017ACDC最后输出通道是4
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef, myo, lv, rv])#2017ACDC
    model.summary()
    return model







