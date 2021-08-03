

from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, AveragePooling3D, UpSampling3D, \
  BatchNormalization, Activation, LeakyReLU, Dropout
from Modules.Networks.UNet.instancenormalization import InstanceNormalization
from keras.optimizers import Adam, Adadelta, SGD
import Modules.LossFunction as lf
import Modules.Common_modules as cm
import numpy as np

#######################################################
# Getting 3D U-net:


def get_3d_unet_norm(nfeatures):

    nfeatures = nfeatures

    inputs = Input((cm.WMHshape[0], cm.WMHshape[1], cm.WMHshape[2], 2))

    conv1 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(inputs)
    norm1 = InstanceNormalization()(conv1)
    act1 = LeakyReLU(alpha=1e-2)(norm1)
    conv1 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act1)
    norm1 = InstanceNormalization()(conv1)
    act1 = LeakyReLU(alpha=1e-2)(norm1)
    pool1 = AveragePooling3D(pool_size=(2, 2, 2))(act1)

    conv2 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool1)
    norm2 = InstanceNormalization()(conv2)
    act2 = LeakyReLU(alpha=1e-2)(norm2)
    conv2 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act2)
    norm2 = InstanceNormalization()(conv2)
    act2 = LeakyReLU(alpha=1e-2)(norm2)
    pool2 = AveragePooling3D(pool_size=(2, 2, 2))(act2)

    conv3 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool2)
    norm3 = InstanceNormalization()(conv3)
    act3 = LeakyReLU(alpha=1e-2)(norm3)
    conv3 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act3)
    norm3 = InstanceNormalization()(conv3)
    act3 = LeakyReLU(alpha=1e-2)(norm3)
    pool3 = AveragePooling3D(pool_size=(2, 2, 2))(act3)

    conv4 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool3)
    norm4 = InstanceNormalization()(conv4)
    act4 = LeakyReLU(alpha=1e-2)(norm4)
    conv4 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act4)
    norm4 = InstanceNormalization()(conv4)
    act4 = LeakyReLU(alpha=1e-2)(norm4)
    pool4 = AveragePooling3D(pool_size=(2, 2, 2))(act4)

    conv5 = Conv3D(filters=nfeatures*16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool4)
    norm5 = InstanceNormalization()(conv5)
    act5 = LeakyReLU(alpha=1e-2)(norm5)
    conv5 = Conv3D(filters=nfeatures*16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act5)
    norm5 = InstanceNormalization()(conv5)
    act5 = LeakyReLU(alpha=1e-2)(norm5)

    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(act5), act4])
    conv6 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up6)
    norm6 = InstanceNormalization()(conv6)
    act6 = LeakyReLU(alpha=1e-2)(norm6)
    conv6 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act6)
    norm6 = InstanceNormalization()(conv6)
    act6 = LeakyReLU(alpha=1e-2)(norm6)

    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(act6), act3])
    conv7 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up7)
    norm7 = InstanceNormalization()(conv7)
    act7 = LeakyReLU(alpha=1e-2)(norm7)
    conv7 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act7)
    norm7 = InstanceNormalization()(conv7)
    act7 = LeakyReLU(alpha=1e-2)(norm7)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(act7), act2])
    conv8 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up8)
    norm8 = InstanceNormalization()(conv8)
    act8 = LeakyReLU(alpha=1e-2)(norm8)
    conv8 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act8)
    norm8 = InstanceNormalization()(conv8)
    act8 = LeakyReLU(alpha=1e-2)(norm8)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(act8), act1])
    conv9 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up9)
    norm9 = InstanceNormalization()(conv9)
    act9 = LeakyReLU(alpha=1e-2)(norm9)
    conv9 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act9)
    norm9 = InstanceNormalization()(conv9)
    act9 = LeakyReLU(alpha=1e-2)(norm9)

    conv10 = Conv3D(filters=2, kernel_size=(1, 1, 1), strides=(1, 1, 1))(act9)
    act10 = Activation('sigmoid', name='final_result')(conv10)

    model = Model(input=inputs, output=act10)

    weights = np.array([1.0, 1.0])
    loss = lf.focal_weighted_categorical_crossentropy_loss_2_class(weights)
    model.compile(optimizer=Adam(lr=2e-4), loss=loss, metrics=[lf.dice_coef_metrics, "categorical_accuracy"])

    return model


def get_3d_unet(nfeatures):

    nfeatures = nfeatures

    inputs = Input((cm.WMHshape[0], cm.WMHshape[1], cm.WMHshape[2], 2))

    conv1 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(inputs)
    act1 = LeakyReLU(alpha=1e-2)(conv1)
    conv1 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act1)
    act1 = LeakyReLU(alpha=1e-2)(conv1)
    pool1 = AveragePooling3D(pool_size=(2, 2, 2))(act1)

    conv2 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool1)
    act2 = LeakyReLU(alpha=1e-2)(conv2)
    conv2 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act2)
    act2 = LeakyReLU(alpha=1e-2)(conv2)
    pool2 = AveragePooling3D(pool_size=(2, 2, 2))(act2)

    conv3 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool2)
    act3 = LeakyReLU(alpha=1e-2)(conv3)
    conv3 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act3)
    act3 = LeakyReLU(alpha=1e-2)(conv3)
    pool3 = AveragePooling3D(pool_size=(2, 2, 2))(act3)

    conv4 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool3)
    act4 = LeakyReLU(alpha=1e-2)(conv4)
    conv4 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act4)
    act4 = LeakyReLU(alpha=1e-2)(conv4)
    pool4 = AveragePooling3D(pool_size=(2, 2, 2))(act4)

    conv5 = Conv3D(filters=nfeatures*16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool4)
    act5 = LeakyReLU(alpha=1e-2)(conv5)
    conv5 = Conv3D(filters=nfeatures*16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act5)
    act5 = LeakyReLU(alpha=1e-2)(conv5)

    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(act5), act4])
    conv6 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up6)
    act6 = LeakyReLU(alpha=1e-2)(conv6)
    conv6 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act6)
    act6 = LeakyReLU(alpha=1e-2)(conv6)

    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(act6), act3])
    conv7 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up7)
    act7 = LeakyReLU(alpha=1e-2)(conv7)
    conv7 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act7)
    act7 = LeakyReLU(alpha=1e-2)(conv7)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(act7), act2])
    conv8 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up8)
    act8 = LeakyReLU(alpha=1e-2)(conv8)
    conv8 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act8)
    act8 = LeakyReLU(alpha=1e-2)(conv8)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(act8), act1])
    conv9 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up9)
    act9 = LeakyReLU(alpha=1e-2)(conv9)
    conv9 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act9)
    act9 = LeakyReLU(alpha=1e-2)(conv9)

    conv10 = Conv3D(filters=2, kernel_size=(1, 1, 1), strides=(1, 1, 1))(act9)
    act10 = Activation('sigmoid', name='final_result')(conv10)

    model = Model(input=inputs, output=act10)

    weights = np.array([1.0, 1.0])
    loss = lf.focal_weighted_categorical_crossentropy_loss_2_class(weights)
    model.compile(optimizer=Adam(lr=2e-4), loss=loss, metrics=[lf.dice_coef_metrics, "categorical_accuracy"])

    return model


def get_3d_unet_crf_intensity(nfeatures):

    from Modules.Networks.CRF.crsasrnn.crfrnn_layer_intensity import New_CrfRnnLayer_3d_GPU_2_2_intensity

    nfeatures = nfeatures

    inputs = Input((cm.WMHshape[0], cm.WMHshape[1], cm.WMHshape[2], 2))

    conv1 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(inputs)
    norm1 = InstanceNormalization()(conv1)
    act1 = LeakyReLU(alpha=1e-2)(norm1)
    conv1 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act1)
    norm1 = InstanceNormalization()(conv1)
    act1 = LeakyReLU(alpha=1e-2)(norm1)
    pool1 = AveragePooling3D(pool_size=(2, 2, 2))(act1)

    conv2 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool1)
    norm2 = InstanceNormalization()(conv2)
    act2 = LeakyReLU(alpha=1e-2)(norm2)
    conv2 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act2)
    norm2 = InstanceNormalization()(conv2)
    act2 = LeakyReLU(alpha=1e-2)(norm2)
    pool2 = AveragePooling3D(pool_size=(2, 2, 2))(act2)

    conv3 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool2)
    norm3 = InstanceNormalization()(conv3)
    act3 = LeakyReLU(alpha=1e-2)(norm3)
    conv3 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act3)
    norm3 = InstanceNormalization()(conv3)
    act3 = LeakyReLU(alpha=1e-2)(norm3)
    pool3 = AveragePooling3D(pool_size=(2, 2, 2))(act3)

    conv4 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool3)
    norm4 = InstanceNormalization()(conv4)
    act4 = LeakyReLU(alpha=1e-2)(norm4)
    conv4 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act4)
    norm4 = InstanceNormalization()(conv4)
    act4 = LeakyReLU(alpha=1e-2)(norm4)
    pool4 = AveragePooling3D(pool_size=(2, 2, 2))(act4)

    conv5 = Conv3D(filters=nfeatures*16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool4)
    norm5 = InstanceNormalization()(conv5)
    act5 = LeakyReLU(alpha=1e-2)(norm5)
    conv5 = Conv3D(filters=nfeatures*16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act5)
    norm5 = InstanceNormalization()(conv5)
    act5 = LeakyReLU(alpha=1e-2)(norm5)

    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(act5), act4])
    conv6 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up6)
    norm6 = InstanceNormalization()(conv6)
    act6 = LeakyReLU(alpha=1e-2)(norm6)
    conv6 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act6)
    norm6 = InstanceNormalization()(conv6)
    act6 = LeakyReLU(alpha=1e-2)(norm6)

    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(act6), act3])
    conv7 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up7)
    norm7 = InstanceNormalization()(conv7)
    act7 = LeakyReLU(alpha=1e-2)(norm7)
    conv7 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act7)
    norm7 = InstanceNormalization()(conv7)
    act7 = LeakyReLU(alpha=1e-2)(norm7)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(act7), act2])
    conv8 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up8)
    norm8 = InstanceNormalization()(conv8)
    act8 = LeakyReLU(alpha=1e-2)(norm8)
    conv8 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act8)
    norm8 = InstanceNormalization()(conv8)
    act8 = LeakyReLU(alpha=1e-2)(norm8)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(act8), act1])
    conv9 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up9)
    norm9 = InstanceNormalization()(conv9)
    act9 = LeakyReLU(alpha=1e-2)(norm9)
    conv9 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act9)
    norm9 = InstanceNormalization()(conv9)
    act9 = LeakyReLU(alpha=1e-2)(norm9)

    conv10 = Conv3D(filters=2, kernel_size=(1, 1, 1), strides=(1, 1, 1))(act9)
    act10 = Activation('softmax', name='before_crf')(conv10)

    crf = New_CrfRnnLayer_3d_GPU_2_2_intensity(image_dims=(cm.WMHshape[0], cm.WMHshape[1], cm.WMHshape[2]),
                                   num_classes=2,
                                   theta_alpha=4.46,
                                   theta_beta=4.5,
                                   theta_gamma=0.11,
                                   num_iterations=1,
                                   trainable=True,
                                   name='crfrnn3d_trainable')([act10, inputs])

    conv11 = Conv3D(filters=2, kernel_size=(1, 1, 1), strides=(1, 1, 1))(crf)
    act11 = Activation('sigmoid', name='final_result')(conv11)

    model = Model(input=inputs, output=act11)

    #weights = np.array([1.0, 20.0])
    #loss = lf.focal_weighted_categorical_crossentropy_loss_2_class(weights)
    model.compile(optimizer=Adam(lr=2e-4), loss=lf.dice_coef_loss, metrics=[lf.dice_coef_metrics, "categorical_accuracy"])

    return model


def get_3d_unet_crf_spatial(nfeatures):

    from Modules.Networks.CRF.crsasrnn.crfrnn_layer_spatial import New_CrfRnnLayer_3d_GPU_2_2_spatial

    nfeatures = nfeatures

    inputs = Input((cm.WMHshape[0], cm.WMHshape[1], cm.WMHshape[2], 2))
    conv1 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(inputs)
    norm1 = InstanceNormalization()(conv1)
    act1 = LeakyReLU(alpha=1e-2)(norm1)
    conv1 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act1)
    norm1 = InstanceNormalization()(conv1)
    act1 = LeakyReLU(alpha=1e-2)(norm1)
    pool1 = AveragePooling3D(pool_size=(2, 2, 2))(act1)

    conv2 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool1)
    norm2 = InstanceNormalization()(conv2)
    act2 = LeakyReLU(alpha=1e-2)(norm2)
    conv2 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act2)
    norm2 = InstanceNormalization()(conv2)
    act2 = LeakyReLU(alpha=1e-2)(norm2)
    pool2 = AveragePooling3D(pool_size=(2, 2, 2))(act2)

    conv3 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool2)
    norm3 = InstanceNormalization()(conv3)
    act3 = LeakyReLU(alpha=1e-2)(norm3)
    conv3 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act3)
    norm3 = InstanceNormalization()(conv3)
    act3 = LeakyReLU(alpha=1e-2)(norm3)
    pool3 = AveragePooling3D(pool_size=(2, 2, 2))(act3)

    conv4 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool3)
    norm4 = InstanceNormalization()(conv4)
    act4 = LeakyReLU(alpha=1e-2)(norm4)
    conv4 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act4)
    norm4 = InstanceNormalization()(conv4)
    act4 = LeakyReLU(alpha=1e-2)(norm4)
    pool4 = AveragePooling3D(pool_size=(2, 2, 2))(act4)

    conv5 = Conv3D(filters=nfeatures*16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool4)
    norm5 = InstanceNormalization()(conv5)
    act5 = LeakyReLU(alpha=1e-2)(norm5)
    conv5 = Conv3D(filters=nfeatures*16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act5)
    norm5 = InstanceNormalization()(conv5)
    act5 = LeakyReLU(alpha=1e-2)(norm5)

    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(act5), act4])
    conv6 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up6)
    norm6 = InstanceNormalization()(conv6)
    act6 = LeakyReLU(alpha=1e-2)(norm6)
    conv6 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act6)
    norm6 = InstanceNormalization()(conv6)
    act6 = LeakyReLU(alpha=1e-2)(norm6)

    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(act6), act3])
    conv7 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up7)
    norm7 = InstanceNormalization()(conv7)
    act7 = LeakyReLU(alpha=1e-2)(norm7)
    conv7 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act7)
    norm7 = InstanceNormalization()(conv7)
    act7 = LeakyReLU(alpha=1e-2)(norm7)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(act7), act2])
    conv8 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up8)
    norm8 = InstanceNormalization()(conv8)
    act8 = LeakyReLU(alpha=1e-2)(norm8)
    conv8 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act8)
    norm8 = InstanceNormalization()(conv8)
    act8 = LeakyReLU(alpha=1e-2)(norm8)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(act8), act1])
    conv9 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up9)
    norm9 = InstanceNormalization()(conv9)
    act9 = LeakyReLU(alpha=1e-2)(norm9)
    conv9 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act9)
    norm9 = InstanceNormalization()(conv9)
    act9 = LeakyReLU(alpha=1e-2)(norm9)

    conv10 = Conv3D(filters=2, kernel_size=(1, 1, 1), strides=(1, 1, 1))(act9)
    act10 = Activation('softmax', name='before_crf')(conv10)

    crf = New_CrfRnnLayer_3d_GPU_2_2_spatial(image_dims=(cm.WMHshape[0], cm.WMHshape[1], cm.WMHshape[2]),
                                   num_classes=2,
                                   theta_alpha=4.46,
                                   theta_beta=4.5,
                                   theta_gamma=0.11,
                                   num_iterations=1,
                                   trainable=True,
                                   name='crfrnn3d_trainable')([act10, act10])

    conv11 = Conv3D(filters=2, kernel_size=(1, 1, 1), strides=(1, 1, 1))(crf)
    act11 = Activation('sigmoid', name='final_result')(conv11)

    model = Model(input=inputs, output=act11)

    #weights = np.array([1.0, 20.0])
    #loss = lf.focal_weighted_categorical_crossentropy_loss_2_class(weights)
    model.compile(optimizer=Adam(lr=2e-4), loss=lf.dice_coef_loss, metrics=[lf.dice_coef_metrics, "categorical_accuracy"])

    return model


def get_3d_unet_crf_posterior_first_FM(nfeatures):

    from Modules.Networks.CRF.crsasrnn.crfrnn_layer_posterior import New_CrfRnnLayer_3d_GPU_2_2_posterior

    nfeatures = nfeatures

    inputs = Input((cm.WMHshape[0], cm.WMHshape[1], cm.WMHshape[2], 2))

    conv1 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(inputs)
    norm1 = InstanceNormalization()(conv1)
    act1 = LeakyReLU(alpha=1e-2)(norm1)
    conv1 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act1)
    norm1 = InstanceNormalization()(conv1)
    act1 = LeakyReLU(alpha=1e-2)(norm1)

    conv1 = Conv3D(filters=2, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act1)
    ref1 = Activation('softmax', name='reference_map')(conv1)

    pool1 = AveragePooling3D(pool_size=(2, 2, 2))(act1)

    conv2 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool1)
    norm2 = InstanceNormalization()(conv2)
    act2 = LeakyReLU(alpha=1e-2)(norm2)
    conv2 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act2)
    norm2 = InstanceNormalization()(conv2)
    act2 = LeakyReLU(alpha=1e-2)(norm2)
    pool2 = AveragePooling3D(pool_size=(2, 2, 2))(act2)

    conv3 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool2)
    norm3 = InstanceNormalization()(conv3)
    act3 = LeakyReLU(alpha=1e-2)(norm3)
    conv3 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act3)
    norm3 = InstanceNormalization()(conv3)
    act3 = LeakyReLU(alpha=1e-2)(norm3)
    pool3 = AveragePooling3D(pool_size=(2, 2, 2))(act3)

    conv4 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool3)
    norm4 = InstanceNormalization()(conv4)
    act4 = LeakyReLU(alpha=1e-2)(norm4)
    conv4 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act4)
    norm4 = InstanceNormalization()(conv4)
    act4 = LeakyReLU(alpha=1e-2)(norm4)
    pool4 = AveragePooling3D(pool_size=(2, 2, 2))(act4)

    conv5 = Conv3D(filters=nfeatures*16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool4)
    norm5 = InstanceNormalization()(conv5)
    act5 = LeakyReLU(alpha=1e-2)(norm5)
    conv5 = Conv3D(filters=nfeatures*16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act5)
    norm5 = InstanceNormalization()(conv5)
    act5 = LeakyReLU(alpha=1e-2)(norm5)

    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(act5), act4])
    conv6 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up6)
    norm6 = InstanceNormalization()(conv6)
    act6 = LeakyReLU(alpha=1e-2)(norm6)
    conv6 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act6)
    norm6 = InstanceNormalization()(conv6)
    act6 = LeakyReLU(alpha=1e-2)(norm6)

    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(act6), act3])
    conv7 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up7)
    norm7 = InstanceNormalization()(conv7)
    act7 = LeakyReLU(alpha=1e-2)(norm7)
    conv7 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act7)
    norm7 = InstanceNormalization()(conv7)
    act7 = LeakyReLU(alpha=1e-2)(norm7)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(act7), act2])
    conv8 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up8)
    norm8 = InstanceNormalization()(conv8)
    act8 = LeakyReLU(alpha=1e-2)(norm8)
    conv8 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act8)
    norm8 = InstanceNormalization()(conv8)
    act8 = LeakyReLU(alpha=1e-2)(norm8)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(act8), act1])
    up9 = concatenate([up9, ref1])
    conv9 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up9)
    norm9 = InstanceNormalization()(conv9)
    act9 = LeakyReLU(alpha=1e-2)(norm9)
    conv9 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act9)
    norm9 = InstanceNormalization()(conv9)
    act9 = LeakyReLU(alpha=1e-2)(norm9)

    conv10 = Conv3D(filters=2, kernel_size=(1, 1, 1), strides=(1, 1, 1))(act9)
    act10 = Activation('softmax', name='before_crf')(conv10)

    crf = New_CrfRnnLayer_3d_GPU_2_2_posterior(image_dims=(cm.WMHshape[0], cm.WMHshape[1], cm.WMHshape[2]),
                                   num_classes=2,
                                   theta_alpha=4.46,
                                   theta_beta=4.5,
                                   theta_gamma=0.11,
                                   num_iterations=1,
                                   trainable=True,
                                   name='crfrnn3d_trainable')([act10, ref1])

    conv11 = Conv3D(filters=2, kernel_size=(1, 1, 1), strides=(1, 1, 1))(crf)
    act11 = Activation('sigmoid', name='final_result')(conv11)

    model = Model(input=inputs, output=act11)

    #weights = np.array([1.0, 20.0])
    #loss = lf.focal_weighted_categorical_crossentropy_loss_2_class(weights)
    model.compile(optimizer=Adam(lr=2e-4), loss=lf.dice_coef_loss, metrics=[lf.dice_coef_metrics, "categorical_accuracy"])

    return model


def get_3d_unet_crf_posterior(nfeatures):

    from Modules.Networks.CRF.crsasrnn.crfrnn_layer_posterior import New_CrfRnnLayer_3d_GPU_2_2_posterior

    nfeatures = nfeatures

    inputs = Input((cm.WMHshape[0], cm.WMHshape[1], cm.WMHshape[2], 2), name='inputs')

    conv1 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(inputs)
    norm1 = InstanceNormalization()(conv1)
    act1 = LeakyReLU(alpha=1e-2)(norm1)
    conv1 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act1)
    norm1 = InstanceNormalization()(conv1)
    act1 = LeakyReLU(alpha=1e-2)(norm1)

    pool1 = AveragePooling3D(pool_size=(2, 2, 2))(act1)

    conv2 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool1)
    norm2 = InstanceNormalization()(conv2)
    act2 = LeakyReLU(alpha=1e-2)(norm2)
    conv2 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act2)

    norm2 = InstanceNormalization()(conv2)
    act2 = LeakyReLU(alpha=1e-2)(norm2)
    pool2 = AveragePooling3D(pool_size=(2, 2, 2))(act2)

    conv3 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool2)
    norm3 = InstanceNormalization()(conv3)
    act3 = LeakyReLU(alpha=1e-2)(norm3)
    conv3 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act3)
    norm3 = InstanceNormalization()(conv3)
    act3 = LeakyReLU(alpha=1e-2)(norm3)
    pool3 = AveragePooling3D(pool_size=(2, 2, 2))(act3)

    conv4 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool3)
    norm4 = InstanceNormalization()(conv4)
    act4 = LeakyReLU(alpha=1e-2)(norm4)
    conv4 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act4)
    norm4 = InstanceNormalization()(conv4)
    act4 = LeakyReLU(alpha=1e-2)(norm4)
    pool4 = AveragePooling3D(pool_size=(2, 2, 2))(act4)

    conv5 = Conv3D(filters=nfeatures*16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool4)
    norm5 = InstanceNormalization()(conv5)
    act5 = LeakyReLU(alpha=1e-2)(norm5)
    conv5 = Conv3D(filters=nfeatures*16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act5)
    norm5 = InstanceNormalization()(conv5)
    act5 = LeakyReLU(alpha=1e-2)(norm5)

    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(act5), act4])
    conv6 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up6)
    norm6 = InstanceNormalization()(conv6)
    act6 = LeakyReLU(alpha=1e-2)(norm6)
    conv6 = Conv3D(filters=nfeatures*8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act6)
    norm6 = InstanceNormalization()(conv6)
    act6 = LeakyReLU(alpha=1e-2)(norm6)

    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(act6), act3])
    conv7 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up7)
    norm7 = InstanceNormalization()(conv7)
    act7 = LeakyReLU(alpha=1e-2)(norm7)
    conv7 = Conv3D(filters=nfeatures*4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act7)
    norm7 = InstanceNormalization()(conv7)
    act7 = LeakyReLU(alpha=1e-2)(norm7)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(act7), act2])
    conv8 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up8)
    norm8 = InstanceNormalization()(conv8)
    act8 = LeakyReLU(alpha=1e-2)(norm8)
    conv8 = Conv3D(filters=nfeatures*2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act8)
    norm8 = InstanceNormalization()(conv8)
    act8 = LeakyReLU(alpha=1e-2)(norm8)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(act8), act1])
    conv9 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up9)
    norm9 = InstanceNormalization()(conv9)
    act9 = LeakyReLU(alpha=1e-2)(norm9)
    conv9 = Conv3D(filters=nfeatures*1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act9)
    norm9 = InstanceNormalization()(conv9)
    act9 = LeakyReLU(alpha=1e-2)(norm9)

    conv10 = Conv3D(filters=2, kernel_size=(1, 1, 1), strides=(1, 1, 1))(act9)
    act10 = Activation('softmax', name='before_crf')(conv10)

    crf = New_CrfRnnLayer_3d_GPU_2_2_posterior(image_dims=(cm.WMHshape[0], cm.WMHshape[1], cm.WMHshape[2]),
                                   num_classes=2,
                                   theta_alpha=4.46,
                                   theta_beta=4.5,
                                   theta_gamma=0.11,
                                   num_iterations=1,
                                   trainable=True,
                                   name='crfrnn3d_trainable')([act10, act10])

    conv11 = Conv3D(filters=2, kernel_size=(1, 1, 1), strides=(1, 1, 1))(crf)
    act11 = Activation('sigmoid', name='final_result')(conv11)

    model = Model(input=inputs, output=act11)

    #weights = np.array([1.0, 20.0])
    #loss = lf.focal_weighted_categorical_crossentropy_loss_2_class(weights)
    model.compile(optimizer=Adam(lr=2e-4), loss=lf.dice_coef_loss, metrics=[lf.dice_coef_metrics, "categorical_accuracy"])

    return model




