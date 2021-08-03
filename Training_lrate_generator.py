from __future__ import print_function

import random
from keras import callbacks
import Modules.Networks_Antonio.Callbacks as cb
import Modules.Common_modules as cm
from Modules.CommonUtil.ImageGeneratorManager import *
from Modules.CommonUtil.KerasBatchDataGenerator import *
from Modules.CommonUtil.LoadDataManager import *
from Modules.CommonUtil.WorkDirsManager import *
import argparse

import Modules.Networks.UNet.UNet_3D as UNet_3D
from Testing import test_net_dice
from Testing_post_processing import test_net_dice_post


img_rows = cm.WMHshape[1]
img_cols = cm.WMHshape[2]
slices = cm.WMHshape[0]


def train_and_predict(args, job, data_split, seed, model, root_path, pretrain, num_epoch):

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    test_img_list = []
    test_mask_list = []

    # Choose which subset you would like to use:

    # Set random seed
    data_seed = seed
    np.random.seed(data_seed)

    # Create image list
    GE3T_imgList = sorted(glob('data/WMH/GE3T/img_*.npy'))
    GE3T_maskList = sorted(glob('data/WMH/GE3T/mask_*.npy'))

    Singapore_imgList = sorted(glob('data/WMH/Singapore/img_*.npy'))
    Singapore_maskList = sorted(glob('data/WMH/Singapore/mask_*.npy'))

    Utrecht_imgList = sorted(glob('data/WMH/Utrecht/img_*.npy'))
    Utrecht_maskList = sorted(glob('data/WMH/Utrecht/mask_*.npy'))

    # Random selection for training, validation and testing
    GE3T_list = [list(pair) for pair in zip(GE3T_imgList, GE3T_maskList)]
    Singapore_list = [list(pair) for pair in zip(Singapore_imgList, Singapore_maskList)]
    Utrecht_list = [list(pair) for pair in zip(Utrecht_imgList, Utrecht_maskList)]

    np.random.shuffle(GE3T_list)
    np.random.shuffle(Singapore_list)
    np.random.shuffle(Utrecht_list)

    # if data_split == '36L':
    #     x_train_list, y_train_list = map(list, zip(
    #         *(GE3T_list[0:  12] + Singapore_list[0:  12] + Utrecht_list[0:  12])))
    #     x_val_list, y_val_list = map(list, zip(
    #         *(GE3T_list[12: 16] + Singapore_list[12: 16] + Utrecht_list[12: 16])))
    #     test_img_list, test_mask_list = map(list, zip(
    #         *(GE3T_list[16: 20] + Singapore_list[16: 20] + Utrecht_list[16: 20])))

    if data_split == '36L':
        x_train_list, y_train_list = map(list, zip(
            *(GE3T_list[0:  1] + Singapore_list[0:  0] + Utrecht_list[0:  0])))
        x_val_list, y_val_list = map(list, zip(
            *(GE3T_list[12: 13] + Singapore_list[12: 12] + Utrecht_list[12: 12])))
        test_img_list, test_mask_list = map(list, zip(
            *(GE3T_list[16: 20] + Singapore_list[16: 20] + Utrecht_list[16: 20])))

    elif data_split == '54L':
        x_train_list, y_train_list = map(list, zip(
            *(GE3T_list[0:  18] + Singapore_list[0:  18] + Utrecht_list[0:  18])))
        x_val_list, y_val_list = map(list, zip(
            *(GE3T_list[18: 20] + Singapore_list[18: 20] + Utrecht_list[18: 20])))
        test_img_list, test_mask_list = map(list, zip(
            *(GE3T_list[18: 20] + Singapore_list[18: 20] + Utrecht_list[18: 20])))

    # visualize validation results during training
    vis_num = 0
    (x_val_vis, y_val_vis) = LoadDataManager.loadData_ListFiles(x_val_list[vis_num:], y_val_list[vis_num:])

    use_existing = pretrain
    initial_epoch = 0
    model.summary()

    # Callbacks:
    # bestfilepath = root_path + 'model' + '/val_weights.hdf5'
    filepath = root_path + 'model' + '/val_weights.{epoch:03d}-{val_dice_coef_metrics:.5f}.hdf5'
    bestfilepath = root_path + 'model' + '/Best_weights.{epoch:03d}-{val_dice_coef_metrics:.5f}.hdf5'

    model_checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_dice_coef_metrics', verbose=0, save_best_only=False,
                                                      save_weights_only=True)

    model_best_checkpoint = callbacks.ModelCheckpoint(bestfilepath, monitor='val_dice_coef_metrics', verbose=0, save_best_only=True,
                                                      save_weights_only=True)

    record_history = cb.RecordLossHistory(root_path=root_path, filepath=root_path,
                                          vis_file=(x_val_vis, y_val_vis), job=job)

    callbacks_list = [record_history, model_best_checkpoint, model_checkpoint]

    # Should we load existing weights?
    # Set argument for call to train_and_predict to true at end of script
    if use_existing:
        model.load_weights(bestfilepath)

    if (args.multiClassCase):
        num_classes_out = args.numClassesMasks
    else:
        num_classes_out = 2

    if (args.slidingWindowImages or args.transformationImages):

        (train_xData, train_yData) = LoadDataManager.loadData_ListFiles(x_train_list, y_train_list)

        train_images_generator = getImagesDataGenerator3D(args.slidingWindowImages, args.prop_overlap_Z_X_Y, args.transformationImages)

        train_batch_data_generator = KerasTrainingBatchDataGenerator(IMAGES_DIMS_Z_X_Y,
                                                                     train_xData,
                                                                     train_yData,
                                                                     train_images_generator,
                                                                     num_classes_out=num_classes_out,
                                                                     batch_size=1,
                                                                     shuffle=True)

        # print("Number volumes: %s. Total Data batches generated: %s..." %(len(x_train_list), len(train_batch_data_generator)))

    use_validation_data = True
    if use_validation_data:
        print("Load Validation data...")
        if (args.slidingWindowImages or args.transformationImages):

            (valid_xData, valid_yData) = LoadDataManager.loadData_ListFiles(x_val_list, y_val_list)

            valid_images_generator = getImagesDataGenerator3D(args.slidingWindowImages, args.prop_overlap_Z_X_Y_valid, args.transformationImages)

            valid_batch_data_generator = KerasTrainingBatchDataGenerator(IMAGES_DIMS_Z_X_Y,
                                                                         valid_xData,
                                                                         valid_yData,
                                                                         valid_images_generator,
                                                                         num_classes_out=num_classes_out,
                                                                         batch_size=1,
                                                                         shuffle=True)
            validation_data = valid_batch_data_generator

            # print("Number volumes: %s. Total Data batches generated: %s..." %(len(x_val_list), len(valid_batch_data_generator)))

    else:
        validation_data = None

    # save training info:
    train_info = ('train_data_size: {}\n\n'
                  'val_data_size: {}\n\n'
                  'train_list: {}\n\n'
                  'val_list: {}\n\n'
                  'model: {}\n\n'
                  'pretrain: {}\n\n'
                  .format(len(x_train_list), len(x_val_list), x_train_list, x_val_list,
                          model.get_config(), use_existing))

    the_file = open(root_path + 'info.txt', 'w')
    the_file.write(train_info)
    the_file.close()

    # TRAINING MODEL
    # ----------------------------------------------
    print("-" * 30)
    print(job)
    print("-" * 30)

    if (args.slidingWindowImages or args.transformationImages):

        model.fit_generator(train_batch_data_generator,
                            # steps_per_epoch=len(train_batch_data_generator),
                            steps_per_epoch=2,
                            nb_epoch=num_epoch,
                            verbose=1,
                            shuffle=True,
                            validation_data=validation_data,
                            validation_steps=1,
                            callbacks=callbacks_list,
                            initial_epoch=initial_epoch)

    # ----------------------------------------------

    print('training finished')


def train_model(Test_only, job, split, seed, folder_name, num_epoch):
    # Choose whether to train based on the last model:
    # Show runtime:
    test_results = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--multiClassCase', type=str2bool, default=MULTICLASSCASE)
    parser.add_argument('--numClassesMasks', type=int, default=NUMCLASSESMASKS)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    parser.add_argument('--prop_overlap_Z_X_Y_valid', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y_valid)
    parser.add_argument('--transformationImages', type=str2bool, default=TRANSFORMATIONIMAGES)
    parser.add_argument('--use_restartModel', type=str2bool, default=USE_RESTARTMODEL)
    parser.add_argument('--restart_modelFile', default=RESTART_MODELFILE)
    parser.add_argument('--restart_only_weights', type=str2bool, default=RESTART_ONLY_WEIGHTS)
    parser.add_argument('--epoch_restart', type=int, default=EPOCH_RESTART)
    args = parser.parse_args()

    basic_path = folder_name + str(job) + '/' + str(split)

    if job == 'CNN_baseline':
        model = UNet_3D.get_3d_unet_norm(8)

        root_path = basic_path + '/seed' + str(seed) + '/'
        cm.mkdir(root_path + 'model')
        cm.mkdir(root_path + 'preview/val')

        pretrain = False

        if not Test_only:
            train_and_predict(args, job, split, seed, model, root_path, pretrain, num_epoch)

            # Testing model
            test_results = test_net_dice(args, job, split, seed, model, root_path, basic_path)

        else:
            # Testing model
            test_results = test_net_dice(args, job, split, seed, model, root_path, basic_path)

    ##############################################
    elif job == 'Intensity_CRF':

        model = UNet_3D.get_3d_unet_crf_intensity(16)

        root_path = basic_path + '/seed' + str(seed) + '/'
        cm.mkdir(root_path + 'model')
        cm.mkdir(root_path + 'preview/val')

        pretrain = False

        if not Test_only:
            train_and_predict(args, job, split, seed, model, root_path, pretrain, num_epoch)

            # Testing model
            test_results = test_net_dice(args, job, split, seed, model, root_path, basic_path)

        else:
            # Testing model
            test_results = test_net_dice(args, job, split, seed, model, root_path, basic_path)

    ##############################################
    elif job == 'Spatial_CRF':

        model = UNet_3D.get_3d_unet_crf_spatial(16)

        root_path = basic_path + '/seed' + str(seed) + '/'
        cm.mkdir(root_path + 'model')
        cm.mkdir(root_path + 'preview/val')

        pretrain = False

        if not Test_only:
            train_and_predict(args, job, split, seed, model, root_path, pretrain, num_epoch)

            # Testing model
            test_results = test_net_dice(args, job, split, seed, model, root_path, basic_path)

        else:
            # Testing model
            test_results = test_net_dice(args, job, split, seed, model, root_path, basic_path)

    ##############################################
    elif job == 'Posterior_CRF':

        model = UNet_3D.get_3d_unet_crf_posterior(2)

        root_path = basic_path + '/seed' + str(seed) + '/'
        cm.mkdir(root_path + 'model')
        cm.mkdir(root_path + 'preview/val')

        pretrain = False

        if not Test_only:
            train_and_predict(args, job, split, seed, model, root_path, pretrain, num_epoch)

            # Testing model
            test_results = test_net_dice(args, job, split, seed, model, root_path, basic_path)

        else:
            # Testing model
            test_results = test_net_dice(args, job, split, seed, model, root_path, basic_path)


    ##############################################
    elif job == 'Post_CRF':

        model = UNet_3D.get_3d_unet_norm(16)

        root_path = basic_path + '/seed' + str(seed) + '/'
        cm.mkdir(root_path + 'model')
        cm.mkdir(root_path + 'preview/val')

        if not Test_only:
            # Testing model
            test_results = test_net_dice_post(args, job, split, seed, model, root_path, basic_path, Test_only)

        else:
            # Testing model
            test_results = test_net_dice_post(args, job, split, seed, model, root_path, basic_path, Test_only)

    return test_results
