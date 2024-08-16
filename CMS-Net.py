#解决二分类任务
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import random
import itertools
import math
import pandas as pd
import numpy as np
# import shap as shap
import tensorflow as tf
import keras
# from imblearn.over_sampling import RandomOverSampler
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.inspection import permutation_importance
from tensorflow.keras.utils import Sequence
# import keras.backend.tensorflow_backend as ktf
from keras import backend as K
from matplotlib import pyplot as plt
from sklearn import preprocessing
# from keras_contrib.callbacks import CyclicLR
from sklearn.metrics import classification_report,confusion_matrix,\
    precision_score, f1_score, accuracy_score, recall_score,\
    roc_auc_score,roc_curve, auc, consensus_score, brier_score_loss, log_loss
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping,  ModelCheckpoint,TensorBoard,\
    LearningRateScheduler
from sklearn.model_selection import StratifiedKFold,train_test_split
from keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler, StandardScaler, MaxAbsScaler
# from transformer_1 import create_transformer_model,MultiHeadSelfAttention
from transformer import create_transformer_model,MultiHeadSelfAttention
from dl_model import create_dlmodel
from tensorflow.keras.models import load_model
from imblearn.over_sampling import RandomOverSampler,ADASYN,SMOTE, \
    BorderlineSMOTE, SVMSMOTE, SMOTENC
from collections import Counter
import seaborn as sns

#路径设置
data_dir = "./运行1/"
data_dir1 = "./dataset/"
os.makedirs(data_dir, exist_ok=True)


# seed = 2100
random_state = 42#42，1234
cv_seed = 42#1230
num_classes = 2
epochs_cv = 500
img_rows, img_cols = 32,32
epochs = epochs_cv
lr_init = 0.001
batch_size_cv = 8#16
batch_size = batch_size_cv
channels = 1
mdd_num = 516
data_cv_splits = 5
n_splits = 5
total_cv = n_splits
in_cv = True

use_clr = False #True* False
use_augmentation = False#FalseTrue
use_ros = False#True
watch_cls = 1
callbacks_dl = []
shap_values_list = []
hist_acc = []
hist_rocp = []
all_fpr = []
all_tpr = []
sensitivity_list = []
specificity_list = []
y_allcv = np.array([],dtype=int)
rocp_allcv = np.array([],dtype=float)
ft_num = 518
embed_dim = 16
num_heads = 2
head_dim = 16
num_classes = 2
model_name = "CMS_net"#transformer,CMS_net

def get_session(devices="0",gpu_fraction=0.25):
    np.random.seed(random_state)
    tf.set_random_seed(random_state)

#        server = tf.train.Server.create_local_server()
    #"", "0,1", "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)

    sess = tf.Session(
#                            server.target,
                        config=tf.ConfigProto(
                                gpu_options=gpu_options,
                        )
                    )
#        sess = tf.Session()
    return sess



def lr_log(epoch):
    lr = model.optimizer.get_config()["learning_rate"] #lr learning_ate
    print('Learning rate: %0.8f'%lr)
    return lr


def lr_schedule(epoch):
    if epoch < int(epochs*lr_factors[0]):
        lr = lr_init
        return lr
    if epoch < int(epochs*lr_factors[1]):
        lr = lr_init*0.1
        return lr
    if epoch < int(epochs*lr_factors[2]):
        lr = lr_init*0.01
        return lr
    if epoch < int(epochs*lr_factors[3]):
        lr = lr_init*0.001
        return lr
    lr = lr_init*0.0001
    return lr

if not use_clr:

    lr_factors = [0.4, 0.6, 0.8, 0.9]
    # lr_scheduler = LearningRateScheduler(lr_schedule)
    # callbacks_dl.append(lr_scheduler)

else:
    step_mode='triangular'
    epochs_clr_cv = epochs_cv/2 *1/2 #/2* 1**? 4-
    epochs_clr = epochs/2 *1/2 #/2* 1**? 4-
    base_lr=1e-7 #0.0000001
    max_lr_cv=0.01 #0.001*c1? 02**c1? 0005c2
    max_lr=max_lr_cv #0.001*c1? 02**c1? 0005c2


def xentropy_regu_mean(y_true, y_pred):
    e=0.5
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(
            K.ones_like(y_pred)/num_classes, y_pred)
    return (1-e)*loss1 + e*loss2

def xcustom_loss_batch(y_true, y_pred):

    loss = xentropy_regu_mean(y_true, y_pred)

    return loss



def clf_sigmoid(x):
    raw = 1.0/(1+np.exp(-x*1.0))
    norm = preprocessing.normalize(raw, norm="l1")
    return norm

# tf_sess = get_session(devices="0",gpu_fraction=0.9)
# ktf.set_session(tf_sess)

class XTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def make_data_splits(train_data, train_label, data_cv_splits, cv_seed):
        kf_cv = StratifiedKFold(n_splits=data_cv_splits, shuffle=True, random_state=cv_seed)
        splits_cv = []
        for train, test in kf_cv.split(train_data, train_label):
            x_train = train_data[train, :]
            y_train = train_label[train]
            x_test = train_data[test, :]
            y_test = train_label[test]

            splits_cv.append([x_train, y_train, x_test, y_test])
        return splits_cv

def feature_dropout(data, drop_prob=0.2):
    mask = np.random.rand(*data.shape) > drop_prob
    return data * mask
if __name__ == '__main__':


    class GaussianNoiseGenerator(Sequence):
        def __init__(self, mean=0.0, std=0.01, shuffle=True):
            self.mean = mean
            self.std = std
            self.shuffle = shuffle

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return self

        def __call__(self, X, y=None):
            while True:
                # Generate a batch of random noise with the same shape as X
                noise = np.random.normal(self.mean, self.std, size=X.shape)
                # Add the noise to the input data
                noisy_X = X + noise
                # Reshape the input data to match the expected shape of the model input
                noisy_X = np.reshape(noisy_X, (X.shape[0], X.shape[1], 1))

                # If the generator is used for training, also generate labels
                if y is not None:
                    # Reshape the labels to match the expected shape of the model output
                    # noisy_y = np.reshape(y, (y.shape[0],num_classes))#不需要对y进行reshape
                    yield noisy_X, y #noisy_y
                else:
                    yield noisy_X

    # print(np.random.normal(0, 2, size=200))






    data_path = data_dir1 + "sva_result_516_train_age_gender.csv"
    dataset = pd.read_csv(data_path, header=None, dtype=str)

    exp1 = dataset.iloc[1:, 2:].astype(float)

    scaler = StandardScaler()
    exp1_normalized = scaler.fit_transform(exp1.T).T
    exp1_normalized_df = pd.DataFrame(exp1_normalized, columns=dataset.columns[2:])

    train_data = np.array(exp1_normalized_df.iloc[:, :].values, dtype=np.float32)
    train_label = np.array(dataset.iloc[1:, 1].values, dtype=np.int32)

    x_train_val, x_test, y_train_val, y_test = train_test_split(train_data, train_label, test_size=0.3, random_state=random_state)#stratify=train_label,

    early_stopper = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=50, mode='min', verbose=1)
    splits_cv = make_data_splits(x_train_val, y_train_val, data_cv_splits, cv_seed)

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)



    n_cv = 0
    for split_cv in splits_cv:

        print("\n=====================CV[%s]========================\n" % n_cv)
        in_cv = True
        n_cv+=1

        data_dir2 = os.path.join(data_dir, '{}cv'.format(n_cv));
        os.makedirs(data_dir2, exist_ok=True)
        log_dir_tb = data_dir2+'/log_tb_tra'

        (x_train, y_train, x_val, y_val) = split_cv



        oversampling_name = "SMOTENC"
        random_seed = 42
        drop_prob = 0.1
        x_train_augmented = feature_dropout(x_train, drop_prob=drop_prob)
        mask = np.random.rand(*x_train.shape) > drop_prob
        dropout_ratio = np.sum(mask) / np.size(mask)
        retained_ratio = 1 - dropout_ratio
        print("丢弃的特征比例：", retained_ratio)
        if use_ros:
            print("Original class distribution:")
            print(Counter(y_train))
            if oversampling_name == "RandomOverSampler":
                ros = RandomOverSampler(random_state=random_seed)
                x_train_resampled, y_train_resampled = ros.fit_resample(x_train, y_train)
            elif oversampling_name == "SMOTE":
                smote = SMOTE(random_state=random_seed)
                x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
            elif oversampling_name == "ADASYN":
                adasyn = ADASYN(random_state=random_seed)
                x_train_resampled, y_train_resampled = adasyn.fit_resample(x_train, y_train)
            elif oversampling_name == "SMOTENC":
                smotenc = SMOTENC(random_state=random_seed, categorical_features=[0, 1, 2])
                x_train_resampled, y_train_resampled = smotenc.fit_resample(x_train, y_train)
            elif oversampling_name == "BorderlineSMOTE":
                borderline_smote = BorderlineSMOTE(random_state=random_seed)
                x_train_resampled, y_train_resampled = borderline_smote.fit_resample(x_train, y_train)

            x_train_dl1 = np.reshape(x_train_resampled,(x_train_resampled.shape[0], x_train_resampled.shape[1], 1))
            x_val_dl1 = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
            x_test_dl1 = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            y_train_dl1 = to_categorical(y_train_resampled, num_classes)
            y_val_dl1 = to_categorical(y_val, num_classes)
            y_test_dl1 = to_categorical(y_test, num_classes)
            y_test_watch = y_test_dl1[:, watch_cls]

            print("Original training samples:", len(x_train))
            print("Oversampled training samples:", len(x_train_resampled))
            print(x_val.shape[0], 'validation samples')
            print(x_test.shape[0], 'test samples\n')
            print("Oversampled class distribution:")
            print(Counter(y_train_resampled))


            plt.subplot(1, 2, 1)
            sns.countplot(x=y_train)
            plt.title('Original Data Label Distribution')


            plt.subplot(1, 2, 2)
            sns.countplot(x=y_train_resampled, color='lightcoral', label='Resampled Data')
            sns.countplot(x=y_train, color='lightgreen', label='Original Data')
            plt.title('Resampled Data Label Distribution')

            plt.legend()
            # plt.show()

        else:
            # x_train_dl1 = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            # x_val_dl1 = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
            # x_test_dl1 = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            x_train_dl1 = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
            x_val_dl1 = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
            x_test_dl1 = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))




            y_train_dl1 = to_categorical(y_train, num_classes)
            y_val_dl1 = to_categorical(y_val, num_classes)
            y_test_dl1 = to_categorical(y_test, num_classes)
            y_test_watch = y_test_dl1[:, watch_cls]
            print(x_train.shape[0], 'train samples')
            print(x_val.shape[0], 'validation samples')
            print(x_test.shape[0], 'test samples\n')
            # print(y_train_dl1.shape)

        # print(x_train_dl1.shape, 'train samples')
        # print(y_train_dl1.shape, 'train samples\n')

        if model_name == "transformer":
            model = create_transformer_model(ft_num, embed_dim, num_heads,  num_classes)
        if model_name == "transformer_1":
            model = create_transformer_model(ft_num, embed_dim, num_heads, head_dim, num_classes, num_blocks=3)
        if model_name == "CMS_net":
            model = create_dlmodel()
        csv_logger = CSVLogger(data_dir2+'/model_classified.csv')

        # tensorboard log save path
        tb = XTensorBoard(log_dir=log_dir_tb,
                          histogram_freq=0, batch_size=batch_size, write_graph=True,
                          write_grads=False)
        # checkpoint save path
        ck = ModelCheckpoint(data_dir2 + '/model_{epoch:02d}-{val_loss:.2f}.hdf5',period=50)


        Reduce = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.5,
                                   patience=30,#5
                                   verbose=1,
                                   mode='auto',
                                   min_delta=0.00001,
                                   cooldown=100,#20
                                   min_lr=1e-8)
        callbacks_dl = [ck,tb,csv_logger,Reduce,early_stopper]#early_stopper

        # ck = ModelCheckpoint('./{}cv/split_{}'.format(n_cv) + '/best_weights.h5',
        #                      monitor='val_loss', save_best_only=True, verbose=1)
        if use_augmentation:
            data_generator = GaussianNoiseGenerator(mean=0, std=0.5)
            print("data_generator is not None")
            # data_generator = DataGenerator(x_train_dl2, y_train_dl2,batch_size=batch_size_cv)

            # Select a few random samples for visualization
            num_samples_to_visualize = 5
            sample_indices = np.random.randint(0, x_train_dl1.shape[0], num_samples_to_visualize)

            # Visualize the original and noisy data
            plt.figure(figsize=(12, 8))
            for i, idx in enumerate(sample_indices):
                noisy_data= next(data_generator(x_train_dl1[idx:idx + 1]))  # Generate noisy data for the selected sample
                plt.subplot(num_samples_to_visualize, 1, i + 1)
                plt.plot(x_train_dl1[idx], label='Original')
                plt.plot(noisy_data[0], label='Noisy', linestyle='dashed')
                plt.title(f"Sample {idx}")
                plt.legend()

            plt.tight_layout()
            # plt.show()
        else:
            data_generator = None
            print("data_generator is None")

        # Fit the model
        if use_augmentation and data_generator is not None:
            history = model.fit(data_generator(x_train_dl1, y_train_dl1),
                                # steps_per_epoch=x_train_dl1.shape[0],
                                steps_per_epoch=batch_size_cv,
                                epochs=epochs_cv,
                                verbose=1,
                                shuffle=True,
                                callbacks=callbacks_dl,
                                validation_data=(x_val_dl1, y_val_dl1))

        else:
            history = model.fit(x_train_dl1, y_train_dl1,
                            batch_size=batch_size_cv,
                            epochs=epochs_cv,
                            verbose=1,
                            shuffle=True,
                            callbacks=callbacks_dl,
                            validation_data=(x_val_dl1, y_val_dl1))

        # model save path
        model.save(data_dir2+'/model_classes.h5')
        if model_name == "CMS_net":
            model = load_model(data_dir2+'/model_classes.h5')
        # if model_name is "transformer":
        #     model = load_model(data_dir2 + '/model_classes.h5',
        #                    custom_objects={'MultiHeadSelfAttention': MultiHeadSelfAttention})

        # evaluation indicators
        label_classed_pred = model.predict(x_test_dl1)
        label_classed_pred_d = np.argmax(label_classed_pred, axis=1)
        label_classed_pred_m = label_classed_pred_d
        label_classed_pred_prob = label_classed_pred[:, watch_cls]
        label_classed_pred_prob = np.nan_to_num(label_classed_pred_prob)
        cm = confusion_matrix(y_test, label_classed_pred_m)
        print(classification_report(y_test, label_classed_pred_m))
        # print(confusion_matrix(y_test, label_classed_pred_m))
        print(cm)
        print("ACCURACY: {:.4f}".format(accuracy_score(y_test, label_classed_pred_m)))
        print("ROC_AUC(Pr.): {:.4f}".format(roc_auc_score(y_test_watch, label_classed_pred_prob)))
        TP = cm[1, 1]
        FP = cm[0, 1]
        TN = cm[0, 0]
        FN = cm[1, 0]
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)



        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='train_accuracy')#acc / accuracy
        plt.plot(history.history['val_accuracy'], label='val_accuracy')#val_acc / val_accuracy
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.subplots_adjust(wspace=0.5)
        plt.savefig(data_dir2+'/loss_accuracy.png')
        # plt.show()


        hist_acc.append(accuracy_score(y_test, label_classed_pred_m))
        hist_rocp.append(roc_auc_score(y_test_watch, label_classed_pred_prob))
        y_allcv = np.concatenate([y_allcv, y_test_watch])
        rocp_allcv = np.concatenate([rocp_allcv, label_classed_pred_prob])


        fpr, tpr, _ = roc_curve(y_test, label_classed_pred_prob)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)



    print("\n========================CV Total=========================\n")

    avg_sensitivity = np.mean(sensitivity_list)
    avg_specificity = np.mean(specificity_list)
    print("Average Sensitivity: {:.4f}".format(avg_sensitivity))
    print("Average Specificity: {:.4f}".format(avg_specificity))

    df_hist = pd.DataFrame({'Specificity': specificity_list,
                            'Sensitivity': sensitivity_list,
                            'acc': np.array(hist_acc, dtype=float),
                            'rocp': np.array(hist_rocp, dtype=float)})
    print(df_hist.describe())
    df_hist.to_csv(data_dir+'/results.csv',index=False)

    fpr_oc, tpr_oc, _ = roc_curve(y_allcv, rocp_allcv, pos_label=1)
    font_size = 14
    plt.figure(123, figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_oc, tpr_oc, 'r-',
            label='AVG(AUC:%0.4f|ACC:%0.4f)'
                 % (
                    df_hist.rocp.mean(),
                    df_hist.acc.mean()
                    ),linewidth=2)
    plt.xlabel('False positive rate(1-Specificity)',fontsize=font_size)
    plt.ylabel('True positive rate(Sensitivity)',fontsize=font_size)
    plt.title('ROC curve',fontsize=font_size)
    plt.legend(loc='lower right',fontsize=font_size)
    # plt.savefig(roc_auc_dir + "/roc-cvs.png")
    plt.savefig(data_dir+"/roc-cvs.png")
    # plt.show()

    plt.figure(1111,figsize=(8,8))
    for fold in range(total_cv):
        plt.plot(all_fpr[fold], all_tpr[fold],
                 label='Fold {} (ACC={:.4f}, AUC={:.4f})'.format(fold + 1, hist_acc[fold], hist_rocp[fold]),
                 linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate',fontsize=font_size)
    plt.ylabel('True Positive Rate',fontsize=font_size)
    plt.title('ROC Curve for 5-Fold Cross Validation',fontsize=font_size)
    plt.legend(loc='lower right',fontsize=font_size)
    plt.savefig(data_dir+"/5cv_roc-cvs.png")
    # plt.show()


