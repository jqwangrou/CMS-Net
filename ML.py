
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score, confusion_matrix, roc_curve, auc, accuracy_score, \
    roc_auc_score, classification_report
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn import feature_selection
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.utils import to_categorical
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler,ADASYN,SMOTE, \
    BorderlineSMOTE, SVMSMOTE, SMOTENC
data_cv_splits = 5
n_splits = 5
total_cv = n_splits
random_state = 42#1234
cv_seed = 1234
num_classes = 2
watch_cls = 1
model_name = "XGBoost"#,RF,SVM,,KNN,XGBoost
hist_acc = []
hist_rocp = []
all_fpr = []
all_tpr = []
sensitivity_list = []
specificity_list = []
y_allcv = np.array([],dtype=int)
rocp_allcv = np.array([],dtype=float)
data_dir = './XGBoost/3'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

dataset = pd.read_csv(r'.\sva_result_516_train.csv')

exp = dataset.iloc[:, 2:].astype(float)

scaler = StandardScaler()
exp_normalized = scaler.fit_transform(exp.T).T
exp_normalized_df = pd.DataFrame(exp_normalized, columns=dataset.columns[2:])

train_data = exp_normalized_df.iloc[:, :].values
train_label = dataset.iloc[:, 1].values


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
# Data enhancement: Feature discard
def feature_dropout(data, drop_prob=0.2):
    mask = np.random.rand(*data.shape) > drop_prob
    return data * mask

x_train_val, x_test, y_train_val, y_test = train_test_split(train_data, train_label, test_size=0.3, stratify=train_label, random_state=random_state)#
splits_cv = make_data_splits(x_train_val, y_train_val, data_cv_splits, cv_seed)
n_cv = 0
for split_cv in splits_cv:
    print("\n=====================CV[%s]========================\n" % n_cv)
    in_cv = True
    n_cv+=1
    (x_train, y_train, x_val, y_val) = split_cv
    random_seed = 42
    drop_prob = 0.1
    x_train_augmented = feature_dropout(x_train, drop_prob=drop_prob)
    mask = np.random.rand(*x_train.shape) > drop_prob
    dropout_ratio = np.sum(mask) / np.size(mask)
    retained_ratio = 1 - dropout_ratio
    print("Discard the feature ratio：", retained_ratio)

    # ros = RandomOverSampler(random_state=random_seed)
    # x_train_resampled, y_train_resampled = ros.fit_resample(x_train, y_train)

    # smote = SMOTE(random_state=random_seed)
    # x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    # adasyn = ADASYN(random_state=random_seed)
    # x_train_resampled, y_train_resampled = adasyn.fit_resample(x_train, y_train)

    smotenc = SMOTENC(random_state=random_seed, categorical_features=[0, 1, 2])
    x_train_resampled, y_train_resampled = smotenc.fit_resample(x_train, y_train)

    # borderline_smote = BorderlineSMOTE(random_state=random_seed)
    # x_train_resampled, y_train_resampled = borderline_smote.fit_resample(x_train, y_train)

    y_test_dl1 = to_categorical(y_test, num_classes)
    y_test_watch = y_test_dl1[:, watch_cls]
    print("Oversampled training samples:", len(x_train_resampled))
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples\n')
    if model_name == "SVM":
        param_grid = {'kernel': ['linear', 'rbf', 'sigmoid', 'poly'], 'C': np.linspace(0.01, 100, 20),
                      'gamma': np.linspace(0.0001, 0.01, 20)}
        model = SVC()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(x_train_resampled, y_train_resampled)#x_train, y_train
        best_parameters = grid_search.best_estimator_.get_params()

        # model with the best parameters
        model = SVC(kernel=best_parameters['kernel'], C=best_parameters['C'],
                    gamma=best_parameters['gamma'],probability=True)

        model.fit(x_train_resampled, y_train_resampled)#x_train, y_train
        print(best_parameters)
        predictions = model.predict(x_val)
        print("val_ACC：{:.4f}".format(model.score(x_val, y_val)))

        label_classed_pred = model.predict(x_test)
        label_classed_pred_prob = model.predict_proba(x_test)[:, watch_cls]
        cm = confusion_matrix(y_test, label_classed_pred)
        # print(classification_report(y_test, label_classed_pred_m))
        print(cm)
        print("ACCURACY: {:.4f}".format(accuracy_score(y_test, label_classed_pred)))
        print("ROC_AUC(Pr.): {:.4f}".format(roc_auc_score(y_test_watch, label_classed_pred_prob)))
        TP = cm[1, 1]
        FP = cm[0, 1]
        TN = cm[0, 0]
        FN = cm[1, 0]
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
    if model_name == "LDA":
        param_grid = {'n_components': [1, 2], 'solver': ['svd', 'laqr', 'eigen'],
                      'tol': [0.00001, 0.0001, 0.001, 0.01, 1, 10, 100]}
        pca = PCA(n_components=2)
        model = LinearDiscriminantAnalysis()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(x_train_resampled, y_train_resampled)
        best_parameters = grid_search.best_estimator_.get_params()

        # model with the best parameters
        model = LinearDiscriminantAnalysis(n_components=best_parameters['n_components'],
                                           solver=best_parameters['solver'],
                                           tol=best_parameters['tol'])

        model.fit(x_train_resampled, y_train_resampled)
        print(best_parameters)
        predictions = model.predict(x_val)
        print("val_ACC：{:.4f}".format(model.score(x_val, y_val)))
        label_classed_pred = model.predict(x_test)
        label_classed_pred_prob = model.predict_proba(x_test)[:, watch_cls]
        cm = confusion_matrix(y_test, label_classed_pred)
        # print(classification_report(y_test, label_classed_pred_m))
        print(cm)
        print("ACCURACY: {:.4f}".format(accuracy_score(y_test, label_classed_pred)))
        print("ROC_AUC(Pr.): {:.4f}".format(roc_auc_score(y_test_watch, label_classed_pred_prob)))
        TP = cm[1, 1]
        FP = cm[0, 1]
        TN = cm[0, 0]
        FN = cm[1, 0]
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
    if model_name == "KNN":
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],  
            'weights': ['uniform', 'distance'],
            'leaf_size': [10, 30, 50, 70] 
        }
        model = KNeighborsClassifier()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(x_train_resampled, y_train_resampled)
        best_parameters = grid_search.best_estimator_.get_params()

        # model with the best parameters
        model = KNeighborsClassifier(n_neighbors=best_parameters['n_neighbors'],
                                     weights=best_parameters['weights'],
                                     leaf_size=best_parameters['leaf_size'])
        model.fit(x_train_resampled, y_train_resampled)
        print(best_parameters)
        predictions = model.predict(x_val)
        print("val_ACC：{:.4f}".format(model.score(x_val, y_val)))

        label_classed_pred = model.predict(x_test)
        label_classed_pred_prob = model.predict_proba(x_test)[:, watch_cls]
        cm = confusion_matrix(y_test, label_classed_pred)
        # print(classification_report(y_test, label_classed_pred_m))
        print(cm)
        print("ACCURACY: {:.4f}".format(accuracy_score(y_test, label_classed_pred)))
        print("ROC_AUC(Pr.): {:.4f}".format(roc_auc_score(y_test_watch, label_classed_pred_prob)))
        TP = cm[1, 1]
        FP = cm[0, 1]
        TN = cm[0, 0]
        FN = cm[1, 0]
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
    if model_name == "RF":
        param_grid = {
            'max_depth': [1,5,10,20, 50, 80, 100, None],
            'n_estimators':[10,20,50,100,120,150,200,300],
            'max_features': [0.1,0.3,0.6,0.8,0.9],
        }
        model = RandomForestClassifier()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        best_parameters = grid_search.best_estimator_.get_params()

        # model with the best parameters
        model = RandomForestClassifier(n_estimators=best_parameters['n_estimators'],
                                      # max_features=best_parameters['max_features'],
                                       max_depth=best_parameters['max_depth']
                                       )
        model.fit(x_train, y_train)
        print(best_parameters)
        predictions = model.predict(x_val)
        print("val_ACC：{:.4f}".format(model.score(x_val, y_val)))
        
        label_classed_pred = model.predict(x_test)
        label_classed_pred_prob = model.predict_proba(x_test)[:, watch_cls]
        cm = confusion_matrix(y_test, label_classed_pred)
        # print(classification_report(y_test, label_classed_pred_m))
        print(cm)
        print("ACCURACY: {:.4f}".format(accuracy_score(y_test, label_classed_pred)))
        print("ROC_AUC(Pr.): {:.4f}".format(roc_auc_score(y_test_watch, label_classed_pred_prob)))
        TP = cm[1, 1]
        FP = cm[0, 1]
        TN = cm[0, 0]
        FN = cm[1, 0]
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
    if model_name == "XGBoost":
        param_grid = {'n_estimators': [10, 20, 50, 100, 200, 300],
                      'max_depth': [1, 3, 5, 7,9],
                      'learning_rate': [0.01, 0.1, 0.2],
                      # 'subsample': [0.8, 0.9, 1.0],
                      # 'colsample_bytree': [0.8, 0.9, 1.0],
                      'gamma': [0.01,0.01, 1, 5],
                      # 'min_child_weight': [1, 3, 5]
                      }

        model = xgb.XGBClassifier(eval_metric='error',use_label_encoder=False)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        best_parameters = grid_search.best_estimator_.get_params()

        # Model with the best parameters
        model = xgb.XGBClassifier(n_estimators=best_parameters['n_estimators'],
                              max_depth=best_parameters['max_depth'],
                              learning_rate=best_parameters['learning_rate'],
                              subsample=best_parameters['subsample'],
                              colsample_bytree=best_parameters['colsample_bytree'],
                              gamma=best_parameters['gamma'],
                              min_child_weight=best_parameters['min_child_weight'])

        model.fit(x_train, y_train)
        print(best_parameters)
        predictions = model.predict(x_val)
        print("val_ACC: {:.4f}".format(accuracy_score(y_val, predictions)))

        label_classed_pred = model.predict(x_test)
        label_classed_pred_prob = model.predict_proba(x_test)[:, 1]  # Assuming binary classification, adjust as needed
        cm = confusion_matrix(y_test, label_classed_pred)
        print(cm)
        print("ACCURACY: {:.4f}".format(accuracy_score(y_test, label_classed_pred)))
        print("ROC_AUC(Pr.): {:.4f}".format(roc_auc_score(y_test_watch, label_classed_pred_prob)))

        TP = cm[1, 1]
        FP = cm[0, 1]
        TN = cm[0, 0]
        FN = cm[1, 0]
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
    hist_acc.append(accuracy_score(y_test, label_classed_pred))
    hist_rocp.append(roc_auc_score(y_test_watch, label_classed_pred_prob))
    y_allcv = np.concatenate([y_allcv, y_test_watch])  
    rocp_allcv = np.concatenate([rocp_allcv, label_classed_pred_prob]) 
    # print(hist_rocp,hist_acc)
    # print(y_allcv.shape,rocp_allcv.shape)
    fpr, tpr, _ = roc_curve(y_test, label_classed_pred_prob)
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    # print(all_fpr)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    # print(sensitivity_list,specificity_list)
print("\n========================CV Total=========================\n")

avg_sensitivity = np.mean(sensitivity_list)
avg_specificity = np.mean(specificity_list)
print("Average Sensitivity: {:.4f}".format(avg_sensitivity))
print("Average Specificity: {:.4f}".format(avg_specificity))
# print(len(hist_rocp), len(hist_acc), len(sensitivity_list), len(specificity_list))

df_hist = pd.DataFrame({'Specificity': specificity_list,
                        'Sensitivity': sensitivity_list,
                        'acc': hist_acc,
                        'rocp': hist_rocp})
print(df_hist.describe()) 
df_hist.to_csv(data_dir + "/results.csv",index=False)

fpr_oc, tpr_oc, _ = roc_curve(y_allcv, rocp_allcv, pos_label=1)
font_size = 14
plt.figure(123, figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_oc, tpr_oc, 'r-',
             label='AVG(AUC:%0.4f|ACC:%0.4f)'
                   % (
                       df_hist.rocp.mean(),
                       df_hist.acc.mean()
                   ), linewidth=2)
plt.xlabel('False positive rate(1-Specificity)',fontsize=font_size)
plt.ylabel('True positive rate(Sensitivity)',fontsize=font_size)
plt.title('ROC curve',fontsize=font_size)
plt.legend(loc='lower right',fontsize=font_size)
plt.savefig(data_dir + "/roc-cvs.png")
plt.show()
    
plt.figure(1111, figsize=(8, 8))
for fold in range(total_cv):
    plt.plot(all_fpr[fold], all_tpr[fold],
                label='Fold {} (ACC={:.4f}, AUC={:.4f})'.format(fold + 1, hist_acc[fold], hist_rocp[fold]),
                linewidth=2)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate',fontsize=font_size)
plt.ylabel('True Positive Rate',fontsize=font_size)
plt.title('ROC Curve for 5-Fold Cross Validation',fontsize=font_size)
plt.legend(loc='lower right',fontsize=font_size)
plt.savefig(data_dir + "/5cv_roc-cvs.png")
plt.show()
