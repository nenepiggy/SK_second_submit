# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:12:15 2022

@author: 13106
"""

import numpy as np
import scipy.io as scio
import h5py
from scipy.stats import skew, kurtosis
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
import matplotlib.pyplot as plt

long_short_value = 1000 # 区分长寿命与短寿命的阈值
# 各个分类器所用的特征编号
features_idx = {
    "all": range(1,14,1),
    "full": [1, 2, 5, 6, 7, 9, 10, 11],
    "variance": [2],
    "discharge": [1, 2, 5 , 9, 13],
    "short_life":[1, 2, 5, 13],
    "long_life":[1, 2, 9, 13],
    "first_predict":[1,2]
    }

def mape(y_true,y_pred):
    output_errors = np.average(np.abs(y_pred - y_true)/y_true)
    return output_errors

def build_feature_df(batch_dict):
    """
    建立一个DataFrame，包含加载的批处理字典中所有最初使用的特性
    """

    # print("Start building features ...")

    # 124 cells (3 batches)
    n_cells = len(batch_dict.keys())

    ## Initializing feature vectors:
    # numpy vector with 124 zeros
    cycle_life = np.zeros(n_cells)
    # 1. delta_Q_100_10(V)
    minimum_dQ_100_10 = np.zeros(n_cells)
    variance_dQ_100_10 = np.zeros(n_cells)
    skewness_dQ_100_10 = np.zeros(n_cells)
    kurtosis_dQ_100_10 = np.zeros(n_cells)

    # dQ_100_10_2 = np.zeros(n_cells)
    # 2. Discharge capacity fade curve features
    slope_lin_fit_2_100 = np.zeros(
        n_cells)  # Slope of the linear fit to the capacity fade curve, cycles 2 to 100
    intercept_lin_fit_2_100 = np.zeros(
        n_cells)  # Intercept of the linear fit to capavity face curve, cycles 2 to 100
    discharge_capacity_2 = np.zeros(n_cells)  # Discharge capacity, cycle 2
    diff_discharge_capacity_max_2 = np.zeros(n_cells)  # Difference between max discharge capacity and cycle 2
    discharge_capacity_100 = np.zeros(n_cells)  # for Fig. 1.e
    slope_lin_fit_95_100 = np.zeros(n_cells)  # for Fig. 1.f
    # 3. Other features
    mean_charge_time_2_6 = np.zeros(n_cells)  # Average charge time, cycle 1 to 5
    minimum_IR_2_100 = np.zeros(n_cells)  # Minimum internal resistance

    diff_IR_100_2 = np.zeros(n_cells)  # Internal resistance, difference between cycle 100 and cycle 2

    # Classifier features
    minimum_dQ_5_4 = np.zeros(n_cells)
    variance_dQ_5_4 = np.zeros(n_cells)
    cycle_550_clf = np.zeros(n_cells)

    # iterate/loop over all cells.
    for i, cell in enumerate(batch_dict.values()):
        cycle_life[i] = cell['cycle_life']
        # 1. delta_Q_100_10(V)
        c10 = cell['cycles']['10']
        c100 = cell['cycles']['100']
        dQ_100_10 = c100['Qdlin'] - c10['Qdlin']

        minimum_dQ_100_10[i] = np.log10(np.abs(np.min(dQ_100_10)))
        variance_dQ_100_10[i] = np.log(np.abs(np.var(dQ_100_10)))
        skewness_dQ_100_10[i] = np.log(np.abs(skew(dQ_100_10)))
        kurtosis_dQ_100_10[i] = np.log(np.abs(kurtosis(dQ_100_10)))

        # Qdlin_100_10 = cell['cycles']['100']['Qdlin'] - cell['cycles']['10']['Qdlin']
        # dQ_100_10_2[i] = np.var(Qdlin_100_10)

        # 2. Discharge capacity fade curve features
        # Compute linear fit for cycles 2 to 100:
        q = cell['summary']['QD'][1:100].reshape(-1, 1)  # discharge cappacities; q.shape = (99, 1);
        X = cell['summary']['cycle'][1:100].reshape(-1, 1)  # Cylce index from 2 to 100; X.shape = (99, 1)
        linear_regressor_2_100 = LinearRegression()
        linear_regressor_2_100.fit(X, q)

        slope_lin_fit_2_100[i] = linear_regressor_2_100.coef_[0]
        intercept_lin_fit_2_100[i] = linear_regressor_2_100.intercept_
        discharge_capacity_2[i] = q[0][0]
        diff_discharge_capacity_max_2[i] = np.max(q) - q[0][0]

        discharge_capacity_100[i] = q[-1][0]

        q95_100 = cell['summary']['QD'][94:100].reshape(-1, 1)
        q95_100 = q95_100 * 1000  # discharge cappacities; q.shape = (99, 1);
        X95_100 = cell['summary']['cycle'][94:100].reshape(-1,
                                                           1)  # Cylce index from 2 to 100; X.shape = (99, 1)
        linear_regressor_95_100 = LinearRegression()
        linear_regressor_95_100.fit(X95_100, q95_100)
        slope_lin_fit_95_100[i] = linear_regressor_95_100.coef_[0]

        # 3. Other features
        mean_charge_time_2_6[i] = np.mean(cell['summary']['chargetime'][1:6])
        minimum_IR_2_100[i] = np.min(cell['summary']['IR'][1:100])
        diff_IR_100_2[i] = cell['summary']['IR'][100] - cell['summary']['IR'][1]

        # Classifier features
        c4 = cell['cycles']['4']
        c5 = cell['cycles']['5']
        dQ_5_4 = c5['Qdlin'] - c4['Qdlin']
        minimum_dQ_5_4[i] = np.log10(np.abs(np.min(dQ_5_4)))
        variance_dQ_5_4[i] = np.log10(np.var(dQ_5_4))
        cycle_550_clf[i] = cell['cycle_life'] >= 550

    # combining all featues in one big matrix where rows are the cells and colums are the features
    # note last two variables below are labels/targets for ML i.e cycle life and cycle_550_clf
    features_df = pd.DataFrame({
        "cell_key": np.array(list(batch_dict.keys())),
        "minimum_dQ_100_10": minimum_dQ_100_10,
        "variance_dQ_100_10": variance_dQ_100_10,
        "skewness_dQ_100_10": skewness_dQ_100_10,
        "kurtosis_dQ_100_10": kurtosis_dQ_100_10,
        "slope_lin_fit_2_100": slope_lin_fit_2_100,
        "intercept_lin_fit_2_100": intercept_lin_fit_2_100,
        "discharge_capacity_2": discharge_capacity_2,
        "diff_discharge_capacity_max_2": diff_discharge_capacity_max_2,
        "mean_charge_time_2_6": mean_charge_time_2_6,
        "minimum_IR_2_100": minimum_IR_2_100,
        "diff_IR_100_2": diff_IR_100_2,
        "minimum_dQ_5_4": minimum_dQ_5_4,
        "variance_dQ_5_4": variance_dQ_5_4,
        "cycle_life": cycle_life,
        "cycle_550_clf": cycle_550_clf
    })

    # print("Done building features")
    return features_df

def train_val_split(features_df, model="regression"):
    """
    划分train&test数据集。注意：数据集要按照指定方式划分
    :param features_df: 包含最初使用的特性dataframe
    :param regression_type: 回归模型的类型
    :param model: 使用模型的flag
    """
    # get the features for the model version (full, variance, discharge)
    feature_indices = features_idx['all']
    # get all cells with the specified features
    model_features = features_df.iloc[:, feature_indices]
    # get last two columns (cycle life and classification)
    labels = features_df.iloc[:, -2:]
    # labels are (cycle life ) for regression other wise (0/1) for classsification
    labels = labels.iloc[:, 0] if model == "regression" else labels.iloc[:, 1]

    # split data in to train/primary_test/and secondary test
    train_cells = np.arange(1, 84, 2)
    val_cells = np.arange(0, 84, 2)
    val_cells = np.delete(val_cells,np.where(val_cells==42)[0])
    test_cells = np.arange(84, 124, 1)

    # get cells and their features of each set and convert to numpy for further computations
    x_train = np.array(model_features.iloc[train_cells])
    x_val = np.array(model_features.iloc[val_cells])
    x_test = np.array(model_features.iloc[test_cells])

    # target values or labels for training
    y_train = np.array(labels.iloc[train_cells])
    y_val = np.array(labels.iloc[val_cells])
    y_test = np.array(labels.iloc[test_cells])

    # return 3 sets
    return {"train": (x_train, y_train), "val": (x_val, y_val), "test": (x_test, y_test)}

def split_long_short(x_data, y_data, pesudo_y):
    long_idx = np.where(pesudo_y>long_short_value)[0]
    # long_idx = np.where(y_data>long_short_value)[0]
    x_data_long = x_data[long_idx,:][:,np.array(features_idx['long_life'])-1]
    y_data_long = y_data[long_idx]
    short_idx = np.where(pesudo_y<=long_short_value)[0]
    # short_idx = np.where(y_data<=long_short_value)[0]
    x_data_short = x_data[short_idx,:][:,np.array(features_idx['short_life'])-1]
    y_data_short = y_data[short_idx]
    return x_data_long,y_data_long,x_data_short,y_data_short

def my_train(dataset,alpha_train,l1_ratio,log_target,normal_or_not,normal_inform):
    # 构造一个模型实例
    x_train,y_train = dataset['train']
    # x_valid, y_valid = dataset['valid']
    # x_train = np.concatenate((x_train, x_valid))
    # y_train = np.concatenate((y_train, y_valid))
    regr = ElasticNet(random_state=2, alpha=alpha_train, l1_ratio=l1_ratio)
    # 是否要对标签取log
    y_train = np.log(y_train) if log_target else y_train
    # 是否归一化
    if normal_or_not:
        y_train = (y_train - normal_inform[1]) / (normal_inform[0] - normal_inform[1])
    # 拟合模型
    regr.fit(x_train, y_train)
    return regr

def my_eval(dataset,model,log_target,normal_not,normal_inform):
    x_train,y_train = dataset.get("train")
    x_val,y_val = dataset.get('valid')
    x_test,y_test = dataset.get('test')
    # 测试模型输出
    pred_train = model.predict(x_train)
    pred_val = model.predict(x_val)
    pred_test = model.predict(x_test)
    # 反归一化
    if normal_not:
        pred_train = pred_train*(normal_inform[0] - normal_inform[1]) + normal_inform[1]
        pred_val = pred_val*(normal_inform[0] - normal_inform[1]) + normal_inform[1]
        pred_test = pred_test*(normal_inform[0] - normal_inform[1]) + normal_inform[1]
    # 看看是否要变回来
    if log_target:
        pred_train = np.exp(pred_train)
        pred_val = np.exp(pred_val)
        pred_test = np.exp(pred_test)
    return pred_train, pred_val, pred_test

def my_error(dataset,pred_train,pred_val,pred_test):
    x_train,y_train = dataset.get("train")
    x_val,y_val = dataset.get('valid')
    x_test,y_test = dataset.get('test')
    
    error_train = mape(y_train,pred_train)*100
    error_val = mape(y_val,pred_val)*100
    error_test = mape(y_test,pred_test)*100
    return error_train, error_val,  error_test

#1. 加载数据集并提取特征矩阵
dataset = np.load('../data/my_data.npy',allow_pickle=True).item()
features_df = build_feature_df(dataset) # 从batch字典数据中提取特征矩阵
#2. 设置是否需要归一化、取log，并根据此变换输出
normal_or_not = True
log_target = True
if normal_or_not:
    features_df.iloc[:, 1:-2] = features_df.iloc[:, 1:-2].apply(lambda x: (x-x.min())/(x.max()-x.min()) if x.max()!=x.min() else 1, axis=0) # 对输入做归一化
if log_target:
    normal_max_min = (np.log(features_df.iloc[:,-2]).max(),np.log(features_df.iloc[:,-2]).min())
else:
    normal_max_min = (features_df.iloc[:,-2].max(),features_df.iloc[:,-2].min())
#3. 拆分训练、验证、测试集，并根据CNN初步分类结果拆分长短寿命数据
battery_dataset = train_val_split(features_df) # 将特征矩阵分为训练和测试集

x_train,y_train = battery_dataset["train"]
x_train_long, y_train_long, x_train_short, y_train_short = split_long_short(x_train, y_train, y_train) 
train_long_num = len(x_train_long)
train_short_num = len(x_train_short)

x_valid,y_valid = battery_dataset["val"]
pesudo_valid = np.load('../data/CNN_pred_valid.npy')
x_valid_long, y_valid_long, x_valid_short, y_valid_short = split_long_short(x_valid, y_valid, pesudo_valid) 
valid_long_num = len(x_valid_long)
valid_short_num = len(x_valid_short)

x_test,y_test = battery_dataset["test"]
pesudo_test = np.load('../data/CNN_pred_test.npy')
x_test_long, y_test_long, x_test_short, y_test_short = split_long_short(x_test, y_test, pesudo_test)
test_long_num = len(x_test_long)
test_short_num = len(x_test_short)

max_CAP = np.concatenate((y_train, y_valid, y_test)).max()
min_CAP = np.concatenate((y_train, y_valid, y_test)).min()

long_dataset = {'train':(x_train_long,y_train_long),'valid':(x_valid[:,np.array(features_idx['long_life'])-1],y_valid),'test':(x_test[:,np.array(features_idx['long_life'])-1],y_test)}
short_dataset = {'train':(x_train_short,y_train_short),'valid':(x_valid[:,np.array(features_idx['short_life'])-1],y_valid),'test':(x_test[:,np.array(features_idx['short_life'])-1],y_test)}
pure_long_dataset = {'train':(x_train_long,y_train_long),'valid':(x_valid_long,y_valid_long),'test':(x_test_long,y_test_long)}
pure_short_dataset = {'train':(x_train_short,y_train_short),'valid':(x_valid_short,y_valid_short),'test':(x_test_short,y_test_short)}
##############################################################################
#4. 训练长寿命模型并测试
##############################################################################
error_best_long = 50
for alpha in np.logspace(-3,0,50):
    for lr in np.logspace(-2,0,50):
        my_model = my_train(pure_long_dataset,alpha_train=alpha,l1_ratio=lr,log_target=log_target,normal_or_not=normal_or_not,normal_inform = normal_max_min)
        pred_train,pred_val,pred_test = my_eval(pure_long_dataset,my_model,log_target,normal_or_not,normal_max_min)
        error_train, error_val, error_test = my_error(pure_long_dataset, pred_train, pred_val, pred_test)
        if error_val < error_best_long:
            error_best_long = error_val
            parameter_best_long = [alpha,lr]

alpha = parameter_best_long[0]
lr = parameter_best_long[1]
my_model_long = my_train(pure_long_dataset,alpha_train=alpha,l1_ratio=lr,log_target=log_target,normal_or_not=normal_or_not,normal_inform = normal_max_min)
pred_train_long,pred_val_long,pred_test_long = my_eval(pure_long_dataset,my_model_long,log_target,normal_or_not,normal_max_min)
error_train_long, error_val_long, error_test_long = my_error(pure_long_dataset, pred_train_long, pred_val_long, pred_test_long)

print('———————————————————————————————————————————————')
print('长寿命电池模型：')
print(f"Regression Error (Train): {error_train_long}%")
print(f"Regression Error (Val): {error_val_long}%")
print(f"Regression Error (Test): {error_test_long}%")
# print(my_model_long.coef_)

label_font = {"family" : "Times New Roman",'size':15}

#5. 训练短寿命模型并测试
error_best_short = 50
for alpha in np.logspace(-3,0,50):
    for lr in np.logspace(-2,0,50):
        my_model = my_train(pure_short_dataset,alpha_train=alpha,l1_ratio=lr,log_target=log_target,normal_or_not=normal_or_not,normal_inform = normal_max_min)
        pred_train,pred_val,pred_test = my_eval(pure_short_dataset,my_model,log_target,normal_or_not,normal_max_min)
        error_train, error_val, error_test = my_error(pure_short_dataset, pred_train, pred_val, pred_test)
        if error_val < error_best_short:
            error_best_short = error_val
            parameter_best_short = [alpha,lr]

alpha = parameter_best_short[0]
lr = parameter_best_short[1]
my_model_short = my_train(pure_short_dataset,alpha_train=alpha,l1_ratio=lr,log_target=log_target,normal_or_not=normal_or_not,normal_inform = normal_max_min)
pred_train_short,pred_val_short,pred_test_short = my_eval(pure_short_dataset,my_model_short,log_target,normal_or_not,normal_max_min)
error_train_short, error_val_short, error_test_short = my_error(pure_short_dataset, pred_train_short, pred_val_short, pred_test_short)

print('———————————————————————————————————————————————')
print('短寿命电池模型：')
print(f"Regression Error (Train): {error_train_short}%")
print(f"Regression Error (Val): {error_val_short}%")
print(f"Regression Error (Test): {error_test_short}%")
# print(my_model_short.coef_)

label_font = {"family" : "Times New Roman",'size':15}
plt.figure('辨识结果对比图')
plt.scatter(pure_long_dataset['train'][1],pred_train_long,label='Train',color='steelblue')
plt.scatter(pure_long_dataset['valid'][1],pred_val_long,label='Valid',color='coral')
plt.scatter(pure_long_dataset['test'][1],pred_test_long,label='Test',color='olivedrab')
plt.plot([min_CAP,max_CAP],[min_CAP,max_CAP],color='k')
plt.legend(prop=label_font)
plt.scatter(pure_short_dataset['train'][1],pred_train_short,color='steelblue',marker=',')
plt.scatter(pure_short_dataset['valid'][1],pred_val_short,color='coral',marker=',')
plt.scatter(pure_short_dataset['test'][1],pred_test_short,color='olivedrab',marker=',')
plt.xlabel('Real Life (CYC)', fontsize = 15)
plt.ylabel('Predict Life (CYC)', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.grid()

print('———————————————————————————————————————————————')
print('结合二者的总评价（直接按照CNN的初步分类结果应用模型）：')
# 若使用CNN_short的结果
error_train_short = 3.9725
error_val_short = 10.670
error_test_short = 7.4605
train_error = (error_train_long * train_long_num + error_train_short * train_short_num) / (train_long_num + train_short_num)
val_error = (error_val_long * valid_long_num + error_val_short * valid_short_num) / (valid_long_num + valid_short_num)
test_error = (error_test_long * test_long_num + error_test_short * test_short_num) / (test_long_num + test_short_num)
print(f"Regression Error (Train): {train_error}%")
print(f"Regression Error (Val): {val_error}%")
print(f"Regression Error (Test): {test_error}%")
###############################################################################
#6. 现在开始结合二者
###############################################################################
alpha = parameter_best_long[0]
lr = parameter_best_long[1]
my_model_long = my_train(pure_long_dataset,alpha_train=alpha,l1_ratio=lr,log_target=log_target,normal_or_not=normal_or_not,normal_inform = normal_max_min)

alpha = parameter_best_short[0]
lr = parameter_best_short[1]
my_model_short = my_train(pure_short_dataset,alpha_train=alpha,l1_ratio=lr,log_target=log_target,normal_or_not=normal_or_not,normal_inform = normal_max_min)

pred_train_long,pred_val_long,pred_test_long = my_eval(long_dataset,my_model_long,log_target,normal_or_not,normal_max_min)
pred_train_short,pred_val_short,pred_test_short = my_eval(short_dataset,my_model_short,log_target,normal_or_not,normal_max_min)

print('———————————————————————————————————————————————')
print('结合二者的总评价（按照CNN的初步分类结果并加权应用模型）：')
pred_val = np.zeros((len(pred_val_long)))
pred_test = np.zeros((len(pred_test_long)))
for idx in range(len(pred_val)):
    pesud_life = pesudo_valid[idx]
    if pesud_life > 1200:
        weight_long = 1
    elif pesud_life < 800:
        weight_long = 0
    else:
        weight_long = (pesud_life - 800) / 400
    weight_short = 1 - weight_long
    pred_val[idx] = pred_val_long[idx] * weight_long + pesudo_valid[idx] * weight_short

for idx in range(len(pred_test)):
    pesud_life = pesudo_test[idx]
    if pesud_life > 1200:
        weight_long = 1
    elif pesud_life < 800:
        weight_long = 0
    else:
        weight_long = (pesud_life - 800) / 400
    weight_short = 1 - weight_long
    pred_test[idx] = pred_test_long[idx] * weight_long + pesudo_test[idx] * weight_short

val_error = mape(y_valid,pred_val) * 100
test_error = mape(y_test,pred_test) * 100
print(f"Regression Error (Train): {train_error}%")
print(f"Regression Error (Val): {val_error}%")
print(f"Regression Error (Test): {test_error}%")

label_font = {"family" : "Times New Roman",'size':15}
plt.figure('辨识结果对比图（加权后）')
plt.scatter(y_valid,pred_val,label='Valid',color='coral')
plt.scatter(y_test,pred_test,label='Test',color='olivedrab')
plt.plot([min_CAP,max_CAP],[min_CAP,max_CAP],color='k')
plt.legend(prop=label_font)
plt.xlabel('Real Life (CYC)', fontsize = 15)
plt.ylabel('Predict Life (CYC)', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.grid()
