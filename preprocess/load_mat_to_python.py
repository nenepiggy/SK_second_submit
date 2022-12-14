# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:37:43 2022

@author: 13106
"""

import numpy as np
import scipy.io as scio
import h5py
from scipy.stats import skew, kurtosis
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression

def mape(y_true,y_pred):
    output_errors = np.average(np.abs(y_pred - y_true)/y_true)
    return output_errors

def data_process():
    f = h5py.File('../data/simplified2.mat','r')
    batch = f['batch_simpy']
    label_mat = scio.loadmat('../data/label.mat')
    num_cells = batch['summary'].shape[0]
    bat_dict = {}
    for i in range(num_cells):
        cl = label_mat['bat_label'][i,0]
        profile = f[batch['profile'][i, 0]][()].tobytes()[::2].decode()
        summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
        summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
        summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
        summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
        summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
        summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
        summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
        summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
        summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
            summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
                   'cycle': summary_CY}
        cycles = f[batch['cycles'][i, 0]]
        cycle_dict = {}
        for j in range(cycles['Qdlin'].shape[0]):
            Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]][()]))
            dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]][()]))
            cd = {'Qdlin': Qdlin, 'dQdV': dQdV}
            cycle_dict[str(j)] = cd
        cell_dict = {'cycle_life': cl, 'profile': profile, 'summary': summary, 'cycles': cycle_dict}
        key = 'c' + str(i)
        bat_dict[key] = cell_dict
    return bat_dict

def build_feature_df(batch_dict):
    """
    ????????????DataFrame???????????????????????????????????????????????????????????????
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

def train_val_split(features_df, regression_type="full", model="regression"):
    """
    ??????train&test?????????????????????????????????????????????????????????
    :param features_df: ???????????????????????????dataframe
    :param regression_type: ?????????????????????
    :param model: ???????????????flag
    """
    # only three versions are allowed.
    assert regression_type in ["full", "variance", "discharge"]

    # dictionary to hold the features indices for each model version.
    features = {
        "full": [1, 2, 5, 6, 7, 9, 10, 11],
        "variance": [2],
        "discharge": [1, 2, 3, 4, 7, 8]
    }
    
    # features = {
    #     "full": [0, 1, 4, 5, 6, 8, 9, 10],
    #     "variance": [1],
    #     "discharge": [0, 1, 2, 3, 6, 7]
    # }
    # get the features for the model version (full, variance, discharge)
    feature_indices = features[regression_type]
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

def my_train(dataset,alpha_train,l1_ratio,log_target,normal_not,normal_inform):
    x_train,y_train = dataset.get("train")
    # ????????????????????????
    regr = ElasticNet(random_state=4, alpha=alpha_train, l1_ratio=l1_ratio)
    # ?????????????????????log
    y_train = np.log(y_train) if log_target else y_train
    # ?????????
    if normal_not:
        y_train = (y_train - normal_inform[1]) / (normal_inform[0] - normal_inform[1])
    # ????????????
    regr.fit(x_train, y_train)
    return regr

def my_eval(dataset,model,log_target,normal_not,normal_inform):
    x_train,y_train = dataset.get("train")
    x_val,y_val = dataset.get('val')
    x_test,y_test = dataset.get('test')
    # ????????????????????????
    pred_train = model.predict(x_train)
    pred_val = model.predict(x_val)
    pred_test = model.predict(x_test)
    # ????????????
    if normal_not:
        pred_train = pred_train*(normal_inform[0] - normal_inform[1]) + normal_inform[1]
        pred_val = pred_val*(normal_inform[0] - normal_inform[1]) + normal_inform[1]
        pred_test = pred_test*(normal_inform[0] - normal_inform[1]) + normal_inform[1]
    # ????????????????????????
    if log_target:
        pred_train = np.exp(pred_train)
        pred_val = np.exp(pred_val)
        pred_test = np.exp(pred_test)
    # ?????????
    error_train = mape(y_train, pred_train) * 100
    error_val = mape(y_val, pred_val) * 100
    error_test = mape(y_test, pred_test) * 100
    
    return error_train,error_val,error_test

"""
????????????????????????????????????????????????
"""
dataset = data_process() # ????????????batch????????????
np.save('../data/my_data',dataset)
