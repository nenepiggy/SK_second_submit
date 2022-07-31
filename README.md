# README模板

## 1. 简介
- 本项目用以参加《Data-driven prediction of battery cycle life before capacity degradation》论文复现比赛。
- 为实现容量早期预测，我们使用了CNN模型，达到了9.2%的测试集误差（MAPE）。此外，通过对特征相关性的分析，我们还给出了一种长寿命模型与短寿命模型分离的可行方案，在该方案下，测试集误差有望达到8.7%（MAPE），然而由于某些中间关节没有打通，最后也仅能达到9.2%左右。同样写在这里以供参考。关于本文所用模型的详细信息请参见“昇科比赛开发报告.docx”。


## 2. 数据集划分方法说明

- 本工程按照比赛要求划分了训练集、验证集、测试集：
  - train_cells = np.arange(1, 84, 2)
  - val_cells = np.arange(0, 84, 2)
  -  test_cells = np.arange(84, 124, 1)
- 但是按照原文中的方法剔除掉了验证集中的一个寿命过于短（150 CYC）的电池，以免对训练过程产生误导。
  - val_cells = np.delete(val_cells,np.where(val_cells==42)[0])
  - （参见原文Table1:
 One battery in the test set reaches 80% state-of-health rapidly and does not match other observed patterns. Therefore, the parenthetical primary test results correspond to the exclusion of this battery.）


## 3. 准备数据与环境

### 3.1 准备环境

- 本项目用到的Python第三方库有Numpy, Scipy, Sklearn, Pytorch, Pandas, h5py。

### 3.2 准备数据

- 本项目所用的数据文件对原始数据文件做了精简，为方便验证，可直接下载my_data.npy放在data文件夹中，下载地址：https://share.weiyun.com/Y4onAhzX
- 也可选择通过preprocess文件夹中预处理文件，在原始数据文件（原文作者提供的batch....mat数据集）的基础上获取my_data.npy。
  - LoadMatData.m 用以将batch数据加载到Matlab工作区
  - PickMatData.m 用以将原始数据中不重要的、未用到的数据剔除，并生成一个mat格式的精简数据simplified2.mat
  - load_mat_to_python.py 用以读取simlified2.mat并存储成npy格式


## 4. 开始使用

### 4.1 CNN法

- 将my_data.npy放入data文件夹后，可直接执行withCNN.py，该文件包括训练、验证过程。精度如下：
![image](https://user-images.githubusercontent.com/37606459/182013559-e078b7f0-b6bc-4a4a-b167-b43b166cd405.png)
- 预测结果与实际结果对比如下：
![image](https://user-images.githubusercontent.com/37606459/182013569-e6055600-1a71-41b3-9533-410983650d24.png)

### 4.2 长短寿命双模型法

- 该方法的具体信息请见文档“昇科比赛开发报告.docx”。
- 依次执行withCNN_short.py和double_model_expect.py，是有望得到的长短寿命双模型法的最优精度：
![image](https://user-images.githubusercontent.com/37606459/182013752-4aeae5b7-e275-4d03-86d5-25e6baa9648c.png)
- 依次执行withCNN_short.py和double_model_real.py，是实际得到的长短寿命双模型法的精度，其测试集误差为9.28%。

## 5. 代码结构与简要说明

- 代码目录结构如下

```undefined
./ShengKe_submit              # 项目文件夹名称
|-- data                      # 数据集类文件夹
|   ├── my_data.npy           # 数据集
|   ├── CNN_pred_train.npy    # CNN预测的训练集结果
|   ├── CNN_pred_valid.npy    # CNN预测的验证集结果，用以双模型法的初分类
|   ├── CNN_pred_test.npy     # CNN预测的测试集结果，用以双模型法的初分类
|   ├── CNN_short_pred_valid.npy    # 短寿命CNN预测的验证集结果
|   ├── CNN_short_pred_test.npy     # 短寿命CNN预测的测试集结果
|-- preprocess                # 预处理类文件夹
|   ├── LoadMatData.m         
|   ├── PickMatData.m
|   ├── load_mat_to_python.py
|-- CNN_model                 # CNN法文件夹
|   ├── Elanet.py             # 弹性网络复现
|   ├── withCNN.py            # CNN的训练、测试
|   ├── withCNN_2d.py         # 二维CNN的训练、测试
|-- double_model              # 长短寿命双模型法
|   ├── short_life_model_CNN.py    # 短寿命模型，CNN法
|   ├── double_model_expect.py     # 有望得到的长短寿命双模型法的最优精度
|   ├── double_model_real.py       # 实际得到的长短寿命双模型法的最优精度
|-- README.md                 # 中文用户手册
|-- 昇科比赛开发报告.docx      # 详细开发报告
```

## 6. LICENSE

- 本项目的发布受[Apache 2.0 license](https://github.com/thinkenergy/vloong-nature-energy/blob/master/LICENSE)许可认证。



## 7. 参考链接与文献

- **[vloong-nature-energy/repo_template at master thinkenergy/vloong-nature-energy](https://github.com/thinkenergy/vloong-nature-energy/tree/master/repo_template)**

- **[Data-driven prediction of battery cycle life before capacity degradation](https://doi.org/10.1038/s41560-019-0356-8)**

- **[A convolutional neural network model for battery capacity fade curve prediction using early life data](https://doi.org/10.1038/s41560-019-0356-8)**
  
