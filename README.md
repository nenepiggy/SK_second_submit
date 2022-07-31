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

### 4.1 模型训练

- 简单说明一下训练（train.py）的命令，建议附一些简短的训练日志。

- 可以简要介绍下可配置的超参数以及配置方法。

### 4.2 模型验证

- 在这里简单说明一下验证（eval.py）的命令，需要提供原始数据等内容，并在文档中体现输出结果。

### 4.3 项目主文件

- 在这里简单说明一下项目主文件（main.py）的命令，main.py中可执行全流程（train+eval）过程。

## 5. 代码结构与简要说明

### 5.1 代码结构

- 列出代码目录结构

```undefined
./repo_template               # 项目文件夹名称，可以修改为自己的文件夹名称
|-- config                    # 配置类文件夹
|   ├── competition.json      # 项目配置信息文件
|-- dataset                   # 数据集类文件夹
|   ├── dataset.py            # 数据集代码文件
|-- log                       # 日志类文件夹
|   ├── train.log             # 训练日志文件
|-- model                     # 模型类文件夹
|   ├── full_regression.pkl   # 训练好的模型文件
|-- preprocess                # 预处理类文件夹
|   ├── preprocess.py         # 数据预处理代码文件
|-- tools                     # 工具类文件夹
|   ├── train.py              # 训练代码文件
|   ├── eval.py               # 验证代码文件
|-- main.py                   # 项目主文件
|-- README.md                 # 中文用户手册
|-- LICENSE                   # LICENSE文件
```

### 5.2 代码简要说明

- 说明代码文件中的类以及主要函数功能

```undefined
# 示例
./dataset.py               # 数据集代码文件
|-- class Dataset          # 数据集类
|   ├── get_feature        # 类主函数，返回可用于训练的数据集
|   ├── train_val_split    # 划分train&val数据集
```



## 6. LICENSE

- 本项目的发布受[Apache 2.0 license](https://github.com/thinkenergy/vloong-nature-energy/blob/master/LICENSE)许可认证。



## 7. 参考链接与文献

- **[vloong-nature-energy/repo_template at master thinkenergy/vloong-nature-energy](https://github.com/thinkenergy/vloong-nature-energy/tree/master/repo_template)**

- **[Data-driven prediction of battery cycle life before capacity degradation](https://doi.org/10.1038/s41560-019-0356-8)**

  
