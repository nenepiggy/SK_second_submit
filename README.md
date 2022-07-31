# Readme
## 项目简介
    本项目用以参加《Data-driven prediction of battery cycle life before capacity degradation》论文复现比赛。
    为实现容量早期预测，我们使用了CNN模型，达到了9.2%的测试集误差（MAPE）。此外，通过对特征相关性的分析，我们还给出了一种长寿命模型与短寿命模型分离的可行方案，在该方案下，测试集误差有望达到8.7%（MAPE），然而由于某些中间关节没有打通，最后也仅能达到9.2%左右。同样写在这里以供参考。关于本文所用模型的详细信息请参见“昇科比赛开发报告.docx”
## 数据集划分方法说明
本工程按照比赛要求划分了训练集、验证集、测试集：
train_cells = np.arange(1, 84, 2)
val_cells = np.arange(0, 84, 2)
test_cells = np.arange(84, 124, 1)
但是按照原文中的方法剔除掉了验证集中的一个寿命过于短（150 CYC）的电池，以免对训练过程产生误导。
val_cells = np.delete(val_cells,np.where(val_cells==42)[0])
（参见原文Table1:
 One battery in the test set reaches 80% state-of-health rapidly and does not match other observed patterns. Therefore, the parenthetical primary test results correspond to the exclusion of this battery.）
## 准备数据与环境
### 数据集
