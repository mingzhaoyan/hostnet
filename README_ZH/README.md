# HostNet

**HostNet** 是一种病毒宿主预测工具，可以在狂犬病毒和黄病毒上进行预测，并且能够提供在其他更多的数据集上进行预测的服务。

[HOME](https://github.com/ChenXJer/HostNet)

## 安装

请注意，此实例仅在 Python 3.8.5 上进行了测试，我们建议使用 linux 和 miniconda 进行环境管理。

1. [下载并安装 miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)。
2. 克隆 `HostNet` 存储库：`git clone https://github.com/ChenXJer/HostNet.git` 并进入目录 `cd HostNet`
3. 创建python虚拟环境: `conda create -n hostnet python=3.8.5 -y`
4. 进入python虚拟环境: `conda activate hostnet`
5. 安装pytorch: `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`
6. 安装Python依赖：`pip3 install -r requirements.txt`
7. 下载原始数据集：`cd origin_data` 并参考 README.md 文件
8. 进入目录 `cd train` 并测试：`python3 train_model.py`

HostNet 配置文件命令
---

HostNet采用配置文件来集中配置模型的各个参数，所有的配置文件均保存在`/HostNet/train/config/`文件夹下（也可以根据需要进行调整）

训练时会自动读取文件夹下的所有配置文件依次进行训练。当前文件夹下已有两个示例文件，分别对给定的两个示例模型进行测试。

配置文件中的各个命令以及其具体的含义如下表所示。

|     命令     |                  作用                   |
| :----------: | :-------------------------------------: |
|  input_path  |        加载输入数据的路径[必要]         |
| output_path  | 保存最佳和模型指标描述文件的路径 [必要] |
|  file_name   |             模型名字 [必要]             |
| cut_sequence |       子序列切割方法 (1.RC 2.ASW)       |
|    epochs    |  最大模型训练的Epoch值 (default : 100)  |
|  subseq_len  |       子序列长度 (default : 250)        |
| emb_word_len |      碱基单元长度 (range : 1, 3-9)      |
|   emb_type   |    序列向量化方法 (1.One-hot 2.K2V)     |
|  need_train  | 训练或者测试模型（y : train, n : test)  |
|  batch_size  |            Mini-batch 的大小            |
|     clw      |    是否使用clw来平衡数据集（y or n)     |
| hidden_node  |      隐藏层节点数 (default : 150)       |

创建好你的配置文件后就可以训练或测试你的模型

HostNet 输出文件
---

在你的配置文件中指定的output_path路径下会生成模型训练和测试的部分关键参数值。

各个文件代表的含义如下：

- **file_name**_best_val_acc.pth : 在vaildation dataset下测试性能的最好的模型的隐藏层参数，用于加载模型进行测试。
- key_point.csv : 生成测试模型中准确率、AUC、F1等参数值
- standard(mean)_confusion_matrix : 生成Standard和Aggregated级别下的混淆矩阵
- key_point_per_class_table.csv : 生成每个类别对应的F1和ACC
- training_kpi.csv : 生成训练过程中的Train ACC、Train LOSS和Val ACC

Contribute
---

I would love for you to fork and send me pull request for this project.
Please contribute.

License
---

This software is licensed under the [Apache License](http://www.apache.org/licenses/)
