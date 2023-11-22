# HostNet

**HostNet** is a virus-host prediction tool that can predict rabies lyssavirus and flaviviruses and provide prediction services on other data sets.

[English](https://github.com/ChenXJer/HostNet) [中文](https://github.com/ChenXJer/HostNet/tree/main/README_ZH)

## Install

Note that this implementation has only been tested on Python 3.8.5; we recommend using Linux and Miniconda for enviroment management.

1. [Download and install miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).

2. Clone the `HostNet` repository: `git clone https://github.com/ChenXJer/HostNet.git` and change directory `cd HostNet`

3. Create the python environment: `conda create -n hostnet python=3.8.5 -y`
4. Enter the python environment: `conda activate hostnet`
5. Install pytorch: `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`
6. Install Python dependencies: `pip install -r requirements.txt`

7. Download the original dataset: `cd origin_data` and refer to the README.md file
8. Change directory `cd train`and test the installation: `python3 train_model.py`

HostNet configuration file commands
---

HostNet uses configuration files to centrally configure the parameters of the model.

 All configuration files are stored in the `/HostNet/train/config/` folder (you can adjust it as needed), and the training will automatically read the folder. 

All profiles are trained sequentially. Two example files in the current folder are used to test the two given example models, respectively.



Each command in the configuration file and its specific meaning are shown in the following table.

|   command    |                         what it does                         |
| :----------: | :----------------------------------------------------------: |
|  input_path  |        The load path of the input dataset [required]         |
| output_path  | The path to save the best model and model key indicators [required] |
|  file_name   |                   Model naming [required]                    |
| cut_sequence |         Subsequence segmentation method (1.RC 2.ASW)         |
|    epochs    | Maximum number of epochs used for training the model (default : 100) |
|  subseq_len  |              Subsequence length (default : 250)              |
| emb_word_len |        Gene sequence fragment length (range : 1, 3-9)        |
|   emb_type   |      Gene sequence vectorization type (1.One-hot 2.K2V)      |
|  need_train  |        Train or test the model（y : train, n : test)         |
|  batch_size  |         Mini-batch size of dataset loading per epoch         |
|     clw      |           Use clw to balance the dataset（y or n)            |
| hidden_node  |     Counts of HostNet hidden layer nodes (default : 150)     |



After creating your config file, you can train or test your model.

HostNet output files
---

Some key parameter values for model training and testing will be generated under the output_path path specified in your configuration file. 

The meanings of each file are as follows:

- **file_name**_best_val_acc.pth : The hidden layer parameters of the best model for testing performance under the vaildation dataset, used to load the model for testing.
- key_point.csv : Generate parameter values such as accuracy, AUC, and F1 in the test model
- standard(mean)_confusion_matrix : Generate a confusion matrix at the Standard and Aggregated levels
- key_point_per_class_table.csv : Generate F1 and ACC corresponding to each category
- training_kpi.csv : Generate Train ACC, Train LOSS and Val ACC during training

Datasets
---

We have published the pre-training dataset Vir61 and the prediction dataset Flavivirus at FigShare with DOI: 10.6084/m9.figshare.24604965. 
You can access the datasets at: https://figshare.com/s/131bb316a196c4674207

Contribute
---

I would love for you to fork and send me a pull request for this project.
Please contribute.

License
---

This software is licensed under the [Apache License](http://www.apache.org/licenses/)
