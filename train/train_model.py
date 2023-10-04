from tqdm import tqdm
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sklearn.utils.class_weight as class_weight
from dataProcessing.modules import *
from dataProcessing.dataset import *
from d2l import torch as d2l
import configparser
import torch
import torch.nn as nn
from net import *
import utils

def read_config_and_run(cfg):

    config_path = "config/" + cfg + ".ini"
    print(config_path)
    config = configparser.ConfigParser()
    config.read(config_path)

    # read configuration file
    input_path = config['file']['input_path']
    output_path = config['file']['output_path']
    file_name = config['file']['file_name']
    module = int(config['model']['cut_sequence']) # 1.loop 2.slide
    epochs = int(config['model']['epochs'])
    subseqlength = int(config['model']['subseq_len'])
    emb_word_len = config['model']['emb_word_len']
    emb_type = int(config['model']['emb_type']) # 1.one_hot 2.w2c 3.bert
    need_train=config['model']['need_train'] == 'y' # Whether training is required, you can only load the test
    batch_size = int(config['model']['batch_size'])
    hidden_node = int(config['model']['hidden_node'])
    clw = config['model']['clw'] == 'y'  # Whether to use class_weight loss for oversampling

    if need_train:
        files = os.listdir(input_path)
        assert "Y_train.csv" in files, f"{input_path} must contain Y_train.csv file, but no such file in {files}"

    if not os.path.exists(output_path): os.mkdir(output_path)

    # Define model parameters
    dropout = 0.2
    lr = 1e-3
    devices = d2l.try_all_gpus()

    emb_word_len = emb_word_len if emb_type != 1 else 1

    # Data loading and preprocessing
    Origin_data = Data(input_path, module, subseqlength, emb_word_len, need_train)
    data_shape = Origin_data.X_test.shape
    label_shape = Origin_data.Y_test.shape

    emb_layer = Embeding_layer(emb_type)

    global collector, weights
    collector = utils.Collector(Origin_data.y_encoder, Origin_data.test_samples, output_path)
    weights = None
    if clw and need_train: 
        labels = np.argmax(Origin_data.Y_train, axis=1)
        weights=class_weight.compute_class_weight(
                                 class_weight='balanced',
                                 classes=np.unique(labels),
                                 y=labels)
        weights = torch.tensor(weights).to(devices[0])


    # define model
    model = Net(seq_length=data_shape[1], in_channels=emb_layer.dims, output_dim=label_shape[-1], 
             num_hiddens=hidden_node, dropout=dropout).build_normal_model()
    print(model)
    # train model
    run(model, Origin_data, batch_size, lr, devices, file_name, emb_layer, need_train, epochs)



def run(model, data:Data, batch_size, lr, devices, file_name, emb_layer, need_train=True, epochs=100):
    
    # get dataloader
    if need_train:
        train_iter = data.get_train_DataLoader(batch_size, emb_layer)
        val_iter = data.get_val_DataLoader(batch_size, emb_layer)
        training(model, train_iter, val_iter, epochs, lr, devices, emb_layer, file_name)

    test_iter = data.get_test_DataLoader(batch_size, emb_layer)
    prediction(model, file_name, test_iter, emb_layer)



def training(net, train_iter, test_iter, num_epochs, lr, devices, emb_layer:Embeding_layer, model_name='test'):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    updater = torch.optim.Adam(net.parameters(), lr)
    loss = nn.CrossEntropyLoss(weight=weights)

    best_acc = 0.0

    f = open(collector.output + model_name + '_log.txt', 'w')
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        loop = tqdm(enumerate(train_iter), total=len(train_iter))
        for _, (X, y) in loop:
            X = emb_layer.get_model_input(X, devices[0])
            updater.zero_grad()
            X = X.type(torch.float32)
            X, y = X.to(devices[0]), y.to(devices[0])
            y_hat = net(X)
            l = loss(y_hat, y).mean()
            l.backward()
            d2l.grad_clipping(net, 1)
            updater.step()
            metric.add(utils.get_accuracy(y_hat, y), l * y.shape[0], y.shape[0])
            loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
            loop.set_postfix(loss = metric[1] / metric[2],acc = metric[0] / metric[2])
        collector.add_train_point_standard(metric[0] / metric[2], metric[1] / metric[2])
        net.eval()
        val_acc = utils.evaluation(net, test_iter, devices, emb_layer)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), collector.output + model_name + '_best_val_acc.pth')
            print("No" + str(epoch + 1) + "epoch save the model")
        collector.val_acc_standard.append(val_acc)
        print(f'epoch {epoch + 1}  train_acc: {metric[0] / metric[2]: .3f}  train_loss: {metric[1] / metric[2]:.3f}  val_acc: {val_acc: .3f}', file=f)
    f.close()
    

def prediction(net, model_name, test_iter, emb_layer, devices=d2l.try_all_gpus()):
    net = nn.DataParallel(net, device_ids=devices[:1]).to(devices[0])
    print(collector.output)
    net.load_state_dict(torch.load(collector.output + model_name + '_best_val_acc.pth'))
    net.eval()
    loop = tqdm(enumerate(test_iter), total=len(test_iter))
    for  _, (X, y) in loop:
        X = emb_layer.get_model_input(X, devices[0])
        X = X.type(torch.float32)
        X, y = X.to(devices[0]), y.to(devices[0])
        y_hat = net(X)
        collector.add_test_result(y_hat.detach().cpu(), y.detach().cpu())
    collector.key_point_generate()

for cfg in os.listdir('./config/'): read_config_and_run(cfg[:-4])
