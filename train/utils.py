from d2l import torch as d2l
import torch
import sklearn.metrics as metrics
from sklearn.metrics import auc, roc_curve, f1_score, precision_score
import pandas as pd
import numpy as np
from dataProcessing.modules import to_categorical

def get_accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
        y = torch.argmax(y, axis=1)
    cmp = (y_hat.type_as(y) == y)
    return torch.sum(cmp.type_as(y))

def evaluation(net, test_iter, devices, emb_layer):
    metric = d2l.Accumulator(2)
    for  _, (X, y) in enumerate(test_iter):
        X = emb_layer.get_model_input(X, devices[0])
        X = X.type(torch.float32)
        X, y = X.to(devices[0]), y.to(devices[0])
        y = torch.argmax(y, axis=1)
        y_hat = torch.argmax(net(X), axis=1)
        metric.add(get_accuracy(y_hat, y), X.shape[0])
    return metric[0] / metric[1]

class Collector:
    # Used to calculate various performance indicators and store them in the specified output folder
    def __init__(self, y_encoder, test_sample, out_put):
        self.y_test_ture = None
        self.y_test_pred = None
        self.train_acc_standard = []
        self.train_loss_standard = []
        self.val_acc_standard = []
        self.y_encoder = y_encoder
        self.test_sample = test_sample
        self.y_true_full = []
        self.y_pred_mean = []
        self.y_pred_mean_exact = []
        self.output = out_put
        self.key_point = {}
        self.key_point_pre_class = {'class_name':[], 'f1_per_class':[], 'acc_per_class':[]}

    def add_train_point_standard(self, acc_standard, loss_standard): 
        self.train_acc_standard.append(acc_standard)
        self.train_loss_standard.append(loss_standard)

    def add_test_result(self, y_test_pred, y_test_ture):
        self.y_test_pred = y_test_pred if self.y_test_pred == None else torch.cat([self.y_test_pred, y_test_pred], 0)
        self.y_test_ture = y_test_ture if self.y_test_ture == None else torch.cat([self.y_test_ture, y_test_ture], 0)

    def _mean_acc_generate(self):
        pred = self.y_test_pred.numpy()
        true = self.y_test_ture.numpy()
        test_pred_counts = 0
        for i in self.test_sample:
            sample_pred_mean = np.array(
                np.sum(pred[test_pred_counts:i + test_pred_counts],  # Calculate the average prediction accuracy
                        axis=0) / i)
            self.y_pred_mean.append(np.argmax(sample_pred_mean))
            self.y_pred_mean_exact.append(sample_pred_mean)
            self.y_true_full.append(np.argmax(true[test_pred_counts]))
            test_pred_counts += i

    def _cal_f1_and_acc_per_class(self, y_true, y_pred, true_label):
        bin_true = []
        bin_pred = []
        for i in range(len(y_pred)):
            bin_true.append(1 if y_true[i] == true_label else 0)
            bin_pred.append(1 if y_pred[i] == true_label else 0)
        
        f1 = f1_score(bin_true, bin_pred)
        acc = precision_score(bin_true, bin_pred)
        return f1, acc

    def key_point_generate(self):
        self._mean_acc_generate()  # 计算mean的结果
        y_encoder = self.y_encoder
        true = np.argmax(self.y_test_ture, axis=1)
        pred = np.argmax(self.y_test_pred, axis=1)
        # standard 混淆矩阵
        table = pd.crosstab(
                    pd.Series(y_encoder.inverse_transform(pred)),
                    pd.Series(y_encoder.inverse_transform(true)),
                    rownames=['True'],
                    colnames=['Predicted'],
                    margins=True)
        print("The confusion matrix of the model at the subsequence level (Standard): ")
        print(table.to_string())
        standard_acc = metrics.accuracy_score(true, pred)
        # standard_acc = get_accuracy(pred, true)
        self.key_point["standard_acc"] = standard_acc
        print("Standard ACC: ", standard_acc)
        print("Standard CM Saved")
        print()
        table.to_csv(self.output + '/standard_confusion_matrix.csv')

        # mean 混淆矩阵
        table_mean = pd.crosstab(
            pd.Series(y_encoder.inverse_transform(self.y_true_full)),
            pd.Series(y_encoder.inverse_transform(self.y_pred_mean)),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print("The confusion matrix of the model at the complete sequence level (Aggregated): ")
        print(table_mean.to_string())
        mean_acc = metrics.accuracy_score(self.y_true_full, self.y_pred_mean)
        # mean_acc = get_accuracy(torch.tensor(self.y_pred_mean), torch.tensor(self.y_true_full))
        self.key_point["mean_acc"] = mean_acc
        print("Aggregated ACC: ", mean_acc)
        print("Aggregated CM Saved")
        print()
        table_mean.to_csv(self.output + '/aggregated_confusion_matrix.csv')

        # 计算f1 score
        standard_f1 = f1_score(true, pred, average='weighted')
        mean_f1 = f1_score(self.y_true_full, self.y_pred_mean, average='weighted')
        self.key_point["standard_f1"] = standard_f1
        self.key_point["aggregated_f1"] = mean_f1
        print("Standard f1 score: ", standard_f1)
        print("Aggregated f1 score: ", mean_f1)
        print()

        # 计算每个类的 f1 score 和 acc
        for index, class_name in enumerate(self.y_encoder.classes_):
            f1, acc = self._cal_f1_and_acc_per_class(self.y_true_full, self.y_pred_mean, index)
            self.key_point_pre_class['class_name'].append(class_name)
            self.key_point_pre_class['f1_per_class'].append(f1)
            self.key_point_pre_class['acc_per_class'].append(acc)

        key_point_pre_class_table = pd.DataFrame(self.key_point_pre_class)
        key_point_pre_class_table.to_csv(self.output + '/key_point_per_class_table.csv')

        # 计算并保存 AUC ROC
        self.plot_roc(self.y_true_full, np.array(self.y_pred_mean_exact))

        # 保存训练中指标
        if len(self.train_acc_standard) != 0:
            kpi_traing = pd.DataFrame({'train_acc':self.train_acc_standard, 'train_loss':self.train_loss_standard, 'val_acc':self.val_acc_standard})
            kpi_traing.to_csv(self.output + '/' + 'training_kpi.csv')

        # 保存核心指标
        kpi = pd.DataFrame(self.key_point.items())
        kpi.to_csv(self.output + '/' + 'key_point.csv')

    def plot_roc(self, y_true_list, y_pred_list):

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        classes_count = len(self.y_encoder.classes_)


        y_true_bin = to_categorical(y_true_list, classes_count)
        y_pred_bin = y_pred_list
        for i in range(classes_count):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr['micro'], tpr['micro'], _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
        self.key_point['micro_auc'] = roc_auc['micro']

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes_count)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(classes_count):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= classes_count
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        self.key_point['macro_auc'] = roc_auc['macro']
