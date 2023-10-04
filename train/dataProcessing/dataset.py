from dataProcessing.modules import *
import pandas as pd
from torch.utils.data import DataLoader


class Data:
    def __init__(self, directory, module, subseqlength, emb_word_len, need_train):
        print("#################################################")
        print("Read training, validation, test datasets, and process long sequences")
        print("#################################################")

        self.y_encoder = None

        if need_train:
            Y_train_old = pd.read_csv(directory + '/Y_train.csv', delimiter='\t', dtype='str', header=None)[1].values
            X_train_old = pd.read_csv(directory + '/X_train.csv', delimiter='\t', dtype='str', header=None)[1].values
            Y_val_old = pd.read_csv(directory + '/Y_val.csv', delimiter='\t', dtype='str', header=None)[1].values
            X_val_old = pd.read_csv(directory + '/X_val.csv', delimiter='\t', dtype='str', header=None)[1].values

            X_train, train_seq_length = DealSequence(module, X_train_old, None, True, emb_word_len).run()
            X_val, val_seq_length = DealSequence(module, X_val_old, None, True, emb_word_len).run()
            Y_train, self.y_encoder = DealSequence(module, Y_train_old, None, False, emb_word_len).run()
            Y_val, _ = DealSequence(module, Y_val_old, self.y_encoder, False, emb_word_len).run()
        


        Y_test_old = pd.read_csv(directory + '/Y_test.csv', delimiter='\t', dtype='str', header=None)[1].values
        X_test_old = pd.read_csv(directory + '/X_test.csv', delimiter='\t', dtype='str', header=None)[1].values

        X_test, test_seq_length = DealSequence(2, X_test_old, None, True, emb_word_len).run()
        Y_test, self.y_encoder = DealSequence(2, Y_test_old, self.y_encoder, False, emb_word_len).run()

        print("#################################################")
        print("process data as subsequences")
        print("#################################################")


        if need_train:
            self.X_train, self.Y_train, self.number_subsequences, _ = Split(module, X_train, Y_train, train_seq_length, subseq_length=subseqlength).run()
            self.X_val, self.Y_val, self.number_subsequences, _ = Split(module, X_val, Y_val, val_seq_length, subseq_length=subseqlength).run()
        self.X_test, self.Y_test, _, self.test_samples = Split(module, X_test, Y_test, test_seq_length, subseq_length=subseqlength).test_gen()

    def get_train_DataLoader(self, batch_size, emb_layer):
        return DataLoader(Dataset(self.X_train, self.Y_train, emb_layer), shuffle=True, batch_size=batch_size, num_workers=8)

    def get_val_DataLoader(self, batch_size, emb_layer):
        return DataLoader(Dataset(self.X_val, self.Y_val, emb_layer), shuffle=False, batch_size=batch_size, num_workers=8)
    
    def get_test_DataLoader(self, batch_size, emb_layer):
        return DataLoader(Dataset(self.X_test, self.Y_test, emb_layer), shuffle=False, batch_size=batch_size, num_workers=8)

class Dataset:
    def __init__(self, x, y, emb_layer):
        self.X = emb_layer.transfer_word_to_index(x)
        self.Y = y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return (self.X[idx], self.Y[idx])