import torch.nn as nn
from torch.nn import functional as F
import torch

class CNN_Block(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, pooling_size):
        super().__init__()
        self.conv1d = nn.Conv1d(input_channels, num_channels, kernel_size=kernel_size)
        self.pooling = nn.MaxPool1d(pooling_size)

    def forward(self, X):
        Y = self.conv1d(X)
        Y = F.leaky_relu(Y, negative_slope=0.01)
        Y = self.pooling(Y)
        return Y

class Attention(nn.Module):
    def __init__(self, dims, head, layers):
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dims, nhead=head), layers)
    
    def forward(self, X):
        X = self.encoder(X)
        return X.permute(0, 2, 1)


class BiGRU_Block(nn.Module):
    def __init__(self, input_dims, num_hiddens, num_layers, dropout=0):
        super(BiGRU_Block, self).__init__()
        self.lstm = nn.GRU(input_dims, num_hiddens, num_layers, bidirectional=True, dropout=dropout)

    def forward(self, inputs):
        self.lstm.flatten_parameters()
        # before permute Y: (batch_size, feature_dims, seq_length)
        # after permute Y: (seq_length, batch_size, feature_dims)
        Y = inputs.permute(2, 0, 1)
        
        # Y: (seq_length, batch_size, hidden_size * 2)
        # hidden_state: (num_layers * num_directions, seq_length, hidden_size), cell_state: (num_layers * num_directions, seq_length, hidden_size)
        # Y, (final_hidden_state, final_cell_state) = self.lstm(Y, None)
        Y, final_hidden_state = self.lstm(Y, None)

        # Y: (batch_size, seq_length(num_step), hidden_size * 2)
        Y = Y.transpose(0, 1)
        # print(f'Y shape: {Y.shape}, hidden_state: {final_hidden_state.shape}, cell_state: {final_hidden_state.shape}')
        output = Y[:, -1, :]

        return output


class Embeding_layer:
    def __init__(self, emb_type):
        self.emb_type = emb_type # 1.one-hot 2.w2v
        self.embedding_weight_matrix = None
        self.word_to_index_dict = {}
        self.index_to_word_dict = {}
        self.dims = 0
        self.prepare()

    def transfer_word_to_index(self, X) -> torch.Tensor:
        ans = []
        for data in X:
            cur = []
            for element in data:
                q = self.word_to_index_dict.get(element)
                if q == None: q = 0
                cur.append(q)
            ans.append(cur)
        return torch.tensor(ans).type(torch.long)

    def one_hot(self):
        for index, i in enumerate("ATCGN-"):
            self.word_to_index_dict[i] = index
            self.index_to_word_dict[index] = i
        self.embedding_weight_matrix = torch.eye(6).type(torch.float32)
        self.dims = 6

    def w2v_embedding(self):
        root = "../origin_data/75d.w2v"
        self.embedding_weight_matrix = []
        with open(root) as f:
            count, dims = f.readline().split(" ")
            for idx in range(int(count)):
                data = f.readline().split(" ")
                data[-1] = data[-1].split("\n")[0]
                word = data[0]
                vec = []
                for i in data[1:]:
                    vec.append(float(i))
                self.index_to_word_dict[idx] = word
                self.word_to_index_dict[word] = idx
                self.embedding_weight_matrix.append(vec)
            f.close()
        self.embedding_weight_matrix = torch.tensor(self.embedding_weight_matrix, dtype=torch.float32)
        self.dims = int(dims)
    
    def get_model_input(self, X, device):
        embedding = torch.nn.Embedding(len(self.word_to_index_dict), self.embedding_weight_matrix.shape[1])
        embedding.weight.data = self.embedding_weight_matrix
        embedding.to(device)
        X = X.to(device)
        return embedding(X)

    def prepare(self):
        if self.emb_type == 1: self.one_hot()
        else: self.w2v_embedding()
       


class Net:
    def __init__(self, seq_length, in_channels, output_dim, num_CNN=2, out_channels=150, num_LSTM=3, num_hiddens=150, dropout=0.2):
        self.seq_length = seq_length
        self.in_channels = in_channels
        self.output_dim = output_dim
        self.n_CNN_layer = num_CNN
        self.out_channels = num_hiddens
        self.n_LSTM_layer  = num_LSTM
        self.num_hiddens = num_hiddens
        self.dropout = dropout
        
    def build_normal_model(self, kernel_size=6, pooling_size=2):
        n = nn.Sequential()
        n.append(Attention(self.in_channels, 1, 2))
        for _ in range(self.n_CNN_layer):
            n.append(CNN_Block(input_channels=self.in_channels, num_channels=self.out_channels, kernel_size=kernel_size, pooling_size=pooling_size))
            self.in_channels = self.out_channels
            self.seq_length -= (6 - 1)
            self.seq_length //= 2
        n.append(BiGRU_Block(input_dims=self.out_channels, num_hiddens=self.num_hiddens, num_layers=self.n_LSTM_layer, dropout=self.dropout))
        n.append(nn.Dropout(self.dropout))
        n.append(nn.Linear(self.num_hiddens * 2, self.num_hiddens))
        n.append(nn.ELU())
        n.append(nn.Dropout(self.dropout))
        n.append(nn.Linear(self.num_hiddens, self.output_dim))
        return n
