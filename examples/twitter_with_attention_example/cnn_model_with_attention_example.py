import math
import os
import numpy as np

from tweeter_covid19 import read_pickle_data
from tweeter_covid19.classification import Classification
from tweeter_covid19.utils.modelutils import optimize_model

import os
import pickle

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from tweeter_covid19.cnn.pruning import scale_and_multiply, tile_pruning, row_pruning_with_zeros



class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        WO = torch.randn(size=q.shape, dtype=torch.float32)
        bias = torch.randn(size=q.shape, dtype=torch.float32)

        WQ_ = tile_pruning(q, bias, 2, 2, milesone=True).transpose(-2, -1)
        WK_ = tile_pruning(k, bias, 2, 2, milesone=True).transpose(-2, -1)
        WV_ = row_pruning_with_zeros(v, 2).transpose(-2, -1)

        x_ = torch.reshape(x, q.shape)

        q_ = torch.matmul(x_.float(), WQ_.float())
        k_ = torch.matmul(x_.float(), WK_.float())
        v_ = torch.matmul(x_.float(), WV_.float())

        k_transpose = k_.transpose(-2, -1)
        d_k = q.size()[-1]

        scale_values = scale_and_multiply(q_, d_k, k_transpose)

        z = torch.matmul(scale_values, v_)
        z[z != z] = 0
        WO = tile_pruning(WO, bias, 2, 2, milesone=True)

        output = torch.matmul(z.float(), WO.float())
        # add_val = torch.zeros(x_.shape, dtype=torch.float32)
        # for val in range(output.shape[0]):
        #     add_val += output[val]
        add_val = torch.reshape(output, x.shape)
        return add_val
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(10, 300, 5)
        self.conv2 = nn.Conv2d(5, 150, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

SETS = 10
if __name__ == '__main__':
        read_main_path = os.path.join('data', 'fold_train_test_collector')
        read_path = os.path.join('data', 'model_save_update')

        data_structure = {
        'sets': [],
        'f1_score': [],
        'precision_score': [],
        'recall_score': [],
        'accuracy_score': [],
        }
        for n_set in range(SETS):
            read_joiner_path = os.path.join(read_main_path, 'set_' + str(n_set + 1))
            model = Classification()
            train_x = read_pickle_data(os.path.join(read_joiner_path, 'train_x.pkl'))
            train_y = read_pickle_data(os.path.join(read_joiner_path, 'train_y.pkl'))
            test_x = read_pickle_data(os.path.join(read_joiner_path, 'test_x.pkl'))
            test_y = read_pickle_data(os.path.join(read_joiner_path, 'test_y.pkl'))
            print("---------------- Loading Set {} completed -------------------".format(n_set + 1))

            print(np.shape(train_x), np.shape(train_y))
            print(np.shape(test_x), np.shape(test_y))
            exit(0)
            net = Net()
            print(net)
