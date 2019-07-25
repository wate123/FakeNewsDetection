#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:04:43 2019

@author: mgy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
from mask import mask_softmax, mask_mean, mask_max

def save_model(model, model_path, grid):
    """Save model."""
    torch.save(model.state_dict(), model_path)
    with open("hyper.pkl",'wb') as f:
        pickle.dump(grid,f)
    print("checkpoint saved")
    return

def load_model(model, model_path):
    """Load model."""
    map_location = 'cpu'
    if torch.cuda.is_available():
        map_location = 'cuda:0'
        if round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1) > 0.3:
            map_location = 'cuda:1'
    model.load_state_dict(torch.load(model_path, map_location))
    return model

def get_model_setting(**args):
    model_args = {
            'num_classes': 401, # eventual prediction size
            'out1': args["out_size1"], # temporal second to the last layer size
            'out2': args["out_size2"],
            "dropout1": args["drop1"], # first dropout
            "dropout2": args["drop2"], # second dropout
            "dropout3": args["drop3"],
            }
    lstm_args = {
        "embed_dim": 401, # input size
        "num_layers": 2, # layers to map to the last result
        "hidden_dim": 512, # hidden state size
        "dropout": args["lstm_drop"], # dropout rate
    }

    model = phoneticModel(model_args,lstm_args)
    return model

class myLSTM(nn.Module):
    """
    LSTM module.

    Parameters
    ----------
    input_size : input size
    hidden_size : hidden size. Default: 100
    num_layers : number of hidden layers. Default: 1
    dropout : dropout rate. Default: 0.
    bidirectional : If True, becomes a bidirectional RNN. Default: False.

    Inputs
    ------
    input: tensor, shaped [sequence, batch, input_size]

    Outputs
    -------
    output: tensor, shaped [batch, num_directions * hidden_size],
         tensor containing the output features (h_t) from the last layer
         of the LSTM, for the last t.
    """

    def __init__(self, input_size, hidden_size=100,
                 num_layers=1, dropout=0., bidirectional=False):
        super(myLSTM, self).__init__()
        
        self.num_layers = num_layers
        self.batch_size = 128
        self.hidden_dim = hidden_size
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bias=True,
            batch_first=True, dropout=dropout, bidirectional=bidirectional)
    
    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x)
        return lstm_out
    
class phoneticModel(nn.Module):
    """Model for phonetic based model.
    """

    def __init__(self, args, args1):
        super(phoneticModel, self).__init__()
        
        embed_dim = args1['embed_dim']
        hidden_dim = args1['hidden_dim']
        num_layers = args1['num_layers']
        dropout1 = args['dropout1']
        dropout2 = args['dropout2']
        dropout3 = args['dropout3']
        dropout = args1['dropout']
        num_classes = args['num_classes']
        linear_in = args1['hidden_dim']
        linear_out1 = args['out1']
        linear_out2 = args['out2']
        
        # forward lstm
        self.lstm1 = myLSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout, bidirectional=False)
        
        self.fc_att = nn.Linear(hidden_dim, 1)

        self.fc = nn.Linear(linear_in,linear_out1)
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(dropout1)
        self.drop2 = nn.Dropout(dropout2)
        self.drop3 = nn.Dropout(dropout3)
        self.out1 = nn.Linear(linear_out1, linear_out2)
        self.out2 = nn.Linear(linear_out2, num_classes)

        self.loss = nn.MSELoss()
        
    def forward(self, x):
        lstm_attn = self.lstm1(x[0])
        
        # attention
        att = self.fc_att(lstm_attn).squeeze(-1)  # [b,sl,h]->[b,sl]
        att = mask_softmax(att)  # [b,sl]
        r_att = torch.sum(att.unsqueeze(-1) * lstm_attn, dim=1)  # [b,h]
        
        # pooling
        r_avg = mask_mean(lstm_attn)  # [b,h]
        r_max = mask_max(lstm_attn)  # [b,h]
        r = torch.cat([r_avg, r_max, r_att], -1)  # [b,h*3]
        
        # concatenate with local part
        
        r = self.act(self.fc(self.drop1(r)))
        
        r = self.act(self.out1(self.drop2(r)))
        
        r = self.out2(self.drop3(r))

        return r
