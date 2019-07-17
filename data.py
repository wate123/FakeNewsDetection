# -*- coding: utf-8 -*-
"""
Created on Sun May 26 21:30:32 2019

@author: mgy92
"""
# pytorch
from torch.utils.data import Dataset
import torch

# numpy
import numpy as np

# scikit-learn
from sklearn.model_selection import train_test_split as TTS

class myDataset(Dataset):
    ''' dataset reader
    '''
    
    def __init__(self,X,y):
        self.data_X, self.data_y = X,y
        
    def __len__(self):
        return len(self.data_y)
    
    def __getitem__(self, index):
        
        X = torch.tensor(self.data_X[index][0], dtype=torch.float)
        y = torch.tensor(self.data_y[index], dtype=torch.float)

        return X, y

def readData(seed):
    ''' method for reading different parts of data
    '''
    
    path = '../Data/0626_'
    
    X = np.load(path+'fw.npy')
    
    # correct/target/actual embeddings
    y = np.load(path+'target.npy')
    
    # split them into 80% training, 10% testing, 10% validation
    X_train, X_test, y_train, y_test = TTS(X, y, test_size = 0.2, random_state = seed)
    X_valid, X_test, y_valid, y_test = TTS(X_test, y_test, test_size = 0.5, random_state = seed)
    
    return X_train,X_valid,X_test,y_train,y_valid,y_test
