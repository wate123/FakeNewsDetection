from torch.utils.data import Dataset, DataLoader
import torch
from h5py import File
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd, numpy as np
import pickle


class FakenewsDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        X = torch.tensor(self.X[item][0], dtype=torch.float)
        y = torch.tensor(self.y[item], dtype=torch.float)
        return X, y


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
    model.load_state_dict(torch.load(model_path, map_location))
    return model


def read_data():
    X = File("w2v_feature_pad.hdf5", "r")["w2v"][:]
    y = pd.read_csv("data.csv")["label"]
    y = LabelEncoder().fit_transform(y)
    # split them into 80% training, 10% testing, 10% validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def data_preparation():
    model_path = 'model/model.pkl'
    batch_size = 128
    X_train, X_valid, X_test, y_train, y_valid, y_test = read_data()

    X_train = np.swapaxes(X_train, 0, 1)
    X_valid = np.swapaxes(X_valid, 0, 1)
    X_test = np.swapaxes(X_test, 0, 1)


    train_set = FakenewsDataset(X_train, y_train)
    print('Training size =', len(train_set.y))
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=16)

    test_set = FakenewsDataset(X_test, y_test)
    print('Testing size =', len(test_set.y))
    # Create a loder for the test set, note that shuffle is set to false for the test loader
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16)

    valid_set = FakenewsDataset(X_valid, y_valid)
    print('valid_size =', len(valid_set.y))
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=16)
    return {"train_data": train_loader, "test_data": test_loader, "validation_data": valid_loader, "model_path": model_path}

