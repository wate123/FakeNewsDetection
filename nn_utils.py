from torch.utils.data import Dataset, DataLoader
import torch
from h5py import File
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd, numpy as np
import pickle
from sklearn.utils import shuffle
import random
class FakenewsDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.ml,  self.y = X[0], X[1], y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        X = torch.tensor(self.X[item], dtype=torch.float)
        ml = torch.tensor(self.ml[item], dtype=torch.float)
        y = torch.tensor(self.y[item], dtype=torch.float)
        return X, ml, y


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


def read_data(seed):
    X = File("w2v_feature_pad.hdf5", "r")["w2v"][:10000]
    # X = np.load("w2v_feature_pad.npy")[3000:9000]
    ml_features = pd.read_csv("final_features_ml.csv").values[:10000]
    # ml_features = np.pad(ml_features, ((0, 0), (0, X.shape[1]-ml_features.shape[1])), 'constant', constant_values=0)
    # ml_features = ml_features[..., np.newaxis]
    # ml_features = np.broadcast_to(ml_features, (ml_features.shape[0], ml_features.shape[1], 300))
    # ml_features = np.broadcast_to(ml_features, X.shape)

    labels = pd.read_csv("data.csv")["label"][:10000]
    y = labels.values
    real_fake_count = labels.value_counts()
    class_weight = [labels.shape[0]/real_fake_count['fake'], labels.shape[0]/real_fake_count['real']]
    # class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

    # y = LabelEncoder().fit_transform(y).reshape(-1,1)
    y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
    # random sample n rows
    # X, y = zip(*random.sample(list(zip(X,y)), 5000))
    # split them into 80% training, 10% testing, 10% validation
    X_train, X_test, ml_train, ml_test, y_train, y_test = train_test_split(list(X), list(ml_features), list(y), test_size=0.2, random_state=seed)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_valid, X_test, ml_valid, ml_test, y_valid, y_test = train_test_split(X_test, ml_test, y_test, test_size=0.5, random_state=seed)
    X_train = [np.array(X_train), ml_train]
    X_test = [np.array(X_test), ml_test]
    X_valid = [np.array(X_valid), ml_valid]
    return X_train, X_valid, X_test, y_train, y_valid, y_test, class_weight


def data_preparation(seed):
    model_path = 'model.pkl'
    batch_size = 64
    X_train, X_valid, X_test, y_train, y_valid, y_test, class_weight = read_data(seed)

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
    return {"train_data": train_loader, "test_data": test_loader, "validation_data": valid_loader,
            "class_weight": class_weight, "model_path": model_path, "seed":seed, "batch_size": batch_size}

