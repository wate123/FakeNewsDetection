import numpy as np
import torch, time
from nn_utils import data_preparation
from nn_model import Train_Model, predict
from itertools import product
from sklearn.metrics import classification_report
from utils import NewsContent
from Word2VecFeature import Word2VecFeatureGenerator

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

dataset = ['politifact', 'gossipcop']
# dataset = ['politifact']
# dataset = ['gossipcop']
print("Preparing Dataset")
start = time.time()
data = NewsContent('../fakenewsnet_dataset', dataset, ['fake', 'real'])
Word2VecFeatureGenerator(data.get_features("pair"), dataset).get_nn_vecs()

data_dict = data_preparation(dataset, seed)
end = time.time()
duration = end - start
print("Data preparation took {} hour {} min {} sec".format(duration // 3600, (duration % 3600) // 60, int(duration % 60)))

#
grid_search = {
    # "out_size1": [2 ** i for i in range(5, 10)],
    # "out_size2": [2 ** i for i in range(4, 9)],
    "out_size1": [512],
    "out_size2": [256],
    # pretrain w2v
    # "out_size1": [512],
    # "out_size2": [256],
    # [0.05 * i for i in range(11)]
    "lstm_drop": [0.01],
    "drop1": [0.05],
    # "lstm_drop": [0.05 * i for i in range(11)],
    # "drop1": [0.05 * i for i in range(11)],
    # "drop2": [0.05 * i for i in range(11)],
    # "drop3": [0.05 * i for i in range(11)],
    "drop2": [0.05],
    "drop3": [0.05],
    "lr": [1.2e-3]
}

device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    if round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1) > 0.3:
        device = torch.device('cuda:1')
data_dict.update({"device": device})

start = time.time()
train = Train_Model()
for grid in [dict(zip(grid_search.keys(), v)) for v in product(*grid_search.values())]:
    if grid["out_size1"] > grid["out_size2"]:
        new_args = {**data_dict, **grid}
        train.train_model(**new_args)

        y_pred, y_true = predict(**data_dict)

        preRecF1 = classification_report(y_true, y_pred, target_names=['fake', 'real'])
        print(preRecF1)
        np.save('prediction.npy', y_pred)
        np.save('ground_truth.npy', y_true)
        end = time.time()
        period = end - start
        print("predicting took {} hour {} min {} sec".format(period // 3600, (period % 3600) // 60, int(period % 60)))

end = time.time()
duration = end - start
print("training took {} hour {} min {} sec".format(duration // 3600, (duration % 3600) // 60, int(duration % 60)))

start = time.time()
y_pred,y_true = predict(**data_dict)

preRecF1 = classification_report(y_true, y_pred, target_names=['fake', 'real'])
print(preRecF1)
np.save('prediction.npy',y_pred)
np.save('ground_truth.npy',y_true)
end = time.time()
period = end - start
print("predicting took {} hour {} min {} sec".format(period // 3600, (period % 3600) // 60, int(period % 60)))
