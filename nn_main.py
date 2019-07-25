import numpy as np
import torch, time
from nn_utils import read_data, data_preparation
from nn_model import Train_Model, predict
from itertools import product
from sklearn.metrics import classification_report
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

print("Preparing Dataset")
start = time.time()
data_dict = data_preparation(seed)
end = time.time()
duration = end - start
print("Data preparation took {} hour {} min {} sec".format(duration // 3600, (duration % 3600) // 60, int(duration % 60)))

grid_search = {
    "out_size1": [2 ** i for i in range(5, 10)],
    "out_size2": [2 ** i for i in range(4, 9)],
    # "out_size1": [256],
    # "out_size2": [64],
    # [0.05 * i for i in range(11)]
    "lstm_drop": [0.05],
    "drop1": [0.05],
    # "lstm_drop": [0.05 * i for i in range(11)],
    # "drop1": [0.05 * i for i in range(11)],
    "drop2": [0.05],
    "drop3": [0.05],
    "lr": [1e-4]
}

start = time.time()
train = Train_Model()
for grid in [dict(zip(grid_search.keys(), v)) for v in product(*grid_search.values())]:
    if grid["out_size1"] > grid["out_size2"]:
        new_args = {**data_dict, **grid}
        train.train_model(**new_args)
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
