import numpy as np
import torch, time
from nn_utils import read_data, data_preparation
from nn_model import train, predict
from itertools import product
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

GLOBAL_BEST_LOSS = 1e11
print("Preparing Dataset")
start = time.time()
data_dict = data_preparation()
end = time.time()
duration = end - start
print("Data preparation took {} hour {} min {} sec".format(duration // 3600, (duration % 3600) // 60, int(duration % 60)))
#
# grid_search = {
#     "out_size1": [2048],
#     "out_size2": [1536],
#     "lstm_drop": [0.5],
#     "drop1": [0.0],
#     "drop2": [0.05],
#     "drop3": [0.05*i for i in range(11)],
# }
grid_search = {
    "out_size1": [128],
    "out_size2": [64],
    "lstm_drop": [0.4],
    "drop1": [0.05],
    "drop2": [0.05],
    "drop3": [0.05],
}

start = time.time()
for grid in [dict(zip(grid_search.keys(), v)) for v in product(*grid_search.values())]:
    new_args = {**data_dict, **grid}
    train(**new_args)
end = time.time()
duration = end - start
print("training took {} hour {} min {} sec".format(duration // 3600, (duration % 3600) // 60, int(duration % 60)))

start = time.time()
predict(**data_dict)
end = time.time()
period = end - start
print("predicting took {} hour {} min {} sec".format(period // 3600, (period % 3600) // 60, int(period % 60)))
