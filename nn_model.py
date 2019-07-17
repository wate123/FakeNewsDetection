import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Word2VecFeature import Word2VecFeatureGenerator
from sklearn.model_selection import train_test_split

from utils import NewsContent

from gensim.models.word2vec import LineSentence
import numpy as np
import pandas as pd
from nn_utils import save_model, load_model
from torch.autograd import Variable

import pickle

torch.manual_seed(1)
torch.cuda.is_available()


# w2v = Word2VecFeatureGenerator()

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, dropout=0, bidirectional=False, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, bias=True,
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, inputs):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        print(inputs.size())
        lstm_out, _ = self.lstm(inputs)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        # y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return lstm_out


class Model(nn.Module):
    def __init__(self, **params):
        super(Model, self).__init__()

        # num_classes = model_param['num_classes']
        dropout1 = params['dropout1']
        dropout2 = params['dropout2']
        dropout3 = params['dropout3']
        linear_out1 = params['out1']
        linear_out2 = params['out2']

        num_layers = params['num_layers']
        embed_dim = params['embed_dim']

        hidden_dim = params['hidden_dim']
        linear_in = params['hidden_dim']

        dropout = params['dropout']

        self.lstm = LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_size=params["batch_size"],
                         dropout=dropout, bidirectional=False)
        self.fc_att = nn.Linear(hidden_dim, 1)

        self.fc = nn.Linear(linear_in, linear_out1)
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(dropout1)
        self.drop2 = nn.Dropout(dropout2)
        self.drop3 = nn.Dropout(dropout3)
        self.out1 = nn.Linear(linear_out1, linear_out2)
        self.out2 = nn.Linear(linear_out2, 1)
        # sigmoid
        # self.sigmoid = torch.sigmoid()

        self.loss = nn.BCELoss()

    def forward(self, data):
        lstm = self.lstm(data)
        att = self.fc_att(lstm).squeeze(-1)  # [b,sl,h]->[b,sl]

        att = F.softmax(att, dim=-1)  # [b,sl]

        r_att = torch.sum(att.unsqueeze(-1) * lstm, dim=1)  # [b,h]

        # pooling
        r_avg = torch.avg(lstm)  # [b,h]
        r_max = torch.max(lstm)  # [b,h]
        r = torch.cat([r_avg, r_max, r_att], -1)  # [b,h*3]

        # concatenate with local part

        r = self.act(self.fc(self.drop1(r)))

        r = self.act(self.out1(self.drop2(r)))

        # no activation
        r = self.out2(self.drop3(r))
        r = torch.sigmoid(r)

        return r


def train(**kwargs):
    train_args = {
        "epochs": 50,
        "batch_size": 128,
        "validate": True,
        "save_best_dev": True,
        "use_cuda": True,
        "print_every_step": 1000,
        "model_path": kwargs['model_path'],
        "eval_metrics": "bce",
        "embed_dim": 300,
        "num_layers": 1,
        "hidden_dim": 256,

    }
    grid = {
        'out1': kwargs["out_size1"],
        'out2': kwargs["out_size2"],
        "dropout1": kwargs["drop1"],
        "dropout2": kwargs["drop2"],
        "dropout3": kwargs["drop3"],
        "dropout": kwargs["lstm_drop"],
    }

    train_args.update(grid)
    device = 'cpu'
    if torch.cuda.is_available() and train_args["use_cuda"]:
        device = 'cuda:0'
    print("Start training")
    # load train data
    train_data = kwargs["train_data"]

    model = Model(**train_args).to(device)

    # only once tune hyperparamer
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
    # if kwargs["validate"]:
    for epoch in range(1, train_args["epochs"]+1):
        model.train()

        # one forward and backward pass
        for index, (data, label) in enumerate(train_data, 1):
            data = Variable(data.cuda())
            print("data: ", (data.size()))
            label = Variable(label.cuda())
            optimizer.zero_grad()

            logits = model(data)

            loss = model.loss(logits, label)
            loss.backward()
            optimizer.step()

            if train_args["n_print"] > 0 and index % train_args["print_every_step"] == 0:
                print("[epoch: {:>3} step: {:>6}] train loss: {:>4.6}"
                      .format(kwargs["epoch"], index, loss.item()))

            if train_args["validate"]:
                default_valid_args = {
                    "batch_size": 128,  # max(8, self.batch_size // 10),
                    "use_cuda": train_args["use_cuda"]}

                eval_results = test(model, kwargs['validation_data'], **default_valid_args)

                if kwargs['save_best_dev'] and best_eval_result(eval_results, **train_args):
                    save_model(model, "model/model,pkl", grid)
                    print("Saved better model selected by validation.")


def test(model, valid_data, **kwargs):
    device = 'cpu'
    if torch.cuda.is_available() and kwargs["use_cuda"]:
        device = 'cuda:0'

    model = model.to(device)

    model.eval()
    output = []
    ground_truth = []

    for index, (data, label) in enumerate(valid_data):
        data = Variable(data.cuda())
        label = Variable(label.cuda())

        with torch.no_grad():
            prediction = model(data)
        output.append(prediction.detach())
        ground_truth.append(label.detach())

    # evaluation matrics

    pred = torch.cat(output, 0)
    truth = torch.cat(ground_truth, 0)

    result_metrics = {"bce": nn.BCELoss(truth, pred)}
    print("[tester] {}".format(", ".join(
            [str(key) + "=" + "{:.5f}".format(value)
             for key, value in result_metrics.items()])))


def predict(data):
    # define model
    with open('hyper.pkl', 'rb') as f:
        final_grid = pickle.load(f)
    args = {
        "num_classes": 1,
        'out1': final_grid["out_size1"],
        'out2': final_grid["out_size2"],
        "dropout1": final_grid["drop1"],
        "dropout2": final_grid["drop2"],
        "dropout3": final_grid["drop3"],
        "dropout": final_grid["lstm_drop"],
        "embed_dim": 300,
        "num_layers": 1,
        "hidden_dim": 256,
    }

    model = load_model(Model(**args), args['model_path'])

    print("Start Predicting")
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    model = model.to(device)

    model.eval()
    batch_output = []
    truth_list = []

    for i, (datas, labels) in enumerate(data):
        datas = [Variable(datas[0].cuda()), Variable(datas[1].cuda()), Variable(datas[2].cuda()),
                 Variable(datas[3].cuda())]
        labels = Variable(labels.cuda())

        with torch.no_grad():
            prediction = model(datas)

        batch_output.append(prediction.detach())
        truth_list.append(labels.detach())

    predict = torch.cat(batch_output, 0)
    truth = torch.cat(truth_list, 0)

    result_metrics = {"bce": nn.BCELoss(truth, predict)}
    print("[tester] {}".format(", ".join(
        [str(key) + "=" + "{:.5f}".format(value)
         for key, value in result_metrics.items()])))


def best_eval_result(eval_results, **kwargs):
    """Check if the current epoch yields better validation results.

    :param eval_results: dict, format {metrics_name: value}
    :return: bool, True means current results on dev set is the best.
    """
    _best_loss = 1e10

    global GLOBAL_BEST_LOSS
    eval_metrics = kwargs["eval_metrics"]
    assert eval_metrics in eval_results, \
        "Evaluation doesn't contain metrics '{}'." \
            .format(eval_metrics)

    loss = eval_results[eval_metrics]
    if loss < _best_loss:
        _best_loss = loss
        if loss < GLOBAL_BEST_LOSS:
            GLOBAL_BEST_LOSS = loss
            return True
    return False
# model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_size=6, num_layers=1)
#
# loss_fn = torch.nn.MSELoss(size_average=False)
#
# optimiser = torch.optim.Adam(model.parameters(), lr=0.2)
#
# #####################
# # Train model
# #####################
# num_epochs = 10
#
#
# sample_train = pd.read_csv("w2v_feature.csv").sample(800)
# labels = sample_train["label"]
# sample_train = sample_train.drop('label', axis=1).values
# X_train, X_test, y_train, y_test = train_test_split(sample_train, labels, test_size=0.2, random_state=1)
#
# hist = np.zeros(num_epochs)
#
# for t in range(num_epochs):
#     # Clear stored gradient
#     model.zero_grad()
#
#     # Initialise hidden state
#     # Don't do this if you want your LSTM to be stateful
#     model.hidden = model.init_hidden()
#
#     print(type(X_train))
#     # Forward pass
#     y_pred = model(X_train)
#
#     loss = loss_fn(y_pred, y_train)
#     if t % 100 == 0:
#         print("Epoch ", t, "MSE: ", loss.item())
#     hist[t] = loss.item()
#
#     # Zero out gradient, else they will accumulate between epochs
#     optimiser.zero_grad()
#
#     # Backward pass
#     loss.backward()
#
#     # Update parameters
#     optimiser.step()