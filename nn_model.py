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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from nn_utils import save_model, load_model
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import pickle

torch.manual_seed(1)
torch.cuda.is_available()

GLOBAL_BEST_LOSS = 1e11
GLOBAL_BEST_Accuracy = 0

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
        dropout4 = params['dropout4']
        dropout5 = params['dropout5']
        linear_out1 = params['out1']
        linear_out2 = params['out2']
        linear_out3 = params['out3']
        linear_out4 = params['out4']

        num_layers = params['num_layers']
        embed_dim = params['embed_dim']

        hidden_dim = params['hidden_dim']
        linear_in = params['hidden_dim'] * 3

        dropout = params['dropout']

        self.lstm = LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_size=params["batch_size"],
                         dropout=dropout, bidirectional=False)
        self.fc_att = nn.Linear(hidden_dim, 1)

        self.fc = nn.Linear(linear_in, linear_out1)
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(dropout1)
        self.drop2 = nn.Dropout(dropout2)
        self.drop3 = nn.Dropout(dropout3)
        self.drop4 = nn.Dropout(dropout4)
        self.drop5 = nn.Dropout(dropout5)
        self.out1 = nn.Linear(linear_out1, linear_out2)
        self.out2 = nn.Linear(linear_out2, linear_out3)
        self.out3 = nn.Linear(linear_out3, linear_out4)
        self.out4 = nn.Linear(linear_out4, params["num_classes"])
        # sigmoid
        # self.sigmoid = torch.sigmoid()
        # self.loss = nn.BCELoss()
        self.loss = nn.BCELoss(weight=torch.FloatTensor(params['class_weight']))

    def forward(self, data):
        lstm = self.lstm(data)

        att = self.fc_att(lstm).squeeze(-1)  # [b,sl,h]->[b,sl]

        att = F.softmax(att, dim=-1)  # [b,sl]

        r_att = torch.sum(att.unsqueeze(-1) * lstm, dim=1)  # [b,h]

        # pooling
        r_avg = torch.mean(lstm, dim=1)  # [b,h]
        r_max = torch.max(lstm, dim=1)[0]  # [b,h]


        r = torch.cat([r_avg, r_max, r_att], -1)  # [b,h*3]

        # concatenate with local part

        r = self.act(self.fc(self.drop1(r)))

        r = self.act(self.out1(self.drop2(r)))

        r = self.act(self.out2(self.drop3(r)))

        r = self.act(self.out3(self.drop4(r)))

        # no activation
        r = self.out4(self.drop5(r))
        # r = F.softmax(r)
        r = torch.sigmoid(r)
        return r


class Train_Model(object):
    def train_model(self, **kwargs):
        train_args = {
            "epochs": 100,
            "batch_size": 256,
            "validate": True,
            "save_best_dev": True,
            "use_cuda": True,
            "print_every_step": 10,
            "model_path": kwargs['model_path'],
            "eval_metrics": "bce",
            "embed_dim": 300,
            "num_layers": 1,
            "hidden_dim": 256,
            "n_print": 1000,
            "class_weight": kwargs["class_weight"],
            "num_classes": 2
        }
        grid = {
            'out1': kwargs["out_size1"],
            'out2': kwargs["out_size2"],
            'out3': kwargs["out_size3"],
            'out4': kwargs["out_size4"],
            "dropout1": kwargs["drop1"],
            "dropout2": kwargs["drop2"],
            "dropout3": kwargs["drop3"],
            "dropout4": kwargs["drop4"],
            "dropout5": kwargs["drop5"],
            "dropout": kwargs["lstm_drop"],
            "lr": kwargs["lr"]
        }

        train_args.update(grid)
        device = 'cpu'
        if torch.cuda.is_available() and train_args["use_cuda"]:
            device = 'cuda:0'
            if round(torch.cuda.memory_allocated(0)/1024**3,1) > 0.3:
                device = 'cuda:1'
        print("Start training")
        # load train data
        train_data = kwargs["train_data"]

        model = Model(**train_args).to(device)

        # only once tune hyperparamer
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"])
        # if kwargs["validate"]:
        vali_acc = []
        train_acc = 0

        for epoch in range(1, train_args["epochs"] + 1):
            model.train()
            # one forward and backward pass
            for index, (data, label) in enumerate(train_data, 1):
                ml_features = Variable(torch.tensor(np.delete(data[1][:, :, 0].numpy(),
                                                 np.argwhere(np.all(data[1][:, :, 0].numpy()[..., :] == 0, axis=0)),
                                                 axis=1)))
                data = [Variable(data[0].cuda()),Variable(ml_features.cuda())]
                # data = Variable(data.cuda())
                label = Variable(label.cuda())
                optimizer.zero_grad()

                logits = model(data)

                loss = model.loss(logits, label)

                loss.backward()
                optimizer.step()

                _, prediction = torch.max(logits.data, 1)
                _, truth = torch.max(label.data, 1)
                train_acc += torch.sum(prediction == truth).item()
                # train_acc = accuracy_score(truth.cpu(), prediction.cpu())

                if train_args["n_print"] > 0 and index % train_args["print_every_step"] == 0:
                    print("[epoch: {:>3} step: {}] train loss: {:>4.6}"
                          .format(epoch, index, loss.item()))

            train_acc = train_acc / len(train_data.dataset)

            print("Train Accuracy: {:>4.6}".format(train_acc))
            if train_args["validate"]:
                test_model = Test_Model()
                default_valid_args = {
                    "batch_size": 256,  # max(8, self.batch_size // 10),
                    "use_cuda": train_args["use_cuda"],
                    "class_weight": train_args["class_weight"]
                }

                test_acc, truth = test_model.test(model, kwargs['validation_data'], **default_valid_args)
                print("Validate Accuracy: {:>4.6}".format(test_acc))
                global GLOBAL_BEST_Accuracy
                if test_acc > GLOBAL_BEST_Accuracy:
                    GLOBAL_BEST_Accuracy = test_acc
                    save_model(model, "./model.pkl", grid)
                    print("Current Grid Parameters", grid)
                    print("Saved better model selected by validation.")
                vali_acc.append(test_acc)
                # if train_args['save_best_dev'] and best_eval_result(eval_results, **train_args):
                #     save_model(model, "./model.pkl", grid)
                #     print("Current Grid Parameters", grid)
                #     print("Saved better model selected by validation.")
        plt.plot(vali_acc)
        plt.title("lr = "+str(kwargs["lr"]))
        if os.path.exists('./learning_rate/learning_curve_'+str(kwargs["lr"])+'.png'):
            os.remove('./learning_rate/learning_curve_'+str(kwargs["lr"])+'.png')
        plt.savefig('./learning_rate/learning_curve_'+str(kwargs["lr"])+'.png')
        plt.clf()
        torch.cuda.empty_cache()


class Test_Model(object):
    def __init__(self):
        self.device = 'cpu'

    def test(self, model, valid_data, **kwargs):
        if torch.cuda.is_available() and kwargs["use_cuda"]:
            self.device = 'cuda:0'
            if round(torch.cuda.memory_allocated(0)/1024**3,1) > 0.3:
                self.device = 'cuda:1'

        model = model.to(self.device)

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
        weight = torch.FloatTensor(kwargs['class_weight']).cuda()
        loss = nn.BCELoss(weight=weight)
        # result_metrics = {"bce": loss(pred, truth)}
        _, pred = torch.max(pred, 1)
        _, truth = torch.max(truth, 1)

        # print("[tester] {}".format(", ".join(
        #     [str(key) + "=" + "{:.5f}".format(value)
        #      for key, value in result_metrics.items()])))
        # return result_metrics
        test_acc = accuracy_score(truth.cpu(), pred.cpu())
        print(classification_report(truth.cpu(), pred.cpu(), target_names=['fake', 'real']))
        # test_acc = balanced_accuracy_score(truth.cpu(), pred.cpu(), weight.cpu())
        torch.cuda.empty_cache()
        return test_acc, truth

def predict(**kwargs):
    # define model
    with open('hyper.pkl', 'rb') as f:
        final_grid = pickle.load(f)
    print(final_grid)
    args = {
        "num_classes": 2,
        'out1': final_grid["out1"],
        'out2': final_grid["out2"],
        "dropout1": final_grid["dropout1"],
        "dropout2": final_grid["dropout2"],
        "dropout3": final_grid["dropout3"],
        "dropout": final_grid["dropout"],
        "embed_dim": 300,
        "num_layers": 1,
        "hidden_dim": 256,
        "batch_size": 256,
        "class_weight": kwargs["class_weight"]
    }

    model = load_model(Model(**args), kwargs['model_path'])
    data_test = kwargs["test_data"]
    print()
    print("="*50)
    print("Start Predicting")
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1) > 0.3:
            device = 'cuda:1'
    model = model.to(device)

    model.eval()
    batch_output = []
    truth_list = []

    for i, (datas, labels) in enumerate(data_test):
        datas = Variable(datas.cuda())
        labels = Variable(labels.cuda())

        with torch.no_grad():
            prediction = model(datas)

        batch_output.append(prediction.detach())
        truth_list.append(labels.detach())

    pred = torch.cat(batch_output, 0)
    truth = torch.cat(truth_list, 0)

    loss = nn.BCELoss(weight=torch.FloatTensor(kwargs['class_weight']).cuda())
    result_metrics = {"bce": loss(pred, truth)}

    # test_acc = classification_report(truth.cpu(), pred.cpu(), target_names=['fake', 'real'])
    # print("[Final tester] Accuracy: {}".format(test_acc))
    print("[Final tester] {}".format(", ".join(
        [str(key) + "=" + "{:.5f}".format(value)
         for key, value in result_metrics.items()])))
    torch.cuda.empty_cache()
    return pred.cpu(), truth.cpu()

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