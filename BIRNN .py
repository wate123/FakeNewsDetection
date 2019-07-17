#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:21:07 2019

@author: mgy
"""

# torch packages
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

# python's own
import time
import numpy as np
import itertools
import pickle

# scikit-learn
from sklearn.metrics import mean_squared_error as MSE

# I wrote the following
from model import save_model, load_model, get_model_setting
from data import myDataset, readData

GLOBAL_BEST_LOSS = 1e11

class Trainer(object):
    """Trainer."""

    def __init__(self, **kwargs):
        self.n_epochs = kwargs["epochs"]
        self.batch_size = kwargs["batch_size"]
        self.validate = kwargs["validate"]
        self.save_best_dev = kwargs["save_best_dev"]
        self.use_cuda = kwargs["use_cuda"]
        self.print_every_step = kwargs["print_every_step"]
        self.optimizer = kwargs["optimizer"]
        self.model_path = kwargs["model_path"]
        self.eval_metrics = kwargs["eval_metrics"]

        self._best_loss = 1e10 # a huge initial value
        
        self.grid = {
                "lstm_drop": kwargs["lstm_drop"],
                "out_size1": kwargs["out_size1"],
                "out_size2": kwargs["out_size2"],
                "drop1": kwargs["drop1"],
                "drop2": kwargs["drop2"],
                "drop3": kwargs["drop3"],
                }

        self.device = 'cpu'
        if torch.cuda.is_available() and self.use_cuda:
            self.device = 'cuda:0'

    def train(self, network, train_data, dev_data=None):
        # transfer model to gpu if available
        network = network.to(self.device)

        # define Tester over dev data
        if self.validate:
            default_valid_args = {
                "batch_size": 128,#max(8, self.batch_size // 10),
                "use_cuda": self.use_cuda}
            validator = Tester(**default_valid_args)
        
        for epoch in range(1, self.n_epochs + 1):
            # turn on network training mode
            network.train()

            # one forward and backward pass
            self._train_step(
                train_data, network, n_print=self.print_every_step, epoch=epoch)

            # validation
            if self.validate:
                if dev_data is None:
                    raise RuntimeError(
                        "self.validate is True in trainer, "
                        "but dev_data is None."
                        " Please provide the validation data.")
                eval_results = validator.test(network, dev_data)

                if self.save_best_dev and self.best_eval_result(eval_results):
                    save_model(network, self.model_path,self.grid)
                    print("Saved better model selected by validation.")

    def _train_step(self, data_iterator, network, **kwargs):
        """Training process in one epoch.
        """
        step = 0
        for i, (data,label) in enumerate(data_iterator):
            data = [Variable(data[0].cuda()),Variable(data[1].cuda()),Variable(data[2].cuda()), Variable(data[3].cuda())]
            label = Variable(label.cuda())

            self.optimizer.zero_grad()
            logits = network(data)
            #print('logits',type(logits),'label',type(label))
            loss = network.loss(logits, label)
            loss.backward()
            self.optimizer.step()

            if kwargs["n_print"] > 0 and step % kwargs["n_print"] == 0:
                print_output = "[epoch: {:>3} step: {:>6}]" \
                    " train loss: {:>4.6}".format(
                        kwargs["epoch"], step, loss.item())
                print(print_output)

            step += 1
            
    def best_eval_result(self, eval_results):
        """Check if the current epoch yields better validation results.

        :param eval_results: dict, format {metrics_name: value}
        :return: bool, True means current results on dev set is the best.
        """
        
        global GLOBAL_BEST_LOSS
        
        assert self.eval_metrics in eval_results, \
            "Evaluation doesn't contain metrics '{}'." \
            .format(self.eval_metrics)

        loss = eval_results[self.eval_metrics]
        if loss < self._best_loss:
            self._best_loss = loss
            if loss < GLOBAL_BEST_LOSS:
                GLOBAL_BEST_LOSS = loss
                return True
        return False
        
class Tester(object):
    """Tester."""

    def __init__(self, **kwargs):
        self.batch_size = kwargs["batch_size"]
        self.use_cuda = kwargs["use_cuda"]
        self.device = 'cpu'
        if torch.cuda.is_available() and self.use_cuda:
            self.device = 'cuda:0'

    def test(self, network, dev_data):
        # transfer model to gpu if available
        network = network.to(self.device)

        # turn on the testing mode; clean up the history
        network.eval()
        output_list = []
        truth_list = []

        # predict
        for i, (data,label) in enumerate(dev_data):
            data = [Variable(data[0].cuda()),Variable(data[1].cuda()),Variable(data[2].cuda()), Variable(data[3].cuda())]
            label = Variable(label.cuda())

            with torch.no_grad():
                prediction = network(data)

            output_list.append(prediction.detach())
            truth_list.append(label.detach())

        # evaluate
        eval_results = self.evaluate(output_list, truth_list)
        print("[tester] {}".format(self.print_eval_results(eval_results)))

        return eval_results

    def evaluate(self, predict, truth):
        """Compute evaluation metrics.

        :param predict: list of Tensor
        :param truth: list of dict
        :param threshold: threshold of positive probability
        :return eval_results: dict, format {name: metrics}.
        """
        
        predict = torch.cat(predict,0)
        truth = torch.cat(truth,0)

        metrics_dict = {"mse":MSE(truth,predict)}

        return metrics_dict
    
    def print_eval_results(self, results):
        """Override this method to support more print formats.
        :param results: dict, (str: float) is (metrics name: value)
        """
        return ", ".join(
            [str(key) + "=" + "{:.5f}".format(value)
             for key, value in results.items()])

class Predictor(object):
    """An interface for predicting outputs based on trained models.
    """

    def __init__(self, batch_size=128, use_cuda=True):
        self.batch_size = batch_size
        self.use_cuda = use_cuda

        self.device = 'cpu'
        if torch.cuda.is_available() and self.use_cuda:
            self.device = 'cuda:0'

    def predict(self, network, data):
        # transfer model to gpu if available
        network = network.to(self.device)

        # turn on the testing mode; clean up the history
        network.eval()
        batch_output = []
        truth_list = []

        for i, (datas, labels) in enumerate(data):
            datas = [Variable(datas[0].cuda()),Variable(datas[1].cuda()),Variable(datas[2].cuda()), Variable(datas[3].cuda())]
            labels = Variable(labels.cuda())

            with torch.no_grad():
                prediction = network(datas)

            batch_output.append(prediction.detach())
            truth_list.append(labels.detach())
            
        eval_results = self.evaluate(batch_output, truth_list)
        print("[Final tester] {}".format(self.print_eval_results(eval_results)))

        return torch.cat(batch_output,0), torch.cat(truth_list,0)
    
    def evaluate(self, predict, truth):
        """Compute evaluation metrics.

        :param predict: list of Tensor
        :param truth: list of dict
        :param threshold: threshold of positive probability
        :return eval_results: dict, format {name: metrics}.
        """
        
        predict = torch.cat(predict,0)
        truth = torch.cat(truth,0)

        metrics_dict = {"mse":MSE(truth,predict)}

        return metrics_dict
    
    def print_eval_results(self, results):
        """Override this method to support more print formats.
        :param results: dict, (str: float) is (metrics name: value)
        """
        return ", ".join(
            [str(key) + "=" + "{:.5f}".format(value)
             for key, value in results.items()])

def pre(seed):
    """Pre-process model."""

    print("Pre-processing...")
    
    model_path = 'model.pkl'
    batch_size = 128
    
    # get dataset
    X_train,X_valid,X_test,y_train,y_valid,y_test = readData(seed)
    

    #Load the training set
    train_set = myDataset(X_train,y_train)
    print('training_size =', len(train_set.data_y))
    #Create a loder for the training set
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=16)

    #Load the test set
    test_set = myDataset(X_test,y_test)
    print('testing_size =', len(test_set.data_y))
    #Create a loder for the test set, note that shuffle is set to false for the test loader
    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=16)

    valid_set = myDataset(X_valid,y_valid)
    print('valid_size =', len(valid_set.data_y))
    valid_loader = DataLoader(valid_set,batch_size=batch_size,shuffle=False,num_workers=16)

    args_dict = {
        "data_train": train_loader, "data_val": valid_loader,
        "data_test": test_loader, "model_path": model_path}

    return args_dict

def train(**args):
    """Train model.
    """

    print("Training...")

    # load data and embed
    data_train = args["data_train"]
    
    # define model
    model = get_model_setting(**args)

    # define trainer
    trainer_args = {
        "epochs": 50,
        "batch_size": 128,
        "validate": True,
        "save_best_dev": True,
        "use_cuda": True,
        "print_every_step": 1000,
        "optimizer": torch.optim.Adam(model.parameters(), lr=4e-4),
        "model_path": args['model_path'],
        "eval_metrics": "mse",
        "lstm_drop": args["lstm_drop"],
        "out_size1": args["out_size1"],
        "out_size2": args["out_size2"],
        "drop1": args["drop1"],
        "drop2": args["drop2"],
        "drop3": args["drop3"],
    }
    trainer = Trainer(**trainer_args)
    
    # train
    data_val = args["data_val"]
    trainer.train(model, data_train, dev_data=data_val)

    print('')

def infer(**args):
    """Inference using model.
    """

    print("Predicting...")

    # define model
    with open('hyper.pkl','rb') as f:
        final_grid = pickle.load(f)
    model = get_model_setting(**final_grid)
    load_model(model, args['model_path'])

    # define predictor
    predictor = Predictor(batch_size=128, use_cuda=True)

    # predict
    data_test = args["data_test"]
    y_pred,y_true = predictor.predict(model, data_test)
    
    np.save('prediction.npy',y_pred.cpu())
    np.save('ground_truth.npy',y_true.cpu())
    
    return
    

# code starts here
if __name__ == "__main__":
    # setting up seeds
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # set up global variable
    GLOBAL_BEST_LOSS = 1e11
    
    start = time.time()
    args_dict = pre(seed)
    end = time.time()
    period = end-start
    print("preprocessing took {} hour {} min {} sec".format(period//3600,(period%3600)//60,int(period%60)))
    
    gridSearch = {
            "out_size1": [2048],
            "out_size2": [1536],
            "lstm_drop": [0.5],
            "drop1": [0.0],
            "drop2": [0.05],
            "drop3": [0.05*i for i in range(11)],
            }
    start = time.time()
    for grid in [dict(zip(gridSearch.keys(),v)) for v in itertools.product(*gridSearch.values())]:
        new_args = {**args_dict,**grid}
        train(**new_args)
    end = time.time()
    period = end-start
    print("training took {} hour {} min {} sec".format(period//3600,(period%3600)//60,int(period%60)))
    
    start = time.time()
    infer(**args_dict)
    end = time.time()
    period = end-start
    print("predicting took {} hour {} min {} sec".format(period//3600,(period%3600)//60,int(period%60)))
    
