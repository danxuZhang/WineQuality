#!/usr/bin/env python3
""" SGD.py: Stochastic Gradient Descent for Wine Quality dataset """

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.model_selection import train_test_split

__author__ = "Zhang Danxu"
__date__ = "16/04/2022"
__email__ = "dzhang022@e.ntu.edu.sg"
__credits__ = ["Zhang Danxu", "Lohia Vardhan", "Sannabhadti Shikha Deepak"]
__copyright__ = "Copyright 2022, NTU SC1015 - Wine Quality Mini Project"
__license__ = "MIT"
__status__ = "Development"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(file_path) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """ Load data from csv files and return tensors for training and testing
    :param file_path: file path of predictors and response csv files
    :return: tuple of 4 tensors
    """
    # check if data files exist
    try:
        f1 = open(file_path+"predictors.csv")
        f2 = open(file_path+"response.csv")
    except:
        raise FileNotFoundError
    else:
        f1.close()
        f2.close()

    # load csv files to dataframes
    predictors = pd.read_csv(file_path + 'predictors.csv')
    response = pd.read_csv(file_path + 'response.csv')

    variables = list(predictors.columns.values)

    # split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(predictors, response, test_size=0.2, random_state=1234)

    # convert dataframe to tensor
    X_train = torch.tensor(X_train[variables].values, dtype=torch.float32)
    X_test = torch.tensor(X_test[variables].values, dtype=torch.float32)
    y_train = torch.tensor(y_train['quality'].values, dtype=torch.float32)
    y_test = torch.tensor(y_test['quality'].values, dtype=torch.float32)

    # convert y_train and y_test from 1d tensor to 2d tensor
    y_train = y_train.view([y_train.shape[0], 1])
    y_test = y_test.view([y_test.shape[0], 1])

    return X_train, X_test, y_train, y_test


class SGDClassifier(nn.Module):
    def __init__(self, n):
        """ Constructor function
        :param n: number of features
        """
        super(SGDClassifier, self).__init__()
        self.linear = nn.Linear(n, 1).to(device)

    def forward(self, X) -> torch.tensor:
        """ Forward Processing
        :param X: predictor tensors
        :return: predicted response tensor
        """
        y_pred = torch.sigmoid(self.linear(X))
        return y_pred


def do_sgd(X_train, X_test, y_train, y_test, plot=True) -> float:
    """
    Do Stochastic Gradient Descent
    :param X_train: training data for predictors
    :param X_test:testing data for predictors
    :param y_train: training data for response
    :param y_test: testing data for response
    :param plot: plot learning curve of loss function or not
    :return: accuracy of model
    """
    # m samples, n features for training data
    m, n = X_train.shape
    # build model
    model = SGDClassifier(n).to(device)

    # construct parameters
    learning_rate = 1e-4
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    n_epochs = int(1e6)
    history = list()

    # training loop
    for epoch in range(n_epochs):
        # forward passing
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        # backward passing
        loss.backward()
        # update weights
        optimizer.step()
        optimizer.zero_grad()

        if plot:
            history.append(loss.item())

        if epoch % 10000 == 0:
            print(f"epoch: {epoch}, loss = {loss.item():.4f}")

    # plot learning curve
    with torch.no_grad():
        if plot:
            dummy = np.arange(n_epochs)
            plt.plot(dummy, history)
            plt.show()

    # evaluate train result
    with torch.no_grad():
        y_train_pred = model(X_train)
        y_train_pred_cls = y_train_pred.round()
        train_accu = (y_train_pred_cls.eq(y_train)).sum() / y_train.shape[0]
        print(f"Train Accuracy: {train_accu:.4f}.")

    # evaluate test result
    with torch.no_grad():
        y_test_pred = model(X_test)
        y_test_pred_cls = y_test_pred.round()
        test_accu = (y_test_pred_cls.eq(y_test)).sum() / y_test.shape[0]
        print(f"Test Accuracy: {test_accu:.4f}.")

    return test_accu


if __name__ == '__main__':
    path = 'data/'

    try:
        X_train, X_test, y_train, y_test = load_data(path)
    except FileNotFoundError:
        print("Files cannot be found.")
    else:
        do_sgd(X_train, X_test, y_train, y_test)

