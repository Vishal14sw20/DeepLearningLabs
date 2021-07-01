import sys

sys.path.insert(0, '..')
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
from torch.distributions import normal
import random
from sklearn.metrics import confusion_matrix, classification_report

batch_size = 256

trans = torchvision.transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="./", train=True, transform=trans, target_transform=None,
                                                download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="./", train=False, transform=trans, target_transform=None,
                                               download=True)
num_workers = 4

train_iter = DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = DataLoader(mnist_test, batch_size, shuffle=False)


def trousers_vs_t_shirts():
    output_metrics = dict()
    np.random.seed(0)
    torch.manual_seed(0)

    ## write your code here (remember to: 1. create a network with a sigmoid output layer. 2. fill in the correct values in the output_metrics dictionary)




    def get_confusion_matrix(y_actual, y_predicted):
        indices_true = torch.eq(y_actual, y_predicted)
        indices_false = ~torch.eq(y_actual, y_predicted)
        # lets say TP == 11, TN == 00, FP = 01 , FN = 10
        TP = (y_actual[indices_true] == 1).sum().item()
        TN = (y_actual[indices_true] == 0).sum().item()
        FP = (y_actual[indices_false] == 0).sum().item()
        FN = (y_actual[indices_false] == 1).sum().item()
        return TP, FN, FP, TN

    def fill_metrics(output_metrics, y_actual, y_predicted):
        TP, FN, FP, TN = get_confusion_matrix(y_actual.unsqueeze(dim=1), torch.round(torch.sigmoid(y_predicted)))

        output_metrics["true_positive_rate"] = TP/(TP+FN)
        output_metrics["false_positive_rate"] = FP/(FP+TN)
        output_metrics["true_negative_rate"] = 1-output_metrics["false_positive_rate"]
        output_metrics["false_negative_rate"] = 1-output_metrics["true_positive_rate"]
        output_metrics["precision"] = TP/(TP+FP)  # can not devide by zero
        output_metrics["recall"] = output_metrics["true_positive_rate"]  # same as TPR
        output_metrics["sensitivity"] = output_metrics["recall"]  # same as Recall
        output_metrics["specificity"] = output_metrics["true_negative_rate"]  # same as TNR
        output_metrics["f1_score"] = (2*TP)/(2*TP+FP+FN)
        output_metrics["accuracy"] = (TP+TN)/(TP+TN+FP+FN)
        return output_metrics



    class Reshape(torch.nn.Module):
        def forward(self, x):
            return x.view(-1, 784)

    net = nn.Sequential(Reshape(), nn.Linear(784,1))
    loss = nn.BCEWithLogitsLoss()

    #sigmoid = nn.Sigmoid() # use torch sigmoid

    num_epochs = 5
    lr = 0.1
    trainer = torch.optim.SGD(net.parameters(), lr)

    for epoch in range(num_epochs):
        train_loss_epoch = 0.0
        net.train()
        for X, y in train_iter:
            indices_train = (y == 0) | (y == 1)
            X_train = X[indices_train]
            y_train = y[indices_train]
            y_hat = net(X_train.type(torch.FloatTensor))
            l = loss(y_hat, y_train.unsqueeze(dim=1).float())
            train_loss_epoch += l
            trainer.zero_grad()
            l.backward()
            trainer.step()
        train_loss_epoch
        net.eval()
        with torch.no_grad():
            test_loss_epoch = 0.0
            y_predicted = torch.tensor([])
            y_test_actual = torch.tensor([])
            for X_test, y_test1 in test_iter:
                indices_train = (y_test1 == 0) | (y_test1 == 1)
                X_test = X_test[indices_train]
                y_test1 = y_test1[indices_train]
                y_hat = net(X_test.type(torch.FloatTensor))
                b_l = loss(y_hat, y_test1.unsqueeze(dim=1).float())
                y_predicted = torch.cat([y_predicted, y_hat])
                y_test_actual = torch.cat([y_test_actual.long(), y_test1])
                test_loss_epoch += b_l
            test_loss_epoch
            output_metrics = fill_metrics(output_metrics, y_test_actual, y_predicted)
        print('epoch {}, train loss {}, test loss {}'.format(epoch + 1, train_loss_epoch, test_loss_epoch))

    ## end of function
    return output_metrics


print(trousers_vs_t_shirts())