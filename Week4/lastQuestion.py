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


# my_train = torch.utils.data.Subset(mnist_train, np.where(mnist_train.targets <= 1)[0]) my_test = torch.utils.data.Subset(mnist_test, np.where(mnist_test.targets <= 1)[0])

trans = torchvision.transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="./", train=True, transform=trans, target_transform=None,
                                                download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="./", train=False, transform=trans, target_transform=None,
                                               download=True)


def trousers_vs_t_shirts():
    output_metrics = dict()
    np.random.seed(0)
    torch.manual_seed(0)

    ## write your code here (remember to: 1. create a network with a sigmoid output layer. 2. fill in the correct values in the output_metrics dictionary)

    x_train = mnist_train.train_data[(mnist_train.targets == 0) | (mnist_train.targets == 1)]
    y_train = mnist_train.targets[(mnist_train.targets == 0) | (mnist_train.targets == 1)]
    x_test = mnist_test.test_data[(mnist_test.targets == 0) | (mnist_test.targets == 1)]
    y_test = mnist_test.targets[(mnist_test.targets == 0) | (mnist_test.targets == 1)]


    """
    def data_iter(batch_size, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        # The examples are read at random, in no particular order
        random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            j = torch.tensor(indices[i: min(i + batch_size, num_examples)])
            yield features[j], labels[j]
            # The “take” function will then return the corresponding element based
            # on the indices
            """

    def get_confusion_matrix(y_actual, y_predicted):
        indices_true = torch.eq(y_actual, y_predicted)
        indices_false = ~torch.eq(y_actual, y_predicted)
        # lets say TP == 11, TN == 10, FP = 01 , FN = 00
        TP = (y_actual[indices_true] == 1).sum().item()
        FN = (y_actual[indices_true] == 0).sum().item()
        FP = (y_actual[indices_false] == 0).sum().item()
        TN = (y_actual[indices_false] == 1).sum().item()
        return TP, FN, FP, TN

    def fill_metrics(output_metrics, y_actual, y_predicted):
        TP, FN, FP, TN = get_confusion_matrix(y_actual.unsqueeze(dim=1), torch.round(y_predicted))

        output_metrics["true_positive_rate"] = TP/(TP+FN)
        output_metrics["false_positive_rate"] = FP/(FP+TN)
        output_metrics["true_negative_rate"] = 1-output_metrics["false_positive_rate"]
        output_metrics["false_negative_rate"] = 1-output_metrics["true_positive_rate"]
        # output_metrics["precision"] = TP/(TP+FP)  # can not devide by zero
        output_metrics["recall"] = output_metrics["true_positive_rate"]  # same as TPR
        output_metrics["sensitivity"] = output_metrics["recall"]  # same as Recall
        output_metrics["specificity"] = output_metrics["true_negative_rate"]  # same as TNR
        output_metrics["f1_score"] = (2*TP)/(2*TP+FP+FN)
        output_metrics["accuracy"] = (TP+TN)/(TP+TN+FP+FN)
        return output_metrics


    class CustomDataset(Dataset):

        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data

        def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]

        def __len__(self):
            return len(self.X_data)

    train_data = CustomDataset(x_train.type(torch.FloatTensor), y_train.type(torch.FloatTensor))
    train_iter = DataLoader(train_data, batch_size, shuffle=True)
    test_data = CustomDataset(x_test.type(torch.FloatTensor), y_test.type(torch.FloatTensor))
    test_iter = DataLoader(test_data, batch_size, shuffle=True)

    #for X,y in train_iter:
        #print("A")

    class Reshape(torch.nn.Module):
        def forward(self, x):
            return x.view(-1, 784)

    net = nn.Sequential(Reshape(), nn.Linear(784,1))
    loss = nn.BCELoss()

    sigmoid = torch.sigmoid()

    num_epochs = 5
    lr = 8
    trainer = torch.optim.SGD(net.parameters(), lr)

    for epoch in range(num_epochs):
        train_loss_epoch = 0.0
        net.train()
        for X, y in train_iter:
            y_hat = torch.sigmoid(net(X.type(torch.FloatTensor)))
            l = loss(y_hat, y.unsqueeze(dim=1).float())
            train_loss_epoch += l
            trainer.zero_grad()
            l.backward()
            trainer.step()
        train_loss_epoch
        net.eval()
        with torch.no_grad():
            test_loss_epoch = 0.0
            y_predicted = torch.tensor([])
            for X_test, y_test1 in test_iter:
                y_hat = torch.sigmoid(net(X_test.type(torch.FloatTensor)))
                b_l = loss(y_hat, y_test1.unsqueeze(dim=1).float())
                y_predicted = torch.cat([y_predicted, y_hat])
                test_loss_epoch += b_l
            test_loss_epoch
            output_metrics = fill_metrics(output_metrics, y_test, y_predicted)
        print('epoch {}, train loss {}, test loss {}'.format(epoch + 1, train_loss_epoch, test_loss_epoch))

    ## end of function
    return output_metrics


print(trousers_vs_t_shirts())
