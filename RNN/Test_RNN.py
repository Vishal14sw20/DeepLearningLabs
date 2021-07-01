import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# data
prototype = \
np.array([ 8.4608798e-02,  1.8681801e-02,  1.3610494e-02,  5.4180953e-02,
        3.4678795e-03,  4.4038339e-02,  3.8967031e-02,  4.4038339e-02,
        6.9394876e-02,  1.2517926e-01,  1.8603495e-01,  2.4181933e-01,
        2.5196194e-01,  3.1281763e-01,  2.7731848e-01,  1.3025057e-01,
       -6.6747353e-03, -6.2459116e-02, -1.6388526e-01, -1.2838611e-01,
       -1.5881396e-01, -1.9938442e-01, -1.9938442e-01, -1.6388526e-01,
       -1.5881396e-01, -1.7402788e-01, -1.4360004e-01, -1.6388526e-01,
        1.1503665e-01,  1.6161436e+00,  3.8069485e+00,  1.9609925e+00,
       -3.0139600e+00, -6.3914507e+00, -5.4076171e+00, -3.4145933e+00,
       -1.7917750e+00, -7.6737084e-01, -4.2252195e-01, -2.5009749e-01,
       -1.6388526e-01, -8.2744347e-02, -8.2744347e-02, -4.2173887e-02,
        8.5391868e-03,  3.8967031e-02,  3.3895724e-02,  1.0996534e-01,
        1.4039318e-01,  2.8746109e-01,  3.7367332e-01,  5.2581254e-01,
        7.3373615e-01,  9.4165975e-01,  1.1647973e+00,  1.3524356e+00,
        1.4741470e+00,  1.5654305e+00,  1.5705019e+00,  1.4386479e+00,
        1.1445121e+00,  8.5037621e-01,  5.9173954e-01,  3.3310286e-01,
        2.1646279e-01,  1.0996534e-01,  4.9109646e-02, -1.6817350e-02,
       -2.6959965e-02, -6.7530424e-02, -1.1746043e-02, -2.1888657e-02,
       -2.6959965e-02,  1.8681801e-02,  8.5391868e-03,  2.8824417e-02,
        3.3895724e-02,  2.3753110e-02,  5.9252261e-02,  3.8967031e-02])

# create longer sequence
elongated = np.array([*prototype]*20)
elongated_noised = elongated + np.random.random(elongated.shape[0])

# splitting sequence
test_train_split = int(0.8*elongated_noised.shape[0])
train_raw = elongated_noised[:test_train_split]
test_raw = elongated_noised[test_train_split:]

# standardizing
train_seq = torch.FloatTensor((train_raw - train_raw.mean()) / train_raw.std())
test_seq = torch.FloatTensor((test_raw - train_raw.mean()) / train_raw.std())

# building train set
train_sequences = []
pattern_length = prototype.shape[0]
for i in range(train_seq.shape[0]-pattern_length):
    seq = train_seq[i:i+pattern_length]
    label = train_seq[i+pattern_length:i+pattern_length+1]
    train_sequences.append((seq, label))


# build model architecture
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

        self.hidden_state = self._init_hidden(1)

    def forward(self, input_seq):
        # input of shape (seq_len, batch, input_size)
        output, self.hidden_state = self.rnn(input_seq.view(len(input_seq), 1, -1), self.hidden_state)
        predictions = self.linear(output.view(len(input_seq), -1))
        return predictions[-1]

    def _init_hidden(self, batch_size):
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        return hidden


# training routine
epochs = 5
y_preds= []
labels1 = []
for i in range(epochs):
    for seq, labels in train_sequences:
        optimizer.zero_grad()
        rnn_model.hidden_state =  torch.zeros(1, 1, rnn_model.hidden_size)
        y_pred = rnn_model(seq)
        y_preds.append(y_pred)
        labels1.append(labels)

        loss = loss_function(y_pred,labels)*1000000
        loss.backward()
        optimizer.step()

    if i%1 == 0:
        print(f'epoch: {i:3} - loss: {loss.item():10.8f}')

print(f'epoch: {i:3} - loss: {loss.item():10.8f}')