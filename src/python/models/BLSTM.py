"""
Created on Jan 11, 2018

@author: Siyuan Qi

Description of the file.

"""

import torch
import torch.nn
import torch.autograd


class BLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=False, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, features):
        # Set initial states
        h0 = torch.autograd.Variable(torch.zeros(self.num_layers * 2, features.size(1), self.hidden_size).cuda())  # 2 for bidirection
        c0 = torch.autograd.Variable(torch.zeros(self.num_layers * 2, features.size(1), self.hidden_size).cuda())

        # Forward propagate RNN
        out, _ = self.lstm(features, (h0, c0))
        out = self.fc(out)
        return out


def main():
    pass


if __name__ == '__main__':
    main()
