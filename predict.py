from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

import numpy as np
import torch
from torch import nn
import random

import unicodedata
import string

def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# def string_vectorizer(strng, alphabet=string.ascii_letters + " .,;'"):
#     vector = [[0 if char != letter else 1 for char in alphabet] 
#                   for letter in strng]
#     return vector


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), n_letters)
    for i in range(len(line)):
        index = all_letters.find(line[i])
        tensor[i][index] = 1
    return tensor


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]
def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor



class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         
            input_size=57,
            hidden_size=128,         
            num_layers=1,           
            batch_first=True,       
        )

        self.out = nn.Linear(128, n_categories)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


# stochastic version    

# rnn = RNN()
# # rnn.cuda()
# optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
# loss_func = nn.CrossEntropyLoss() 

# for i in range(100000):
#     e = randomTrainingExample()
#     # output = rnn(e[3].cuda())
#     output = rnn(e[3].unsqueeze(0))
#     # loss = loss_func(output, e[2].cuda())  
#     loss = loss_func(output, e[2]) 
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if i % 1000 == 0:
#         print(i)

# for i in range(10):
#     t = randomTrainingExample()
#     test_output = rnn(t[3].unsqueeze(0))
#     pred_y = torch.max(test_output, 1)[1]
#     print(pred_y)
#     print(t[2])
#     print("")



# batch version

rnn = RNN()
rnn.cuda()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss() 

for epoch in range(100):
    max_length = 0
    el = []
    yl = []
    for i in range(1000):
        e = randomTrainingExample()
        el.append(e[3])
        yl.append(e[2])
        max_length = max(max_length, len(e[3]))

    for i in range(len(el)):
        pads = torch.zeros([max_length - len(el[i]), 57])
        el[i] = torch.cat((pads, el[i]))
    data = torch.stack(el)
    ys = torch.cat(yl)

    output = rnn(data.cuda())
    loss = loss_func(output, ys.cuda()) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(epoch)

for i in range(10):
    t = randomTrainingExample()
    test_output = rnn(t[3].unsqueeze(0).cuda())
    pred_y = torch.max(test_output, 1)[1]
    print(pred_y)
    print(t[2].cuda())
    print("")

# future work: https://www.kdnuggets.com/2018/06/taming-lstms-variable-sized-mini-batches-pytorch.html