import os
import subprocess

import numpy as np
np.random.seed(123)

import torch
torch.manual_seed(123)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


MODEL_PATH  = "../model/age_est_model.pt"
DATAPATH    = "../data/"

class twoHiddenNet(nn.Module):

    def __init__(self, input_unit_size=512, hidden_layer1_size=256, hidden_layer2_size=128, output_layer_size=1):
        super(twoHiddenNet, self).__init__()
        self.layer1    = nn.Linear(input_unit_size, hidden_layer1_size)
        nn.init.xavier_uniform(self.layer1.weight)

        self.layer2    = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        nn.init.xavier_uniform(self.layer2.weight)

        self.layer3    = nn.Linear(hidden_layer2_size, output_layer_size)
        nn.init.xavier_uniform(self.layer3.weight)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def main():

    train_np    = np.load(DATAPATH + 'train.npy')
    train_gt_np = np.load(DATAPATH + 'train_gt.npy')

    valid_np    = np.load(DATAPATH + 'valid.npy')
    valid_gt_np = np.load(DATAPATH + 'valid_gt.npy')

    print("----------------------------------------------------")
    print("Number of training samples   : %d" % len(train_np))
    print("Number of validation samples : %d" % len(valid_np))
    print("----------------------------------------------------")
    print()

    EPOCHS  = 1000
    LR      = 1e-4
    REG     = 1e-4

    model     = twoHiddenNet(input_unit_size    = 512,  \
                             hidden_layer1_size = 256,  \
                             hidden_layer2_size = 128,  \
                             output_layer_size  = 1
                             )

    loss_fn   = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop( model.parameters(), lr=LR, weight_decay=REG)

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        x_train, y_train = torch.autograd.Variable(torch.from_numpy(train_np).float(), requires_grad=False),\
                           torch.autograd.Variable(torch.from_numpy(train_gt_np).float(), requires_grad=False)

        x_valid, y_valid = torch.autograd.Variable(torch.from_numpy(valid_np).float(), requires_grad=False),\
                           torch.autograd.Variable(torch.from_numpy(valid_gt_np).float(), requires_grad=False)

        train_output = model(x_train)
        valid_output = model(x_valid)

        train_loss = loss_fn(train_output, y_train)
        valid_loss = loss_fn(valid_output, y_valid)

        train_loss.backward()
        optimizer.step()

        if (epoch+1)%50 == 0:
            print("----------------------------------------------------")
            print("%dth epoch" % int(epoch+1))
            print()

            train_pred = model(x_train)
            train_pred = train_pred.data.numpy()
            np.save('estimations.npy', train_pred)
            print("train accuracy")
            subprocess.call(["python3", "utils/evaluate.py", "estimations.npy", DATAPATH+"train_gt.npy"])
            print("train loss")
            print(str(format(train_loss.data[0], '.8g')))
            print()

            valid_pred = model(x_valid)
            valid_pred = valid_pred.data.numpy()
            np.save('estimations.npy', valid_pred)
            print("valid acc")
            subprocess.call(["python3", "utils/evaluate.py", "estimations.npy", DATAPATH+"valid_gt.npy"])
            print("valid loss")
            print(str(format(valid_loss.data[0], '.8g')))
            print("----------------------------------------------------")
            print()

    try:
        os.remove('estimations.npy')
    except:
        pass

    torch.save(model.state_dict(), MODEL_PATH)
    print("-> age estimation model is saved to %s" % MODEL_PATH)
    return

if __name__ == '__main__':
    main()
