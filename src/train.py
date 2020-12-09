import os
import glob
import time
from datetime import datetime
import random
import argparse
import numpy as np
from scipy.stats import t
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import make_batch, load_data
from models import FCGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=69, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

system_name = '10ptlgo' # input("Enter system name")
dimension = 3


# Load data (only for some information)
data_temp = load_data(system_name, 0)

# Model and optimizer
model = FCGAT(n_input_features=1 + 2*dimension,
              n_hidden_features=args.hidden,
              n_output_features=2*dimension,
              dropout=args.dropout,
              n_heads=args.nb_heads,
              alpha=args.alpha
              )

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

criterion = nn.MSELoss()

if args.cuda:
    model.cuda()
    # features = features.cuda()
    # adj = adj.cuda()
    # labels = labels.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()


def train(batch, epoch, epoch_total, log_dir):  # batch starts from 0
    # global variables : model, system_name
    t = time.time()
    model.train()
    optimizer.zero_grad()

    features = load_data(system_name, batch)
    if features == False:
        return False
    input_features_batch, target_features_batch = make_batch(features)
    input_features_batch = torch.FloatTensor(input_features_batch)
    target_features_batch = torch.FloatTensor(target_features_batch)
    if args.cuda:
        input_features_batch = input_features_batch.cuda()
        target_features_batch = target_features_batch.cuda()

    outputs = []

    for input_feature_minibatch in input_features_batch:
        output_minibatch = model(input_feature_minibatch)
        outputs.append(output_minibatch)

    output_batch = torch.stack(outputs)

    loss_train = F.mse_loss(output_batch, target_features_batch)

    loss_train.backward()
    optimizer.step()

    fd = open(log_dir,'a')
    current_log = "{0}\n{1}\n{2}\n{3}\n".format(epoch, batch, loss_train.data.item(), time.time() - t)
    fd.write(current_log)
    fd.close()

    print('{:6.3f}%'.format(epoch*100/epoch_total),
          ' | ',
          'Epoch: {:08d}'.format(epoch + 1),
          ' | ',
          'Batch: {:08d}'.format(batch + 1),
          ' | ',
          'loss_train: {:10.4f}'.format(loss_train.data.item()),
          ' | ',
          'time: {:7.4f}s'.format(time.time() - t),
          )

    return loss_train.data.item()


def compute_test():
    # global: model, system_name
    dir = "../data/" + system_name + "_eval"
    if not os.path.exists(dir):
        print("test cases do not exist.")
        return
    else:
        model.eval()

    loss_value = 0
    batch = 0
    while (True):
        loss_value_petit = train(batch)
        if not loss_value_petit:
            break
        else:
            loss_value += loss_value_petit
        batch += 1

    loss_values.append(loss_value)


# train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = 0
best_epoch = 0
now = datetime.today().strftime('%Y_%m_%d %H_%M_%S')
if not os.path.exists("../train log/{0}".format(system_name)):
    os.makedirs("../train log/{0}".format(system_name))
log_dir = "../train log/{1}/{0}.txt".format(now, system_name)

for epoch in range(args.epochs):
    loss_value = 0
    batch = 0
    while(True):
        loss_value_petit = train(batch, epoch, args.epochs, log_dir)
        if not loss_value_petit:
            break
        else:
            loss_value += loss_value_petit
        batch += 1

    loss_values.append(loss_value)

    if loss_values[-1] < best or best == 0:
        torch.save(model.state_dict(), '../model_save/{0}_{1}.pkl'.format(system_name, epoch))
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
        print("hit!")
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0].split('_')[1])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0].split('_')[1])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}min".format((time.time() - t_total)/60))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('../model_save/{0}_{1}.pkl'.format(system_name, best_epoch)))