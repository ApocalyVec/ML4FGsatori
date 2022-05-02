# Basic python and data processing imports
import math
import pickle

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from AEClassification.AE import AE
from AEClassification.EPIDataset import EPIDataset
from AEClassification.BetaVAESolver import Solver

np.set_printoptions(suppress=True)  # Suppress scientific notation when printing small
import h5py


# import matplotlib.pyplot as plt
from datetime import datetime


model = 'AE'
# model = 'BetaVAE'
# cell_lines = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
cell_lines = ['HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']

# Model training parameters
epochs = 3
batch_size = 100
train_ratio = 0.9  # fraction of data to use for training

t = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
# opt = Adam(lr=1e-5)  # opt = RMSprop(lr = 1e-6)

data_path = 'data/all_sequence_data.h5'
use_cuda = True
device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
lr = 1e-3

training_histories = {}
for cell_line in cell_lines:
    dataset = EPIDataset(data_path, cell_line, use_cuda=use_cuda, is_onehot_labels=True)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    train_losses = []
    val_losses = []
    best_loss = np.inf

    net = AE(3000).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()
    print('Training on {} samples, validating on {} samples'.format(len(train_data_loader.dataset),
                                                                    len(val_data_loader.dataset)))
    # try:
    for epoch in range(epochs):
        mini_batch_i = 0
        mini_batch_i_val = 0

        pbar = tqdm(total=math.ceil(len(train_data_loader.dataset) / train_data_loader.batch_size),
                    desc='Training AE Net')
        pbar.update(mini_batch_i)
        batch_losses_train = []
        net.train()
        for input_p, input_e, y in train_data_loader:
            mini_batch_i += 1
            pbar.update(1)

            x_recon = net(input_p)
            loss = criteria(input_p, x_recon)
            optim.zero_grad()
            loss.backward()
            optim.step()

            pbar.set_description('Training [{}] loss:{:.5f}'.format(mini_batch_i, loss.item()))
            batch_losses_train.append(loss.item())
        train_losses.append(np.mean(batch_losses_train))
        pbar.close()

        net.eval()
        with torch.no_grad():
            pbar = tqdm(total=math.ceil(len(val_data_loader.dataset) / val_data_loader.batch_size),
                        desc='Validating AE Net')
            pbar.update(mini_batch_i_val)
            batch_losses_val = []
            for input_p, input_e, y in val_data_loader:
                mini_batch_i_val += 1
                pbar.update(1)

                x_recon = net(input_p)
                loss = criteria(input_p, x_recon)
                batch_losses_val.append(loss.item())
                pbar.set_description('Validating [{}] loss:{:.5f}'.format(mini_batch_i, loss.item()))

            val_losses.append(np.mean(batch_losses_val))
            pbar.close()
        print("Epoch {} - train recon loss:{:.5f}, , val recon loss:{:.5f}".format(epoch, np.mean(
            batch_losses_train), np.mean(batch_losses_val)))

        if np.mean(batch_losses_val) < best_loss:
            torch.save(net.state_dict(), 'AEClassification/models/net_AE_{}'.format(cell_line))
            print('Best loss improved from {} to {}, saved best model to {}'.format(best_loss, np.mean(batch_losses_val),
                                                                                  'AEClassification/models/net_AE_{}'.format(
                                                                                      cell_line)))
            best_loss = np.mean(batch_losses_val)

        # Save training histories after every epoch
        training_histories[cell_line] = {'train_losss': train_losses,'val_losses': val_losses}
        pickle.dump(training_histories, open('AEClassification/models/AE_training_histories.pickle', 'wb'))

    # except:
    #     print('Training terminated for cell line {} because of exception'.format(cell_line))
    print('Training completed for cell line {}, training history saved'.format(cell_line))




