# Basic python and data processing imports
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from AEClassification.EPIDataset import EPIDataset
from AEClassification.TrainBetaVAE import Solver

np.set_printoptions(suppress=True)  # Suppress scientific notation when printing small
import h5py


# import matplotlib.pyplot as plt
from datetime import datetime


# model = 'AE'
model = 'BetaVAE'
cell_lines = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']

# Model training parameters
num_epochs = 32
batch_size = 100
training_frac = 0.9  # fraction of data to use for training

t = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
# opt = Adam(lr=1e-5)  # opt = RMSprop(lr = 1e-6)

data_path = 'data/all_sequence_data.h5'
use_cuda = True
# for cell_line in cell_lines:
#     print('Loading ' + cell_line + ' data from ' + data_path)
#     X_enhancers = None
#     X_promoters = None
#     labels = None
#     with h5py.File(data_path, 'r') as hf:
#         X_enhancers = np.array(hf.get(cell_line + '_X_enhancers')).transpose((0, 2, 1))
#         X_promoters = np.array(hf.get(cell_line + '_X_promoters')).transpose((0, 2, 1))
#         labels = np.array(hf.get(cell_line + 'labels'))
#
#     print(
#         "Cell line {0} has {1} EP-pairs, number of positive samples is {2}, negative is {3}, percentage postive is {4}".format(
#             cell_line, len(X_enhancers), np.sum(labels == 1), np.sum(labels == 0), np.sum(labels == 1) / len(labels)))
training_histories = {}
for cell_line in cell_lines:
    dataset = EPIDataset(data_path, cell_line, use_cuda=use_cuda)
    data_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    solver = Solver(data_loader=data_loader, use_cuda=use_cuda, beta=4, lr=1e-3, z_dim=10, objective='H', model=model, max_iter=150)
    training_histories[cell_line] = solver.train()

    pickle.dump(training_histories, open('AEClassification/training_histories.pickle', 'wb'))
    torch.save(solver.net_0.state_dict(), 'AEClassification/models/net_{}_0_{}'.format(model, cell_line))
    torch.save(solver.net_1.state_dict(), 'AEClassification/models/net_{}_1_{}'.format(model, cell_line))

    break


