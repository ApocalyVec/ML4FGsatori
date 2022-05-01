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


model = 'AE'
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
    torch.save(solver.net_0.state_dict(), 'AEClassification/models/net_0_{}'.format(cell_line))
    torch.save(solver.net_1.state_dict(), 'AEClassification/models/net_1_{}'.format(cell_line))

    break
    # model = bm.build_model(use_JASPAR=False)
    #
    # model.compile(loss='binary_crossentropy',
    #               optimizer=opt,
    #               metrics=["accuracy"])
    #
    # model.summary()

    # # Define AEClassification callback that prints/plots performance at end of each epoch
    # class ConfusionMatrix(Callback):
    #     def on_train_begin(self, logs={}):
    #         self.epoch = 0
    #         self.precisions = []
    #         self.recalls = []
    #         self.f1_scores = []
    #         self.losses = []
    #         self.training_losses = []
    #         self.training_accs = []
    #         self.accs = []
    #         plt.ion()
    #
    #     def on_epoch_end(self, batch, logs={}):
    #         self.training_losses.append(logs.get('loss'))
    #         self.training_accs.append(logs.get('acc'))
    #         self.epoch += 1
    #         val_predict = model.predict_classes([X_enhancers, X_promoters], batch_size=batch_size, verbose=0)
    #         util.print_live(self, labels, val_predict, logs)
    #         if self.epoch > 1:  # need at least two time points to plot
    #             util.plot_live(self)
    #
    #
    # # print '\nlabels.mean(): ' + str(labels.mean())
    # print
    # 'Data sizes: '
    # print
    # '[X_enhancers, X_promoters]: [' + str(np.shape(X_enhancers)) + ', ' + str(np.shape(X_promoters)) + ']'
    # print
    # 'labels: ' + str(np.shape(labels))
    #
    # # Instantiate callbacks
    # confusionMatrix = ConfusionMatrix()
    # checkpoint_path = "/home/sss1/Desktop/projects/DeepInteractions/weights/test-delete-this-" + cell_line + "-basic-" + t + ".hdf5"
    # checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1)
    #
    # print
    # 'Running fully trainable model for exactly ' + str(num_epochs) + ' epochs...'
    # model.fit([X_enhancers, X_promoters],
    #           [labels],
    #           # validation_data = ([X_enhancer, X_promoter], y_val),
    #           batch_size=batch_size,
    #           nb_epoch=num_epochs,
    #           shuffle=True,
    #           callbacks=[confusionMatrix, checkpointer]
    #           )

