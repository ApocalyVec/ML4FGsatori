import h5py
import numpy as np
import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class AttentionNet(nn.Module):
    def __init__(self):
        super(AttentionNet,self).__init__()

        # model parameters
        self.input_channels = 4
        self.enhancer_length = 3000 #
        self.promoter_length = 2000 #
        self.n_kernels = 200 # Number of kernels; used to be 1024
        self.filter_length = 13 #SPEID says 40 # Length of each kernel
        self.cnn_pool_size = 6 # from SATORI
        self.LSTM_out_dim = 50 # Output direction of ONE DIRECTION of LSTM; used to be 512
        self.dense_layer_size = 800
        self.drop_out = 0.2

        self.RNN_hiddenSize = 100
        self.n_rnn_layers = 2
        self.lstm_dropout_p = 0.4

        self.SingleHeadSize = 32
        self.numMultiHeads = 8
        self.MultiHeadSize = 100
        self.genPAttn = True

        self.readout_strategy = 'normalize'

        self.batch_size=256 #batch size
        self.num_epochs=50 #number of epochs


        # Convolutional/maxpooling layers to extract prominent motifs
        # Separate identically initialized convolutional layers are trained for
        # enhancers and promoters
        # Define enhancer layers

        self.enhancer_conv_layer = nn.Sequential(
                        nn.Conv1d(in_channels=self.input_channels,out_channels=self.n_kernels, kernel_size = self.filter_length, padding = 0, bias= False),
                        nn.BatchNorm1d(num_features=self.n_kernels), # tends to help give faster convergence: https://arxiv.org/abs/1502.03167
                        # nn.Dropout2d(), # popular form of regularization: https://jmlr.org/papers/v15/srivastava14a.html
                        nn.ReLU(inplace=True),
                        nn.MaxPool1d(kernel_size=self.cnn_pool_size),
                        nn.Dropout(p=self.drop_out)
                    )

        self.promoter_conv_layer = nn.Sequential(
                        nn.Conv1d(in_channels=self.input_channels,out_channels=self.n_kernels, kernel_size=self.filter_length, padding = 0, bias= False),
                        nn.BatchNorm1d(num_features=self.n_kernels), # tends to help give faster convergence: https://arxiv.org/abs/1502.03167
                        # nn.Dropout2d(), # popular form of regularization: https://jmlr.org/papers/v15/srivastava14a.html
                        nn.ReLU(inplace=True),
                        nn.MaxPool1d(kernel_size=self.cnn_pool_size),
                        nn.Dropout(p=self.drop_out)
                    )
        self.lstm_layer = nn.Sequential(
                                nn.LSTM(self.n_kernels, self.RNN_hiddenSize, num_layers=self.n_rnn_layers, bidirectional=True)
                            )
        self.lstm_dropout = nn.Dropout(p=self.lstm_dropout_p)

        self.Q = nn.ModuleList(
            [nn.Linear(in_features=2 * self.RNN_hiddenSize, out_features=self.SingleHeadSize) for i in
             range(0, self.numMultiHeads)])
        self.K = nn.ModuleList(
            [nn.Linear(in_features=2 * self.RNN_hiddenSize, out_features=self.SingleHeadSize) for i in
             range(0, self.numMultiHeads)])
        self.V = nn.ModuleList(
            [nn.Linear(in_features=2 * self.RNN_hiddenSize, out_features=self.SingleHeadSize) for i in
             range(0, self.numMultiHeads)])

        self.RELU = nn.ModuleList([nn.ReLU() for i in range(0, self.numMultiHeads)])
        self.MultiHeadLinear = nn.Linear(in_features=self.SingleHeadSize * self.numMultiHeads,
                                         out_features=self.MultiHeadSize)  # 50
        self.MHReLU = nn.ReLU()

        self.fc3 = nn.Linear(in_features=self.MultiHeadSize, out_features=2)

    def attention(self, query, key, value, mask=None, dropout=0.0):
        #based on: https://nlp.seas.harvard.edu/2018/04/03/attention.html
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim = -1)
        p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.matmul(p_attn, value), p_attn


    def forward(self, input_p, input_e):

        #First we need to do CNN for each promoter and enhancer sequences
        p_output = self.enhancer_conv_layer(input_p)
        e_output = self.promoter_conv_layer(input_e)

        # Now Merge the two layers
        output = torch.cat([p_output, e_output], dim=1)

        #not sure why this, but the SATORI authors
        output = output.permute(0,2,1)

        output, _ = self.lstm_layer(output)
        F_RNN = output[:,:,:self.RNN_hiddenSize]
        R_RNN = output[:,:,self.RNN_hiddenSize:]
        output = torch.cat((F_RNN,R_RNN),2)
        output = self.lstm_dropout(output)

        pAttn_concat = torch.Tensor([]).to(self.device)
        attn_concat = torch.Tensor([]).to(self.device)
        for i in range(0,self.numMultiHeads):
            query, key, value = self.Q[i](output), self.K[i](output), self.V[i](output)
            attnOut,p_attn = self.attention(query, key, value, dropout=0.2)
            attnOut = self.RELU[i](attnOut)
            # if self.usepooling:
            #     attnOut = self.MAXPOOL[i](attnOut.permute(0,2,1)).permute(0,2,1)
            attn_concat = torch.cat((attn_concat,attnOut),dim=2)
            if self.genPAttn:
                pAttn_concat = torch.cat((pAttn_concat, p_attn), dim=2)

        output = self.MultiHeadLinear(attn_concat)
        output = self.MHReLU(output)

        if self.readout_strategy == 'normalize':
            output = output.sum(axis=1)
            output = (output-output.mean())/output.std()

        output = self.fc3(output)
        assert not torch.isnan(output).any()
        if self.genPAttn:
            return output,pAttn_concat
        else:
            return output


class EPIDataset(Dataset):
    def __init__(self, data_path, cell_line, use_cuda):
        with h5py.File(data_path, 'r') as hf:
            self.X_enhancers = np.array(hf.get(cell_line + '_X_enhancers')).transpose((0, 2, 1))
            self.X_promoters = np.array(hf.get(cell_line + '_X_promoters')).transpose((0, 2, 1))
            self.labels = np.array(hf.get(cell_line + 'labels'))
            print(
                "Cell line {0} has {1} EP-pairs, number of positive samples is {2}, negative is {3}, percentage postive is {4}".format(
                    cell_line, len(self.X_enhancers), np.sum(self.labels == 1), np.sum(self.labels == 0), np.sum(self.labels == 1) / len(self.labels)))
            self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'

            # change the data to channel first
            self.X_enhancers = np.moveaxis(self.X_enhancers, -1, 1)
            self.X_promoters = np.moveaxis(self.X_promoters, -1, 1)

            self.X_enhancers = torch.Tensor(self.X_enhancers).to(self.device)
            self.X_promoters = torch.Tensor(self.X_promoters).to(self.device)
            self.labels = torch.Tensor(self.labels).to(self.device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.X_enhancers[idx],  self.X_promoters[idx], self.labels[idx]

    def get_input_length(self):
        return self.X_enhancers.shape[2]


if __name__ == '__main__':

    use_cuda = True
    batch_size = 512

    cell_lines = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
    data_path = '/data/all_sequence_data.h5'

    for cell_line in cell_lines:
        dataset = EPIDataset(data_path, cell_line, use_cuda=use_cuda)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        solver = Solver(data_loader=data_loader, use_cuda=use_cuda, beta=4, lr=1e-3, z_dim=10, objective='H', model='H',
                        max_iter=1)
        a = solver.train()

        torch.save(solver.net_0.state_dict(), 'BetaVAEClassifier/models/net_0_{}'.format(cell_line))
        torch.save(solver.net_1.state_dict(), 'BetaVAEClassifier/models/net_1_{}'.format(cell_line))
