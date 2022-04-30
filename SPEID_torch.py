import torch
from torch import nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

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
        self.lstm_dropout = 0.4

        self.SingleHeadSize = 32
        self.numMultiHeads = 8
        self.MultiHeadSize = 100


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
                                nn.LSTM(self.n_kernels, self.RNN_hiddenSize, num_layers=self.n_rnn_layers, bidirectional=True),
                                nn.Dropout(p=self.lstm_dropout)
                            )
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


    def forward(self, input1, input2):




        ## Now Merge the two layers
        merge = torch.cat([layer_1, layer_2], dim=1)



        output = self.layer1(input)
        output = self.dropout1(output)
        output = output.permute(0,2,1)

        if self.useRNN:
            output, _ = self.RNN(output)
            F_RNN = output[:,:,:self.RNN_hiddenSize]
            R_RNN = output[:,:,self.RNN_hiddenSize:]
            output = torch.cat((F_RNN,R_RNN),2)
            output = self.dropoutRNN(output)

        pAttn_concat = torch.Tensor([]).to(self.device)
        attn_concat = torch.Tensor([]).to(self.device)
        for i in range(0,self.numMultiHeads):
            query, key, value = self.Q[i](output), self.K[i](output), self.V[i](output)
            attnOut,p_attn = self.attention(query, key, value, dropout=0.2)
            attnOut = self.RELU[i](attnOut)
            if self.usepooling:
                attnOut = self.MAXPOOL[i](attnOut.permute(0,2,1)).permute(0,2,1)
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