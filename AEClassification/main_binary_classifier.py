import random

import torch
import torch as torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from AEClassification.BetaVAE import BetaVAE_EP
from AEClassification.EPIDataset import EPIDataset

data_path = 'data/all_sequence_data.h5'
device = 'cuda'
use_cuda = True
model_path = 'AEClassification/models/net_BetaVAE_0_GM12878'
cell_line  = 'GM12878'
z_dim = 10

dataset = EPIDataset(data_path, cell_line, use_cuda=use_cuda)
data_loader = DataLoader(dataset, batch_size=512, shuffle=True)
vae_model = BetaVAE_EP(z_dim=10, input_length=3000).to(device)
vae_model.load_state_dict(torch.load(model_path))
vae_model.eval() # set the model to eval mode

rand_idx = random.randint(1, len(dataset) - 1)
random_sample = data_loader.dataset.__getitem__(rand_idx)[0].unsqueeze(0)
random_sample_z = vae_model.encoder(random_sample)[:, :z_dim]
recon_sample = torch.sigmoid(vae_model.decoder(random_sample_z))

plt.imshow(random_sample.squeeze(0).detach().cpu().numpy()[:, :10])
plt.imshow(recon_sample.squeeze(0).detach().cpu().numpy()[:, :10], vmin=0, vmax=1)
plt.show()