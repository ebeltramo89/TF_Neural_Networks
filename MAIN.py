import netFunctions
import dataSetFunctions
import auxFunctions

import torch
import numpy as np
from torch import nn


# Network type, Optimizer and Loss Function
optimizer = 'Adam'
lossFuncName = 'CrossEntropy'
net_type = 'classifier'

# Training parameters
momentum = 0.1
num_epochs = 30
learning_rate = 0.001
batch_size_train = 100
batch_size_valid = 10000

# Network parameters
N = 64
p = 0.2

#-----------------------------------------------------------------------
 
# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device used: {device}")

# Loss function
loss_fn = netFunctions.selectLossFunction(lossFuncName)

# Creation of Data Loaders
labels_names, train_set, valid_set = dataSetFunctions.imageSet (device)
train_loader, valid_loader = dataSetFunctions.createDataLoaders (batch_size_train, batch_size_valid, train_set, valid_set)

# Network 
#myNetwork = netFunctions.autoEncoder(N, p) 
myNetwork = netFunctions.classifier(N, p, encoder = None) 
myNetwork = myNetwork.to(device)

trainable_params = sum(p.numel() for p in myNetwork.parameters() if p.requires_grad)

print(myNetwork)
print('Number of trainable parameters:', trainable_params)

# Load parameters of trainned network
# myNetwork.load_state_dict(torch.load('Conv1D_50_Epocas'))

# Loops over each epoch
train_avg_loss_in, train_avg_loss, valid_avg_loss, train_precision, valid_precision = netFunctions.epoch_loop (num_epochs, train_loader, valid_loader, myNetwork, myNetwork.parameters(),
loss_fn, net_type, netFunctions.train_loop, netFunctions.valid_loop, optimizer, learning_rate, momentum)

# Save network parameters
#torch.save(myNetwork.state_dict(), 'Conv1D_50_Epocas')

# Graphs
match net_type:
    case 'autoencoder':
        auxFunctions.plotErrors (train_avg_loss_in, train_avg_loss, valid_avg_loss, num_epochs)
        auxFunctions.plotOutImag (valid_loader, device, myNetwork)
    case 'classifier':
        auxFunctions.plotErrors (train_avg_loss_in, train_avg_loss, valid_avg_loss, num_epochs)
        # auxFunctions.plotPrecision (train_precision, valid_precision, num_epochs)
        auxFunctions.evalNetwork (valid_loader, myNetwork, batch_size_valid)