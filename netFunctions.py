import numpy as np

import copy
import torch
import torch.optim as optim
from torch import nn

# Network
class autoEncoder (nn.Module):

    def __init__ (self, N, p): 

        super(autoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 1,  out_channels = 16, kernel_size = 3), 
            nn.ReLU(),
            nn.Dropout(p),
            nn.MaxPool2d(kernel_size = 2, stride = 2),  
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3),  
            nn.ReLU(),
            nn.Dropout(p),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, N),
            nn.ReLU(),
            nn.Dropout(p)  
        )

        self.decoder = nn.Sequential(
            nn.Linear(N, 32 * 5 * 5),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Unflatten(1, (32, 5, 5)),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 4, stride = 2, output_padding = 1),
            nn.ReLU(),
            nn.Dropout(p),
            nn.ConvTranspose2d(in_channels = 16, out_channels = 1,  kernel_size = 3, stride = 2, output_padding = 1),
            nn.Sigmoid()
        )
    
    def forward (self, x):

        x = self.encoder(x)
        x = self.decoder(x)
  
        return x
  
class classifier (nn.Module):

    def __init__ (self, N, p, encoder):

        super(classifier, self).__init__()

        if encoder is None:

            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels = 1,  out_channels = 16, kernel_size = 3), 
                nn.ReLU(),
                nn.Dropout(p),
                nn.MaxPool2d(kernel_size = 2, stride = 2),  
                nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3),  
                nn.ReLU(),
                nn.Dropout(p),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                nn.Flatten(),
                nn.Linear(32 * 5 * 5, N),
                nn.ReLU(),
                nn.Dropout(p)                  
            )

        else:

            self.encoder = copy.deepcopy(encoder)

        self.classifier = nn.Sequential(
            nn.Linear(N, 10),
            nn.ReLU(),
            nn.Dropout(p)
        )

    def forward (self, x):

        x = self.encoder(x)
        x = self.classifier(x)

        return x

# Training routine
def train_loop (dataloader, model, loss_fn, optimizer, net_type):

    # Se activa el entrenamiento
    model.train()

    # Parámetros
    num_batches = len(dataloader)
    sum_loss    = 0.0
    avg_loss    = 0.0

    match net_type:

        case 'autoencoder':

            # Iteracion sobre cada batch
            for batch, (X, Y) in enumerate(dataloader):
                
                # Predicción del Modelo y la función de pérdida
                X_pred = model (X)
                loss   = loss_fn (X_pred, X)

                # Backpropagation
                optimizer.zero_grad ()
                loss.backward ()
                optimizer.step ()

                # Error acumulado
                sum_loss = sum_loss + loss.item()

        case 'classifier':

            # Iteracion sobre cada batch
            for batch, (X, Y) in enumerate(dataloader):
                
                # Predicción del Modelo y la función de pérdida
                Y_pred = model (X)
                loss   = loss_fn (Y_pred, Y)
                
                # Backpropagation
                optimizer.zero_grad ()
                loss.backward ()
                optimizer.step ()

                # Error acumulado
                sum_loss = sum_loss + loss.item()

    # Error promedio a medida que se entrena la red
    avg_loss = sum_loss / num_batches
    print(f"Average training loss: {avg_loss}")

    return avg_loss

# Validation routine
def valid_loop (dataloader, model, loss_fn, net_type):

    # Se desactiva el entrenamiento sobre el modelo
    model.eval ()

    # Parametros
    num_batches = len(dataloader)
    size        = len(dataloader.dataset)
    sum_loss    = 0.0
    sum_correct = 0.0
    precision   = 0.0

    # Se desactiva el cálculo de gradientes
    with torch.no_grad ():

        match net_type:

            case 'autoencoder':

                # Iteracion sobre el conjunto de validacion
                    for batch, (X, Y) in enumerate(dataloader):

                        # Calculo de las predicciones del modelo ya entrenado
                        X_pred = model (X)
                        loss   = loss_fn (X_pred, X) 

                        # Calculo de los errores
                        sum_loss = sum_loss + loss.item()

            case 'classifier':

                # Iteracion sobre el conjunto de validacion
                for batch, (X, Y) in enumerate(dataloader):

                    # Calculo de las predicciones del modelo ya entrenado
                    Y_pred = model (X)
                    loss   = loss_fn (Y_pred, Y) 

                    # Calculo de los errores
                    sum_loss = sum_loss + loss.item()

    # Error promedio a medida que se entrena la red
    avg_loss = sum_loss / num_batches

    return avg_loss
            
# Optimizer selection
def selectOptimizer (opt_type, eta, gamma, parameters):
    
    match opt_type:
    
        case 'SGD':
            # Optimizador: Stochastic Gradient Descent
            optimizer = torch.optim.SGD(parameters, lr = eta, momentum = gamma)

        case 'Adam':
            # Optimizador: Adam method
            optimizer = torch.optim.Adam(parameters, lr = eta, eps = 1e-08, weight_decay = 0.001, amsgrad = False)

    print(f"Used Optimizer: {opt_type}")

    return optimizer

# Loss function selection
def selectLossFunction (func_name):
    
    match func_name:
    
        case 'MSE':
            lossFunction = nn.MSELoss()

        case 'CrossEntropy':
            lossFunction = nn.CrossEntropyLoss()

    print(f"Loss function: {func_name}")

    return lossFunction

# Loop over training epoch
def epoch_loop (num_epochs, train_loader, valid_loader, network, parameters, loss_fn, net_type, train_loop, valid_loop, opt, learning_rate, momentum):

    train_avg_loss    = []
    train_avg_loss_in = []
    valid_avg_loss    = []
    train_precision   = []
    valid_precision   = []

    # Selección del optimizador
    optimizer = selectOptimizer (opt, learning_rate, momentum, parameters)
    #scheduler = torch.optim.lr_scheduler.CyclicLR (optimizer, base_lr = 1e-6, max_lr = 1e0, step_size_up = 10, step_size_down = 10, mode = 'triangular', gamma = 0.1, cycle_momentum = False)

    for epoch in range(num_epochs):
            
        print (f"Epoch: {epoch}")

        train_loss_in = train_loop (train_loader, network, loss_fn, optimizer, net_type)
        train_loss    = valid_loop (train_loader, network, loss_fn, net_type)
        valid_loss    = valid_loop (valid_loader, network, loss_fn, net_type)

        train_avg_loss_in.append(train_loss_in)
        train_avg_loss.append(train_loss)
        valid_avg_loss.append(valid_loss)

        #scheduler.step()

    print ("Done!")
    return train_avg_loss_in, train_avg_loss, valid_avg_loss, train_precision, valid_precision



    # def __init__ (self, N, p): 

    #     super(autoEncoder, self).__init__()

    #     self.encoder = nn.Sequential(

    #         nn.Conv2d(in_channels = 1,  out_channels = 16, kernel_size = 3), 
    #         nn.ReLU(),
    #         nn.Dropout(p),
    #         nn.MaxPool2d(kernel_size = 2), 

    #         nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3), 
    #         nn.ReLU(),
    #         nn.Dropout(p),
    #         nn.MaxPool2d(kernel_size = 2) 
    #     )

    #     self.linear_encoder = nn.Sequential(

    #         nn.Flatten(),
    #         nn.Linear(32 * 5 * 5, N),
    #         nn.ReLU(),
    #         nn.Dropout(p)
    #     )

    #     self.linear_decoder = nn.Sequential(

    #         nn.Linear(N, 32 * 5 * 5),
    #         nn.ReLU(),
    #         nn.Dropout(p),
    #         nn.Unflatten(1, (32, 5, 5))
    #     )

    #     self.decoder = nn.Sequential(

    #         nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 4, stride = 2, output_padding = 1),
    #         nn.ReLU(),
    #         nn.Dropout(p),
    #         nn.ConvTranspose2d(in_channels = 16, out_channels = 1,  kernel_size = 3, stride = 2, output_padding = 1),
    #         nn.Sigmoid()
    #     )
    
    # def forward (self, x):
                
    #     x = self.encoder(x)
    #     x = self.linear_encoder(x)
    #     x = self.linear_decoder(x)
    #     x = self.decoder(x)
 
    #     return x