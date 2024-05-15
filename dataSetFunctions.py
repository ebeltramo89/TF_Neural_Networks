import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose

# Definicion clase para personalizar el conjunto de datos
class customDataset (Dataset):

    def __init__ (self, dataset):
        self.dataset = dataset

    def __len__ (self):
        return len(self.dataset)
    
    def __getitem__ (self, i):
        image, label  = self.dataset [i]
        return image, label 

def imageSet (device):

    # Transform to apply to data
    transform = transforms.Compose([transforms.ToTensor()])
    # transforms.Normalize((0.5,),(0.5,),),  , Lambda(lambda x: x.to(device))

    # Download and load the training data (Construye tensores con las imagenes)
    train_set_original = datasets.FashionMNIST('MNIST_data/', download = True, train = True,  transform = transform)
    valid_set_original = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)
 
    print(f"Lenght of train_set_original: {len(train_set_original)}, Lenght of valid_set_original {len(valid_set_original)}")
  
    # Labels
    labels_names = {

        0: "T-Shirt",    # Remera manga corta
        1: "Trouser",    # Pantalon
        2: "Pullover",   # Buzo
        3: "Dress",      # Vestido
        4: "Coat",       # Abrigo
        5: "Sandal",     # Sandalia
        6: "Shirt",      # Remera manga larga
        7: "Sneaker",    # Zapatilla
        8: "Bag",        # Bolso
        9: "Ankle Boot", # Bota

    }

    # Dataset customized (Genera el conjunto de entrenamiento y validación personalizado)
    train_set = customDataset (train_set_original) 
    valid_set = customDataset (valid_set_original)      

    return labels_names, train_set, valid_set

def collate_gpu (batch):
    x, y = torch.utils.data.default_collate(batch)
    return x.to(device = "cuda"), y.to(device = "cuda")

def createDataLoaders (batch_size_train, batch_size_valid, train_set, valid_set):
    
    # Conjunto de entrenamiento y de validacion: Dataloaders ()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size_train, shuffle = True, collate_fn = collate_gpu)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = batch_size_valid, shuffle = True, collate_fn = collate_gpu)

    print(f"Number of training batches: {len(train_loader)}, Number of validation batches {len(valid_loader)}")

    return train_loader, valid_loader