import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from torch import nn

def plotErrors (train_avg_loss_in, train_avg_loss, valid_avg_loss, num_epochs):

    plt.figure(1)
    # plt.plot(train_avg_loss_in, label = "Training - VW", linewidth = 1.25, linestyle = '-',  c = 'red', marker = 'o', markersize = 5)
    plt.plot(train_avg_loss, label = "Training", linewidth = 1.25, linestyle = '-',  c = 'red', marker = 'o', markersize = 5)
    plt.plot(valid_avg_loss, label = "Validation", linewidth = 1.25, linestyle = '--', c = 'blue', marker = 'o', markersize = 5)

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)

    plt.grid(True)
    plt.xlim(0, num_epochs - 1)
    plt.xlabel("Epochs")
    plt.ylabel("Errors")

    plt.show()

    return

def plotPrecision (train_precision, valid_precision, num_epochs):

    plt.figure()
    plt.plot(train_precision, label = "Training", linewidth = 1.25, linestyle = '-',  c = 'red')
    plt.plot(valid_precision, label = "Validation", linewidth = 1.25, linestyle = '-',  c = 'black')

    plt.grid(True)
    plt.xlim(0, num_epochs)
    plt.xlabel("Epochs")
    plt.ylabel("Precision")

    plt.show()

    return

def evalNetwork (valid_loader, model, batch_size):

    model.eval()
    size = len(valid_loader)
    Accuracy = []

    labels = ["T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

    Sf = nn.Softmax(dim = 1)
    Y_class = np.zeros(batch_size, dtype = int)
 
    # Se desactiva el cálculo de gradientes
    with torch.no_grad ():

        # Iteracion sobre el conjunto de validacion
        for batch, (X, Y) in enumerate(valid_loader):

            # Calculo de las predicciones del modelo ya entrenado
            Y_pred = model (X)

            # Applied softmax to Y_pred
            Y = Y.to('cpu').numpy()
            Y_pred = Y_pred.to('cpu')
            Y_prob = Sf(Y_pred)
                   
            for i in range(0, batch_size):
                if Y[i] == Y_pred[i].argmax():
                    #if Y_prob[i, Y[i]] > 0.95:
                    Y_class[i] = Y_pred[i].argmax()
                                
            # Accuracy
            accuracy = accuracy_score(Y, Y_class)
            print("Accuracy:", accuracy * 100, "%")
                     
                   
            # Precision score
            #Precision = precision_score(Y, Y_class)
            #print("Precision:", Precision)

            # Recall score
            #Recall = recall_score(Y, Y_class)
            #print("Recall:", Recall)

            # Confusion matrix
            ConfMatrix = confusion_matrix(Y, Y_class)

            ConfusionMatrixDisplay.from_predictions(
                Y,
                Y_class,
                display_labels = labels,
                cmap = plt.cm.Blues,
                xticks_rotation = 'vertical',
                normalize = 'true'
            )

            #print(disp.confusion_matrix)
            #ConfusionMatrixDisplay(confusion_matrix = ConfMatrix, cmap=plt.cm.Blues, display_labels = [i for i in labels]).plot() 
            plt.show()   
    return

def plotOutImag (valid_loader, device, network):


    # Activate eval mode
    network.eval()

    train_iter = iter(valid_loader)
    images, labels = next(train_iter)

    for j in range(2,3):
    
        figure = plt.figure(j)

        for i in range(1, 4):


            imgOrig = images[30 - i]
            
            imgPred = network(imgOrig.unsqueeze(0))
            imgPred = imgPred.to('cpu')
            imgOrig = imgOrig.to('cpu')

            figure.add_subplot(3, 2, 2*i-1)
            if (i == 1): plt.title("Original")
            plt.imshow(imgOrig.squeeze(0,1), cmap = "Greys_r")
            plt.axis("off")

            figure.add_subplot(3, 2, 2*i)
            if (i == 1): plt.title("Reconstructed")
            plt.imshow(imgPred.detach().numpy().reshape([1,28,28]).squeeze(0), cmap = "Greys_r")
            plt.axis("off")

    plt.show()

    return


    