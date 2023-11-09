import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

#################
########### DADES
#################

# El label del dataset és l'índex de la llista labels. Cada posició de la llista és un codi ASCII. Podeu emprar la funció chr per fer la transformació

# Definim una seqüència (composició) de transformacions
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # mitjana, desviacio tipica (precalculats)
    ])

# Descarregam un dataset ja integrat en la llibreria Pytorch:
train = datasets.EMNIST('data', split="digits", train=True, download=True, transform=transform)  ## Si acabau podeu fer proves amb el split "balanced"
test = datasets.EMNIST('data', split="digits",train=False, transform=transform)

print("Info:")
print("Tipus de la variable train : ", type(train)) # la variable test te les mateixes característiques
print(test.__dict__.keys()) ## Tot objecte Python té un diccionari amb els seus atributs
classes = test.classes ## Obtenim una llista amb les classes del dataset
print("Classes:"+"="*50)
print(classes)

train_batch_size = 16
test_batch_size = 100

# Transformam les dades en l'estructura necessaria per entrenar una xarxa
train_loader = torch.utils.data.DataLoader(train, train_batch_size)
test_loader = torch.utils.data.DataLoader(test, test_batch_size)

iterador =  iter(train_loader) # Un iterador!!

features, labels = next(iterador)

print("Saber l'estructura del batch us ajudarà: ")
print(f"Feature batch shape: {features.size()}")
print(f"Labels batch shape: {labels.size()}")

## Display images in a 4x2 grid
#num_rows = 4
#num_cols = 4
#
#fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
#
#for i in range(num_rows):
#    for j in range(num_cols):
#        idx = i * num_cols + j
#        axes[i, j].imshow(features[idx].squeeze(), cmap='gray')
#        axes[i, j].set_title(f'Label: {labels[idx]}')
#        axes[i, j].axis('off')  # Turn off axis labels for cleaner display
#
#plt.tight_layout()
#plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 1) # 28x28 = 784

        # TODO: definir les capes que necessitem

    def forward(self, x):
            
        x = torch.flatten(x, 1)
        x = self.l1(x) #

        # TODO connectar les capes. El valor de retorn d'una capa és l'entrada de la següent
        output = F.softmax(x, dim=1)
        return output
    
    def train(model, device, train_loader, optimizer, epoch, loss_fn, log_interval=100, verbose=True):
        
        model.train() # Posam la xarxa en mode entrenament

        loss_v = 0 # Per calcular la mitjana (és la vostra)

        # Bucle per entrenar cada un dels batches
        for batch_idx, (data, target) in enumerate(train_loader):
        
            data, target = data.to(device), target.to(device)  ###  Veure ús de CuDA en cel·les inferiors
            optimizer.zero_grad()
            output = model(data)
            loss =  loss_fn(output, target)
            loss.backward()

            optimizer.step()

            ## Informació de debug
            if batch_idx % log_interval == 0 and verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Average: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), loss.item()/ len(data)))
            loss_v += loss.item()

        loss_v /= len(train_loader.dataset)
        print(f'\n Epoch {epoch} Train set: Average loss: {loss_v}')
    
        return loss_v


    def test(model, device, test_loader,loss_fn):
        model.eval() # Posam la xarxa en mode avaluació

        test_loss = 0
        correct = 0

        with torch.no_grad(): # desactiva el càlcul de gradients, no ho necessitam per l'inferència. Estalvia memòria i fa més ràpids els càlculs
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss +=  loss_fn(output, target)
                pred = output.argmax(dim=1, keepdim=True)  # index amb la max probabilitat
                correct += pred.eq(target.view_as(pred)).sum().item()
    
        # Informació de debug
        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        return test_loss
    
torch.manual_seed(33)

# El següent ens permet emprar l'entorn de cuda. Si estam emprant google colab el podem activar a "Entorno de ejecución"
use_cuda = False
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Paràmetres bàsics
epochs = 10
lr =  0.001

model = Net().to(device)

# Stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=lr)

# Guardam el valor de pèrdua mig   de cada època, per fer el gràfic final
train_l = np.zeros((epochs))
test_l = np.zeros((epochs))

loss_fn =  torch.nn.CrossEntropyLoss()

# Bucle d'entrenament
for epoch in range(0, epochs):
    train_l[epoch] = train(model, device, train_loader, optimizer, epoch, loss_fn, verbose=False)
    test_l[epoch]  = test(model, device, test_loader, loss_fn)