import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('PyTorch version:', torch.__version__, ' Device:', device)
# --------------------------------

# -----------------------------------------
# plots a confusion matrix
def plot_confusion_matrix(gt, pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):

    cm = metrics.confusion_matrix(gt, pred)
    np.set_printoptions(precision=2)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, fontsize=3)
    plt.yticks(tick_marks, fontsize=3)

    plt.grid(True)

    plt.ylabel('Ground Truth')
    plt.xlabel('Predictions')
    plt.tight_layout()
    plt.savefig(f"cm.pdf", bbox_inches='tight')
    plt.close()
#-----------------------------------------

# --------------------------------
# Hyper-Parameters
learning_rate = 0.001 # Initial learning rate
training_epochs = 15 # Number of epochs to train
batch_size = 100 # Number of images per batch
display_step = 1 # How often to output model metrics during training

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
# --------------------------------

# --------------------------------
# The dataset
train_dataset = datasets.MNIST('./data', 
                               train=True, 
                               download=True, 
                               transform=transforms.ToTensor())

test_dataset = datasets.MNIST('./data', 
                                    train=False, 
                                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=False)
# --------------------------------




class Multilayer_Perceptron(nn.Module):
    def __init__(self):
        super(Multilayer_Perceptron, self).__init__()

        # Hidden fully connected layer with 256 neurons
        self.fc1 = nn.Linear(n_input, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_classes)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)

# defining model, optimizer and loss:
model = Multilayer_Perceptron().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# training function:
def train():
    # Set model to training mode
    model.train()
    
    for epoch in range(training_epochs):
    # Loop over each batch from the training set
        for batch_idx, (img, lbl) in enumerate(train_loader):
            # Copy image data to GPU if needed
            img = img.to(device)
            lbl = lbl.to(device)

            # Zero gradient buffers
            optimizer.zero_grad() 
            
            # Pass image data through the network
            output = model(img)

            # Calculate loss
            loss = criterion(output, lbl)

            # Backpropagate
            loss.backward()
            
            # Update weights
            optimizer.step()
        
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(loss.item()))
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.data.item()))


def validate():
    preds, gts = [], []
    model.eval()
    val_loss, correct = 0, 0

    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        gts.append(target.item())


        output = model(data)
        val_loss += criterion(output, target).data.item()

        _, pred = torch.max(output, dim=1)
        # pred = output.data.max(1)[1] # get the index of the max log-probability

        preds.append(pred)
        correct += pred.eq(target).cpu().sum()

    # val_loss /= len(test_loader)
    # loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(test_loader.dataset)
    # accuracy_vector.append(accuracy)
    
    print("Accuracy:", accuracy)

    plot_confusion_matrix(gts, preds, test_dataset.classes)

    # print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        # val_loss, correct, len(test_loader.dataset), accuracy))


# lossv, accv = [], []
train()
print("Optimization Finished!")

# validate(lossv, accv)
validate()
