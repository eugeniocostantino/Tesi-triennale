import torch
from torch import nn
import torchvision
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from matplotlib import pyplot
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="logpt/fit")
from torchsummary import summary

train_set = torchvision.datasets.MNIST(
    root="./",
    download=True,
    train=True,
    transform=torchvision.transforms.ToTensor()
)

test_set = torchvision.datasets.MNIST(
    root="./",
    train=False,
    transform=torchvision.transforms.ToTensor()
)
train_indices, val_indices, _, _ = train_test_split(
    range(len(train_set)),
    train_set.targets,
    stratify=train_set.targets,
    test_size=0.3,
)

train_split = Subset(train_set, train_indices)
val_split = Subset(train_set, val_indices)

epochs=15
batch_size = 32
classes=len(np.unique(train_set.targets))
print(classes)
train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)



class Model (nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn_part = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=4, stride=1),
            nn.ReLU()
        )
        self.linear1 = nn.Linear(9*16*16, 500)
        self.lrelu = nn.ReLU()
        self.linear2 = nn.Linear(500, 10)
        self.lsoftmax = nn.Softmax(1)

    def forward(self, data):
        output = self.cnn_part(data)
        output = torch.flatten(output,1)
        output = self.linear1(output)
        output = self.lrelu(output)
        output = self.linear2(output)
        output = self.lsoftmax(output)
        return output

model = Model()
print(Model)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_validate(model, num_epoch):
    train_losses = []
    val_losses = []
    for epoch in range(num_epoch):
        train_running_loss = 0.0
        train_correct = 0.0
        val_correct = 0.0
        # training
        model.train()
        for i, data in enumerate(train_loader, 0):
            X, labels = data
            optimizer.zero_grad()
            outputs = model(X)  # , th_images)
            current_loss = loss(outputs, labels)
            current_loss.backward()
            optimizer.step()
            train_running_loss += current_loss.item()
            train_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                X, labels = data
                outputs = model(X)  # , th_images)
                current_loss = loss(outputs, labels)
                val_running_loss += current_loss.item()
                val_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

        writer.add_scalars('Loss', {
            'train': train_running_loss/len(train_loader),
            'validation': val_running_loss/len(val_loader),
        }, epoch)
        writer.add_scalars('Accuracy', {
            'train': train_correct/len(train_split),
            'validation': val_correct/len(val_split),
        }, epoch)

        print("EPOCH ", epoch+1, "/", num_epoch, f"     |||     train_loss={(train_running_loss/len(train_loader)):.5f}, train_accuracy= {(train_correct/len(train_split)):.4f}%     |||     val_loss={(val_running_loss)/len(val_loader):.5f}, val_accuracy= {(val_correct/len(val_split)):.4f}%")

    writer.flush()
    pyplot.title("Learning losses")
    pyplot.xlabel("epochs")
    pyplot.ylabel("loss")
    pyplot.plot(np.array(val_losses), 'r', label="Validation loss")
    pyplot.plot(np.array(train_losses), 'b', label="Training loss")
    pyplot.legend()
    pyplot.show()

print(model)
summary(model, (1, 28, 28), batch_size)
train_validate(model,epochs)


def test(dataloader):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


test(test_loader)
writer.close()

model.eval()
example_input = torch.rand(1,1,28,28)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("./img_classifier.pt")

