import torch
import torch.nn as nn
from keras.datasets import mnist
import numpy as np
from tqdm import trange
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle

val_losses = []
train_losses = []


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.c1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.p1 = nn.MaxPool2d(2, stride=2)
        self.c2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.p2 = nn.MaxPool2d(2, stride=2)
        self.f = nn.Flatten()
        self.l1 = nn.Linear(3136, 256)
        self.d = nn.Dropout(0.2)
        self.act1 = nn.Tanh()
        self.l2 = nn.Linear(256, 10)
        self.act2 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.c1(x)
        x = self.p1(F.relu(x))

        x = self.c2(x)
        x = self.p2(F.relu(x))

        x = self.f(x)

        x = self.l1(x)
        x = self.d(x)
        x = self.act1(x)
        x = self.l2(x)
        x = self.act2(x)
        return x


if __name__ == "__main__":
    (x, raw_y), (_, _) = mnist.load_data()
    x = x.astype(np.float64)
    x /= 255.0

    # one hot encode y
    y = []
    for idx, e in enumerate(raw_y):
        y.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y[idx][e] = 1
    y = np.array(y)

    model = ConvNet()

    n_iters = 1500
    BS = 32
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    x = torch.Tensor(x.reshape(-1, 1, 28, 28))
    y = torch.Tensor(y.reshape(-1, 10))

    for i in (t := trange(n_iters)):
        optim.zero_grad()
        sample = np.random.randint(0, x.shape[0], size=(BS))
        X = x[sample].reshape(-1, 1, 28, 28)
        Y = y[sample]
        predictions = model(X)
        cost = loss(predictions, Y)
        cost.backward()
        optim.step()
        t.set_description(f"Cost: {cost}")
        train_losses.append(cost.detach().numpy())

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, color="black")
    plt.show()

    val_predictions = model(x[0:3])
    print(val_predictions)
    print(y[0:3])

    with open("convnet.bin", "wb") as f:
        pickle.dump(model, f)
