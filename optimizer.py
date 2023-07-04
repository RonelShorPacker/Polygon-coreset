from create_data_synthetic import polygon_data
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from algo import computeCostToPolygon
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from set_of_points import SetOfPoints
from algo import exhaustive_search


class PolygonModel(nn.Module):
    def __init__(self, k, convex=True, h=500, d=2, init=None):
        super(PolygonModel, self).__init__()
        self.k = k
        self.d = d
        self.batch_size = 32# self.d
        self.h = h
        self.h1 = 1000
        self.h2 = 500
        self.layers = nn.Sequential(nn.Linear(self.batch_size, self.h), nn.ReLU(), nn.Linear(self.h, self.h1), nn.ReLU(), nn.Linear(self.h1, self.k))
        self.layers.apply(self.init_weights)
        x = SetOfPoints(data)
        x.parameters_config.k = self.k
        init, _ = exhaustive_search(x, iters=10, plot=True)

        # self.polyPoints = nn.Linear(d * self.k, 1)
        # init, _ = exhaustive_search(SetOfPoints(data), iters=10, plot=True)
        # opt = init.points[init.vertices]
        # self.polyPoints.weight = nn.Parameter(torch.Tensor(opt))
        self.convex = convex

    def forward(self, data):
        return self.layers(data)

    def backward(self, x, data):
        """try:
            y = ConvexHull(x.T.cpu().detach().numpy())
        except:
            print(x.shape)
        if len(y.vertices) != self.k:
            print("fuck")
            return 10000000
        else:
            print("great")"""
        return computeCostToPolygon(SetOfPoints(data), ConvexHull(x.T.cpu().detach().numpy()))

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal(m.weight, gain=1.9)
            m.bias.data.fill_(0.01)

def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr


def train_epochs(model, optimizer, train_dl, scheduler, val_dl=None, epochs=10, C=1000):
    idx = 0
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for x in train_dl:
            if x.shape[0] != model.batch_size:
                continue
            polygonPoints = model(x.T.to(torch.float32))

            optimizer.zero_grad()
            loss = model.backward(polygonPoints, data)
            optimizer.step()
            idx += 1
            total += model.batch_size
            sum_loss += loss #model.item()
        scheduler.step()
        train_loss = sum_loss / total
        print("train_loss %.3f" % (train_loss))
        if i % 100 == 0:
            # polygonPoints_ = ConvexHull(polygonPoints.T.cpu().detach().numpy())
            plt.plot(polygonPoints.T.cpu().detach().numpy()[:, 0], polygonPoints.T.cpu().detach().numpy()[:, 1], 'r--', lw=2)
            plt.scatter(data[:, 0], data[:, 1])
            plt.show()
    return sum_loss / total


if __name__ == "__main__":
    k = 5
    epochs = 1000
    data = polygon_data(k)
    model = PolygonModel(k=k)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    train_dl = DataLoader(data, batch_size=model.batch_size, shuffle=True)
    train_epochs(model, optimizer, train_dl, epochs=epochs, scheduler=scheduler)
