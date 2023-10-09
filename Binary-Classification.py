import os
import pickle
import time
from collections import Counter

import matplotlib
import sys
import torch
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
import sklearn.model_selection
from torch.utils.data import Subset

plt.rcParams["figure.figsize"] = (12, 12) # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
TRAIN_TEST_SPLIT = 0.7
EMBEDDING_SIZE = 8


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/cardekho_india_dataset_bce.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1645110979-deep-learning-intro-2022-q1/cardekho_india_dataset_bce.pkl',
                path_dataset,
                progress=True
            )
        with open(f'{path_dataset}', 'rb') as fp:
            X, Y, self.labels = pickle.load(fp)

        X = np.array(X)
        self.Y_idx = Y
        self.Y_labels = self.labels[3]
        self.Y_len = len(self.Y_labels)

        Y_counter = Counter(Y)
        Y_counter_val = np.array(list(Y_counter.values()))
        self.Y_weights = (0.77 / Y_counter_val) * np.sum(Y_counter_val)
        self.X_classes = np.array(X[:, :3])

        self.X = np.array(X[:, 3:]).astype(np.float32)  # VERY IMPORTANT OTHERWISE NOT ENOUGH CAPACITY
        X_mean = np.mean(self.X, axis=0)
        X_std = np.std(self.X, axis=0)
        self.X = (self.X - X_mean) / X_std
        self.X = self.X.astype(np.float32)

        # x_brands,
        # x_fuel,
        # x_seller_type,

        # x_year,
        # x_km_driven,
        # y_owner,
        # y_selling_price

        # self.Y_idx

        # x_transmission
            # Manual
            # Automatic

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], np.array(self.X_classes[idx]), self.Y_idx[idx]

dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)

idxes_train, idxes_test = sklearn.model_selection.train_test_split(
    np.arange(len(dataset_full)),
    train_size=train_test_split,
    test_size=len(dataset_full) - train_test_split,
    stratify=dataset_full.Y_idx,
    random_state=0
)


dataloader_train = torch.utils.data.DataLoader(
    dataset=Subset(dataset_full, idxes_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=Subset(dataset_full, idxes_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class BatchNorm1d(torch.nn.Module):
    def __init__(
            self,
            num_features
    ):
        super().__init__()

        self.num_features = num_features
        self.gamma = torch.nn.Parameter(torch.ones(1, self.num_features))
        self.beta = torch.nn.Parameter(torch.zeros(1, self.num_features))

        self.train_mean_list = []
        self.train_var_list = []

        self.train_mean = torch.zeros(self.num_features,)
        self.train_var = torch.ones(self.num_features, )

    def forward(self, x):

        if self.training:
            mean = torch.mean(x, dim=0, keepdim=True)
            var = torch.var(x, dim=0, keepdim=True)

            self.train_mean = (1 - 0.1) * self.train_mean + 0.1 * mean
            self.train_var = (1 - 0.1) * self.train_var + 0.1 * var

            x_norm = (x - mean) / torch.sqrt(var + 1e-8)

            out = self.gamma * x_norm + self.beta

            self.train_mean_list.append(mean)
            self.train_var_list.append(var)
        else:
            x_norm = (x - self.train_mean) / torch.sqrt(self.train_var + 1e-8)

            out = self.gamma * x_norm + self.beta
        return out


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=4 + EMBEDDING_SIZE * 3, out_features=256),
            BatchNorm1d(num_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=64),
            BatchNorm1d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=1),
            torch.nn.Sigmoid()
        )

        self.embs = torch.nn.ModuleList()
        for i in range(3):
            self.embs.append(
                torch.nn.Embedding(embedding_dim=EMBEDDING_SIZE, num_embeddings=len(dataset_full.labels[i]))
            )

    def forward(self, x, x_classes):
        x_enc = []
        for i in range(len(self.embs)):
            x_enc.append(self.embs[i].forward(x_classes[:, i]))
        x_enc = torch.cat(x_enc, dim=-1)
        x = torch.cat([x, x_enc], dim=-1)
        y_prim = self.layers.forward(x)
        return y_prim

class LossBCE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prim, y):
        bce = -(y * torch.log(y_prim) + (1 - y) * torch.log(1 - y_prim))
        loss = torch.mean(bce)
        return loss

model = Model()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
loss_fn = LossBCE()

loss_plot_train = []
loss_plot_test = []

acc_plot_train = []
acc_plot_test = []

f1_plot_train = []
f1_plot_test = []

conf_matrix_train = np.zeros((dataset_full.Y_len, dataset_full.Y_len))
conf_matrix_test = np.zeros((dataset_full.Y_len, dataset_full.Y_len))

for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:

        if dataloader == dataloader_test:
            model = model.eval()
            torch.set_grad_enabled(False)
        else:
            model = model.train()
            torch.set_grad_enabled(True)

        losses = []
        accs = []

        conf_matrix = np.zeros((dataset_full.Y_len, dataset_full.Y_len))

        for x, x_classes, y_idx in dataloader:

            y_prim_idx = model.forward(x, x_classes)

            loss = loss_fn(y_prim_idx, y_idx.float())

            losses.append(loss.item())

            if dataloader == dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            y_idx = y_idx.data.numpy()
            y_prim_idx = np.round(y_prim_idx.data.numpy()).astype(np.int)
            acc = np.mean((y_idx == y_prim_idx) * 1.0)
            accs.append(acc)

            for idx in range(len(y_prim_idx)):
                conf_matrix[y_prim_idx[idx], y_idx[idx]] += 1

        f1s = []
        for y_idx in range(len(conf_matrix)):
            TP = conf_matrix[y_idx, y_idx]
            FP = np.sum(conf_matrix[y_idx, :]) - TP
            FN = np.sum(conf_matrix[:, y_idx]) - TP
            TN = np.sum(conf_matrix) - TP - FP - FN
            f1 = 2 * TP / (2 * TP + FP + FN)
            f1s.append(f1)


        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
            acc_plot_train.append(np.mean(accs))
            conf_matrix_train = conf_matrix
            f1_plot_train.append(np.mean(f1s))
        else:
            loss_plot_test.append(np.mean(losses))
            acc_plot_test.append(np.mean(accs))
            conf_matrix_test = conf_matrix
            f1_plot_test.append(np.mean(f1s))

    print(
        f'epoch: {epoch} '
        f'loss_train: {loss_plot_train[-1]} '
        f'loss_test: {loss_plot_test[-1]} '
        f'acc_train: {acc_plot_train[-1]} '
        f'acc_test: {acc_plot_test[-1]} '
        f'f1_train: {f1_plot_train[-1]} '
        f'f1_test: {f1_plot_test[-1]} '
    )

    if epoch % 100 == 0:

        plt.tight_layout(pad=0)

        fig, axes = plt.subplots(nrows=2, ncols=2)

        ax1 = axes[0, 0]
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax1 = axes[0, 1]
        ax1.plot(acc_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(acc_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")

        for ax, conf_matrix in [(axes[1, 0], conf_matrix_train), (axes[1, 1], conf_matrix_test)]:
            ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Greys'))
            ax.set_xticks(np.arange(dataset_full.Y_len), dataset_full.Y_labels, rotation=45)
            ax.set_yticks(np.arange(dataset_full.Y_len), dataset_full.Y_labels)
            for x in range(dataset_full.Y_len):
                for y in range(dataset_full.Y_len):
                    perc = round(100 * conf_matrix[x, y] / np.sum(conf_matrix))
                    ax.annotate(
                        str(int(conf_matrix[x, y])),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        backgroundcolor=(1., 1., 1., 0.),
                        color='black' if perc < 50 else 'white',
                        fontsize=10
                    )
            ax.set_xlabel('True')
            ax.set_ylabel('Predicted')


        plt.show()