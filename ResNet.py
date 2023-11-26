import math
from scipy import signal
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils import data
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import metrics
from torchsummary import summary

SEED = 3
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 定义SE-ResNet
class flattenlayer(nn.Module):
    def __init__(self):
        super(flattenlayer, self).__init__()

    @staticmethod
    def forward(x):
        return x.view(x.shape[0], -1)


class GlobalAvgPooling1d(nn.Module):
    def __init__(self):
        super(GlobalAvgPooling1d, self).__init__()

    @staticmethod
    def forward(x):
        return F.avg_pool1d(x, kernel_size=x.shape[2])


class GlobalMaxPooling1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPooling1d, self).__init__()

    @staticmethod
    def forward(x):
        return F.max_pool1d(x, kernel_size=x.shape[2])


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.SE = nn.Sequential(
            GlobalAvgPooling1d(),
            nn.Conv1d(out_channels, out_channels//4, 1),
            # nn.ELU(),
            nn.Conv1d(out_channels//4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        Y = F.elu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        # Y = Y * self.SE(Y)
        if self.conv3:
            x = self.conv3(x)
        return F.elu(x+Y)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for m in range(num_residuals):
        if m == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


class SE_ResNet(nn.Module):
    def __init__(self):
        super(SE_ResNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(4),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        )
        self.residual_block = nn.Sequential(
            resnet_block(in_channels=4, out_channels=4, num_residuals=2, first_block=True),
            resnet_block(4, 16, 2),
            resnet_block(16, 32, 2),
            resnet_block(32, 64, 2),
        )
        self.pool_fc = nn.Sequential(
            GlobalAvgPooling1d(),
            flattenlayer(),
            nn.Linear(64, 16),
            nn.ELU(),
            nn.Linear(16, 2)
        )

    def forward(self, Z):
        feature_1 = self.conv(Z)
        feature_2 = self.residual_block(feature_1)
        output = self.pool_fc(feature_2)
        return output


all_data = pd.read_csv('data/all_data.csv')
data_x = all_data.iloc[:, :-1]
data_y = all_data.iloc[:, -1]
# ofcTrain_x = pd.DataFrame(msc(ofcTrain_x.mean().values, ofcTrain_x.values, 1))
# ofcTrain_x = signal.savgol_filter(ofcTrain_x, 7, 3)
# ofcTrain_x = pd.DataFrame(signal.savgol_filter(ofcTrain_x, 7, 3, deriv=1))


# 绘制光谱图
'''fig, ax = plt.subplots()
ax.set_xlabel('Wavelength(nm)')
ax.set_ylabel('Absorbance')
wavelength = data_x.columns[:].values.astype(float)
p = 0
for i in range(data_x.shape[0]):
    ax.plot(wavelength, data_x.iloc[i].values)
    p += 1
print(p)
plt.show()'''

# myTrain_X, myTest_X, myTrain_Y, myTest_Y = train_test_split(ofcTrain_x, ofcTrain_y, test_size=1/6, random_state=2021)
# myTrain_X, myValid_X, myTrain_Y, myValid_Y = train_test_split(myTrain_X, myTrain_Y, test_size=1/5, random_state=2021)
# torch.set_printoptions(precision=10)
# np.set_printoptions(precision=15)
# print(myValid_X.values)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kf = KFold(n_splits=5, shuffle=True, random_state=0)
test_acc_best = [0, 0, 0, 0, 0]
for k, (train_index, test_index) in enumerate(kf.split(data_x)):
    print('第', k+1, '折')
    X_train, X_test = data_x.values[train_index], data_x.values[test_index]
    y_train, y_test = data_y.values[train_index], data_y.values[test_index]
    model = SE_ResNet().to(device)  # 定义模型
    # summary(model, (1, 125), batch_size=1)
    model = model.double()
    myTrain_X = torch.tensor(X_train, dtype=torch.double)
    myTest_X = torch.tensor(X_test, dtype=torch.double)

    myTrain_X = myTrain_X.view(myTrain_X.shape[0], 1, 125).to(device)
    myTest_X = myTest_X.view(myTest_X.shape[0], 1, 125).to(device)

    myTrain_Y = torch.tensor(y_train).to(device)
    myTest_Y = torch.tensor(y_test).to(device)

    # print(myTrain_X)

    myTrainData = torch.utils.data.TensorDataset(myTrain_X, myTrain_Y)
    batch_size = 500
    train_iter = torch.utils.data.DataLoader(myTrainData, batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    train_loss_list = []
    train_acc_list = []


    def train(net, train_data, list_loss, acc_list):
        net.train()
        for batch_idx, (x, y) in enumerate(train_data):
            y_hat = net(x)
            loss = nn.CrossEntropyLoss()(y_hat, y)
            list_loss.append(loss.item())
            y_hat = nn.Softmax(dim=1)(y_hat).argmax(dim=1).tolist()
            acc = metrics.accuracy_score(y_hat, y.tolist())
            acc_list.append(acc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def test(net, test_X, test_Y):
        net.eval()
        yhat = net(test_X)
        loss = nn.CrossEntropyLoss()(yhat, test_Y)
        yhat = nn.Softmax(dim=1)(yhat).argmax(dim=1).tolist()
        acc = metrics.accuracy_score(yhat, y_test)
        # precision = metrics.precision_score(yhat, y_test)
        # recall = metrics.recall_score(yhat, y_test)
        return loss.item(), acc


    test_loss_list = []
    test_acc_list = []
    best_score = 0
    epoch = 1000
    for i in range(epoch):
        train(model, train_iter, train_loss_list, train_acc_list)
        test_loss, test_acc = test(model, myTest_X, myTest_Y)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        ExpLR.step()
        if test_acc > best_score:
            best_score = test_acc
            test_acc_best[k] = best_score
            model_name = 'model/model' + str(k+1) + '.pt'
            torch.save(model.state_dict(), model_name)
        if (i+1) % 100 == 0:
            # print('epoch =', i + 1, '; train-loss =', train_loss_list[-1], '; test-loss =', test_loss)
            print('epoch =', i + 1, '; train-acc =', train_acc_list[-1], '; test-acc =', test_acc)

    # print('min-valid-loss =', best_score)
    plt.rc("font", family='Microsoft YaHei')
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.plot(np.array(range(1, epoch+1)), np.array(train_loss_list))
    ax.plot(np.array(range(1, epoch+1)), np.array(test_loss_list))
    ax.legend(labels=['训练集', '测试集'])
    plt.show()
print(sum(test_acc_best)/len(test_acc_best))
