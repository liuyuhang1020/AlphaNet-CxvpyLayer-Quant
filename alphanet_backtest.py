from AlphaNet import AlphaNet
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch import optim
import numpy as np
import pandas as pd
import joblib
from chinese_calendar import is_workday
from datetime import date
from datetime import timedelta
from alphalens import performance
from alphalens import plotting
from alphalens import tears
from alphalens import utils

AN = torch.load("alphanet_model.pt")
data1 = joblib.load("all_stock_hfq_trade_data")
data2 = joblib.load("tushare_turnover_rate")
data1_group = dict(list(data1.groupby("code")))
data2_group = dict(list(data2.groupby("code")))

def chinese_trade_dates(start_date="20110201", end_date="20200529"):
    start_date = date(int(start_date[:4]), int(start_date[4:6]), int(start_date[6:8]))
    end_date = date(int(end_date[:4]), int(end_date[4:6]), int(end_date[6:8]))
    trade_date = list()
    for i in range((end_date - start_date).days + 1):
        day = start_date + timedelta(days=i)
        if is_workday(day):
            trade_date.append(day.strftime(r"%Y%m%d"))
    trade_date = pd.DataFrame(trade_date, columns=["date"])
    return trade_date

trade_date = chinese_trade_dates()

stock1 = list()
for key in data1_group.keys():
    stock1.append(key)
    data1_group[key] = pd.merge(trade_date, data1_group[key], how="left")

stock2 = list()
for key in data2_group.keys():
    stock2.append(key)
    data2_group[key] = pd.merge(trade_date, data2_group[key], how="left")

empty1 = pd.DataFrame(index=list(trade_date["date"]), columns=["open", "high", "low", "close", "vol", "amount"])
empty2 = pd.DataFrame(index=list(trade_date["date"]), columns=["turnover_rate"])

stocks_feature = dict()
for stock in list(set(stock1 + stock2)):
    feature1 = data1_group.get(stock, empty1)
    feature2 = data2_group.get(stock, empty2)
    feature1.index = list(trade_date["date"])
    feature2.index = list(trade_date["date"])
    stocks_feature[stock] = pd.concat([feature1[["open", "high", "low", "close", "vol", "amount"]], feature2["turnover_rate"]], axis=1)

data = dict()
for key in stocks_feature.keys():
    if sum(stocks_feature[key].isnull().any(axis=1)) < 100:
        stocks_feature[key] = stocks_feature[key].fillna(axis=0, method='ffill')
        if not stocks_feature[key].isnull().any().any():
            data[key] = stocks_feature[key]

for key, value in data.items():
    close1 = np.array(value.iloc[1:, 3])
    close0 = np.array(value.iloc[:-1, 3])
    ret = (close1 - close0)/close0
    data[key]["return"] = 0
    data[key].iloc[1:, -1] = ret

batch_size = 100
channel = 1
height = 7
width = 30
pred_day = 5
test_rate = 0.5

group = int((len(trade_date) - pred_day - 1)/width)
test_group = int(group*test_rate)
train_group = group - test_group
last_day = group*width + 1
X_ix = np.arange(1, last_day).reshape(-1, channel, width)
Y_ix = np.arange(width + pred_day, last_day + pred_day, width)

X_train = np.empty((len(data)*train_group, channel, height, width))
X_test = np.empty((len(data)*test_group, channel, height, width))
Y_train = np.empty((len(data)*train_group, 1))
Y_test = np.empty((len(data)*test_group, 1))
for i, df in enumerate(data.values()):
    X = df.values[X_ix, :-1].transpose(0, 1, 3, 2)
    Y = df.values[Y_ix, -1].reshape(-1, 1)
    X_train[i*train_group:(i + 1)*train_group] = X[:train_group]
    X_test[i*test_group:(i + 1)*test_group] = X[train_group:]
    Y_train[i*train_group:(i + 1)*train_group] = Y[:train_group]
    Y_test[i*test_group:(i + 1)*test_group] = Y[train_group:]

train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float())
test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).float())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

AN = AlphaNet(batch_size, channel, height, width)
optimizer = optim.RMSprop(AN.parameters(), lr=0.0005)
criterion = nn.MSELoss()
n_epochs = 2
verbose = 15
for epoch in range(n_epochs):
    running_loss = 0
    for i, train_data in enumerate(train_loader):
        X_train, y_train = train_data
        if np.isnan(X_train).sum() != 0:
            print(np.isnan(X_train).sum())
        optimizer.zero_grad()
        output = AN(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i%verbose == verbose - 1:
            print("epoch: %d, batch: %d, loss: %5f"%(epoch + 1, i + 1, running_loss/verbose))
            running_loss = 0

y_test_series = list()
y_pred_series = list()
with torch.no_grad():
     bias = 0
     for i, test_data in enumerate(test_loader):
          X_test, y_test = test_data
          y_pred = AN(X_test)
          bias += abs((y_test - y_pred)).mean()
          y_test_series.append(y_test)
          y_pred_series.append(y_pred)
     bias = bias/(i + 1)
     print(bias)
y_test_series = pd.Series(y_test_series)
y_pred_series = pd.Series(y_pred_series)

factor_return = utils.get_clean_factor_and_forward_returns(y_test_series.astype(float), y_pred_series.astype(float))
IC = performance.factor_information_coefficient(factor_return)
print(IC.head())
plotting.plot_ic_ts(IC)
plotting.plot_ic_hist(IC)
plotting.plot_ic_qq(IC)
a = IC.iloc[:, 0]
print(len(a[a > 0.02]) / len(a))