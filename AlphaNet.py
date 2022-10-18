import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

class AlphaNet(nn.Module):

    def __init__(self, batch_size, channel, height, width, d=10, stride=10, pool_d=3, pool_stride=3, hidden_neuron=30, dropout_rate=0.5):
        super(AlphaNet, self).__init__()
        self.batch_size = batch_size
        self.channel = channel
        self.height = height
        self.width = width
        self.d = d
        self.stride = stride
        self.pool_d = pool_d
        self.pool_stride = pool_stride
        self.hidden_neuron = hidden_neuron
        self.dropout_rate = dropout_rate
        self.X_ix = torch.arange(self.height - 1).repeat_interleave(torch.arange(self.height - 1, 0, -1)).long()
        self.Y_ix = (torch.arange(self.X_ix.size()[0]) - self.height*self.X_ix + (0.5*self.X_ix + 1)*(self.X_ix + 1)).long()
        self.step_ix = (torch.arange(0, self.width - d + 1, stride)[:, None] + torch.arange(d)).long()
        self.X_ix_full = self.X_ix.repeat(self.step_ix.size()[0]), self.step_ix.repeat_interleave(self.X_ix.size()[0], axis=0).T
        self.Y_ix_full = self.Y_ix.repeat(self.step_ix.size()[0]), self.step_ix.repeat_interleave(self.Y_ix.size()[0], axis=0).T
        self.Z_ix_full = torch.arange(self.height).repeat(self.step_ix.size()[0]), self.step_ix.repeat_interleave(self.height, axis=0).T
        self.batchnorm = nn.BatchNorm2d(channel)
        self.hidden_layer = nn.Linear(channel*(2*self.X_ix.size()[0] + 5*height)*(self.step_ix.size()[0] + 3*(int((self.step_ix.size()[0] - pool_d)/pool_stride) + 1)), hidden_neuron)
        self.output_layer = nn.Linear(hidden_neuron, 1)

    def forward(self, data):
        X = data[:, :, self.X_ix_full[0], self.X_ix_full[1]].transpose(2, 3).float()
        Y = data[:, :, self.Y_ix_full[0], self.Y_ix_full[1]].transpose(2, 3).float()
        Z = data[:, :, self.Z_ix_full[0], self.Z_ix_full[1]].transpose(2, 3).float()
        input = list()
        for conv in [self.ts_cov(X, Y), self.ts_corr(X, Y), self.ts_stddev(Z), self.ts_decaylinear(Z), self.ts_zscore(Z), self.ts_return(Z), self.ts_mean(Z)]:
            conv_bn = self.batchnorm(conv)
            input.append(conv_bn.flatten(start_dim=1))
            input.append(self.batchnorm(F.max_pool2d(conv_bn, (1, self.pool_d), (1, self.pool_stride))).flatten(start_dim=1))
            input.append(self.batchnorm(F.avg_pool2d(conv_bn, (1, self.pool_d), (1, self.pool_stride))).flatten(start_dim=1))
            input.append(self.batchnorm(-F.max_pool2d(-conv_bn, (1, self.pool_d), (1, self.pool_stride))).flatten(start_dim=1))
        input = torch.cat(input, dim=1)
        output = self.output_layer(F.dropout(F.relu((self.hidden_layer(input))), self.dropout_rate))
        return output

    def ts_cov(self, X, Y): 
        return (torch.sum((X.T - torch.mean(X.T, axis=0))*(Y.T - torch.mean(Y.T, axis=0)), axis=0)/(self.d - 1)).view(self.step_ix.size()[0], -1, self.channel, self.batch_size).T

    def ts_corr(self, X, Y): 
        return (self.ts_cov(X, Y).T.view(-1, self.channel, self.batch_size)/(torch.std(X.T, axis=0)*torch.std(Y.T, axis=0))).view(self.step_ix.size()[0], -1, self.channel, self.batch_size).T

    def ts_stddev(self, Z): 
        return torch.std(Z.T, axis=0).view(self.step_ix.size()[0], -1, self.channel, self.batch_size).T

    def ts_zscore(self, Z): 
        return (torch.mean(Z.T, axis=0)/torch.std(Z.T, axis=0)).view(self.step_ix.size()[0], -1, self.channel, self.batch_size).T

    def ts_return(self, Z): 
        return (Z.T[-1, :, :, :]/Z.T[0, :, :, :] - 1).view(self.step_ix.size()[0], -1, self.channel, self.batch_size).T

    def ts_decaylinear(self, Z): 
        return torch.sum((Z*(torch.arange(self.d) + 1)/(0.5*self.d*(self.d + 1))).T, axis=0).view(self.step_ix.size()[0], -1, self.channel, self.batch_size).T

    def ts_mean(self, Z): 
        return torch.mean(Z.T, axis=0).view(self.step_ix.size()[0], -1, self.channel, self.batch_size).T

data = torch.randn(3000*50, 1, 9, 30)
target = torch.randn(3000*50, 1)
batch_size = 1000
train_dataset = TensorDataset(data[:int(data.size()[0]/2)], target[:int(data.size()[0]/2)])
test_dataset = TensorDataset(data[int(data.size()[0]/2):], target[int(data.size()[0]/2):])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

AN = AlphaNet(batch_size, 1, 9, 30)
optimizer = optim.RMSprop(AN.parameters(), lr=0.0001)
criterion = nn.MSELoss()
n_epochs = 2
verbose = 15
for epoch in range(n_epochs):
    running_loss = 0
    for i, train_data in enumerate(train_loader):
        X_train, y_train = train_data
        optimizer.zero_grad()
        output = AN(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i%verbose == verbose - 1:
            print("epoch: %d, batch: %d, loss: %5f"%(epoch + 1, i + 1, running_loss/verbose))
            running_loss = 0
print("Finished Training")

torch.save(AN, "alphanet_model.pt")
