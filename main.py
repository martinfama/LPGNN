import numpy as np
from src import network_analysis as na

import torch
import torch_geometric as pyg

from src import DataSetup
from src import GraphNeuralNet
from src import LinkPrediction
from src import Logger

import matplotlib.pyplot as plt

N = 200
avg_k = 6
gamma = 2.4
Temp = 0.1
seed = 100
PS, PS_nx, data = DataSetup.getPSNetwork(N=N, avg_k=avg_k, gamma=gamma, Temp=Temp, seed=seed)
train, val, test = DataSetup.TrainTestSplit(data, test_ratio=0.1, val_ratio=0.1, neg_samples=True)

## Generate LaBNE embedding and Precision-Recall on it
PS_train = PS.copy()
test_edges = np.transpose(test.pos_edge_label_index)
PS_train.delete_edges([(x[0], x[1]) for x in test_edges])
PR_LaBNE = na.precision_recall_snapshots(PS_train, PS, metric='LaBNE', step=1, plot=False)

## GraphSAGE model and PR on it
graphSAGE_model = pyg.nn.GraphSAGE(in_channels=data.num_features, hidden_channels=128, out_channels=32, num_layers=3)
optimizer = torch.optim.SGD(graphSAGE_model.parameters(), lr=0.01)

epochs = 100
print(f'Training model: {graphSAGE_model} for {epochs} epochs')
loss = GraphNeuralNet.train_model(model=graphSAGE_model, optimizer=optimizer, train_data=train, test_data=test, val_data=val, epochs=epochs)
print(f'Train loss: {loss}')
R_SAGE, P_SAGE, predictions = LinkPrediction.PrecisionRecallTrainedModel(model=graphSAGE_model, train_data=train, test_data=test)

## Plot results, and log
fig, ax = plt.subplots(figsize=(10, 10))
ax.grid(alpha=0.5)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.plot(PR_LaBNE['recall'], PR_LaBNE['precision'], label='LaBNE')
ax.plot(R_SAGE, P_SAGE, label='GraphSAGE')
ax.legend(loc='upper right')

ax.set_title(f'PS network: N={N}, avg_k={avg_k}, gamma={gamma}, Temp={Temp}, seed={seed}')

fig.savefig('./figs/PR/PR_PS_Net_N{}_avg_k{}_gamma{}_Temp{}_seed{}.pdf'.format(N, avg_k, gamma, Temp, seed), bbox_inches='tight')
plt.close()