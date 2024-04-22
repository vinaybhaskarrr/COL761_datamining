import torch

# Assuming edge_attr and x are currently integer tensors
for data in GC_D1:
    data.edge_attr = data.edge_attr.float()
    data.x = data.x.float()

# zero, one = 0, 0
zero_graphs, one_graphs = [], []

for graph in GC_D1:
    if graph.y == 0:
        # zero += 1
        zero_graphs.append(graph)
    else:
        # one += 1
        one_graphs.append(graph)

# print(zero, one)

import random

undersampled_zero_graphs = random.sample(zero_graphs, 1232)
new_dataset = undersampled_zero_graphs + one_graphs

random.shuffle(new_dataset)
print(len(new_dataset))

test_dataset = new_dataset[:197]
train_dataset = new_dataset[197:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(9, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)  # Dropout before Linear
        x = self.lin(x)
        
        return x

model = GCN(hidden_channels=1)
print(model)

from tqdm.auto import tqdm
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


model = GCN(hidden_channels=64)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         data.to(device)
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y.to(device))  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

    save_path = "Q1"

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,  
    }
    
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    
    torch.save(checkpoint, os.path.join(save_path, f'GC_D1_checkpoint_{epoch}.pt'))

def test(loader):
    model.eval()

    correct = 0
    all_preds = []
    all_labels = []

    for data in loader:  
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())  

        # Collect predictions and labels for ROC-AUC calculation
        all_preds.extend(pred.detach().cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    accuracy = correct / len(loader.dataset) 

    # Calculate ROC-AUC metrics
    # print(all_preds)
    # print(all_labels)
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)  # Assuming class 1 is the positive class
    roc_auc = auc(fpr, tpr)

    return accuracy, roc_auc

# training loop

for epoch in range(1, 171):
    train()
    train_acc, train_roc_auc = test(train_loader)
    test_acc, test_roc_auc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    print(f'Epoch: {epoch:03d}, Train Roc_Auc: {train_roc_auc:.4f}, Test ROC_AUC: {test_roc_auc:.4f}')