import sys
import torch
import numpy as np
from tqdm.auto import tqdm
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import random
from torch_geometric.loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = str(sys.argv[1])
dataset_path = str(sys.argv[2])
output_path = str(sys.argv[3])

dataset = torch.load(dataset_path)

# Assuming edge_attr and x are currently integer tensors
for data in dataset:
    if data.edge_attr:
        data.edge_attr = data.edge_attr.float()
    data.x = data.x.float()

# print(dataset[0].x.ndimension())

if dataset[0].x.ndimension() != 1:
    num_features = len(dataset[0].x[0])
else:
    num_features = 1
    for data in dataset:
        data.x = data.x[:, np.newaxis]



# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset, batch_size=64, shuffle=False)


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
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



model = GCN(num_features = num_features, hidden_channels=128)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


# Load the saved state
if not  torch.cuda.is_available():
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
else:
    checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



def test(loader):
    model.eval()
    probs = []
    for data in tqdm(loader):  
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)  
        prob = out.softmax(dim=1)
        probs.append(prob)

    return probs


output = test(test_loader)

# Extracting and saving probabilities
with open(output_path, "w") as f:
    for probs in output:
        for probabilities in probs:
            second_prob = probabilities[1].item()  # Convert to a regular float
            f.write(str(second_prob) + "\n")