
import sys
import torch
from torch_geometric.datasets import Planetoid
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

model_path = str(sys.argv[1])
dataset_path = str(sys.argv[2])
output_path = str(sys.argv[3])

# data
dataset = torch.load(dataset_path)
data_y = dataset.y.reshape(-1, 1)  # Reshape for OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)  # Dense output is usually easier to work with
dataset.y = (encoder.fit_transform(data_y))

num_nodes = len(dataset.x)
num_node_features = len(dataset.x[0])
# print(f'num_nodes: {num_nodes}, num_node_features: {num_node_features}')

test_mask = torch.zeros(num_nodes, dtype=torch.bool)
for i in range(len(test_mask)):
    test_mask[i] = True

dataset.test_mask = test_mask
dataset.y = torch.tensor(dataset.y)
num_classes = len(dataset.y[0])


# model
class MLP(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(num_features, 512)
        self.lin2 = Linear(512, 128)
        self.lin3 = Linear(128, num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x


model = MLP(num_features=num_node_features, num_classes=num_classes, hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)  # Define optimizer.


# Load the saved state
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def test(data):
      model.eval()
      with torch.no_grad():  # Disable gradient computation for efficiency
          out = model(data.x)
          probs = out.softmax(dim=1)  # Get probabilities for each class

      return probs

output = test(dataset)

output_tensor = np.array(output) 

with open(output_path, "w") as file:
    for row in output_tensor[:-1]:
        probabilities = row.tolist()  # Convert to a list for easier formatting
        line = ",".join(str(p) for p in probabilities)  # Format as comma-separated
        file.write(line + "\n")  # Write to file with a newlin

    probabilities = output_tensor[-1].tolist()  # Convert to a list for easier formatting
    line = ",".join(str(p) for p in probabilities)  # Format as comma-separated
    file.write(line)