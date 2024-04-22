import numpy as np
from sklearn.preprocessing import OneHotEncoder

data_y = NC_D2.y.reshape(-1, 1)  # Reshape for OneHotEncoder

encoder = OneHotEncoder(sparse=False)  # Dense output is usually easier to work with
NC_D2.y = (encoder.fit_transform(data_y))

print(NC_D2.y)

import torch
from torch_geometric.datasets import Planetoid
from sklearn.model_selection import train_test_split

def stratified_split(data, test_size=0.2, random_state=42):
    """Creates train and test masks with stratified sampling.

    Args:
        data (torch_geometric.data.Data): The input PyTorch Geometric Data object.
        test_size (float, optional): Proportion of data for the test set. Defaults to 0.2.
        random_state (int, optional): Seed for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing (data with updated masks, train_index, test_index)
    """
  
    num_nodes = data.y.size//9
    print(num_nodes)
    train_index, test_index = train_test_split(
        range(num_nodes), test_size=test_size, random_state=random_state, 
        stratify=data.y
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_index] = True
    test_mask[test_index] = True

    data.train_mask = train_mask
    data.test_mask = test_mask

    return data, train_index, test_index

# Apply the split to your data 
NC_D2, train_index, test_index = stratified_split(NC_D2)
print(NC_D2)


NC_D2.y = torch.tensor(NC_D2.y)

import torch
from torch.nn import Linear
import torch.nn.functional as F


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
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.lin3(x)
        return x

# from IPython.display import Javascript  # Restrict height of output cell.
# display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

from sklearn.metrics import roc_auc_score

model = MLP(num_features=745, num_classes=9, hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)  # Define optimizer.

def train(data):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.


    save_path = "Q2"

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,  
    }
    
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    
    torch.save(checkpoint, os.path.join(save_path, f'NC_D2_checkpoint_{epoch}.pt'))

    return loss



def test(data):
      model.eval()
      with torch.no_grad():  # Disable gradient computation for efficiency
          out = model(data.x)
          probs = out.softmax(dim=1)  # Get probabilities for each class
          pred = out.argmax(dim=1)  
          test_correct = pred[data.test_mask] == data.y.argmax(dim=1)[data.test_mask] 
          test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  

          # ROC AUC calculation (macro-averaged)
          y_true = data.y[data.test_mask].cpu().numpy() 
          y_scores = probs[data.test_mask].cpu().numpy()
          roc_auc = roc_auc_score(y_true, y_scores, multi_class='ovo') 


      return test_acc, roc_auc

max_acc = -1
max_roc_auc = -1
for epoch in range(1, 501):
    loss = train(NC_D2)
    test_acc, test_roc_auc = test(NC_D2)
    max_acc = max(test_acc, max_acc)
    max_roc_auc = max(test_roc_auc, max_roc_auc)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test acc: {test_acc}, Test roc_auc: {test_roc_auc}')

print(f'max acc: {max_acc}')