import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from model_id import load_model
from data_open import create_datasets

# Load data
train_loader, _, make_dataset, vals, gals = create_datasets()

# Model setup
model = load_model()
model = model.cuda()

# Set up criterion and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(loader, model, device):
    """Extract features from the model for a given data loader."""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for data, _, _, label in loader:
            data = data.to(device)
            output = model(data)
            features.append(output)
            labels.append(label.to(device))
    
    return torch.cat(features), torch.cat(labels)

def compute_accuracy(val_features, val_labels, gal_features, gal_labels):
    """Compare labels of each output in vals with the 3 nearest outputs in gals."""
    # Compute pairwise distances on GPU
    distances = torch.cdist(val_features, gal_features)
    
    # Find the 3 nearest neighbors on GPU
    nearest_indices = distances.topk(2, largest=False).indices
    nearest_labels = gal_labels[nearest_indices]
    
    # Check if the correct label is among the top 3 nearest neighbors
    correct = (nearest_labels == val_labels.unsqueeze(1)).any(dim=1).sum().item()
    total = val_labels.size(0)

    return correct / total

def validate(model, val_loader, gals_loader, device):
    # Extract features for validation and gallery datasets
    val_features, val_labels = extract_features(val_loader, model, device)
    gal_features, gal_labels = extract_features(gals_loader, model, device)
    
    # Compute accuracy based on nearest neighbors
    accuracy = compute_accuracy(val_features, val_labels, gal_features, gal_labels)
    #print(f'Validation Accuracy: {accuracy:.4f}')
    return accuracy

def validation(model_inp):
    model.load_state_dict(torch.load(model_inp, map_location=device))
    
    # Validate on the entire validation dataset
    names = ('p_r', 'p_l', 'd_r', 'd_l')
    total_accuracy = 0
    for name, val_loader, gal_loader in zip(names, vals, gals):
        acc = validate(model, val_loader, gal_loader, device)
        print(f'Acc for {name}: {acc}')
        total_accuracy += acc
    print()
    return -1*total_accuracy

if __name__ == "__main__":
    validation("./net/pre_trained_weights/best.pth")
