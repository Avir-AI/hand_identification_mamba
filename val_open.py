import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from model_id import load_model
from data_open import create_datasets, HandDataset

# Load data
train_loader, val_loader, make_dataset = create_datasets()

# Model setup
model = load_model()
model = model.cuda()

# Set up criterion and device
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(loader, model, device):
    """Extract features from the model for a given data loader."""
    model.eval()
    anchor_features = []
    positive_features = []
    negative_features = []
    labels = []
    
    with torch.no_grad():
        for anchor, positive, negative, anchor_label in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)
            
            anchor_features.append(anchor_output.cpu())
            positive_features.append(positive_output.cpu())
            negative_features.append(negative_output.cpu())
            labels.append(anchor_label.cpu())
    
    return (torch.cat(anchor_features), torch.cat(positive_features), torch.cat(negative_features)), torch.cat(labels)

def compute_triplet_accuracy(anchor_features, positive_features, negative_features):
    """Compute accuracy based on triplet constraints."""
    pos_dist = F.pairwise_distance(anchor_features, positive_features)
    neg_dist = F.pairwise_distance(anchor_features, negative_features)
    
    correct = (pos_dist < neg_dist).sum().item()
    total = anchor_features.size(0)
    
    return correct / total

def validate(model, val_loader, device):
    # Extract features for validation dataset
    (val_anchor_features, val_positive_features, val_negative_features), val_labels = extract_features(val_loader, model, device)
    
    # Compute triplet accuracy
    triplet_accuracy = compute_triplet_accuracy(val_anchor_features, val_positive_features, val_negative_features)
    print(f'Triplet Accuracy: {triplet_accuracy:.4f}')
    return -1 * triplet_accuracy

def validation(model_inp):
    model.load_state_dict(torch.load(model_inp, map_location="cuda"))
    
    # Validate on the entire validation dataset
    total_accuracy = validate(model, val_loader, device)
    
    return total_accuracy
