import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from model import load_model
from data_open import create_datasets, HandDataset

# Load data
train_loader, val_loader, make_dataset = create_datasets()

# Model setup
model = load_model(make_dataset.num_classes1, make_dataset.num_classes2, make_dataset.num_classes3)
model = model.cuda()

# Set up criterion and device
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model weights
model.load_state_dict(torch.load("./net/pre_trained_weights/best_no_schedule.pth", map_location="cuda"))

def extract_features(loader, model, device, sia = False):
    """Extract features from the model for a given data loader."""
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, lbls in loader:
            inputs = inputs.to(device)
            outputs, _, _ = model(inputs)
            features.append(outputs.cpu())
            for lb in lbls:
                if lb.shape[1] > 30:
                    labels.append(lb.cpu())  # Assuming you are interested in labels1
    return torch.cat(features), torch.cat(labels)

def compute_cosine_similarity(features1, features2):
    """Compute cosine similarity between two sets of features."""
    # print(features1.shape)
    # print(features2.shape)
    # print(F.cosine_similarity(features1.unsqueeze(1), features2.unsqueeze(0), dim=-1).shape)
    return F.cosine_similarity(features1.unsqueeze(1), features2.unsqueeze(0), dim=-1)

def rank_1_accuracy(gallery_features, gallery_labels, val_features, val_labels):
    """Compute rank-1 accuracy based on cosine distance."""
    cosine_sim = compute_cosine_similarity(val_features, gallery_features)
    _, top1_indices = cosine_sim.topk(1, dim=1, largest=True, sorted=True)
    top1_indices = top1_indices.squeeze(1)
    #print(gallery_labels.shape)
    #print(torch.eq(gallery_labels[top1_indices], val_labels).all(dim = 1))
    correct = (torch.eq(gallery_labels[top1_indices], val_labels).all(dim = 1)).sum().item()
    return correct / val_labels.size(0)

def validate(model, key, val_loader, gallery_loader, device):
    # Extract features for validation and gallery datasets
    val_features, val_labels = extract_features(val_loader, model, device)
    gallery_features, gallery_labels = extract_features(gallery_loader, model, device, True)
    
    # Compute rank-1 accuracy
    rank1_acc = rank_1_accuracy(gallery_features, gallery_labels, val_features, val_labels)
    print(f'Acc_id: {rank1_acc:.4f}', end=', ')


def val_result(key, val_df, device, model = model, criterion = criterion,
               data_transforms = make_dataset.data_transforms, val_id_one_hot = make_dataset.val_id_one_hot, 
               val_age_one_hot = make_dataset.val_age_one_hot, val_gender_one_hot = make_dataset.val_gender_one_hot):
    val_id_one_hot = val_id_one_hot.loc[val_df.index, :]
    val_age_one_hot = val_age_one_hot.loc[val_df.index, :]
    val_gender_one_hot = val_gender_one_hot.loc[val_df.index, :]
    # Create datasets
    val_dataset = HandDataset(val_df, [val_id_one_hot, val_age_one_hot, val_gender_one_hot], 
                              make_dataset.image_directory, transform=data_transforms['val'])
    # Create data loaders
    test_loader = DataLoader(val_dataset, batch_size=make_dataset.batch_size, shuffle=False, num_workers=0)
    torch.cuda.empty_cache()
    model.eval()  # Set model to evaluate mode
    #running_loss = 0.0
    running_corrects1 = 0
    running_corrects2 = 0
    running_corrects3 = 0
    # Disable gradient computation for testing
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels1, labels2, labels3 = labels
            #labels1 = labels1.to(device)
            labels2 = labels2.to(device)
            labels3 = labels3.to(device)

            # Forward
            outputs1, outputs2, outputs3 = model(inputs)
            #_, preds1 = torch.max(outputs1, 1)
            _, preds2 = torch.max(outputs2, 1)
            _, preds3 = torch.max(outputs3, 1)
            # loss1 = criterion(outputs1, labels1.argmax(dim=1))
            # loss2 = criterion(outputs2, labels2.argmax(dim=1))
            # loss3 = criterion(outputs3, labels3.argmax(dim=1))
            # loss = loss1 + loss2 + loss3

            # Statistics
            #running_loss += loss.item() * inputs.size(0)
            #running_corrects1 += torch.sum(preds1 == labels1.argmax(dim=1).data)
            running_corrects2 += torch.sum(preds2 == labels2.argmax(dim=1).data)
            running_corrects3 += torch.sum(preds3 == labels3.argmax(dim=1).data)

    #test_loss = running_loss / len(test_loader.dataset)
    #test_acc1 = running_corrects1.double() / len(test_loader.dataset)
    test_acc2 = running_corrects2.double() / len(test_loader.dataset)
    test_acc3 = running_corrects3.double() / len(test_loader.dataset)

    print(f'Acc_Age: {test_acc2:.4f} Acc_Gender: {test_acc3:.4f}')
    
    return -1 * (test_acc2 + test_acc3)



def validation(model_inp = None):
    global model
    if model_inp:
        model = model_inp
    # Prepare validation and gallery data loaders
    tot = 0
    for key in make_dataset.val_dic:
        val_df = make_dataset.val_dic[key].copy(deep=True)
        gallery_df = make_dataset.gal_dic[key].copy(deep=True)
        
        val_id_one_hot = make_dataset.val_id_one_hot.loc[val_df.index, :]
        gallery_id_one_hot = make_dataset.gal_id_one_hot.loc[gallery_df.index, :]
        
        val_dataset = HandDataset(val_df, [val_id_one_hot, make_dataset.val_age_one_hot, make_dataset.val_gender_one_hot], 
                                make_dataset.image_directory, transform=make_dataset.data_transforms['val'])
        gallery_dataset = HandDataset(gallery_df, [gallery_id_one_hot, make_dataset.val_age_one_hot, make_dataset.val_gender_one_hot], 
                                    make_dataset.image_directory, transform=make_dataset.data_transforms['val'])
        
        val_loader = DataLoader(val_dataset, batch_size=make_dataset.batch_size, shuffle=False, num_workers=0)
        gallery_loader = DataLoader(gallery_dataset, batch_size=make_dataset.batch_size, shuffle=False, num_workers=0)
        print(f'Results for {key}:', end=' ')
        validate(model, key, val_loader, gallery_loader, device)
        if key == 'Total':
            tot = val_result(key, val_df, device, model)
        else:
            val_result(key, val_df, device, model)
    return tot
