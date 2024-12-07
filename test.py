import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, precision_score
from model import load_model
from data import create_datasets, HandDataset
from schedule import *

# Load data
train_loader, val_loader, make_dataset = create_datasets()

# Model setup
model = load_model(make_dataset.num_classes1, make_dataset.num_classes2, make_dataset.num_classes3)
model = model.cuda()
model.load_state_dict(torch.load("./net/pre_trained_weights/best_schedule.pth", map_location="cuda"))

# Set up criterion, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Use CrossEntropyLoss with label smoothing

def val_result(key, val_df, device, model=model, criterion=criterion,
               data_transforms=make_dataset.data_transforms, val_id_one_hot=make_dataset.val_id_one_hot, 
               val_age_one_hot=make_dataset.val_age_one_hot, val_gender_one_hot=make_dataset.val_gender_one_hot):
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
    running_loss = 0.0

    all_preds1 = []
    all_preds2 = []
    all_preds3 = []
    all_labels1 = []
    all_labels2 = []
    all_labels3 = []

    # Disable gradient computation for testing
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels1, labels2, labels3 = labels
            labels1 = labels1.to(device)
            labels2 = labels2.to(device)
            labels3 = labels3.to(device)

            # Forward
            outputs1, outputs2, outputs3 = model(inputs)
            _, preds1 = torch.max(outputs1, 1)
            _, preds2 = torch.max(outputs2, 1)
            _, preds3 = torch.max(outputs3, 1)

            loss1 = criterion(outputs1, labels1.argmax(dim=1))
            loss2 = criterion(outputs2, labels2.argmax(dim=1))
            loss3 = criterion(outputs3, labels3.argmax(dim=1))
            loss = loss1 + loss2 + loss3

            # Statistics
            running_loss += loss.item() * inputs.size(0)

            # Store predictions and labels for F1 and recall calculation
            all_preds1.extend(preds1.cpu().numpy())
            all_labels1.extend(labels1.argmax(dim=1).cpu().numpy())
            all_preds2.extend(preds2.cpu().numpy())
            all_labels2.extend(labels2.argmax(dim=1).cpu().numpy())
            all_preds3.extend(preds3.cpu().numpy())
            all_labels3.extend(labels3.argmax(dim=1).cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)

    # Calculate F1 scores and recall
    f1_id = f1_score(all_labels1, all_preds1, average='weighted')
    f1_age = f1_score(all_labels2, all_preds2, average='weighted')
    f1_gender = f1_score(all_labels3, all_preds3, average='weighted')
    
    recall_id = recall_score(all_labels1, all_preds1, average='weighted')
    recall_age = recall_score(all_labels2, all_preds2, average='weighted')
    recall_gender = recall_score(all_labels3, all_preds3, average='weighted')

    pre_id = precision_score(all_labels1, all_preds1, average='weighted')
    pre_age = precision_score(all_labels2, all_preds2, average='weighted')
    pre_gender = precision_score(all_labels3, all_preds3, average='weighted')

    print(f'{key} Results:\nTest_Loss: {test_loss:.4f}, Pre_Id: {pre_id:.4f}, pre_Age: {pre_age:.4f}, Pre_Gender: {pre_gender:.4f}', end=', ')
    print(f'F1_Id: {f1_id:.4f} , F1_Age: {f1_age:.4f}, F1_Gender: {f1_gender:.4f}, Recall_Id: {recall_id:.4f}, Recall_Age: {recall_age:.4f}, Recall_Gender: {recall_gender:.4f}')

    return test_loss

# Set device
def validation(model_inp=False):
    global model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_inp:
        model = model_inp
    for key in make_dataset.val_dic:
        val_result(key, make_dataset.val_dic[key].copy(deep=True), device)

if __name__ == "__main__":
    validation()
