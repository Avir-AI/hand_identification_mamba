import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model import load_model
from data import create_datasets, HandDataset
from schedule import *

#load data
train_loader, val_loader, make_dataset = create_datasets()

# Model setup
model = load_model(make_dataset.num_classes1, make_dataset.num_classes2, make_dataset.num_classes3)
model = model.cuda()

# Set up criterion, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Use CrossEntropyLoss with label smoothing

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
    running_loss = 0.0
    running_corrects1 = 0
    running_corrects2 = 0
    running_corrects3 = 0
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
            running_corrects1 += torch.sum(preds1 == labels1.argmax(dim=1).data)
            running_corrects2 += torch.sum(preds2 == labels2.argmax(dim=1).data)
            running_corrects3 += torch.sum(preds3 == labels3.argmax(dim=1).data)

    test_loss = running_loss / len(test_loader.dataset)
    test_acc1 = running_corrects1.double() / len(test_loader.dataset)
    test_acc2 = running_corrects2.double() / len(test_loader.dataset)
    test_acc3 = running_corrects3.double() / len(test_loader.dataset)

    print(f'{key} results: Test Loss: {test_loss:.4f} Acc_Id: {test_acc1:.4f} Acc_Age: {test_acc2:.4f} Acc_Gender: {test_acc3:.4f}')
    
    return test_loss, test_acc1, test_acc2, test_acc3

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("./net/pre_trained_weights/best_no_schedule.pth", map_location="cuda"))
for key in make_dataset.val_dic:
    val_result(key, make_dataset.val_dic[key].copy(deep = True), device)



