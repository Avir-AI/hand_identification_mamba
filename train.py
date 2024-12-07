import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from model import load_model
from data import create_datasets
import copy
import sys
from schedule import *
from val import validation


# Parameters
lr = 8e-6
num_epochs = 70

#load data
train_loader, val_loader, make_dataset = create_datasets()

# Model setup
model = load_model(make_dataset.num_classes1, make_dataset.num_classes2, make_dataset.num_classes3)
model = model.cuda()

# Set up criterion, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Use CrossEntropyLoss with label smoothing
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
scheduler = CustomLRScheduler(optimizer, lr, 8e-4, 4e-4, 10, 30)
# Initialize TensorBoard SummaryWriter
writer = SummaryWriter('runs/experiment_2')

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                scheduler, num_epochs=25, patience=10, base_model_path="./net/pre_trained_weights/best_schedule.pth"):
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'val':
                lss = validation(model)
                scheduler.step()
                if lss < best_val_loss:
                    print(f'Current best is in epoch {epoch}.')
                    best_val_loss = lss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), base_model_path)  # Save the base model
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    model.load_state_dict(best_model_wts)
                    writer.close()
                    return model
            else:
                model.train()  # Set model to training mode
                data_loader = train_loader
            running_loss = 0.0
            running_corrects1 = 0
            running_corrects2 = 0
            running_corrects3 = 0

            # Iterate over data
            for batch_idx, (inputs, labels) in enumerate(data_loader):
                inputs = inputs.to(device)
                labels1, labels2, labels3 = labels
                labels1 = labels1.to(device)
                labels2 = labels2.to(device)
                labels3 = labels3.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs1, outputs2, outputs3 = model(inputs)
                    _, preds1 = torch.max(outputs1, 1)
                    _, preds2 = torch.max(outputs2, 1)
                    _, preds3 = torch.max(outputs3, 1)
                    loss1 = criterion(outputs1, labels1.argmax(dim=1))
                    loss2 = criterion(outputs2, labels2.argmax(dim=1))
                    loss3 = criterion(outputs3, labels3.argmax(dim=1))
                    loss = loss1 + loss2 + loss3

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects1 += torch.sum(preds1 == labels1.argmax(dim=1).data)
                running_corrects2 += torch.sum(preds2 == labels2.argmax(dim=1).data)
                running_corrects3 += torch.sum(preds3 == labels3.argmax(dim=1).data)

                # Print loss status after each batch
                if phase == 'train':
                    sys.stdout.write(f'\rBatch {batch_idx}/{len(data_loader) - 1} Loss: {loss.item():.4f}')
                    sys.stdout.flush()

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc1 = running_corrects1.double() / len(data_loader.dataset)
            epoch_acc2 = running_corrects2.double() / len(data_loader.dataset)
            epoch_acc3 = running_corrects3.double() / len(data_loader.dataset)

            # Log to TensorBoard
            writer.add_scalar(f'{phase}/Loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}/Accuracy1', epoch_acc1, epoch)
            writer.add_scalar(f'{phase}/Accuracy2', epoch_acc2, epoch)
            writer.add_scalar(f'{phase}/Accuracy3', epoch_acc3, epoch)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc1: {epoch_acc1:.4f} Acc2: {epoch_acc2:.4f} Acc3: {epoch_acc3:.4f}')
            torch.save(model.state_dict(), base_model_path)

            # Deep copy the model
            # if phase == 'val':
            #     scheduler.step()
            #     avg_acc = (epoch_acc1 + epoch_acc2 + epoch_acc3) / 3
            #     if avg_acc > best_acc:
            #         print(f'Current best is in epoch {epoch}.')
            #         best_acc = avg_acc
            #         best_model_wts = copy.deepcopy(model.state_dict())
            #         epochs_no_improve = 0
            #         torch.save(model.state_dict(), base_model_path)  # Save the base model
            #     else:
            #         epochs_no_improve += 1

            #     if epochs_no_improve >= patience:
            #         print(f'Early stopping at epoch {epoch}')
            #         model.load_state_dict(best_model_wts)
            #         writer.close()
            #         return model

        print()

    print(f'Best val Acc: {best_val_loss:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    writer.close()
    return model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Train the model
model = train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                    scheduler, num_epochs=num_epochs, patience=5)