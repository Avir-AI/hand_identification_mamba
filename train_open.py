import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from model_id import load_model
from data_open import create_datasets
import copy
import sys
from schedule import CustomLRScheduler  # Ensure this is correctly imported
from val_open import validation  # Ensure this is correctly imported

# Parameters
lr = 8e-6
num_epochs = 70

# Load data
train_loader, val_loader, make_dataset = create_datasets()

# Model setup
model = load_model()
model = model.cuda()

# Set up criterion, optimizer, and learning rate scheduler
criterion = nn.TripletMarginLoss(margin=1.0, p=2)  # Use TripletMarginLoss for triplet-based training
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
scheduler = CustomLRScheduler(optimizer, lr, 8e-4, 10, 30, 4e-4)

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter('runs/experiment_2')

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                scheduler, num_epochs=25, patience=10, base_model_path="./net/pre_trained_weights/best.pth"):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (anchor, positive, negative, _) in enumerate(train_loader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)
            loss = criterion(anchor_output, positive_output, negative_output)
            
            # Backward + optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * anchor.size(0)
            
            # Print loss status after each batch
            sys.stdout.write(f'\rBatch {batch_idx}/{len(train_loader) - 1} Loss: {loss:.4f}')
            sys.stdout.flush()

        epoch_loss = running_loss / len(train_loader.dataset)

        # Log to TensorBoard
        writer.add_scalar('train/Loss', epoch_loss, epoch)

        print(f'Train Loss: {epoch_loss:.4f}')

        torch.save(model.state_dict(), base_model_path)  # Save the best model
        scheduler.step()
        # Validation phase
        if epoch % 5 == 0:
            model.eval()
            val_loss = validation(base_model_path)
            
            
            if val_loss < best_val_loss:
                print(f'Current best is in epoch {epoch}.')
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                model.load_state_dict(best_model_wts)
                writer.close()
                return model

        print()

    print(f'Best val Loss: {best_val_loss:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), base_model_path)  # Save the best model
    writer.close()
    return model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
model = train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                    scheduler, num_epochs=num_epochs, patience=5)
