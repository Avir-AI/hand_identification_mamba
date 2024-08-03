# Define a custom learning rate scheduler
class CustomLRScheduler:
    def __init__(self, optimizer, initial_lr, final_lr, increase_epochs, decay_epoch, decay_lr):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.increase_epochs = increase_epochs
        self.decay_epoch = decay_epoch
        self.decay_lr = decay_lr
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.increase_epochs:
            # Linearly increase the learning rate
            lr = self.initial_lr + (self.final_lr - self.initial_lr) * (self.current_epoch / self.increase_epochs)
        elif self.current_epoch < self.decay_epoch:
            # Keep the learning rate constant
            lr = self.final_lr
        else:
            # Decay the learning rate
            lr = self.decay_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1