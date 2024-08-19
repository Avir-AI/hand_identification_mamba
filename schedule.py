# Define a custom learning rate scheduler
class CustomLRScheduler:
    def __init__(self, optimizer, initial_lr, end_lr, last_lr, tresh_decay, tresh_last):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.last_lr = last_lr
        self.tresh_decay = tresh_decay
        self.tresh_last = tresh_last
        self.current_epoch = 0
        self.current_lr = initial_lr

    def step(self):
        if (self.current_epoch < self.tresh_decay):
            self.current_lr = self.initial_lr + (self.end_lr - self.initial_lr) * self.current_epoch / self.tresh_decay

        elif self.current_epoch > self.tresh_last:
            # Keep the learning rate constant
            self.current_lr = self.last_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_epoch
        
        self.current_epoch += 1