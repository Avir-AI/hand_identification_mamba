# Define a custom learning rate scheduler
class CustomLRScheduler:
    def __init__(self, optimizer, initial_lr, decay_lr, decay_epoch):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.decay_epoch = decay_epoch
        self.decay_lr = decay_lr
        self.current_epoch = 0

    def step(self):
        if ((self.current_epoch % self.decay_epoch) != 0) or (self.current_epoch == 0):
            # Linearly increase the learning rate
            lr = self.initial_lr
        elif (self.current_epoch % self.decay_epoch) == 0:
            # Keep the learning rate constant
            self.initial_lr *= self.decay_lr
            lr = self.initial_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1