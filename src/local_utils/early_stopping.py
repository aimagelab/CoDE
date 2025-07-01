class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.validation_accuracy = 0

    def early_stop(self, validation_accuracy):
        if validation_accuracy > self.validation_accuracy:
            self.validation_accuracy = validation_accuracy
            self.counter = 0
        elif validation_accuracy < (self.validation_accuracy - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False