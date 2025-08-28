class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as improvement.
            verbose (bool): If True, prints a message when stopping early.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, val_loss):
        score = -val_loss  # minimize loss → maximize (-loss)

        if self.best_score is None:
            self.best_score = score
            return False  # continue training

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"⏳ EarlyStopping patience {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print("🛑 Early stopping triggered.")
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0  # reset counter
        return False