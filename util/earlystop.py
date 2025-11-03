from typing import Optional


class EarlyStop:
    def __init__(self, patience: int = 5) -> None:
        self.patience: int = patience
        self.counter: int = 0
        self.best_score: Optional[float] = None

    def __call__(self, score: float) -> bool:

        if self.best_score is None:
            self.best_score = score
            return False

        elif score >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False

    def is_best_score(self, score: float) -> bool:
        return score < self.best_score if self.best_score is not None else True
