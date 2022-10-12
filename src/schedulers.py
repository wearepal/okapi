from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

__all__ = [
    "FixMatchLR",
]


class FixMatchLR(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        max_steps: int,
    ) -> None:
        self.max_steps = max_steps
        super().__init__(optimizer, lr_lambda=self._lr_lambda)

    def _lr_lambda(self, step: int) -> float:
        return (1.0 + 10 * float(step) / self.max_steps) ** -0.75
