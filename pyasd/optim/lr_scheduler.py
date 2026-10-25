import math
import torch
from torch.optim.lr_scheduler import LRScheduler

def _flat_cosine_schedule(current_iter, warmup_iter, flat_iter, total_iter, new_lr_iter, init_lr, min_lr):
    """Compute the learning rate for one iteration of the flat-cosine schedule.

    Phases: linear warmup from 0 to ``init_lr`` until ``warmup_iter``, constant
    ``init_lr`` until ``flat_iter``, cosine decay to ``min_lr``, and constant
    ``min_lr`` for the last ``new_lr_iter`` iterations.

    Args:
        current_iter: Current iteration.
        warmup_iter: Number of linear warmup iterations.
        flat_iter: Iteration at which the flat phase ends.
        total_iter: Total number of iterations.
        new_lr_iter: Number of final iterations held at ``min_lr``.
        init_lr: Peak learning rate.
        min_lr: Final learning rate.

    Returns:
        float: Learning rate for ``current_iter``.
    """
    # (This function remains unchanged)
    if current_iter < warmup_iter:
        return init_lr * current_iter / warmup_iter
    elif warmup_iter <= current_iter < flat_iter:
        return init_lr
    elif current_iter >= total_iter - new_lr_iter:
        return min_lr
    else:
        cosine_decay_iters = total_iter - flat_iter - new_lr_iter
        iters_since_flat = current_iter - flat_iter
        cosine_decay = 0.5 * (1 + math.cos(math.pi * iters_since_flat / cosine_decay_iters))
        return min_lr + (init_lr - min_lr) * cosine_decay


class FlatCosineLRScheduler(LRScheduler):
    """
    A PyTorch learning rate scheduler that implements a flat-cosine schedule.
    This scheduler is designed to be updated at each training iteration.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 total_iter: int,
                 warmup_iter: int,
                 flat_iter: int = 0,
                 new_lr_iter: int = 0,
                 lr_gamma: float = 0.05,
                 last_epoch: int = -1,
                 verbose: bool = False):
        """Set up the schedule and print a summary of its phases.

        Args:
            optimizer: Wrapped optimizer. Its current ``lr`` values are the peak rates.
            total_iter: Total number of training iterations.
            warmup_iter: Number of linear warmup iterations.
            flat_iter: Iteration at which the flat phase ends (at least ``warmup_iter``).
            new_lr_iter: Number of final iterations held at the minimum rate.
            lr_gamma: Minimum rate as a fraction of each group's peak rate.
            last_epoch: Index of the last iteration; -1 starts a new schedule.
            verbose: Unused.
        """

        self.total_iter = total_iter
        self.warmup_iter = warmup_iter
        self.flat_iter = max(flat_iter, warmup_iter)
        self.new_lr_iter = new_lr_iter
        
        # 1. Get base_lrs directly from the optimizer
        base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # 2. Create self.min_lrs BEFORE the super() call
        self.min_lrs = [base_lr * lr_gamma for base_lr in base_lrs]

        # FIX: Call the parent constructor FIRST to initialize self.base_lrs
        super().__init__(optimizer, last_epoch)
        
        self._log_schedule()
        
        # Set the initial learning rate
        if last_epoch == -1:
            self.step()


    def _log_schedule(self):
        """Uses the library's logger to print the schedule details."""
        # This function requires self.base_lrs, which is why super() must be called first
        base_lr = self.base_lrs[0]
        min_lr = self.min_lrs[0]

        warmup_duration = self.warmup_iter
        flat_duration = max(0, self.flat_iter - self.warmup_iter)
        cosine_end_iter = self.total_iter - self.new_lr_iter
        cosine_duration = max(0, cosine_end_iter - self.flat_iter)
        final_duration = self.new_lr_iter

        total_from_phases = warmup_duration + flat_duration + cosine_duration + final_duration
        check_mark = "✅" if self.total_iter == total_from_phases else "⚠️"

        print("FlatCosineLRScheduler Initialized")
        print(f"  - LR Range: {base_lr:.1e} -> {min_lr:.1e}")
        print("  - Schedule Breakdown (in iterations):")
        print(f"    - Warmup Phase:  {warmup_duration:>7,} iters (ends at {self.warmup_iter:,})")
        if flat_duration > 0:
            print(f"    - Flat Phase:    {flat_duration:>7,} iters (ends at {self.flat_iter:,})")
        if cosine_duration > 0:
            print(f"    - Cosine Decay:  {cosine_duration:>7,} iters (ends at {cosine_end_iter:,})")
        print(f"    - Final Phase:   {final_duration:>7,} iters (at min_lr)")
        print("  - Verification:")
        print(f"    - Sum of Phases: {total_from_phases:,} iters")
        print(f"    - Expected Total: {self.total_iter:,} iters {check_mark}")
        print("-----------------------------------------")

    def get_lr(self):
        """
        Compute the learning rate for the current step.
        `self.last_epoch` is used as the current iteration counter.

        Returns:
            list[float]: Learning rate for each parameter group.
        """
        current_iter = self.last_epoch

        return [
            _flat_cosine_schedule(
                current_iter=current_iter,
                warmup_iter=self.warmup_iter,
                flat_iter=self.flat_iter,
                total_iter=self.total_iter,
                new_lr_iter=self.new_lr_iter,
                init_lr=base_lr,
                min_lr=min_lr
            )
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]