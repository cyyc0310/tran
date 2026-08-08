"""Lightweight training progress utilities (no external deps).

Provides a simple callback-based progress bar that prints epoch-level loss
trends to stderr with a ``\\r`` carriage return, so long training loops show
live progress without pulling in tqdm.
"""
import sys


class TrainProgress:
    """Callback passed into ``train_*`` functions.

    Call ``progress(epoch, total, loss)`` every few epochs from inside the
    training loop; it renders a single-line bar to stderr::

        [RAG] 120/200 [=====>     ] loss=0.0831

    The label is fixed at construction so each method gets its own line prefix.
    """

    def __init__(self, label="", width=20, every=10):
        self.label = label
        self.width = width
        self.every = every
        self._last_loss = None

    def __call__(self, epoch, total, loss, extra=""):
        self._last_loss = loss
        if (epoch + 1) % self.every != 0 and (epoch + 1) != total:
            return
        frac = (epoch + 1) / total
        filled = int(self.width * frac)
        bar = "=" * filled + ">" + " " * (self.width - filled - 1)
        msg = f"\r    [{self.label}] {epoch+1:>4}/{total} [{bar}] loss={loss:.4f}"
        if extra:
            msg += f"  {extra}"
        sys.stdout.write(msg)
        sys.stdout.flush()

    def finish(self):
        sys.stdout.write("\n")
        sys.stdout.flush()
