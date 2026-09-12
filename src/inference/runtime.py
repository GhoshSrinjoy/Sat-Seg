"""One GPU owner across language, inference and classifier training in this server."""

import threading
from contextlib import contextmanager

GPU_LOCK = threading.RLock()
_training = None


def set_training(process):
    global _training
    _training = process


def training_active():
    return _training is not None and _training.poll() is None


@contextmanager
def gpu_work():
    with GPU_LOCK:
        if training_active():
            raise RuntimeError("Training is using the GPU. Wait for it or cancel it in Advanced.")
        yield
