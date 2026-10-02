"""RNG seed policy and state serialization."""

import random
from typing import Any

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> dict[str, Any]:
    """Sets random seeds across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    rng_info: dict[str, Any] = {
        "seed": seed,
        "deterministic": deterministic,
        "torch": False,
    }

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        rng_info["torch"] = True
    except ImportError:
        pass

    return rng_info


def get_rng_states() -> dict[str, Any]:
    """Serializes current random states for checkpoint saving."""
    states: dict[str, Any] = {
        "python": str(random.getstate()),
        "numpy": [str(x) for x in np.random.get_state() if not isinstance(x, np.ndarray)],
    }
    try:
        import torch
        states["torch"] = str(torch.get_rng_state().tolist()[:16])
        if torch.cuda.is_available():
            states["torch_cuda"] = [
                str(s.tolist()[:16]) for s in torch.cuda.get_rng_state_all()
            ]
    except ImportError:
        pass
    return states
