import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_rng_state():
    np_state = np.random.get_state()

    return {
        "python": random.getstate(),
        "numpy": (
            np_state[0],
            np_state[1].tolist(),
            int(np_state[2]),
            int(np_state[3]),
            float(np_state[4]),
        ),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

def restore_rng_state(state):
    random.setstate(state["python"])

    name, keys, pos, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((
        name,
        np.array(keys, dtype=np.uint32),
        pos,
        has_gauss,
        cached_gauss,
    ))

    torch.set_rng_state(state["torch"].cpu())

    if state["cuda"] is not None and torch.cuda.is_available():
        cuda_states = [s.cpu() for s in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)