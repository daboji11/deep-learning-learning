import torch


# helper function to use mps but not gpu
# substitution for try_gpu :)
def try_mps():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")