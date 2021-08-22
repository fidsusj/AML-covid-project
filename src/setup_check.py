import torch

assert torch.rand(5, 3).numel()
assert torch.cuda.is_available()
print('Installation complete')
