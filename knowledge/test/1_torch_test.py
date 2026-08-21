import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())


"""
返回:
    True
    NVIDIA GeForce RTX 3050 Laptop GPU
"""