import torch
from torchvision import transforms

class MinMaxNormalize(object):
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        min_x = tensor.min()
        max_x = tensor.max()

        if max_x == min_x:  
            return tensor
        
        normalized = (tensor - min_x) / (max_x - min_x) * (self.max_val - self.min_val) + self.min_val
        
        return normalized