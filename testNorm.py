import normalizer
import numpy as np

path = 'data/train_partition3_data.json'

print(f'Total of file three: {normalizer.counter(path)}')
weights = normalizer.counter('data/train_partition3_data.json', 1000)
print(f'Weights of the first 1000: {np.sum(weights)}')

tnsr, ys = normalizer.subSample(path, earlyStop=6, device='cpu')
print(f'Shape of tensor: should be 30, 33, 60: {tnsr.shape}')
print(f'lenght of ys: should be 30: {len(ys)}')
print(ys)
