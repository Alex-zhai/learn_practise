import numpy as np

img = np.random.normal(size=(2, 3))


max_ax = max((0, 1), key=lambda i: img.shape[i]) 
print(max_ax)
