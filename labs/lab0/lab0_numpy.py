import numpy as np

a = np.array([0, 0, 4])
b = np.array([0, 3, 0])

dist = np.linalg.norm(a - b)
print(dist)
