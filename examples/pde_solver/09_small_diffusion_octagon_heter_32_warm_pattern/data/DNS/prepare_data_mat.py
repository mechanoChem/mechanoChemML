import numpy as np


mat = [[0.5], [1.0], [2.0], [3.0], [4.0], [5.0]]
mat = np.array(mat)
print(np.shape(mat))
np.save('np-mats.npy', mat)

