import numpy as np

lists = [10, 16, 22, 28, 34, 40, 46, 4, 52, 58]

for f0 in lists:
    f0 = 'np-features-' +str(f0)+'.npy'
    data = np.load(f0)
    d1 = data[:,:,:,0:1]
    n1 = data[:,:,:,1:2]
    n2 = data[:,:,:,2:3]
    print( np.max(n1), np.max(n2))
    # if np.max(n1) > 0.000001 and np.max(n2) <= 0.000001:
        # new_n2 = n2 + np.where(n1 > 0.000001, 0.5, 0.0) 
    # else:
        # new_n2 = n2
    # if np.max(n1) <= 0.000001 and np.max(n2) > 0.000001:
        # new_n1 = n1 + np.where(n2 > 0.000001, 0.5, 0.0) 
    # else: 
        # new_n1 = n1

    # new_data = np.concatenate([d1, new_n1, new_n2], axis=3)
    # np.save(f0, new_data)
