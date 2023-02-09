import numpy as np
import matplotlib.pyplot as plt 

def example_plot(ax, img0):
    img0=np.squeeze(img0)
    # img0 = np.ma.masked_where(img0==-2, img0)
    c_img = ax.imshow(img0)
    return c_img

all_data = np.load('old/np-features.npy')

data = all_data[0:1,:,:,:]

d_BC = data[:,:,:, 0:1]

d_BC = np.where(d_BC==0.8, 0.7, d_BC)

n_BC_x = data[:,:,:, 1:2]
n_BC_y = data[:,:,:, 2:3]
y_s = 10
y_e = 22
n_BC_y[0,0,y_s:y_e] = n_BC_y[0,0,y_s:y_e] + 1.25

# fig, axs = plt.subplots(1, 1)
# # c_img = example_plot(axs, data[0,:,:,0])
# c_img = example_plot(axs, d_BC)
# # c_img = example_plot(axs, n_BC_x)
# # c_img = example_plot(axs, n_BC_y)
# plt.show()
# # exit(0)
# # print(data)

data = np.concatenate([d_BC, n_BC_x, n_BC_y], axis=3)
new_data = np.concatenate([data, data, data, data, data, data], axis=0)
print(np.shape(data), np.shape(new_data))
np.save('np-features.npy', new_data)
