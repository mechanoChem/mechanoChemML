import numpy as np
import pickle
import sys

# all_fields = {'inputs':inputs, 'labels':labels, 'mean':mean, 'var':var, 'std':std}
# pickle_out = open('all_fields.pickle', "wb")
# pickle.dump(all_fields, pickle_out)
# pickle_out.close()

scale_factor = 2.0 # 
scale_factor = 0.1 # 

###########################################################################
def write_vtk(shp,data=None,fname='grid.vtk'):
###########################################################################
  nx,ny,nz = shp
  f = open(fname,'w')
  f.write("# vtk DataFile Version 3.0\n")
  f.write("# {0}\n".format(fname))
  f.write("ASCII\n")
  f.write("DATASET UNSTRUCTURED_GRID\n")
  npts = (nx+1)*(ny+1)*(nz+1)
  f.write("POINTS {0} float\n".format(npts))
  # C order
  for j in range(ny,-1,-1):
    for i in range(nx+1):
      f.write("{0} {1} {2}\n".format(float(i/(nx)),float(j/(ny)),float(0.0)))
  f.write("\n")
  nelms = nx*ny
  f.write("CELLS {0} {1}\n".format(nelms,5*nelms))
  mx  = (nx+1)
  my  = (ny+1)
  mz  = (nz+1)
  mxy = mx*my
  myz = my*mz
  for i in range(nx):
    for j in range(ny):
        n1 = myz*i+mz*j  # o
        n2 = n1+1          # o + 1k
        n3 = n2+my         # o + 1k + 1j
        n4 = n3-1          # o      + 1j
        f.write("{0} {1} {2} {3} {4}\n".format(4,n1,n2,n3,n4))
  f.write("\n")
  f.write("CELL_TYPES {0}\n".format(nelms))
  for i in range(nelms): f.write("{0} ".format(9))
  f.write("\n\n")
  ## C order: last index is continuous
  f.write("POINT_DATA {0}\n".format(npts))
  # for name,ds in data.items():
    # f.write("SCALARS {0} float\n".format(name))
    # f.write("LOOKUP_TABLE default\n")
    # for d in ds: f.write("{0}\n".format(d))
  f.write("VECTORS {0} float\n".format('NN'))

  for i0 in range(0, len(data['ux'])):
    f.write("{0} {1} {2}\n".format( scale_factor*(data['ux'][i0]-0.5), scale_factor*(data['uy'][i0]-0.5), 0.0))
  f.write("\n")

  f.write("VECTORS {0} float\n".format('BNN'))

  for i0 in range(0, len(data['ux'])):
    f.write("{0} {1} {2}\n".format( scale_factor*(data['uxb'][i0]-0.5), scale_factor*(data['uyb'][i0]-0.5), 0.0))
  f.write("\n")

  f.write("SCALARS {0} float\n".format('markx')) # name for the variable
  f.write("LOOKUP_TABLE default\n")
  for d in data['markx']: f.write("{0}\n".format(d))
  f.write("\n")

  f.write("SCALARS {0} float\n".format('marky')) # name for the variable
  f.write("LOOKUP_TABLE default\n")
  for d in data['marky']: f.write("{0}\n".format(d))
  f.write("\n")

  f.write("SCALARS {0} float\n".format('uxl')) # name for the variable
  f.write("LOOKUP_TABLE default\n")
  for d in data['markx']: f.write("{0}\n".format( scale_factor*(d-0.5) ))
  f.write("\n")

  f.write("SCALARS {0} float\n".format('uyl')) # name for the variable
  f.write("LOOKUP_TABLE default\n")
  for d in data['marky']: f.write("{0}\n".format( scale_factor*(d-0.5) ))
  f.write("\n")

  f.write("VECTORS {0} float\n".format('DNS'))

  for i0 in range(0, len(data['markx'])):
    f.write("{0} {1} {2}\n".format( scale_factor*(data['markx'][i0]-0.5), scale_factor*(data['marky'][i0]-0.5), 0.0))
  f.write("\n")


  f.close()

print("Usage: " + sys.argv[0] + " pickle_file " + " load_index ")
load_index = 0

# pickle_file = 'all_fields.pickle'
pickle_file = sys.argv[1]
if len(sys.argv) > 2:
    load_index = int(sys.argv[2])

saved_data = pickle.load(open(pickle_file, "rb"))
for key, item in saved_data.items():
    print(key)

means = saved_data['mean']
print('means: ', np.shape(means))

labels = saved_data['labels']
print('labels:', np.shape(labels))
labels_cnn = labels[0]
labels_bnn = labels[1]
one_mark = labels_cnn[load_index]

mean_cnn = means[0]
mean_bnn = means[1]
print('mean_cnn: ', np.shape(mean_cnn))
print('mean_bnn: ', np.shape(mean_bnn))

one_cnn = mean_cnn[load_index]
print('one_cnn: ', np.shape(one_cnn))
one_bnn = mean_bnn[load_index]

shp_info = np.shape(one_cnn)

# losses = saved_data['losses']
# for key, item in losses.items():
    # print(key)
write_vtk(shp=(shp_info[0]-1,shp_info[1]-1,0), data={'ux':one_cnn.numpy()[:,:,0].reshape(-1), 'uy':one_cnn.numpy()[:,:,1].reshape(-1), 'markx':one_mark[:,:,0].reshape(-1), 'marky':one_mark[:,:,1].reshape(-1), 'uxb':one_bnn.numpy()[:,:,0].reshape(-1), 'uyb':one_bnn.numpy()[:,:,1].reshape(-1)} )
