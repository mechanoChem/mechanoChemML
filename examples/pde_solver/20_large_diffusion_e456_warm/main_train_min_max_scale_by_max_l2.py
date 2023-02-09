from mechanoChemML.workflows.pde_solver.pde_system_diffusion_steady_state import WeakPDESteadyStateDiffusion as thisPDESystem
import tensorflow as tf
import numpy as np

def write_file(filepath,text):
    with open(filepath, "w") as myfile:
        myfile.write(text)

def statistics_data(_inputs, _labels, _mean, _var, _std, _edge):
    print('labels:',tf.shape(_labels))
    print('mean:', tf.shape(_mean))
    _dirichlet_bc = _inputs[:,:,:,0:1]
    _domain_from_dirichlet_bc = tf.where(_dirichlet_bc != -1, tf.ones_like(_dirichlet_bc, dtype=tf.float32), tf.zeros_like(_dirichlet_bc, dtype=tf.float32))
    _domain_from_label = tf.where(_labels > 0, tf.ones_like(_labels, dtype=tf.float32), tf.zeros_like(_labels, dtype=tf.float32))
    # print(_domain_from_dirichlet_bc[0,0:30,0:30,0])
    # print(_domain_from_label[0,0:30,0:30,0])
    _domain_union = tf.multiply(_domain_from_dirichlet_bc, _domain_from_label)
    # print(_domain_union[0,0:30,0:30,0])
    # pixel_count = tf.reduce_sum (_domain_from_label, axis=[1,2,3])
    pixel_count = tf.reduce_sum (_domain_union, axis=[1,2,3])
    # print(tf.reduce_sum (_domain_union, axis=[1,2,3]), tf.reduce_sum (_domain_from_dirichlet_bc, axis=[1,2,3]), tf.reduce_sum (_domain_from_label, axis=[1,2,3]))
    # print('pixel_count: ', tf.shape(pixel_count))
    # exit(0)
    # print(tf.squeeze(pixel_count))

    # _labels = tf.where(_labels > 0, _labels, tf.zeros_like(_labels, dtype=tf.float32))
    # _mean = tf.where(_labels > 0, _mean, tf.zeros_like(_mean, dtype=tf.float32))

    _labels = tf.multiply(_labels, _domain_union)
    _mean = tf.multiply(_mean, _domain_union)

    _labels_max = tf.reduce_max(_labels, axis=[1,2,3], keepdims=True)
    # print(_labels_max)

    # dy_DNS_NN = _labels - _mean
    dy_DNS_NN = tf.divide((_labels - _mean), _labels_max) # 
    # print('dy_DNS_NN error:', tf.reduce_sum (dy_DNS_NN, axis=[1,2,3]))
    # print('dy_DNS_NN error2:', tf.reduce_sum (tf.multiply(dy_DNS_NN, dy_DNS_NN), axis=[1,2,3]))
    # print('dy_DNS_NN error2 ave:', tf.reduce_sum (tf.multiply(dy_DNS_NN, dy_DNS_NN), axis=[1,2,3])/pixel_count )
    # print('dy_DNS_NN error2 sqrt:', tf.sqrt(tf.reduce_sum (tf.multiply(dy_DNS_NN, dy_DNS_NN), axis=[1,2,3])/pixel_count))
    l2_error = tf.sqrt(tf.reduce_sum (tf.multiply(dy_DNS_NN, dy_DNS_NN), axis=[1,2,3])/pixel_count)
    print(tf.shape(l2_error), l2_error)
    # l2_error = tf.reduce_sum(l2_error)/tf.shape(_labels).to_numpy()[0]
    # print('all BVP l2 error:', l2_error.numpy())
    # _inputs = _inputs.numpy()

    _inputs = np.ma.masked_where(_inputs <= 0, _inputs)
    _labels = np.ma.masked_where(_labels <= 0, _labels)
    l2_error = l2_error.numpy()
    dy_DNS_NN = dy_DNS_NN.numpy()

    label_mean = tf.reduce_sum (_labels, axis=[1,2,3])/pixel_count

    text_entries = []
    for i0 in range(0, tf.shape(_labels).numpy()[0]):
        Dirichlet_bc = _inputs[i0,:,:,0:1]
        Neumann_bc_1 = _inputs[i0,:,:,1:2]
        Neumann_bc_2 = _inputs[i0,:,:,2:3]
        one_label = _labels[i0,:,:,0]
        # print(one_label[30:35, 30:35])

        Dirichlet_min = np.amin(Dirichlet_bc)
        Dirichlet_max = np.amax(Dirichlet_bc)
        Neumann_min_1 = np.amin(Neumann_bc_1)
        Neumann_max_1 = np.amax(Neumann_bc_1)
        Neumann_min_2 = np.amin(Neumann_bc_2)
        Neumann_max_2 = np.amax(Neumann_bc_2)
        label_min = np.amin(one_label)
        label_max = np.amax(one_label)

        error = dy_DNS_NN[i0,:,:,:]
        error_min = np.amin(error)
        error_max = np.amax(error)

        print(i0, [Dirichlet_min, Dirichlet_max], [Neumann_min_1, Neumann_max_1], [Neumann_min_2, Neumann_max_2], l2_error[i0], label_mean[i0])
        one_entry = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}".format(_edge, i0, l2_error[i0], error_min, error_max,Dirichlet_min,Dirichlet_max,Neumann_min_1, Neumann_max_1, Neumann_min_2, Neumann_max_2,label_min,label_max,label_mean[i0])
        text_entries.append(one_entry)
    return text_entries

if __name__ == '__main__':
    # _inputs = np.load('../data/diffusion/new_test/s4/np-features-all-run_0.npy')
    # _labels = np.load('../data/diffusion/new_test/s4/np-labels-all-run_0.npy')
    # _mean = _labels
    # _var = _labels
    # _std = _labels
    # text_entries = statistics_data(_inputs, _labels, _mean, _var, _std, _edge=4)
    # exit(0)

    problem = thisPDESystem()
    # problem.run()
    # problem.test(test_folder='test-240')

    csv_header = 'edge_number,test_id,l2_error,error_min,error_max,Dirichlet_min,Dirichlet_max,Neumann_min_1,Neumann_max_1,Neumann_min_2,Neumann_max_2,DNS_min,DNS_max,DNS_mean'
    text_output = [csv_header]
    # _inputs, _labels, _mean, _var, _std = problem.test(test_folder='new_test/extreme')
    # text_entries = statistics_data(_inputs, _labels, _mean, _var, _std, _edge=5)
    # text_output.extend(text_entries)
    # _inputs, _labels, _mean, _var, _std = problem.test(test_folder='new_test/regular')
    # text_entries = statistics_data(_inputs, _labels, _mean, _var, _std, _edge=5)
    # text_output.extend(text_entries)


    _inputs, _labels, _mean, _var, _std = problem.test(test_folder='FOLDER', plot_png=False)
    text_entries = statistics_data(_inputs, _labels, _mean, _var, _std, _edge=EDGE)
    text_output.extend(text_entries)

    # _inputs, _labels, _mean, _var, _std = problem.test(test_folder='new_test/s5')
    # text_entries = statistics_data(_inputs, _labels, _mean, _var, _std, _edge=5)
    # text_output.extend(text_entries)
    # _inputs, _labels, _mean, _var, _std = problem.test(test_folder='new_test/s6')
    # text_entries = statistics_data(_inputs, _labels, _mean, _var, _std, _edge=6)
    # text_output.extend(text_entries)
    # _inputs, _labels, _mean, _var, _std = problem.test(test_folder='new_test/s7')
    # text_entries = statistics_data(_inputs, _labels, _mean, _var, _std, _edge=7)
    # text_output.extend(text_entries)
    # _inputs, _labels, _mean, _var, _std = problem.test(test_folder='new_test/s8')
    # text_entries = statistics_data(_inputs, _labels, _mean, _var, _std, _edge=8)
    # text_output.extend(text_entries)
    # _inputs, _labels, _mean, _var, _std = problem.test(test_folder='new_test/s9')
    # text_entries = statistics_data(_inputs, _labels, _mean, _var, _std, _edge=9)
    # text_output.extend(text_entries)
    # _inputs, _labels, _mean, _var, _std = problem.test(test_folder='new_test/s10')
    # text_entries = statistics_data(_inputs, _labels, _mean, _var, _std, _edge=10)
    # text_output.extend(text_entries)
    # _inputs, _labels, _mean, _var, _std = problem.test(test_folder='new_test/s11')
    # text_entries = statistics_data(_inputs, _labels, _mean, _var, _std, _edge=11)
    # text_output.extend(text_entries)
    # _inputs, _labels, _mean, _var, _std = problem.test(test_folder='new_test/s12')
    # text_entries = statistics_data(_inputs, _labels, _mean, _var, _std, _edge=12)
    # text_output.extend(text_entries)
    write_file('stats.csv', ('\n').join(text_output))


    #problem.test(test_folder='DNS')
    #problem.test(test_folder='Test')
    #problem.test(test_folder='Inter')
    #problem.test(test_folder='Extra')
    # problem.debug_problem(use_label=False)
    # problem.debug_problem(use_label=True)
    # problem.test_residual_gaussian(noise_std=1e-4, sample_num=1000)
