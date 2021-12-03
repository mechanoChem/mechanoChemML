import numpy as np
import tensorflow as tf

def GetElementResidualMask(data):
    """ 
    Create a mask [batch, node_height, node_width, 1] for data [batch, node_height, node_width, m] where only the actual residual region is 1, the remaining part is zero.

    args: 
        data (numpy array): [batch, node_height, node_width, m] (scalar/vector)
    
    return:
        numpy array: mask [batch, elem_height, elem_width, 1] (same padding is used.)
    """
    pflag = False
    if pflag: print('data', np.shape(data))
    if pflag: print('data', data[0,:,:,0])

    mask_original = tf.where( data < 0, tf.fill(tf.shape(data), 0.0), tf.fill(tf.shape(data), 1.0))
    if pflag: print('mask (original)', mask_original[0,:,:,0])
    n1 = np.array([[0, 0], [0, 1]])
    n1 = np.expand_dims(n1, axis=2)
    n1 = np.expand_dims(n1, axis=3)
    mask_shift = tf.nn.conv2d(mask_original[:,:,:,0:1], n1, [1,1,1,1], 'SAME')
    if pflag: print('mask (shift)', mask_shift[0,:,:,0])
    mask = tf.multiply(mask_original[:,:,:,0:1], mask_shift)
    if pflag: print('mask (final)', mask[0,:,:,0])
    return mask


def ComputeBoundaryMaskNodalData(data_input, dof, opt=1):
    """ 
    Create Dirichlet mask or Neumann mask based on the inputs, where only the boundary part is 0.0, margin and the body part is 1.0.

    args:
        data_input (numpy array): size of [batch, node_height, node_width, dof*2]
        dof (int): dof per node
        opt (int): Dirichlet Mask (opt=1), Neumann mask (opt=2)
    return:
        numpy array: boundary mask with size of [batch, node_height, node_width, dof]

    todo:
        make this function to work with (1S, 1V), 2S, 1V1S, 3S, 2V, etc.
    """
    if dof == 1 :
        pflag = False
        # data_input = tf.convert_to_tensor(data_input, dtype=tf.float32)
        if pflag: print('data_input', np.shape(data_input))
        #---------------- Dirichlet BCs-----------------
        dirichlet_data = data_input[:,:,:,0:1]
        if pflag: print('dirichlet_data', dirichlet_data[0,:,:,0])
        dirichlet_reverse_mask = tf.where( dirichlet_data < 0, tf.fill(tf.shape(dirichlet_data), 1.0), tf.fill(tf.shape(dirichlet_data), 0.0))
        if pflag: print('dirichlet_reverse_mask', dirichlet_reverse_mask[0,:,:,0])

        #---------------- Neumann BCs-----------------
        neumann_data = data_input[:,:,:,1:2]
        if pflag:  print('neumann_data', neumann_data[0,:,:,0])
        if pflag:  print('attention, NM should not be scaled ')
        neumann_reverse_mask = tf.where( neumann_data > 0.0, tf.fill(tf.shape(neumann_data), 0.0), tf.fill(tf.shape(neumann_data), 1.0))
        if pflag:  print('neumann_reverse_mask', neumann_reverse_mask[0,:,:,0])

        # bc_mask = tf.multiply(dirichlet_reverse_mask, neumann_reverse_mask)
        bc_mask = dirichlet_reverse_mask
        # re-index the y-axis to make sure the bcs look correct on the plot
        # bc_mask = tf.reverse(bc_mask, [1]) # check on 2020-07-14: seems wrong
        if opt == 1:
          return dirichlet_reverse_mask
        elif opt == 2:
          return neumann_reverse_mask 
    elif dof == 2 :
        """ 
        input: feature inputs with Dirichlet and Neumann BCs
        output: Dirichlet mask, and Neumann mask
        """
        pflag = False
        # data_input = tf.convert_to_tensor(data_input, dtype=tf.float32)
        if pflag: print('data_input', np.shape(data_input))
        #---------------- Dirichlet BCs-----------------
        dirichlet_x_data = data_input[:,:,:,0:1]
        dirichlet_y_data = data_input[:,:,:,1:2]
        if pflag: print('dirichlet_x_data', dirichlet_x_data[0,:,:,0])
        if pflag: print('dirichlet_y_data', dirichlet_y_data[0,:,:,0])
        dirichlet_x_reverse_mask = tf.where( dirichlet_x_data < 0, tf.fill(tf.shape(dirichlet_x_data), 1.0), tf.fill(tf.shape(dirichlet_x_data), 0.0))
        dirichlet_y_reverse_mask = tf.where( dirichlet_y_data < 0, tf.fill(tf.shape(dirichlet_y_data), 1.0), tf.fill(tf.shape(dirichlet_y_data), 0.0))
        if pflag: print('dirichlet_x_reverse_mask', dirichlet_x_reverse_mask[0,:,:,0])
        if pflag: print('dirichlet_y_reverse_mask', dirichlet_y_reverse_mask[0,:,:,0])

        #---------------- Neumann BCs-----------------
        neumann_x_data = data_input[:,:,:,2:3]
        neumann_y_data = data_input[:,:,:,3:4]
        if pflag: print('neumann_x_data', neumann_x_data[0,:,:,0])
        if pflag: print('neumann_y_data', neumann_y_data[0,:,:,0])
        if pflag: print('attention, NM should not be scaled ')
        neumann_x_reverse_mask = tf.where( neumann_x_data > 0.0, tf.fill(tf.shape(neumann_x_data), 0.0), tf.fill(tf.shape(neumann_x_data), 1.0))
        neumann_y_reverse_mask = tf.where( neumann_y_data > 0.0, tf.fill(tf.shape(neumann_y_data), 0.0), tf.fill(tf.shape(neumann_y_data), 1.0))
        if pflag: print('neumann_x_reverse_mask', neumann_x_reverse_mask[0,:,:,0])
        if pflag: print('neumann_y_reverse_mask', neumann_y_reverse_mask[0,:,:,0])
        dirichlet_reverse_mask = tf.concat([dirichlet_x_reverse_mask, dirichlet_y_reverse_mask], axis=3)
        neumann_reverse_mask = tf.concat([neumann_x_reverse_mask, neumann_y_reverse_mask], axis=3)
        # bc_mask = tf.multiply(dirichlet_reverse_mask, neumann_reverse_mask)
        bc_mask = dirichlet_reverse_mask
        # bc_mask = tf.reverse(bc_mask, [1])
        # if pflag: print('bc_mask_x :', np.shape(bc_mask), bc_mask[0,:,:,0])
        # if pflag: print('bc_mask_y :', np.shape(bc_mask), bc_mask[0,:,:,1])
        # return bc_mask # disabled on 2020-07-22

        if opt == 1:
          return dirichlet_reverse_mask
        elif opt == 2:
          return neumann_reverse_mask 
    elif dof > 2 :
        """ 
        input: feature inputs with Dirichlet and Neumann BCs
        output: Dirichlet mask, and Neumann mask
        """
        pflag = False
        # data_input = tf.convert_to_tensor(data_input, dtype=tf.float32)
        if pflag: print('data_input', np.shape(data_input))
        #---------------- Dirichlet BCs-----------------
        dirichlet_data = data_input[:,:,:,0:dof]
        dirichlet_reverse_mask = tf.where( dirichlet_data < 0, tf.fill(tf.shape(dirichlet_data), 1.0), tf.fill(tf.shape(dirichlet_data), 0.0))
        if pflag: 
            for d0 in range(0, dof):
                print(' dirichlet_reverse_mask ' + str(d0) + ':', dirichlet_reverse_mask[0,:,:,d0])

        #---------------- Neumann BCs-----------------
        neumann_data = data_input[:,:,:,dof:2*dof]
        neumann_reverse_mask = tf.where( neumann_data > 0.0, tf.fill(tf.shape(neumann_data), 0.0), tf.fill(tf.shape(neumann_data), 1.0))
        if pflag: 
            for d0 in range(0, dof):
                print(' neumann_reverse_mask ' + str(d0) + ':', neumann_reverse_mask[0,:,:,d0])

        if opt == 1:
          return dirichlet_reverse_mask
        elif opt == 2:
          return neumann_reverse_mask 

        raise ValueError("Please check dof = ", dof, " if it is implemented correct or not in pde_layers.ComputeBoundaryMaskNodalData()!")

def ComputeNeumannBoundaryResidualNodalData(data_input, dh, dof, padding='SAME'):
    """ 
    Compute the residual on the Neumann BCs.  The implementation is based on Neumann BCs is scaled between (-1, 1), and Neumann condition should be always > 0 in the domain region. Raise value error if negative value is detected

    args:
        data_input (numpy array): size of [batch, node_height, node_width, dof*2]
        dof (int): dof per node
        dh (float): element size
    return:
        numpy array: nodal Neumann residual value with size of [batch, node_height, node_width, dof]

    todo:
        make this function to work with (1S, 1V), 2S, 1V1S, 3S, 2V, etc.

        loop over each dof, instead of implementing different dof opt.
    """
    # pflag = True
    pflag = False
    if (dof > 4):
        raise ValueError(" dof = ", dof, " is not implemented! Only dof = 1 or 2 or 3 or 4 is coded!")

    data_input = tf.convert_to_tensor(data_input, dtype=tf.float32)
    if pflag: print('data_input', np.shape(data_input))

    # --------------- Dirichlet data------------
    dirichlet_data = data_input[:,:,:,0:dof]
    if pflag: print('dirichlet_data', dirichlet_data[0,:,:,0])

    #--------------- Domain mask ------------------
    # change actual dirichlet BCs to -2, the all -2 will be the domain
    domain_mask = tf.where( dirichlet_data >= 0.0, tf.fill(tf.shape(dirichlet_data), -2.0), dirichlet_data)
    # print('domain_mask', domain_mask[0,:,:,0])
    domain_mask = tf.where( domain_mask < -1.0, tf.fill(tf.shape(domain_mask), 1.0), tf.fill(tf.shape(domain_mask), 0.0))
    if pflag: print('domain_mask', domain_mask[0,:,:,0])

    #--------------- Neumann BCs ------------------
    Neumann_max = 1.0
    Neumann_min = -1.0
    neumann_data = data_input[:,:,:,dof:dof+dof]
    if pflag: print('neumann_data', neumann_data[0,:,:,0])
    if pflag: print('neumann_data(dof-1)', neumann_data[0,:,:,dof-1])

    # #--------------- check Neumann BCs------------ !!!! should be checked at the very beginning of data.
    # # is not allowed during tensor running/training 
    # check_neumann = tf.multiply(neumann_data, domain_mask)
    # if pflag: print('---check_neumann', check_neumann[0,:,:,0])
    # if (tf.reduce_sum(check_neumann) == 0) :
        # print("WARNING: no Neumann BCs is detected!")
    # check_neumann = tf.where( check_neumann < 0.0, tf.fill(tf.shape(check_neumann), 1.0), tf.fill(tf.shape(check_neumann), 0.0))
    # if pflag: print('check_neumann', check_neumann[0,:,:,0])
    # if (tf.reduce_sum(check_neumann) < 0) :
        # raise ValueError("Neumann BCs should NOT be smaller than zero (< 0). Consider use diffusivity or elastic constant to scale the data!")

    #---------------- Neumann BCs-----------------
    # Should consider the scaling as well.  Any mask will not work, as Neumann BCs can be smaller than 0.0
    # Solution: Neumann BCs is not allowed to be smaller than 0 in the input data.

    neumann_mask = tf.where( neumann_data <= 0.0, tf.fill(tf.shape(neumann_data), 0.0), tf.fill(tf.shape(neumann_data), 1.0))
    if pflag: print('neumann_mask', neumann_mask[0,:,:,0])
    if pflag: print('neumann_mask(dof-1)', neumann_mask[0,:,:,dof-1])
    neumann_data = tf.multiply( neumann_data, neumann_mask)
    if pflag: print('neumann_data', neumann_data[0,:,:,0])
    if pflag: print('neumann_data(dof-1)', neumann_data[0,:,:,dof-1])

    # The main idea is to form a element connection as for the bulk residual case.
    # However, if the neumman BCs is location dependent, has both positive and negative 
    # values, then, it is very challenging to make the following to work stably. 
    # On the other hand, if we only consider the bulk region, then the neumman BCs is
    # literally enforced through the residual form, but not explicitly. 
    # Still, the following might still work.

    # form dof-1 level horizontal element
    # the padding zeros will helps to keep the location of surface node unchanged for the bottom edges
    n1 = np.array([[1, 0], [0, 0]])
    n1 = np.expand_dims(n1, axis=2)
    n1 = np.expand_dims(n1, axis=3)
    n2 = np.array([[0, 1], [0, 0]])
    n2 = np.expand_dims(n2, axis=2)
    n2 = np.expand_dims(n2, axis=3)

    # form dof-1 level vertical element:
    n3 = np.array([[1, 0], [0, 0]])
    n3 = np.expand_dims(n3, axis=2)
    n3 = np.expand_dims(n3, axis=3)
    n4 = np.array([[0, 0], [1, 0]])
    n4 = np.expand_dims(n4, axis=2)
    n4 = np.expand_dims(n4, axis=3)


    # create surface elements
    if dof == 1:
        # horizontal element
        c_n1 = tf.nn.conv2d(neumann_data, n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data, n2, [1,1,1,1], padding)
        elem_y = tf.concat([c_n1, c_n2], 3)

        # vertical element
        c_n3 = tf.nn.conv2d(neumann_data, n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data, n4, [1,1,1,1], padding)
        elem_x = tf.concat([c_n3, c_n4], 3)

        if pflag: print('elem_x', np.shape(elem_x))
        if pflag: print('elem_x 1: ', elem_x[0,:,:,0])
        if pflag: print('elem_x 2: ', elem_x[0,:,:,1])

        if pflag: print('elem_y', np.shape(elem_y))
        if pflag: print('elem_y 1: ', elem_y[0,:,:,0])
        if pflag: print('elem_y 2: ', elem_y[0,:,:,1])
    elif dof == 2:
        neumann_data_1 = neumann_data[:,:,:,0:1]
        neumann_data_2 = neumann_data[:,:,:,1:2]

        c_n1 = tf.nn.conv2d(neumann_data_1, n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_1, n2, [1,1,1,1], padding)
        elem_y_1 = tf.concat([c_n1, c_n2], 3)

        c_n1 = tf.nn.conv2d(neumann_data_2, n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_2, n2, [1,1,1,1], padding)
        elem_y_2 = tf.concat([c_n1, c_n2], 3)

        c_n3 = tf.nn.conv2d(neumann_data_1, n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_1, n4, [1,1,1,1], padding)
        elem_x_1 = tf.concat([c_n3, c_n4], 3)

        c_n3 = tf.nn.conv2d(neumann_data_2, n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_2, n4, [1,1,1,1], padding)
        elem_x_2 = tf.concat([c_n3, c_n4], 3)

        if pflag: print('elem_y_1 ', np.shape(elem_y_1))
        if pflag: print('elem_y_1 1: ', elem_y_1[0,:,:,0])
        if pflag: print('elem_y_1 2: ', elem_y_1[0,:,:,1])
        if pflag: print('elem_y_2 ', np.shape(elem_y_2))
        if pflag: print('elem_y_2 1: ', elem_y_2[0,:,:,0])
        if pflag: print('elem_y_2 2: ', elem_y_2[0,:,:,1])

        if pflag: print('elem_x_1 ', np.shape(elem_x_1))
        if pflag: print('elem_x_1 1: ', elem_x_1[0,:,:,0])
        if pflag: print('elem_x_1 2: ', elem_x_1[0,:,:,1])
        if pflag: print('elem_x_2 ', np.shape(elem_x_2))
        if pflag: print('elem_x_2 1: ', elem_x_2[0,:,:,0])
        if pflag: print('elem_x_2 2: ', elem_x_2[0,:,:,1])
    elif dof == 3:
        neumann_data_1 = neumann_data[:,:,:,0:1]
        neumann_data_2 = neumann_data[:,:,:,1:2]
        neumann_data_3 = neumann_data[:,:,:,2:3]

        c_n1 = tf.nn.conv2d(neumann_data_1, n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_1, n2, [1,1,1,1], padding)
        elem_y_1 = tf.concat([c_n1, c_n2], 3)

        c_n1 = tf.nn.conv2d(neumann_data_2, n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_2, n2, [1,1,1,1], padding)
        elem_y_2 = tf.concat([c_n1, c_n2], 3)

        c_n1 = tf.nn.conv2d(neumann_data_3, n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_3, n2, [1,1,1,1], padding)
        elem_y_3 = tf.concat([c_n1, c_n2], 3)

        c_n3 = tf.nn.conv2d(neumann_data_1, n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_1, n4, [1,1,1,1], padding)
        elem_x_1 = tf.concat([c_n3, c_n4], 3)

        c_n3 = tf.nn.conv2d(neumann_data_2, n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_2, n4, [1,1,1,1], padding)
        elem_x_2 = tf.concat([c_n3, c_n4], 3)

        c_n3 = tf.nn.conv2d(neumann_data_3, n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_3, n4, [1,1,1,1], padding)
        elem_x_3 = tf.concat([c_n3, c_n4], 3)

        if pflag: print('elem_y_1 ', np.shape(elem_y_1))
        if pflag: print('elem_y_1 1: ', elem_y_1[0,:,:,0])
        if pflag: print('elem_y_1 2: ', elem_y_1[0,:,:,1])
        if pflag: print('elem_y_2 ', np.shape(elem_y_2))
        if pflag: print('elem_y_2 1: ', elem_y_2[0,:,:,0])
        if pflag: print('elem_y_2 2: ', elem_y_2[0,:,:,1])
        if pflag: print('elem_y_3 ', np.shape(elem_y_3))
        if pflag: print('elem_y_3 1: ', elem_y_3[0,:,:,0])
        if pflag: print('elem_y_3 2: ', elem_y_3[0,:,:,1])

        if pflag: print('elem_x_1 ', np.shape(elem_x_1))
        if pflag: print('elem_x_1 1: ', elem_x_1[0,:,:,0])
        if pflag: print('elem_x_1 2: ', elem_x_1[0,:,:,1])
        if pflag: print('elem_x_2 ', np.shape(elem_x_2))
        if pflag: print('elem_x_2 1: ', elem_x_2[0,:,:,0])
        if pflag: print('elem_x_2 2: ', elem_x_2[0,:,:,1])
        if pflag: print('elem_x_3 ', np.shape(elem_x_3))
        if pflag: print('elem_x_3 1: ', elem_x_3[0,:,:,0])
        if pflag: print('elem_x_3 2: ', elem_x_3[0,:,:,1])

    elif dof == 4:
        neumann_data_1 = neumann_data[:,:,:,0:1]
        neumann_data_2 = neumann_data[:,:,:,1:2]
        neumann_data_3 = neumann_data[:,:,:,2:3]
        neumann_data_4 = neumann_data[:,:,:,2:3]

        c_n1 = tf.nn.conv2d(neumann_data_1, n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_1, n2, [1,1,1,1], padding)
        elem_y_1 = tf.concat([c_n1, c_n2], 3)

        c_n1 = tf.nn.conv2d(neumann_data_2, n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_2, n2, [1,1,1,1], padding)
        elem_y_2 = tf.concat([c_n1, c_n2], 3)

        c_n1 = tf.nn.conv2d(neumann_data_3, n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_3, n2, [1,1,1,1], padding)
        elem_y_3 = tf.concat([c_n1, c_n2], 3)

        c_n1 = tf.nn.conv2d(neumann_data_4, n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_4, n2, [1,1,1,1], padding)
        elem_y_4 = tf.concat([c_n1, c_n2], 3)

        c_n3 = tf.nn.conv2d(neumann_data_1, n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_1, n4, [1,1,1,1], padding)
        elem_x_1 = tf.concat([c_n3, c_n4], 3)

        c_n3 = tf.nn.conv2d(neumann_data_2, n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_2, n4, [1,1,1,1], padding)
        elem_x_2 = tf.concat([c_n3, c_n4], 3)

        c_n3 = tf.nn.conv2d(neumann_data_3, n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_3, n4, [1,1,1,1], padding)
        elem_x_3 = tf.concat([c_n3, c_n4], 3)

        c_n3 = tf.nn.conv2d(neumann_data_4, n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_4, n4, [1,1,1,1], padding)
        elem_x_4 = tf.concat([c_n3, c_n4], 3)



    if dof == 1:
        # create a mask to delete data that are not properly aligned
        c_n1_mask = tf.nn.conv2d(neumann_mask, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask, n2, [1,1,1,1], padding)
        elem_y_mask = tf.multiply(c_n1_mask, c_n2_mask)
        if pflag: print('c_n1_mask: ', c_n1_mask[0,:,:,0])
        if pflag: print('c_n2_mask: ', c_n2_mask[0,:,:,0])
        if pflag: print('elem_y_mask: ', elem_y_mask[0,:,:,0])
    
        # create a mask to delete data that are not properly aligned
        c_n3_mask = tf.nn.conv2d(neumann_mask, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask, n4, [1,1,1,1], padding)
        elem_x_mask = tf.multiply(c_n3_mask, c_n4_mask)
        if pflag: print('c_n3_mask: ', c_n3_mask[0,:,:,0])
        if pflag: print('c_n4_mask: ', c_n4_mask[0,:,:,0])
        if pflag: print('elem_x_mask: ', elem_x_mask[0,:,:,0])
        # exit(0)
    elif dof == 2:
        # For the 3D case, it would be impossible to perform task like this.
        # Thus, how to come up with a 3D implementation, or sparse pattern 
        # would be extremely useful.

        # create a mask to delete data that are not properly aligned
        neumann_mask_1 = neumann_mask[:,:,:,0:1]
        neumann_mask_2 = neumann_mask[:,:,:,1:2]

        c_n1_mask = tf.nn.conv2d(neumann_mask_1, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_1, n2, [1,1,1,1], padding)
        elem_y_mask_1 = tf.multiply(c_n1_mask, c_n2_mask)

        if pflag: print('c_n1_mask: ', c_n1_mask[0,:,:,0])
        if pflag: print('c_n2_mask: ', c_n2_mask[0,:,:,0])
        if pflag: print('elem_y_mask_1: ', elem_y_mask_1[0,:,:,0])

        c_n1_mask = tf.nn.conv2d(neumann_mask_2, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_2, n2, [1,1,1,1], padding)
        elem_y_mask_2 = tf.multiply(c_n1_mask, c_n2_mask)

        if pflag: print('c_n1_mask: ', c_n1_mask[0,:,:,0])
        if pflag: print('c_n2_mask: ', c_n2_mask[0,:,:,0])
        if pflag: print('elem_y_mask_2: ', elem_y_mask_2[0,:,:,0])

    
        # create a mask to delete data that are not properly aligned
        c_n3_mask = tf.nn.conv2d(neumann_mask_1, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_1, n4, [1,1,1,1], padding)
        elem_x_mask_1 = tf.multiply(c_n3_mask, c_n4_mask)
        if pflag: print('c_n3_mask: ', c_n3_mask[0,:,:,0])
        if pflag: print('c_n4_mask: ', c_n4_mask[0,:,:,0])
        if pflag: print('elem_x_mask_1: ', elem_x_mask_1[0,:,:,0])

        c_n3_mask = tf.nn.conv2d(neumann_mask_2, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_2, n4, [1,1,1,1], padding)
        elem_x_mask_2 = tf.multiply(c_n3_mask, c_n4_mask)
        if pflag: print('c_n3_mask: ', c_n3_mask[0,:,:,0])
        if pflag: print('c_n4_mask: ', c_n4_mask[0,:,:,0])
        if pflag: print('elem_x_mask_2: ', elem_x_mask_2[0,:,:,0])

    elif dof == 3:

        # create a mask to delete data that are not properly aligned
        neumann_mask_1 = neumann_mask[:,:,:,0:1]
        neumann_mask_2 = neumann_mask[:,:,:,1:2]
        neumann_mask_3 = neumann_mask[:,:,:,2:3]

        c_n1_mask = tf.nn.conv2d(neumann_mask_1, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_1, n2, [1,1,1,1], padding)
        elem_y_mask_1 = tf.multiply(c_n1_mask, c_n2_mask)

        if pflag: print('c_n1_mask: ', c_n1_mask[0,:,:,0])
        if pflag: print('c_n2_mask: ', c_n2_mask[0,:,:,0])
        if pflag: print('elem_y_mask_1: ', elem_y_mask_1[0,:,:,0])

        c_n1_mask = tf.nn.conv2d(neumann_mask_2, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_2, n2, [1,1,1,1], padding)
        elem_y_mask_2 = tf.multiply(c_n1_mask, c_n2_mask)

        if pflag: print('c_n1_mask: ', c_n1_mask[0,:,:,0])
        if pflag: print('c_n2_mask: ', c_n2_mask[0,:,:,0])
        if pflag: print('elem_y_mask_2: ', elem_y_mask_2[0,:,:,0])

        c_n1_mask = tf.nn.conv2d(neumann_mask_3, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_3, n2, [1,1,1,1], padding)
        elem_y_mask_3 = tf.multiply(c_n1_mask, c_n2_mask)

        if pflag: print('c_n1_mask: ', c_n1_mask[0,:,:,0])
        if pflag: print('c_n2_mask: ', c_n2_mask[0,:,:,0])
        if pflag: print('elem_y_mask_3: ', elem_y_mask_3[0,:,:,0])

    
        # create a mask to delete data that are not properly aligned
        c_n3_mask = tf.nn.conv2d(neumann_mask_1, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_1, n4, [1,1,1,1], padding)
        elem_x_mask_1 = tf.multiply(c_n3_mask, c_n4_mask)
        if pflag: print('c_n3_mask: ', c_n3_mask[0,:,:,0])
        if pflag: print('c_n4_mask: ', c_n4_mask[0,:,:,0])
        if pflag: print('elem_x_mask_1: ', elem_x_mask_1[0,:,:,0])

        c_n3_mask = tf.nn.conv2d(neumann_mask_2, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_2, n4, [1,1,1,1], padding)
        elem_x_mask_2 = tf.multiply(c_n3_mask, c_n4_mask)
        if pflag: print('c_n3_mask: ', c_n3_mask[0,:,:,0])
        if pflag: print('c_n4_mask: ', c_n4_mask[0,:,:,0])
        if pflag: print('elem_x_mask_2: ', elem_x_mask_2[0,:,:,0])

        c_n3_mask = tf.nn.conv2d(neumann_mask_3, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_3, n4, [1,1,1,1], padding)
        elem_x_mask_3 = tf.multiply(c_n3_mask, c_n4_mask)
        if pflag: print('c_n3_mask: ', c_n3_mask[0,:,:,0])
        if pflag: print('c_n4_mask: ', c_n4_mask[0,:,:,0])
        if pflag: print('elem_x_mask_3: ', elem_x_mask_3[0,:,:,0])

    elif dof == 4:

        # create a mask to delete data that are not properly aligned
        neumann_mask_1 = neumann_mask[:,:,:,0:1]
        neumann_mask_2 = neumann_mask[:,:,:,1:2]
        neumann_mask_3 = neumann_mask[:,:,:,2:3]
        neumann_mask_4 = neumann_mask[:,:,:,2:3]

        c_n1_mask = tf.nn.conv2d(neumann_mask_1, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_1, n2, [1,1,1,1], padding)
        elem_y_mask_1 = tf.multiply(c_n1_mask, c_n2_mask)

        c_n1_mask = tf.nn.conv2d(neumann_mask_2, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_2, n2, [1,1,1,1], padding)
        elem_y_mask_2 = tf.multiply(c_n1_mask, c_n2_mask)

        c_n1_mask = tf.nn.conv2d(neumann_mask_3, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_3, n2, [1,1,1,1], padding)
        elem_y_mask_3 = tf.multiply(c_n1_mask, c_n2_mask)

        c_n1_mask = tf.nn.conv2d(neumann_mask_4, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_4, n2, [1,1,1,1], padding)
        elem_y_mask_4 = tf.multiply(c_n1_mask, c_n2_mask)

    
        # create a mask to delete data that are not properly aligned
        c_n3_mask = tf.nn.conv2d(neumann_mask_1, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_1, n4, [1,1,1,1], padding)
        elem_x_mask_1 = tf.multiply(c_n3_mask, c_n4_mask)

        c_n3_mask = tf.nn.conv2d(neumann_mask_2, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_2, n4, [1,1,1,1], padding)
        elem_x_mask_2 = tf.multiply(c_n3_mask, c_n4_mask)

        c_n3_mask = tf.nn.conv2d(neumann_mask_3, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_3, n4, [1,1,1,1], padding)
        elem_x_mask_3 = tf.multiply(c_n3_mask, c_n4_mask)

        c_n3_mask = tf.nn.conv2d(neumann_mask_4, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_4, n4, [1,1,1,1], padding)
        elem_x_mask_4 = tf.multiply(c_n3_mask, c_n4_mask)
    
    


    if dof == 1:
        # Scale the Neumann BC value back to the original one
        # original scale in VtuDataGenerateFixedc.py: 
        #   - data = (data + (self.upperlimit - self.lowerlimit) * 0.5 ) * 0.5
        elem_x = 2.0 * elem_x - (Neumann_max - Neumann_min) * 0.5
        elem_y = 2.0 * elem_y - (Neumann_max - Neumann_min) * 0.5

        clean_elem_y = tf.multiply(elem_y, elem_y_mask)
        clean_elem_x = tf.multiply(elem_x, elem_x_mask)
        if pflag: print('clean_elem_y (node1, 2)', np.shape(clean_elem_y), clean_elem_y[0,:,:,0], clean_elem_y[0,:,:,1])
        if pflag: print('clean_elem_x (node1, 2)', np.shape(clean_elem_x), clean_elem_x[0,:,:,0], clean_elem_x[0,:,:,1])
    elif dof == 2:
        elem_x_1 = 2.0 * elem_x_1 - (Neumann_max - Neumann_min) * 0.5
        elem_y_1 = 2.0 * elem_y_1 - (Neumann_max - Neumann_min) * 0.5
        elem_x_2 = 2.0 * elem_x_2 - (Neumann_max - Neumann_min) * 0.5
        elem_y_2 = 2.0 * elem_y_2 - (Neumann_max - Neumann_min) * 0.5

        clean_elem_y_1 = tf.multiply(elem_y_1, elem_y_mask_1)
        clean_elem_x_1 = tf.multiply(elem_x_1, elem_x_mask_1)
        clean_elem_y_2 = tf.multiply(elem_y_2, elem_y_mask_2)
        clean_elem_x_2 = tf.multiply(elem_x_2, elem_x_mask_2)

        if pflag: print('clean_elem_y_1 (node1, 2)', np.shape(clean_elem_y_1), clean_elem_y_1[0,:,:,0], clean_elem_y_1[0,:,:,1])
        if pflag: print('clean_elem_x_1 (node1, 2)', np.shape(clean_elem_x_1), clean_elem_x_1[0,:,:,0], clean_elem_x_1[0,:,:,1])
        if pflag: print('clean_elem_y_2 (node1, 2)', np.shape(clean_elem_y_2), clean_elem_y_2[0,:,:,0], clean_elem_y_2[0,:,:,1])
        if pflag: print('clean_elem_x_2 (node1, 2)', np.shape(clean_elem_x_2), clean_elem_x_2[0,:,:,0], clean_elem_x_2[0,:,:,1])
    elif dof == 3:
        elem_x_1 = 2.0 * elem_x_1 - (Neumann_max - Neumann_min) * 0.5
        elem_y_1 = 2.0 * elem_y_1 - (Neumann_max - Neumann_min) * 0.5
        elem_x_2 = 2.0 * elem_x_2 - (Neumann_max - Neumann_min) * 0.5
        elem_y_2 = 2.0 * elem_y_2 - (Neumann_max - Neumann_min) * 0.5
        elem_x_3 = 2.0 * elem_x_3 - (Neumann_max - Neumann_min) * 0.5
        elem_y_3 = 2.0 * elem_y_3 - (Neumann_max - Neumann_min) * 0.5

        clean_elem_y_1 = tf.multiply(elem_y_1, elem_y_mask_1)
        clean_elem_x_1 = tf.multiply(elem_x_1, elem_x_mask_1)
        clean_elem_y_2 = tf.multiply(elem_y_2, elem_y_mask_2)
        clean_elem_x_2 = tf.multiply(elem_x_2, elem_x_mask_2)
        clean_elem_y_3 = tf.multiply(elem_y_3, elem_y_mask_3)
        clean_elem_x_3 = tf.multiply(elem_x_3, elem_x_mask_3)

        if pflag: print('clean_elem_y_1 (node1, 2)', np.shape(clean_elem_y_1), clean_elem_y_1[0,:,:,0], clean_elem_y_1[0,:,:,1])
        if pflag: print('clean_elem_x_1 (node1, 2)', np.shape(clean_elem_x_1), clean_elem_x_1[0,:,:,0], clean_elem_x_1[0,:,:,1])
        if pflag: print('clean_elem_y_2 (node1, 2)', np.shape(clean_elem_y_2), clean_elem_y_2[0,:,:,0], clean_elem_y_2[0,:,:,1])
        if pflag: print('clean_elem_x_2 (node1, 2)', np.shape(clean_elem_x_2), clean_elem_x_2[0,:,:,0], clean_elem_x_2[0,:,:,1])
        if pflag: print('clean_elem_y_3 (node1, 2)', np.shape(clean_elem_y_3), clean_elem_y_3[0,:,:,0], clean_elem_y_3[0,:,:,1])
        if pflag: print('clean_elem_x_3 (node1, 2)', np.shape(clean_elem_x_3), clean_elem_x_3[0,:,:,0], clean_elem_x_3[0,:,:,1])

    elif dof == 4:
        elem_x_1 = 2.0 * elem_x_1 - (Neumann_max - Neumann_min) * 0.5
        elem_y_1 = 2.0 * elem_y_1 - (Neumann_max - Neumann_min) * 0.5
        elem_x_2 = 2.0 * elem_x_2 - (Neumann_max - Neumann_min) * 0.5
        elem_y_2 = 2.0 * elem_y_2 - (Neumann_max - Neumann_min) * 0.5
        elem_x_3 = 2.0 * elem_x_3 - (Neumann_max - Neumann_min) * 0.5
        elem_y_3 = 2.0 * elem_y_3 - (Neumann_max - Neumann_min) * 0.5
        elem_x_4 = 2.0 * elem_x_4 - (Neumann_max - Neumann_min) * 0.5
        elem_y_4 = 2.0 * elem_y_4 - (Neumann_max - Neumann_min) * 0.5

        clean_elem_y_1 = tf.multiply(elem_y_1, elem_y_mask_1)
        clean_elem_x_1 = tf.multiply(elem_x_1, elem_x_mask_1)
        clean_elem_y_2 = tf.multiply(elem_y_2, elem_y_mask_2)
        clean_elem_x_2 = tf.multiply(elem_x_2, elem_x_mask_2)
        clean_elem_y_3 = tf.multiply(elem_y_3, elem_y_mask_3)
        clean_elem_x_3 = tf.multiply(elem_x_3, elem_x_mask_3)
        clean_elem_y_4 = tf.multiply(elem_y_4, elem_y_mask_4)
        clean_elem_x_4 = tf.multiply(elem_x_4, elem_x_mask_4)



    if dof == 1 :
        shape=elem_x.get_shape()[0:].as_list()    
        new_shape = shape[1:3]
        if pflag: print('new_shape:', new_shape)
    elif dof == 2 :
        shape=elem_x_1.get_shape()[0:].as_list()    
        new_shape = shape[1:3]
        if pflag: print('new_shape:', new_shape)
    elif dof == 3 :
        shape=elem_x_1.get_shape()[0:].as_list()    
        new_shape = shape[1:3]
        if pflag: print('new_shape:', new_shape)
    elif dof == 4 :
        shape=elem_x_1.get_shape()[0:].as_list()    
        new_shape = shape[1:3]


    # get the 1D info, and then perform a N h calculation
    # and then unfold everything to the nodal value
    # 
    N, B, jxw = Get1DGaussPointInfo(dh=dh, GPs=2, dof=1)
    if pflag: print("N", np.shape(N))
    if pflag: print("B", np.shape(B))
    if pflag: print("jxw", jxw)

    if dof == 1:
        elem_x2 = tf.reshape(clean_elem_x,[-1, 2])
        elem_y2 = tf.reshape(clean_elem_y,[-1, 2])

        if pflag: print('elem_x2', np.shape(elem_x2), elem_x2)
        if pflag: print('elem_y2', np.shape(elem_y2), elem_y2)
        # exit(0)

    elif dof == 2:
        elem_x2_1 = tf.reshape(clean_elem_x_1,[-1, 2])
        elem_y2_1 = tf.reshape(clean_elem_y_1,[-1, 2])
        elem_x2_2 = tf.reshape(clean_elem_x_2,[-1, 2])
        elem_y2_2 = tf.reshape(clean_elem_y_2,[-1, 2])

        if pflag: print('elem_x2_1', np.shape(elem_x2_1), elem_x2_1)
        if pflag: print('elem_y2_1', np.shape(elem_y2_1), elem_y2_1)
        if pflag: print('elem_x2_2', np.shape(elem_x2_2), elem_x2_2)
        if pflag: print('elem_y2_2', np.shape(elem_y2_2), elem_y2_2)
    elif dof == 3:
        elem_x2_1 = tf.reshape(clean_elem_x_1,[-1, 2])
        elem_y2_1 = tf.reshape(clean_elem_y_1,[-1, 2])
        elem_x2_2 = tf.reshape(clean_elem_x_2,[-1, 2])
        elem_y2_2 = tf.reshape(clean_elem_y_2,[-1, 2])
        elem_x2_3 = tf.reshape(clean_elem_x_3,[-1, 2])
        elem_y2_3 = tf.reshape(clean_elem_y_3,[-1, 2])

        if pflag: print('elem_x2_1', np.shape(elem_x2_1), elem_x2_1)
        if pflag: print('elem_y2_1', np.shape(elem_y2_1), elem_y2_1)
        if pflag: print('elem_x2_2', np.shape(elem_x2_2), elem_x2_2)
        if pflag: print('elem_y2_2', np.shape(elem_y2_2), elem_y2_2)
        if pflag: print('elem_x2_3', np.shape(elem_x2_3), elem_x2_3)
        if pflag: print('elem_y2_3', np.shape(elem_y2_3), elem_y2_3)
    elif dof == 4:
        elem_x2_1 = tf.reshape(clean_elem_x_1,[-1, 2])
        elem_y2_1 = tf.reshape(clean_elem_y_1,[-1, 2])
        elem_x2_2 = tf.reshape(clean_elem_x_2,[-1, 2])
        elem_y2_2 = tf.reshape(clean_elem_y_2,[-1, 2])
        elem_x2_3 = tf.reshape(clean_elem_x_3,[-1, 2])
        elem_y2_3 = tf.reshape(clean_elem_y_3,[-1, 2])
        elem_x2_4 = tf.reshape(clean_elem_x_4,[-1, 2])
        elem_y2_4 = tf.reshape(clean_elem_y_4,[-1, 2])


    if dof == 1:
        # int(N^T h) dA: h是nodal value，必须通过shape fcn来分析正确的值, 但是它也是scale了的值， 
        # Calculate the hbar at the GPs based on nodal info.
        # GP1
        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2, N[0,:]) 
        # GP2
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2, N[1,:]) 
        # GP1
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2, N[0,:]) 
        # GP2
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])

        # if pflag: print('elem_x2_hbar_gp1', np.shape(elem_x2_hbar_gp1),tf.reshape(elem_x2_hbar_gp1, new_shape)) # work for [1, 16, 16, 1], but not [8, 16, 16, 1]
        # if pflag: print('elem_x2_hbar_gp2', np.shape(elem_x2_hbar_gp2),tf.reshape(elem_x2_hbar_gp2, new_shape))
        # if pflag: print('elem_y2_hbar_gp1', np.shape(elem_y2_hbar_gp1),tf.reshape(elem_y2_hbar_gp1, new_shape))
        # if pflag: print('elem_y2_hbar_gp2', np.shape(elem_y2_hbar_gp2),tf.reshape(elem_y2_hbar_gp2, new_shape))

        if pflag: print("N1", N[0,:])
        if pflag: print("N2", N[1,:])

        #-------------------- WARNING --------------------------
        # Since here we start to distinguish x-, y- traction/flux, if the residual contains
        # the gradient term, then we can use different B function for either x-direction
        # or reversed y-direction as the operator to calculate the residual on the edge.
        # For now, we are good.
        #
        #---------------- END OF WARNING -----------------------
        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        # print('Rx1', np.shape(Rx1), Rx1*jxw)
        # print('Rx2', np.shape(Rx2), Rx2*jxw)
        # print('Ry1', np.shape(Ry1), Ry1*jxw)
        # print('Ry2', np.shape(Ry2), Ry2*jxw)
        Rx = jxw * (Rx1 + Rx2)
        Ry = jxw * (Ry1 + Ry2)

        if pflag: print('jxw', jxw)
        if pflag: print('elem_x2_hbar_gp1', elem_x2_hbar_gp1)
        # if pflag: print(N[0:1, :])

        # element level residual for traction in either x or y direction
        Rx = tf.reshape(Rx, [-1, new_shape[0], new_shape[1], 2])
        Ry = tf.reshape(Ry, [-1, new_shape[0], new_shape[1], 2])

        if pflag: print('Rx1', np.shape(Rx), Rx[0,:,:,0])
        if pflag: print('Rx2', np.shape(Rx), Rx[0,:,:,1])
        if pflag: print('Ry1', np.shape(Ry), Ry[0,:,:,0])
        if pflag: print('Ry2', np.shape(Ry), Ry[0,:,:,1])
    elif dof == 2:
        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_1, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_1, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_1, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_1, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])
        if pflag: print('elem_x2_hbar_gp1', np.shape(elem_x2_hbar_gp1),tf.reshape(elem_x2_hbar_gp1, new_shape))
        if pflag: print('elem_x2_hbar_gp2', np.shape(elem_x2_hbar_gp2),tf.reshape(elem_x2_hbar_gp2, new_shape))
        if pflag: print('elem_y2_hbar_gp1', np.shape(elem_y2_hbar_gp1),tf.reshape(elem_y2_hbar_gp1, new_shape))
        if pflag: print('elem_y2_hbar_gp2', np.shape(elem_y2_hbar_gp2),tf.reshape(elem_y2_hbar_gp2, new_shape))

        if pflag: print("N1", N[0,:])
        if pflag: print("N2", N[1,:])

        #-------------------- WARNING --------------------------
        # Since here we start to distinguish x-, y- traction/flux, if the residual contains
        # the gradient term, then we can use different B function for either x-direction
        # or reversed y-direction as the operator to calculate the residual on the edge.
        # For now, we are good.
        #
        #---------------- END OF WARNING -----------------------
        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_1 = jxw * (Rx1 + Rx2)
        Ry_1 = jxw * (Ry1 + Ry2)

        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_2, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_2, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_2, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_2, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])

        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_2 = jxw * (Rx1 + Rx2)
        Ry_2 = jxw * (Ry1 + Ry2)

        # element level residual for traction in either x or y direction
        Rx_1 = tf.reshape(Rx_1, [-1, new_shape[0], new_shape[1], 2])
        Ry_1 = tf.reshape(Ry_1, [-1, new_shape[0], new_shape[1], 2])
        Rx_2 = tf.reshape(Rx_2, [-1, new_shape[0], new_shape[1], 2])
        Ry_2 = tf.reshape(Ry_2, [-1, new_shape[0], new_shape[1], 2])

        if pflag: print('Rx1_1', np.shape(Rx_1), Rx_1[0,:,:,0])
        if pflag: print('Rx2_1', np.shape(Rx_1), Rx_1[0,:,:,1])
        if pflag: print('Ry1_1', np.shape(Ry_1), Ry_1[0,:,:,0])
        if pflag: print('Ry2_1', np.shape(Ry_1), Ry_1[0,:,:,1])
        if pflag: print('Rx1_2', np.shape(Rx_2), Rx_2[0,:,:,0])
        if pflag: print('Rx2_2', np.shape(Rx_2), Rx_2[0,:,:,1])
        if pflag: print('Ry1_2', np.shape(Ry_2), Ry_2[0,:,:,0])
        if pflag: print('Ry2_2', np.shape(Ry_2), Ry_2[0,:,:,1])
    elif dof == 3:
        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_1, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_1, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_1, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_1, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])
        if pflag: print('elem_x2_hbar_gp1', np.shape(elem_x2_hbar_gp1),tf.reshape(elem_x2_hbar_gp1, new_shape))
        if pflag: print('elem_x2_hbar_gp2', np.shape(elem_x2_hbar_gp2),tf.reshape(elem_x2_hbar_gp2, new_shape))
        if pflag: print('elem_y2_hbar_gp1', np.shape(elem_y2_hbar_gp1),tf.reshape(elem_y2_hbar_gp1, new_shape))
        if pflag: print('elem_y2_hbar_gp2', np.shape(elem_y2_hbar_gp2),tf.reshape(elem_y2_hbar_gp2, new_shape))

        if pflag: print("N1", N[0,:])
        if pflag: print("N2", N[1,:])

        #-------------------- WARNING --------------------------
        # Since here we start to distinguish x-, y- traction/flux, if the residual contains
        # the gradient term, then we can use different B function for either x-direction
        # or reversed y-direction as the operator to calculate the residual on the edge.
        # For now, we are good.
        #
        #---------------- END OF WARNING -----------------------
        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_1 = jxw * (Rx1 + Rx2)
        Ry_1 = jxw * (Ry1 + Ry2)

        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_2, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_2, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_2, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_2, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])

        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_2 = jxw * (Rx1 + Rx2)
        Ry_2 = jxw * (Ry1 + Ry2)

        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_3, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_3, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_3, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_3, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])

        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_3 = jxw * (Rx1 + Rx2)
        Ry_3 = jxw * (Ry1 + Ry2)


        # element level residual for traction in either x or y direction
        Rx_1 = tf.reshape(Rx_1, [-1, new_shape[0], new_shape[1], 2])
        Ry_1 = tf.reshape(Ry_1, [-1, new_shape[0], new_shape[1], 2])
        Rx_2 = tf.reshape(Rx_2, [-1, new_shape[0], new_shape[1], 2])
        Ry_2 = tf.reshape(Ry_2, [-1, new_shape[0], new_shape[1], 2])
        Rx_3 = tf.reshape(Rx_3, [-1, new_shape[0], new_shape[1], 2])
        Ry_3 = tf.reshape(Ry_3, [-1, new_shape[0], new_shape[1], 2])

        if pflag: print('Rx1_1', np.shape(Rx_1), Rx_1[0,:,:,0])
        if pflag: print('Rx2_1', np.shape(Rx_1), Rx_1[0,:,:,1])
        if pflag: print('Ry1_1', np.shape(Ry_1), Ry_1[0,:,:,0])
        if pflag: print('Ry2_1', np.shape(Ry_1), Ry_1[0,:,:,1])
        if pflag: print('Rx1_2', np.shape(Rx_2), Rx_2[0,:,:,0])
        if pflag: print('Rx2_2', np.shape(Rx_2), Rx_2[0,:,:,1])
        if pflag: print('Ry1_2', np.shape(Ry_2), Ry_2[0,:,:,0])
        if pflag: print('Ry2_2', np.shape(Ry_2), Ry_2[0,:,:,1])
        if pflag: print('Rx1_3', np.shape(Rx_3), Rx_3[0,:,:,0])
        if pflag: print('Rx2_3', np.shape(Rx_3), Rx_3[0,:,:,1])
        if pflag: print('Ry1_3', np.shape(Ry_3), Ry_3[0,:,:,0])
        if pflag: print('Ry2_3', np.shape(Ry_3), Ry_3[0,:,:,1])
    elif dof == 4:
        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_1, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_1, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_1, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_1, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])

        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_1 = jxw * (Rx1 + Rx2)
        Ry_1 = jxw * (Ry1 + Ry2)

        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_2, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_2, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_2, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_2, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])

        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_2 = jxw * (Rx1 + Rx2)
        Ry_2 = jxw * (Ry1 + Ry2)

        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_3, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_3, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_3, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_3, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])

        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_3 = jxw * (Rx1 + Rx2)
        Ry_3 = jxw * (Ry1 + Ry2)

        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_4, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_4, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_4, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_4, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])

        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_4 = jxw * (Rx1 + Rx2)
        Ry_4 = jxw * (Ry1 + Ry2)

        # element level residual for traction in either x or y direction
        Rx_1 = tf.reshape(Rx_1, [-1, new_shape[0], new_shape[1], 2])
        Ry_1 = tf.reshape(Ry_1, [-1, new_shape[0], new_shape[1], 2])
        Rx_2 = tf.reshape(Rx_2, [-1, new_shape[0], new_shape[1], 2])
        Ry_2 = tf.reshape(Ry_2, [-1, new_shape[0], new_shape[1], 2])
        Rx_3 = tf.reshape(Rx_3, [-1, new_shape[0], new_shape[1], 2])
        Ry_3 = tf.reshape(Ry_3, [-1, new_shape[0], new_shape[1], 2])
        Rx_4 = tf.reshape(Rx_4, [-1, new_shape[0], new_shape[1], 2])
        Ry_4 = tf.reshape(Ry_4, [-1, new_shape[0], new_shape[1], 2])



    if dof == 1:
        c_x1 = Rx[:,:,:,0:1]
        c_x2 = tf.roll(Rx[:,:,:,1:2], [1], [1])

        # on 2020-07-16, was not sure, why this is not shift to the row axis to get nodal value. Right now, it's still nodal information 

        c_y1 = Ry[:,:,:,0:1]
        c_y2 = tf.roll(Ry[:,:,:,1:2], [1], [2])
        if pflag: print('Rx 1 (before): ', Rx[0,:,:,0])
        if pflag: print('Rx 1 (after ): ', c_x1[0,:,:,0])
        if pflag: print('Rx 2 (before): ', Rx[0,:,:,1])
        if pflag: print('Rx 2 (after ): ', c_x2[0,:,:,0])

        if pflag: print('Ry 1 (before): ', Ry[0,:,:,0])
        if pflag: print('Ry 1 (after ): ', c_y1[0,:,:,0])
        if pflag: print('Ry 2 (before): ', Ry[0,:,:,1])
        if pflag: print('Ry 2 (after ): ', c_y2[0,:,:,0])

        Rx = c_x1 + c_x2
        Ry = c_y1 + c_y2
        # Add on 2020-07-16. Note on 2020-07-17, not working well
        # Rx = tf.roll(Rx[:,:,:,0:1], [1], [2])
        # Ry = tf.roll(Ry[:,:,:,0:1], [1], [1])
        #---------------------
        if pflag: print('Pay attention to potential errors here')
        if pflag: print('Rx : ', Rx[0,:,:,0])
        if pflag: print('Ry : ', Ry[0,:,:,0])
    elif dof == 2:
        c_x1 = Rx_1[:,:,:,0:1]
        c_x2 = tf.roll(Rx_1[:,:,:,1:2], [1], [1])

        c_y1 = Ry_1[:,:,:,0:1]
        c_y2 = tf.roll(Ry_1[:,:,:,1:2], [1], [2])

        Rx_1 = c_x1 + c_x2
        Ry_1 = c_y1 + c_y2
        if pflag: print('Rx_1 : ', Rx_1[0,:,:,0])
        if pflag: print('Ry_1 : ', Ry_1[0,:,:,0])

        c_x1 = Rx_2[:,:,:,0:1]
        c_x2 = tf.roll(Rx_2[:,:,:,1:2], [1], [1])

        c_y1 = Ry_2[:,:,:,0:1]
        c_y2 = tf.roll(Ry_2[:,:,:,1:2], [1], [2])

        Rx_2 = c_x1 + c_x2
        Ry_2 = c_y1 + c_y2
        if pflag: print('Rx_2 : ', Rx_2[0,:,:,0])
        if pflag: print('Ry_2 : ', Ry_2[0,:,:,0])
        if pflag: print('Pay attention to potential errors here')

    elif dof == 3:
        c_x1 = Rx_1[:,:,:,0:1]
        c_x2 = tf.roll(Rx_1[:,:,:,1:2], [1], [1])

        c_y1 = Ry_1[:,:,:,0:1]
        c_y2 = tf.roll(Ry_1[:,:,:,1:2], [1], [2])

        Rx_1 = c_x1 + c_x2
        Ry_1 = c_y1 + c_y2
        if pflag: print('Rx_1 : ', Rx_1[0,:,:,0])
        if pflag: print('Ry_1 : ', Ry_1[0,:,:,0])

        c_x1 = Rx_2[:,:,:,0:1]
        c_x2 = tf.roll(Rx_2[:,:,:,1:2], [1], [1])

        c_y1 = Ry_2[:,:,:,0:1]
        c_y2 = tf.roll(Ry_2[:,:,:,1:2], [1], [2])

        Rx_2 = c_x1 + c_x2
        Ry_2 = c_y1 + c_y2
        if pflag: print('Rx_2 : ', Rx_2[0,:,:,0])
        if pflag: print('Ry_2 : ', Ry_2[0,:,:,0])
        if pflag: print('Pay attention to potential errors here')

        c_x1 = Rx_3[:,:,:,0:1]
        c_x2 = tf.roll(Rx_3[:,:,:,1:2], [1], [1])

        c_y1 = Ry_3[:,:,:,0:1]
        c_y2 = tf.roll(Ry_3[:,:,:,1:2], [1], [2])

        Rx_3 = c_x1 + c_x2
        Ry_3 = c_y1 + c_y2
        if pflag: print('Rx_3 : ', Rx_3[0,:,:,0])
        if pflag: print('Ry_3 : ', Ry_3[0,:,:,0])
        if pflag: print('Pay attention to potential errors here')
    elif dof == 4:
        c_x1 = Rx_1[:,:,:,0:1]
        c_x2 = tf.roll(Rx_1[:,:,:,1:2], [1], [1])

        c_y1 = Ry_1[:,:,:,0:1]
        c_y2 = tf.roll(Ry_1[:,:,:,1:2], [1], [2])

        Rx_1 = c_x1 + c_x2
        Ry_1 = c_y1 + c_y2

        c_x1 = Rx_2[:,:,:,0:1]
        c_x2 = tf.roll(Rx_2[:,:,:,1:2], [1], [1])

        c_y1 = Ry_2[:,:,:,0:1]
        c_y2 = tf.roll(Ry_2[:,:,:,1:2], [1], [2])

        Rx_2 = c_x1 + c_x2
        Ry_2 = c_y1 + c_y2

        c_x1 = Rx_3[:,:,:,0:1]
        c_x2 = tf.roll(Rx_3[:,:,:,1:2], [1], [1])

        c_y1 = Ry_3[:,:,:,0:1]
        c_y2 = tf.roll(Ry_3[:,:,:,1:2], [1], [2])

        Rx_3 = c_x1 + c_x2
        Ry_3 = c_y1 + c_y2

        c_x1 = Rx_4[:,:,:,0:1]
        c_x2 = tf.roll(Rx_4[:,:,:,1:2], [1], [1])

        c_y1 = Ry_4[:,:,:,0:1]
        c_y2 = tf.roll(Ry_4[:,:,:,1:2], [1], [2])

        Rx_4 = c_x1 + c_x2
        Ry_4 = c_y1 + c_y2


    if dof == 1 :
        R = Rx + Ry
        R = tf.multiply(R, neumann_mask) # remove other edge left R due to conv operation: do test edge (1,3) and (2,4)
        if pflag: print('R: ', np.shape(R), R[0,:,:,0])
        # R = tf.reverse(R, [1]) # disabled on 2020-07-16
        if pflag: print('R: ', np.shape(R), R[0,:,:,0])
    elif dof == 2:
        # R for dof=x
        R_1 = Rx_1 + Ry_1
        if pflag: print('R_1: ', np.shape(R_1), R_1[0,:,:,0])
        # R for dof=y
        R_2 = Rx_2 + Ry_2
        if pflag: print('R_2: ', np.shape(R_2), R_2[0,:,:,0])

        R = tf.concat([R_1, R_2], axis=3)
        R = tf.multiply(R, neumann_mask) # remove other edge left R due to conv operation: do test edge (1,3) and (2,4)
        # R = tf.reverse(R, [1]) # disabled on 2020-07-22
        if pflag: print('R: ', np.shape(R), R[0,:,:,0], R[0,:,:,1])
    elif dof == 3:
        # R for dof=x
        R_1 = Rx_1 + Ry_1
        if pflag: print('R_1: ', np.shape(R_1), R_1[0,:,:,0])
        # R for dof=y
        R_2 = Rx_2 + Ry_2
        if pflag: print('R_2: ', np.shape(R_2), R_2[0,:,:,0])

        R_3 = Rx_3 + Ry_3
        if pflag: print('R_3: ', np.shape(R_3), R_3[0,:,:,0])

        R = tf.concat([R_1, R_2, R_3], axis=3)
        R = tf.multiply(R, neumann_mask) # remove other edge left R due to conv operation: do test edge (1,3) and (2,4)
        # R = tf.reverse(R, [1]) # disabled on 2020-07-22
        if pflag: print('R: ', np.shape(R), R[0,:,:,0], R[0,:,:,1], R[0,:,:,2])
    elif dof == 4:
        # R for dof=x
        R_1 = Rx_1 + Ry_1
        # R for dof=y
        R_2 = Rx_2 + Ry_2

        R_3 = Rx_3 + Ry_3

        R_4 = Rx_4 + Ry_4
        R = tf.concat([R_1, R_2, R_3, R_4], axis=3)
        R = tf.multiply(R, neumann_mask) # remove other edge left R due to conv operation: do test edge (1,3) and (2,4)
        if pflag: print('R: ', np.shape(R), R[0,:,:,0], R[0,:,:,1], R[0,:,:,2], R[0,:,:,3])

    # exit(0)

    return R

def ComputeNeumannBoundaryResidualNodalDataNew(data_input, dh, dof, padding='SAME'):
    """ 
    Compute the residual on the Neumann BCs.  The implementation is based on Neumann BCs is scaled between (-1, 1), and Neumann condition should be always > 0 in the domain region. Raise value error if negative value is detected

    args:
        data_input (numpy array): size of [batch, node_height, node_width, dof*3]
        dof (int): dof per node
        dh (float): element size
    return:
        numpy array: nodal Neumann residual value with size of [batch, node_height, node_width, dof]

    todo:
        make this function to work with (1S, 1V), 2S, 1V1S, 3S, 2V, etc.

        loop over each dof, instead of implementing different dof opt.
    """
    # pflag = True
    pflag = False
    if (dof > 4):
        raise ValueError(" dof = ", dof, " is not implemented! Only dof = 1 or 2 or 3 or 4 is coded!")

    data_input = tf.convert_to_tensor(data_input, dtype=tf.float32)
    if pflag: print('data_input', np.shape(data_input))

    # --------------- Dirichlet data------------
    dirichlet_data = data_input[:,:,:,0:dof]
    if pflag: print('dirichlet_data', dirichlet_data[0,:,:,0])

    #--------------- Domain mask ------------------
    # change actual dirichlet BCs to -2, the all -2 will be the domain
    domain_mask = tf.where( dirichlet_data >= 0.0, tf.fill(tf.shape(dirichlet_data), -2.0), dirichlet_data)
    # print('domain_mask', domain_mask[0,:,:,0])
    domain_mask = tf.where( domain_mask < -1.0, tf.fill(tf.shape(domain_mask), 1.0), tf.fill(tf.shape(domain_mask), 0.0))
    if pflag: print('domain_mask', domain_mask[0,:,:,0])

    #--------------- Neumann BCs ------------------
    Neumann_max = 1.0
    Neumann_min = -1.0
    neumann_data = data_input[:,:,:,dof:dof+dof+dof]
    if pflag: print('neumann_data', neumann_data[0,:,:,0])
    if pflag: print('neumann_data(dof-1)', neumann_data[0,:,:,1])

    # #--------------- check Neumann BCs------------ !!!! should be checked at the very beginning of data.
    # # is not allowed during tensor running/training 
    # check_neumann = tf.multiply(neumann_data, domain_mask)
    # if pflag: print('---check_neumann', check_neumann[0,:,:,0])
    # if (tf.reduce_sum(check_neumann) == 0) :
        # print("WARNING: no Neumann BCs is detected!")
    # check_neumann = tf.where( check_neumann < 0.0, tf.fill(tf.shape(check_neumann), 1.0), tf.fill(tf.shape(check_neumann), 0.0))
    # if pflag: print('check_neumann', check_neumann[0,:,:,0])
    # if (tf.reduce_sum(check_neumann) < 0) :
        # raise ValueError("Neumann BCs should NOT be smaller than zero (< 0). Consider use diffusivity or elastic constant to scale the data!")

    #---------------- Neumann BCs-----------------
    # Should consider the scaling as well.  Any mask will not work, as Neumann BCs can be smaller than 0.0
    # Solution: Neumann BCs is not allowed to be smaller than 0 in the input data.

    neumann_mask = tf.where( neumann_data <= 0.0, tf.fill(tf.shape(neumann_data), 0.0), tf.fill(tf.shape(neumann_data), 1.0))
    if pflag: print('neumann_mask', neumann_mask[0,:,:,0])
    if pflag: print('neumann_mask(dof-1)', neumann_mask[0,:,:,1])
    neumann_data = tf.multiply( neumann_data, neumann_mask)
    if pflag: print('neumann_data', neumann_data[0,:,:,0])
    if pflag: print('neumann_data(dof-1)', neumann_data[0,:,:,1])

    # The main idea is to form a element connection as for the bulk residual case.
    # However, if the neumman BCs is location dependent, has both positive and negative 
    # values, then, it is very challenging to make the following to work stably. 
    # On the other hand, if we only consider the bulk region, then the neumman BCs is
    # literally enforced through the residual form, but not explicitly. 
    # Still, the following might still work.

    # form dof-1 level horizontal element
    # the padding zeros will helps to keep the location of surface node unchanged for the bottom edges
    # n1    n2    -> t_y
    # 1 0   0 1
    # 0 0   0 0
    n1 = np.array([[1, 0], [0, 0]])
    n1 = np.expand_dims(n1, axis=2)
    n1 = np.expand_dims(n1, axis=3)
    n2 = np.array([[0, 1], [0, 0]])
    n2 = np.expand_dims(n2, axis=2)
    n2 = np.expand_dims(n2, axis=3)

    # n3   n4     -> t_x
    # 1 0  0 0
    # 0 0  1 0
    # form dof-1 level vertical element:
    n3 = np.array([[1, 0], [0, 0]])
    n3 = np.expand_dims(n3, axis=2)
    n3 = np.expand_dims(n3, axis=3)
    n4 = np.array([[0, 0], [1, 0]])
    n4 = np.expand_dims(n4, axis=2)
    n4 = np.expand_dims(n4, axis=3)


    if (dof != 1):
        raise ValueError(" t_x and t_y might need to be reversed for dof>1, was not tested in the following implementation!")

    # create surface elements
    if dof == 1:
        # horizontal element
        c_n1 = tf.nn.conv2d(neumann_data[:,:,:,1:2], n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data[:,:,:,1:2], n2, [1,1,1,1], padding)
        elem_y = tf.concat([c_n1, c_n2], 3)

        # vertical element
        c_n3 = tf.nn.conv2d(neumann_data[:,:,:,0:1], n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data[:,:,:,0:1], n4, [1,1,1,1], padding)
        elem_x = tf.concat([c_n3, c_n4], 3)

        if pflag: print('elem_x', np.shape(elem_x))
        if pflag: print('elem_x 1: ', elem_x[0,:,:,0])
        if pflag: print('elem_x 2: ', elem_x[0,:,:,1])

        if pflag: print('elem_y', np.shape(elem_y))
        if pflag: print('elem_y 1: ', elem_y[0,:,:,0])
        if pflag: print('elem_y 2: ', elem_y[0,:,:,1])
    elif dof == 2:
        neumann_data_1 = neumann_data[:,:,:,0:2]
        neumann_data_2 = neumann_data[:,:,:,2:4]

        c_n1 = tf.nn.conv2d(neumann_data_1[:,:,:,0:1], n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_1[:,:,:,0:1], n2, [1,1,1,1], padding)
        elem_y_1 = tf.concat([c_n1, c_n2], 3)

        c_n1 = tf.nn.conv2d(neumann_data_2[:,:,:,0:1], n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_2[:,:,:,0:1], n2, [1,1,1,1], padding)
        elem_y_2 = tf.concat([c_n1, c_n2], 3)

        c_n3 = tf.nn.conv2d(neumann_data_1[:,:,:,1:2], n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_1[:,:,:,1:2], n4, [1,1,1,1], padding)
        elem_x_1 = tf.concat([c_n3, c_n4], 3)

        c_n3 = tf.nn.conv2d(neumann_data_2[:,:,:,1:2], n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_2[:,:,:,1:2], n4, [1,1,1,1], padding)
        elem_x_2 = tf.concat([c_n3, c_n4], 3)

        if pflag: print('elem_y_1 ', np.shape(elem_y_1))
        if pflag: print('elem_y_1 1: ', elem_y_1[0,:,:,0])
        if pflag: print('elem_y_1 2: ', elem_y_1[0,:,:,1])
        if pflag: print('elem_y_2 ', np.shape(elem_y_2))
        if pflag: print('elem_y_2 1: ', elem_y_2[0,:,:,0])
        if pflag: print('elem_y_2 2: ', elem_y_2[0,:,:,1])

        if pflag: print('elem_x_1 ', np.shape(elem_x_1))
        if pflag: print('elem_x_1 1: ', elem_x_1[0,:,:,0])
        if pflag: print('elem_x_1 2: ', elem_x_1[0,:,:,1])
        if pflag: print('elem_x_2 ', np.shape(elem_x_2))
        if pflag: print('elem_x_2 1: ', elem_x_2[0,:,:,0])
        if pflag: print('elem_x_2 2: ', elem_x_2[0,:,:,1])
    elif dof == 3:
        neumann_data_1 = neumann_data[:,:,:,0:2]
        neumann_data_2 = neumann_data[:,:,:,2:4]
        neumann_data_3 = neumann_data[:,:,:,4:6]

        c_n1 = tf.nn.conv2d(neumann_data_1[:,:,:,0:1], n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_1[:,:,:,0:1], n2, [1,1,1,1], padding)
        elem_y_1 = tf.concat([c_n1, c_n2], 3)

        c_n1 = tf.nn.conv2d(neumann_data_2[:,:,:,0:1], n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_2[:,:,:,0:1], n2, [1,1,1,1], padding)
        elem_y_2 = tf.concat([c_n1, c_n2], 3)

        c_n1 = tf.nn.conv2d(neumann_data_3[:,:,:,0:1], n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_3[:,:,:,0:1], n2, [1,1,1,1], padding)
        elem_y_3 = tf.concat([c_n1, c_n2], 3)

        c_n3 = tf.nn.conv2d(neumann_data_1[:,:,:,1:2], n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_1[:,:,:,1:2], n4, [1,1,1,1], padding)
        elem_x_1 = tf.concat([c_n3, c_n4], 3)

        c_n3 = tf.nn.conv2d(neumann_data_2[:,:,:,1:2], n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_2[:,:,:,1:2], n4, [1,1,1,1], padding)
        elem_x_2 = tf.concat([c_n3, c_n4], 3)

        c_n3 = tf.nn.conv2d(neumann_data_3[:,:,:,1:2], n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_3[:,:,:,1:2], n4, [1,1,1,1], padding)
        elem_x_3 = tf.concat([c_n3, c_n4], 3)

        if pflag: print('elem_y_1 ', np.shape(elem_y_1))
        if pflag: print('elem_y_1 1: ', elem_y_1[0,:,:,0])
        if pflag: print('elem_y_1 2: ', elem_y_1[0,:,:,1])
        if pflag: print('elem_y_2 ', np.shape(elem_y_2))
        if pflag: print('elem_y_2 1: ', elem_y_2[0,:,:,0])
        if pflag: print('elem_y_2 2: ', elem_y_2[0,:,:,1])
        if pflag: print('elem_y_3 ', np.shape(elem_y_3))
        if pflag: print('elem_y_3 1: ', elem_y_3[0,:,:,0])
        if pflag: print('elem_y_3 2: ', elem_y_3[0,:,:,1])

        if pflag: print('elem_x_1 ', np.shape(elem_x_1))
        if pflag: print('elem_x_1 1: ', elem_x_1[0,:,:,0])
        if pflag: print('elem_x_1 2: ', elem_x_1[0,:,:,1])
        if pflag: print('elem_x_2 ', np.shape(elem_x_2))
        if pflag: print('elem_x_2 1: ', elem_x_2[0,:,:,0])
        if pflag: print('elem_x_2 2: ', elem_x_2[0,:,:,1])
        if pflag: print('elem_x_3 ', np.shape(elem_x_3))
        if pflag: print('elem_x_3 1: ', elem_x_3[0,:,:,0])
        if pflag: print('elem_x_3 2: ', elem_x_3[0,:,:,1])

    elif dof == 4:
        neumann_data_1 = neumann_data[:,:,:,0:2]
        neumann_data_2 = neumann_data[:,:,:,2:4]
        neumann_data_3 = neumann_data[:,:,:,4:6]
        neumann_data_4 = neumann_data[:,:,:,6:8]

        c_n1 = tf.nn.conv2d(neumann_data_1[:,:,:,0:1], n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_1[:,:,:,0:1], n2, [1,1,1,1], padding)
        elem_y_1 = tf.concat([c_n1, c_n2], 3)

        c_n1 = tf.nn.conv2d(neumann_data_2[:,:,:,0:1], n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_2[:,:,:,0:1], n2, [1,1,1,1], padding)
        elem_y_2 = tf.concat([c_n1, c_n2], 3)

        c_n1 = tf.nn.conv2d(neumann_data_3[:,:,:,0:1], n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_3[:,:,:,0:1], n2, [1,1,1,1], padding)
        elem_y_3 = tf.concat([c_n1, c_n2], 3)

        c_n1 = tf.nn.conv2d(neumann_data_4[:,:,:,0:1], n1, [1,1,1,1], padding )
        c_n2 = tf.nn.conv2d(neumann_data_4[:,:,:,0:1], n2, [1,1,1,1], padding)
        elem_y_4 = tf.concat([c_n1, c_n2], 3)

        c_n3 = tf.nn.conv2d(neumann_data_1[:,:,:,1:2], n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_1[:,:,:,1:2], n4, [1,1,1,1], padding)
        elem_x_1 = tf.concat([c_n3, c_n4], 3)

        c_n3 = tf.nn.conv2d(neumann_data_2[:,:,:,1:2], n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_2[:,:,:,1:2], n4, [1,1,1,1], padding)
        elem_x_2 = tf.concat([c_n3, c_n4], 3)

        c_n3 = tf.nn.conv2d(neumann_data_3[:,:,:,1:2], n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_3[:,:,:,1:2], n4, [1,1,1,1], padding)
        elem_x_3 = tf.concat([c_n3, c_n4], 3)

        c_n3 = tf.nn.conv2d(neumann_data_4[:,:,:,1:2], n3, [1,1,1,1], padding )
        c_n4 = tf.nn.conv2d(neumann_data_4[:,:,:,1:2], n4, [1,1,1,1], padding)
        elem_x_4 = tf.concat([c_n3, c_n4], 3)



    if dof == 1:
        # channel 0:1 == channel 1:2
        # create a mask to delete data that are not properly aligned
        c_n1_mask = tf.nn.conv2d(neumann_mask[:,:,:,1:2], n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask[:,:,:,1:2], n2, [1,1,1,1], padding)
        elem_y_mask = tf.multiply(c_n1_mask, c_n2_mask)
        if pflag: print('c_n1_mask: ', c_n1_mask[0,:,:,0])
        if pflag: print('c_n2_mask: ', c_n2_mask[0,:,:,0])
        if pflag: print('elem_y_mask: ', elem_y_mask[0,:,:,0])
    
        # create a mask to delete data that are not properly aligned
        c_n3_mask = tf.nn.conv2d(neumann_mask[:,:,:,0:1], n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask[:,:,:,0:1], n4, [1,1,1,1], padding)
        elem_x_mask = tf.multiply(c_n3_mask, c_n4_mask)
        if pflag: print('c_n3_mask: ', c_n3_mask[0,:,:,0])
        if pflag: print('c_n4_mask: ', c_n4_mask[0,:,:,0])
        if pflag: print('elem_x_mask: ', elem_x_mask[0,:,:,0])
    elif dof == 2:
        # For the 3D case, it would be impossible to perform task like this.
        # Thus, how to come up with a 3D implementation, or sparse pattern 
        # would be extremely useful.

        # create a mask to delete data that are not properly aligned
        neumann_mask_1 = neumann_mask[:,:,:,0:1] # 0:1 = 1:2
        neumann_mask_2 = neumann_mask[:,:,:,2:3]

        c_n1_mask = tf.nn.conv2d(neumann_mask_1, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_1, n2, [1,1,1,1], padding)
        elem_y_mask_1 = tf.multiply(c_n1_mask, c_n2_mask)

        if pflag: print('c_n1_mask: ', c_n1_mask[0,:,:,0])
        if pflag: print('c_n2_mask: ', c_n2_mask[0,:,:,0])
        if pflag: print('elem_y_mask_1: ', elem_y_mask_1[0,:,:,0])

        c_n1_mask = tf.nn.conv2d(neumann_mask_2, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_2, n2, [1,1,1,1], padding)
        elem_y_mask_2 = tf.multiply(c_n1_mask, c_n2_mask)

        if pflag: print('c_n1_mask: ', c_n1_mask[0,:,:,0])
        if pflag: print('c_n2_mask: ', c_n2_mask[0,:,:,0])
        if pflag: print('elem_y_mask_2: ', elem_y_mask_2[0,:,:,0])

    
        # create a mask to delete data that are not properly aligned
        c_n3_mask = tf.nn.conv2d(neumann_mask_1, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_1, n4, [1,1,1,1], padding)
        elem_x_mask_1 = tf.multiply(c_n3_mask, c_n4_mask)
        if pflag: print('c_n3_mask: ', c_n3_mask[0,:,:,0])
        if pflag: print('c_n4_mask: ', c_n4_mask[0,:,:,0])
        if pflag: print('elem_x_mask_1: ', elem_x_mask_1[0,:,:,0])

        c_n3_mask = tf.nn.conv2d(neumann_mask_2, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_2, n4, [1,1,1,1], padding)
        elem_x_mask_2 = tf.multiply(c_n3_mask, c_n4_mask)
        if pflag: print('c_n3_mask: ', c_n3_mask[0,:,:,0])
        if pflag: print('c_n4_mask: ', c_n4_mask[0,:,:,0])
        if pflag: print('elem_x_mask_2: ', elem_x_mask_2[0,:,:,0])

    elif dof == 3:

        # create a mask to delete data that are not properly aligned
        neumann_mask_1 = neumann_mask[:,:,:,0:1]
        neumann_mask_2 = neumann_mask[:,:,:,2:3]
        neumann_mask_3 = neumann_mask[:,:,:,4:5]

        c_n1_mask = tf.nn.conv2d(neumann_mask_1, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_1, n2, [1,1,1,1], padding)
        elem_y_mask_1 = tf.multiply(c_n1_mask, c_n2_mask)

        if pflag: print('c_n1_mask: ', c_n1_mask[0,:,:,0])
        if pflag: print('c_n2_mask: ', c_n2_mask[0,:,:,0])
        if pflag: print('elem_y_mask_1: ', elem_y_mask_1[0,:,:,0])

        c_n1_mask = tf.nn.conv2d(neumann_mask_2, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_2, n2, [1,1,1,1], padding)
        elem_y_mask_2 = tf.multiply(c_n1_mask, c_n2_mask)

        if pflag: print('c_n1_mask: ', c_n1_mask[0,:,:,0])
        if pflag: print('c_n2_mask: ', c_n2_mask[0,:,:,0])
        if pflag: print('elem_y_mask_2: ', elem_y_mask_2[0,:,:,0])

        c_n1_mask = tf.nn.conv2d(neumann_mask_3, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_3, n2, [1,1,1,1], padding)
        elem_y_mask_3 = tf.multiply(c_n1_mask, c_n2_mask)

        if pflag: print('c_n1_mask: ', c_n1_mask[0,:,:,0])
        if pflag: print('c_n2_mask: ', c_n2_mask[0,:,:,0])
        if pflag: print('elem_y_mask_3: ', elem_y_mask_3[0,:,:,0])

    
        # create a mask to delete data that are not properly aligned
        c_n3_mask = tf.nn.conv2d(neumann_mask_1, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_1, n4, [1,1,1,1], padding)
        elem_x_mask_1 = tf.multiply(c_n3_mask, c_n4_mask)
        if pflag: print('c_n3_mask: ', c_n3_mask[0,:,:,0])
        if pflag: print('c_n4_mask: ', c_n4_mask[0,:,:,0])
        if pflag: print('elem_x_mask_1: ', elem_x_mask_1[0,:,:,0])

        c_n3_mask = tf.nn.conv2d(neumann_mask_2, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_2, n4, [1,1,1,1], padding)
        elem_x_mask_2 = tf.multiply(c_n3_mask, c_n4_mask)
        if pflag: print('c_n3_mask: ', c_n3_mask[0,:,:,0])
        if pflag: print('c_n4_mask: ', c_n4_mask[0,:,:,0])
        if pflag: print('elem_x_mask_2: ', elem_x_mask_2[0,:,:,0])

        c_n3_mask = tf.nn.conv2d(neumann_mask_3, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_3, n4, [1,1,1,1], padding)
        elem_x_mask_3 = tf.multiply(c_n3_mask, c_n4_mask)
        if pflag: print('c_n3_mask: ', c_n3_mask[0,:,:,0])
        if pflag: print('c_n4_mask: ', c_n4_mask[0,:,:,0])
        if pflag: print('elem_x_mask_3: ', elem_x_mask_3[0,:,:,0])

    elif dof == 4:

        # create a mask to delete data that are not properly aligned
        neumann_mask_1 = neumann_mask[:,:,:,0:1]
        neumann_mask_2 = neumann_mask[:,:,:,2:3]
        neumann_mask_3 = neumann_mask[:,:,:,4:5]
        neumann_mask_4 = neumann_mask[:,:,:,6:7]

        c_n1_mask = tf.nn.conv2d(neumann_mask_1, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_1, n2, [1,1,1,1], padding)
        elem_y_mask_1 = tf.multiply(c_n1_mask, c_n2_mask)

        c_n1_mask = tf.nn.conv2d(neumann_mask_2, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_2, n2, [1,1,1,1], padding)
        elem_y_mask_2 = tf.multiply(c_n1_mask, c_n2_mask)

        c_n1_mask = tf.nn.conv2d(neumann_mask_3, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_3, n2, [1,1,1,1], padding)
        elem_y_mask_3 = tf.multiply(c_n1_mask, c_n2_mask)

        c_n1_mask = tf.nn.conv2d(neumann_mask_4, n1, [1,1,1,1], padding )
        c_n2_mask = tf.nn.conv2d(neumann_mask_4, n2, [1,1,1,1], padding)
        elem_y_mask_4 = tf.multiply(c_n1_mask, c_n2_mask)

    
        # create a mask to delete data that are not properly aligned
        c_n3_mask = tf.nn.conv2d(neumann_mask_1, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_1, n4, [1,1,1,1], padding)
        elem_x_mask_1 = tf.multiply(c_n3_mask, c_n4_mask)

        c_n3_mask = tf.nn.conv2d(neumann_mask_2, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_2, n4, [1,1,1,1], padding)
        elem_x_mask_2 = tf.multiply(c_n3_mask, c_n4_mask)

        c_n3_mask = tf.nn.conv2d(neumann_mask_3, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_3, n4, [1,1,1,1], padding)
        elem_x_mask_3 = tf.multiply(c_n3_mask, c_n4_mask)

        c_n3_mask = tf.nn.conv2d(neumann_mask_4, n3, [1,1,1,1], padding )
        c_n4_mask = tf.nn.conv2d(neumann_mask_4, n4, [1,1,1,1], padding)
        elem_x_mask_4 = tf.multiply(c_n3_mask, c_n4_mask)
    
    


    if dof == 1:
        # Scale the Neumann BC value back to the original one
        # original scale in VtuDataGenerateFixedc.py: 
        #   - data = (data + (self.upperlimit - self.lowerlimit) * 0.5 ) * 0.5
        elem_x = 2.0 * elem_x - (Neumann_max - Neumann_min) * 0.5
        elem_y = 2.0 * elem_y - (Neumann_max - Neumann_min) * 0.5

        clean_elem_y = tf.multiply(elem_y, elem_y_mask)
        clean_elem_x = tf.multiply(elem_x, elem_x_mask)
        if pflag: print('clean_elem_y (node1, 2)', np.shape(clean_elem_y), clean_elem_y[0,:,:,0], clean_elem_y[0,:,:,1])
        if pflag: print('clean_elem_x (node1, 2)', np.shape(clean_elem_x), clean_elem_x[0,:,:,0], clean_elem_x[0,:,:,1])

    elif dof == 2:
        elem_x_1 = 2.0 * elem_x_1 - (Neumann_max - Neumann_min) * 0.5
        elem_y_1 = 2.0 * elem_y_1 - (Neumann_max - Neumann_min) * 0.5
        elem_x_2 = 2.0 * elem_x_2 - (Neumann_max - Neumann_min) * 0.5
        elem_y_2 = 2.0 * elem_y_2 - (Neumann_max - Neumann_min) * 0.5

        clean_elem_y_1 = tf.multiply(elem_y_1, elem_y_mask_1)
        clean_elem_x_1 = tf.multiply(elem_x_1, elem_x_mask_1)
        clean_elem_y_2 = tf.multiply(elem_y_2, elem_y_mask_2)
        clean_elem_x_2 = tf.multiply(elem_x_2, elem_x_mask_2)

        if pflag: print('clean_elem_y_1 (node1, 2)', np.shape(clean_elem_y_1), clean_elem_y_1[0,:,:,0], clean_elem_y_1[0,:,:,1])
        if pflag: print('clean_elem_x_1 (node1, 2)', np.shape(clean_elem_x_1), clean_elem_x_1[0,:,:,0], clean_elem_x_1[0,:,:,1])
        if pflag: print('clean_elem_y_2 (node1, 2)', np.shape(clean_elem_y_2), clean_elem_y_2[0,:,:,0], clean_elem_y_2[0,:,:,1])
        if pflag: print('clean_elem_x_2 (node1, 2)', np.shape(clean_elem_x_2), clean_elem_x_2[0,:,:,0], clean_elem_x_2[0,:,:,1])
    elif dof == 3:
        elem_x_1 = 2.0 * elem_x_1 - (Neumann_max - Neumann_min) * 0.5
        elem_y_1 = 2.0 * elem_y_1 - (Neumann_max - Neumann_min) * 0.5
        elem_x_2 = 2.0 * elem_x_2 - (Neumann_max - Neumann_min) * 0.5
        elem_y_2 = 2.0 * elem_y_2 - (Neumann_max - Neumann_min) * 0.5
        elem_x_3 = 2.0 * elem_x_3 - (Neumann_max - Neumann_min) * 0.5
        elem_y_3 = 2.0 * elem_y_3 - (Neumann_max - Neumann_min) * 0.5

        clean_elem_y_1 = tf.multiply(elem_y_1, elem_y_mask_1)
        clean_elem_x_1 = tf.multiply(elem_x_1, elem_x_mask_1)
        clean_elem_y_2 = tf.multiply(elem_y_2, elem_y_mask_2)
        clean_elem_x_2 = tf.multiply(elem_x_2, elem_x_mask_2)
        clean_elem_y_3 = tf.multiply(elem_y_3, elem_y_mask_3)
        clean_elem_x_3 = tf.multiply(elem_x_3, elem_x_mask_3)

        if pflag: print('clean_elem_y_1 (node1, 2)', np.shape(clean_elem_y_1), clean_elem_y_1[0,:,:,0], clean_elem_y_1[0,:,:,1])
        if pflag: print('clean_elem_x_1 (node1, 2)', np.shape(clean_elem_x_1), clean_elem_x_1[0,:,:,0], clean_elem_x_1[0,:,:,1])
        if pflag: print('clean_elem_y_2 (node1, 2)', np.shape(clean_elem_y_2), clean_elem_y_2[0,:,:,0], clean_elem_y_2[0,:,:,1])
        if pflag: print('clean_elem_x_2 (node1, 2)', np.shape(clean_elem_x_2), clean_elem_x_2[0,:,:,0], clean_elem_x_2[0,:,:,1])
        if pflag: print('clean_elem_y_3 (node1, 2)', np.shape(clean_elem_y_3), clean_elem_y_3[0,:,:,0], clean_elem_y_3[0,:,:,1])
        if pflag: print('clean_elem_x_3 (node1, 2)', np.shape(clean_elem_x_3), clean_elem_x_3[0,:,:,0], clean_elem_x_3[0,:,:,1])

    elif dof == 4:
        elem_x_1 = 2.0 * elem_x_1 - (Neumann_max - Neumann_min) * 0.5
        elem_y_1 = 2.0 * elem_y_1 - (Neumann_max - Neumann_min) * 0.5
        elem_x_2 = 2.0 * elem_x_2 - (Neumann_max - Neumann_min) * 0.5
        elem_y_2 = 2.0 * elem_y_2 - (Neumann_max - Neumann_min) * 0.5
        elem_x_3 = 2.0 * elem_x_3 - (Neumann_max - Neumann_min) * 0.5
        elem_y_3 = 2.0 * elem_y_3 - (Neumann_max - Neumann_min) * 0.5
        elem_x_4 = 2.0 * elem_x_4 - (Neumann_max - Neumann_min) * 0.5
        elem_y_4 = 2.0 * elem_y_4 - (Neumann_max - Neumann_min) * 0.5

        clean_elem_y_1 = tf.multiply(elem_y_1, elem_y_mask_1)
        clean_elem_x_1 = tf.multiply(elem_x_1, elem_x_mask_1)
        clean_elem_y_2 = tf.multiply(elem_y_2, elem_y_mask_2)
        clean_elem_x_2 = tf.multiply(elem_x_2, elem_x_mask_2)
        clean_elem_y_3 = tf.multiply(elem_y_3, elem_y_mask_3)
        clean_elem_x_3 = tf.multiply(elem_x_3, elem_x_mask_3)
        clean_elem_y_4 = tf.multiply(elem_y_4, elem_y_mask_4)
        clean_elem_x_4 = tf.multiply(elem_x_4, elem_x_mask_4)



    if dof == 1 :
        shape=elem_x.get_shape()[0:].as_list()    
        new_shape = shape[1:3]
        if pflag: print('new_shape:', new_shape)
    elif dof == 2 :
        shape=elem_x_1.get_shape()[0:].as_list()    
        new_shape = shape[1:3]
        if pflag: print('new_shape:', new_shape)
    elif dof == 3 :
        shape=elem_x_1.get_shape()[0:].as_list()    
        new_shape = shape[1:3]
        if pflag: print('new_shape:', new_shape)
    elif dof == 4 :
        shape=elem_x_1.get_shape()[0:].as_list()    
        new_shape = shape[1:3]


    # get the 1D info, and then perform a N h calculation
    # and then unfold everything to the nodal value
    # 
    N, B, jxw = Get1DGaussPointInfo(dh=dh, GPs=2, dof=1)
    if pflag: print("N", np.shape(N))
    if pflag: print("B", np.shape(B))
    if pflag: print("jxw", jxw)

    if dof == 1:
        elem_x2 = tf.reshape(clean_elem_x,[-1, 2])
        elem_y2 = tf.reshape(clean_elem_y,[-1, 2])

        if pflag: print('elem_x2', np.shape(elem_x2), elem_x2)
        if pflag: print('elem_y2', np.shape(elem_y2), elem_y2)

    elif dof == 2:
        elem_x2_1 = tf.reshape(clean_elem_x_1,[-1, 2])
        elem_y2_1 = tf.reshape(clean_elem_y_1,[-1, 2])
        elem_x2_2 = tf.reshape(clean_elem_x_2,[-1, 2])
        elem_y2_2 = tf.reshape(clean_elem_y_2,[-1, 2])

        if pflag: print('elem_x2_1', np.shape(elem_x2_1), elem_x2_1)
        if pflag: print('elem_y2_1', np.shape(elem_y2_1), elem_y2_1)
        if pflag: print('elem_x2_2', np.shape(elem_x2_2), elem_x2_2)
        if pflag: print('elem_y2_2', np.shape(elem_y2_2), elem_y2_2)
    elif dof == 3:
        elem_x2_1 = tf.reshape(clean_elem_x_1,[-1, 2])
        elem_y2_1 = tf.reshape(clean_elem_y_1,[-1, 2])
        elem_x2_2 = tf.reshape(clean_elem_x_2,[-1, 2])
        elem_y2_2 = tf.reshape(clean_elem_y_2,[-1, 2])
        elem_x2_3 = tf.reshape(clean_elem_x_3,[-1, 2])
        elem_y2_3 = tf.reshape(clean_elem_y_3,[-1, 2])

        if pflag: print('elem_x2_1', np.shape(elem_x2_1), elem_x2_1)
        if pflag: print('elem_y2_1', np.shape(elem_y2_1), elem_y2_1)
        if pflag: print('elem_x2_2', np.shape(elem_x2_2), elem_x2_2)
        if pflag: print('elem_y2_2', np.shape(elem_y2_2), elem_y2_2)
        if pflag: print('elem_x2_3', np.shape(elem_x2_3), elem_x2_3)
        if pflag: print('elem_y2_3', np.shape(elem_y2_3), elem_y2_3)
    elif dof == 4:
        elem_x2_1 = tf.reshape(clean_elem_x_1,[-1, 2])
        elem_y2_1 = tf.reshape(clean_elem_y_1,[-1, 2])
        elem_x2_2 = tf.reshape(clean_elem_x_2,[-1, 2])
        elem_y2_2 = tf.reshape(clean_elem_y_2,[-1, 2])
        elem_x2_3 = tf.reshape(clean_elem_x_3,[-1, 2])
        elem_y2_3 = tf.reshape(clean_elem_y_3,[-1, 2])
        elem_x2_4 = tf.reshape(clean_elem_x_4,[-1, 2])
        elem_y2_4 = tf.reshape(clean_elem_y_4,[-1, 2])


    if dof == 1:
        # int(N^T h) dA: h是nodal value，必须通过shape fcn来分析正确的值, 但是它也是scale了的值， 
        # Calculate the hbar at the GPs based on nodal info.
        # GP1
        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2, N[0,:]) 
        # GP2
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2, N[1,:]) 
        # GP1
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2, N[0,:]) 
        # GP2
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])

        # if pflag: print('elem_x2_hbar_gp1', np.shape(elem_x2_hbar_gp1),tf.reshape(elem_x2_hbar_gp1, new_shape)) # work for [1, 16, 16, 1], but not [8, 16, 16, 1]
        # if pflag: print('elem_x2_hbar_gp2', np.shape(elem_x2_hbar_gp2),tf.reshape(elem_x2_hbar_gp2, new_shape))
        # if pflag: print('elem_y2_hbar_gp1', np.shape(elem_y2_hbar_gp1),tf.reshape(elem_y2_hbar_gp1, new_shape))
        # if pflag: print('elem_y2_hbar_gp2', np.shape(elem_y2_hbar_gp2),tf.reshape(elem_y2_hbar_gp2, new_shape))

        if pflag: print("N1", N[0,:])
        if pflag: print("N2", N[1,:])

        #-------------------- WARNING --------------------------
        # Since here we start to distinguish x-, y- traction/flux, if the residual contains
        # the gradient term, then we can use different B function for either x-direction
        # or reversed y-direction as the operator to calculate the residual on the edge.
        # For now, we are good.
        #
        #---------------- END OF WARNING -----------------------
        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        # print('Rx1', np.shape(Rx1), Rx1*jxw)
        # print('Rx2', np.shape(Rx2), Rx2*jxw)
        # print('Ry1', np.shape(Ry1), Ry1*jxw)
        # print('Ry2', np.shape(Ry2), Ry2*jxw)
        Rx = jxw * (Rx1 + Rx2)
        Ry = jxw * (Ry1 + Ry2)

        if pflag: print('jxw', jxw)
        if pflag: print('elem_x2_hbar_gp1', elem_x2_hbar_gp1)
        # if pflag: print(N[0:1, :])

        # element level residual for traction in either x or y direction
        Rx = tf.reshape(Rx, [-1, new_shape[0], new_shape[1], 2])
        Ry = tf.reshape(Ry, [-1, new_shape[0], new_shape[1], 2])

        if pflag: print('Rx1', np.shape(Rx), Rx[0,:,:,0])
        if pflag: print('Rx2', np.shape(Rx), Rx[0,:,:,1])
        if pflag: print('Ry1', np.shape(Ry), Ry[0,:,:,0])
        if pflag: print('Ry2', np.shape(Ry), Ry[0,:,:,1])

    elif dof == 2:
        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_1, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_1, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_1, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_1, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])
        if pflag: print('elem_x2_hbar_gp1', np.shape(elem_x2_hbar_gp1),tf.reshape(elem_x2_hbar_gp1, new_shape))
        if pflag: print('elem_x2_hbar_gp2', np.shape(elem_x2_hbar_gp2),tf.reshape(elem_x2_hbar_gp2, new_shape))
        if pflag: print('elem_y2_hbar_gp1', np.shape(elem_y2_hbar_gp1),tf.reshape(elem_y2_hbar_gp1, new_shape))
        if pflag: print('elem_y2_hbar_gp2', np.shape(elem_y2_hbar_gp2),tf.reshape(elem_y2_hbar_gp2, new_shape))

        if pflag: print("N1", N[0,:])
        if pflag: print("N2", N[1,:])

        #-------------------- WARNING --------------------------
        # Since here we start to distinguish x-, y- traction/flux, if the residual contains
        # the gradient term, then we can use different B function for either x-direction
        # or reversed y-direction as the operator to calculate the residual on the edge.
        # For now, we are good.
        #
        #---------------- END OF WARNING -----------------------
        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_1 = jxw * (Rx1 + Rx2)
        Ry_1 = jxw * (Ry1 + Ry2)

        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_2, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_2, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_2, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_2, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])

        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_2 = jxw * (Rx1 + Rx2)
        Ry_2 = jxw * (Ry1 + Ry2)

        # element level residual for traction in either x or y direction
        Rx_1 = tf.reshape(Rx_1, [-1, new_shape[0], new_shape[1], 2])
        Ry_1 = tf.reshape(Ry_1, [-1, new_shape[0], new_shape[1], 2])
        Rx_2 = tf.reshape(Rx_2, [-1, new_shape[0], new_shape[1], 2])
        Ry_2 = tf.reshape(Ry_2, [-1, new_shape[0], new_shape[1], 2])

        if pflag: print('Rx1_1', np.shape(Rx_1), Rx_1[0,:,:,0])
        if pflag: print('Rx2_1', np.shape(Rx_1), Rx_1[0,:,:,1])
        if pflag: print('Ry1_1', np.shape(Ry_1), Ry_1[0,:,:,0])
        if pflag: print('Ry2_1', np.shape(Ry_1), Ry_1[0,:,:,1])
        if pflag: print('Rx1_2', np.shape(Rx_2), Rx_2[0,:,:,0])
        if pflag: print('Rx2_2', np.shape(Rx_2), Rx_2[0,:,:,1])
        if pflag: print('Ry1_2', np.shape(Ry_2), Ry_2[0,:,:,0])
        if pflag: print('Ry2_2', np.shape(Ry_2), Ry_2[0,:,:,1])
    elif dof == 3:
        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_1, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_1, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_1, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_1, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])
        if pflag: print('elem_x2_hbar_gp1', np.shape(elem_x2_hbar_gp1),tf.reshape(elem_x2_hbar_gp1, new_shape))
        if pflag: print('elem_x2_hbar_gp2', np.shape(elem_x2_hbar_gp2),tf.reshape(elem_x2_hbar_gp2, new_shape))
        if pflag: print('elem_y2_hbar_gp1', np.shape(elem_y2_hbar_gp1),tf.reshape(elem_y2_hbar_gp1, new_shape))
        if pflag: print('elem_y2_hbar_gp2', np.shape(elem_y2_hbar_gp2),tf.reshape(elem_y2_hbar_gp2, new_shape))

        if pflag: print("N1", N[0,:])
        if pflag: print("N2", N[1,:])

        #-------------------- WARNING --------------------------
        # Since here we start to distinguish x-, y- traction/flux, if the residual contains
        # the gradient term, then we can use different B function for either x-direction
        # or reversed y-direction as the operator to calculate the residual on the edge.
        # For now, we are good.
        #
        #---------------- END OF WARNING -----------------------
        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_1 = jxw * (Rx1 + Rx2)
        Ry_1 = jxw * (Ry1 + Ry2)

        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_2, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_2, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_2, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_2, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])

        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_2 = jxw * (Rx1 + Rx2)
        Ry_2 = jxw * (Ry1 + Ry2)

        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_3, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_3, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_3, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_3, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])

        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_3 = jxw * (Rx1 + Rx2)
        Ry_3 = jxw * (Ry1 + Ry2)


        # element level residual for traction in either x or y direction
        Rx_1 = tf.reshape(Rx_1, [-1, new_shape[0], new_shape[1], 2])
        Ry_1 = tf.reshape(Ry_1, [-1, new_shape[0], new_shape[1], 2])
        Rx_2 = tf.reshape(Rx_2, [-1, new_shape[0], new_shape[1], 2])
        Ry_2 = tf.reshape(Ry_2, [-1, new_shape[0], new_shape[1], 2])
        Rx_3 = tf.reshape(Rx_3, [-1, new_shape[0], new_shape[1], 2])
        Ry_3 = tf.reshape(Ry_3, [-1, new_shape[0], new_shape[1], 2])

        if pflag: print('Rx1_1', np.shape(Rx_1), Rx_1[0,:,:,0])
        if pflag: print('Rx2_1', np.shape(Rx_1), Rx_1[0,:,:,1])
        if pflag: print('Ry1_1', np.shape(Ry_1), Ry_1[0,:,:,0])
        if pflag: print('Ry2_1', np.shape(Ry_1), Ry_1[0,:,:,1])
        if pflag: print('Rx1_2', np.shape(Rx_2), Rx_2[0,:,:,0])
        if pflag: print('Rx2_2', np.shape(Rx_2), Rx_2[0,:,:,1])
        if pflag: print('Ry1_2', np.shape(Ry_2), Ry_2[0,:,:,0])
        if pflag: print('Ry2_2', np.shape(Ry_2), Ry_2[0,:,:,1])
        if pflag: print('Rx1_3', np.shape(Rx_3), Rx_3[0,:,:,0])
        if pflag: print('Rx2_3', np.shape(Rx_3), Rx_3[0,:,:,1])
        if pflag: print('Ry1_3', np.shape(Ry_3), Ry_3[0,:,:,0])
        if pflag: print('Ry2_3', np.shape(Ry_3), Ry_3[0,:,:,1])
    elif dof == 4:
        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_1, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_1, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_1, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_1, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])

        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_1 = jxw * (Rx1 + Rx2)
        Ry_1 = jxw * (Ry1 + Ry2)

        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_2, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_2, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_2, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_2, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])

        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_2 = jxw * (Rx1 + Rx2)
        Ry_2 = jxw * (Ry1 + Ry2)

        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_3, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_3, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_3, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_3, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])

        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_3 = jxw * (Rx1 + Rx2)
        Ry_3 = jxw * (Ry1 + Ry2)

        elem_x2_hbar_gp1 = tf.linalg.matvec(elem_x2_4, N[0,:]) 
        elem_x2_hbar_gp2 = tf.linalg.matvec(elem_x2_4, N[1,:]) 
        elem_y2_hbar_gp1 = tf.linalg.matvec(elem_y2_4, N[0,:]) 
        elem_y2_hbar_gp2 = tf.linalg.matvec(elem_y2_4, N[1,:]) 

        elem_x2_hbar_gp1 = tf.reshape(elem_x2_hbar_gp1,[-1, 1])
        elem_x2_hbar_gp2 = tf.reshape(elem_x2_hbar_gp2,[-1, 1])
        elem_y2_hbar_gp1 = tf.reshape(elem_y2_hbar_gp1,[-1, 1])
        elem_y2_hbar_gp2 = tf.reshape(elem_y2_hbar_gp2,[-1, 1])

        Rx1 = tf.matmul(elem_x2_hbar_gp1, N[0:1,:]) 
        Rx2 = tf.matmul(elem_x2_hbar_gp2, N[1:2,:]) 
        Ry1 = tf.matmul(elem_y2_hbar_gp1, N[0:1,:]) 
        Ry2 = tf.matmul(elem_y2_hbar_gp2, N[1:2,:]) 

        Rx_4 = jxw * (Rx1 + Rx2)
        Ry_4 = jxw * (Ry1 + Ry2)

        # element level residual for traction in either x or y direction
        Rx_1 = tf.reshape(Rx_1, [-1, new_shape[0], new_shape[1], 2])
        Ry_1 = tf.reshape(Ry_1, [-1, new_shape[0], new_shape[1], 2])
        Rx_2 = tf.reshape(Rx_2, [-1, new_shape[0], new_shape[1], 2])
        Ry_2 = tf.reshape(Ry_2, [-1, new_shape[0], new_shape[1], 2])
        Rx_3 = tf.reshape(Rx_3, [-1, new_shape[0], new_shape[1], 2])
        Ry_3 = tf.reshape(Ry_3, [-1, new_shape[0], new_shape[1], 2])
        Rx_4 = tf.reshape(Rx_4, [-1, new_shape[0], new_shape[1], 2])
        Ry_4 = tf.reshape(Ry_4, [-1, new_shape[0], new_shape[1], 2])



    if dof == 1:
        c_x1 = Rx[:,:,:,0:1]
        c_x2 = tf.roll(Rx[:,:,:,1:2], [1], [1])

        # on 2020-07-16, was not sure, why this is not shift to the row axis to get nodal value. Right now, it's still nodal information 

        c_y1 = Ry[:,:,:,0:1]
        c_y2 = tf.roll(Ry[:,:,:,1:2], [1], [2])
        if pflag: print('Rx 1 (before): ', Rx[0,:,:,0])
        if pflag: print('Rx 1 (after ): ', c_x1[0,:,:,0])
        if pflag: print('Rx 2 (before): ', Rx[0,:,:,1])
        if pflag: print('Rx 2 (after ): ', c_x2[0,:,:,0])

        if pflag: print('Ry 1 (before): ', Ry[0,:,:,0])
        if pflag: print('Ry 1 (after ): ', c_y1[0,:,:,0])
        if pflag: print('Ry 2 (before): ', Ry[0,:,:,1])
        if pflag: print('Ry 2 (after ): ', c_y2[0,:,:,0])

        Rx = c_x1 + c_x2
        Ry = c_y1 + c_y2
        # Add on 2020-07-16. Note on 2020-07-17, not working well
        # Rx = tf.roll(Rx[:,:,:,0:1], [1], [2])
        # Ry = tf.roll(Ry[:,:,:,0:1], [1], [1])
        #---------------------
        if pflag: print('Pay attention to potential errors here')
        if pflag: print('Rx : ', Rx[0,:,:,0])
        if pflag: print('Ry : ', Ry[0,:,:,0])

    elif dof == 2:
        c_x1 = Rx_1[:,:,:,0:1]
        c_x2 = tf.roll(Rx_1[:,:,:,1:2], [1], [1])

        c_y1 = Ry_1[:,:,:,0:1]
        c_y2 = tf.roll(Ry_1[:,:,:,1:2], [1], [2])

        Rx_1 = c_x1 + c_x2
        Ry_1 = c_y1 + c_y2
        if pflag: print('Rx_1 : ', Rx_1[0,:,:,0])
        if pflag: print('Ry_1 : ', Ry_1[0,:,:,0])

        c_x1 = Rx_2[:,:,:,0:1]
        c_x2 = tf.roll(Rx_2[:,:,:,1:2], [1], [1])

        c_y1 = Ry_2[:,:,:,0:1]
        c_y2 = tf.roll(Ry_2[:,:,:,1:2], [1], [2])

        Rx_2 = c_x1 + c_x2
        Ry_2 = c_y1 + c_y2
        if pflag: print('Rx_2 : ', Rx_2[0,:,:,0])
        if pflag: print('Ry_2 : ', Ry_2[0,:,:,0])
        if pflag: print('Pay attention to potential errors here')

    elif dof == 3:
        c_x1 = Rx_1[:,:,:,0:1]
        c_x2 = tf.roll(Rx_1[:,:,:,1:2], [1], [1])

        c_y1 = Ry_1[:,:,:,0:1]
        c_y2 = tf.roll(Ry_1[:,:,:,1:2], [1], [2])

        Rx_1 = c_x1 + c_x2
        Ry_1 = c_y1 + c_y2
        if pflag: print('Rx_1 : ', Rx_1[0,:,:,0])
        if pflag: print('Ry_1 : ', Ry_1[0,:,:,0])

        c_x1 = Rx_2[:,:,:,0:1]
        c_x2 = tf.roll(Rx_2[:,:,:,1:2], [1], [1])

        c_y1 = Ry_2[:,:,:,0:1]
        c_y2 = tf.roll(Ry_2[:,:,:,1:2], [1], [2])

        Rx_2 = c_x1 + c_x2
        Ry_2 = c_y1 + c_y2
        if pflag: print('Rx_2 : ', Rx_2[0,:,:,0])
        if pflag: print('Ry_2 : ', Ry_2[0,:,:,0])
        if pflag: print('Pay attention to potential errors here')

        c_x1 = Rx_3[:,:,:,0:1]
        c_x2 = tf.roll(Rx_3[:,:,:,1:2], [1], [1])

        c_y1 = Ry_3[:,:,:,0:1]
        c_y2 = tf.roll(Ry_3[:,:,:,1:2], [1], [2])

        Rx_3 = c_x1 + c_x2
        Ry_3 = c_y1 + c_y2
        if pflag: print('Rx_3 : ', Rx_3[0,:,:,0])
        if pflag: print('Ry_3 : ', Ry_3[0,:,:,0])
        if pflag: print('Pay attention to potential errors here')
    elif dof == 4:
        c_x1 = Rx_1[:,:,:,0:1]
        c_x2 = tf.roll(Rx_1[:,:,:,1:2], [1], [1])

        c_y1 = Ry_1[:,:,:,0:1]
        c_y2 = tf.roll(Ry_1[:,:,:,1:2], [1], [2])

        Rx_1 = c_x1 + c_x2
        Ry_1 = c_y1 + c_y2

        c_x1 = Rx_2[:,:,:,0:1]
        c_x2 = tf.roll(Rx_2[:,:,:,1:2], [1], [1])

        c_y1 = Ry_2[:,:,:,0:1]
        c_y2 = tf.roll(Ry_2[:,:,:,1:2], [1], [2])

        Rx_2 = c_x1 + c_x2
        Ry_2 = c_y1 + c_y2

        c_x1 = Rx_3[:,:,:,0:1]
        c_x2 = tf.roll(Rx_3[:,:,:,1:2], [1], [1])

        c_y1 = Ry_3[:,:,:,0:1]
        c_y2 = tf.roll(Ry_3[:,:,:,1:2], [1], [2])

        Rx_3 = c_x1 + c_x2
        Ry_3 = c_y1 + c_y2

        c_x1 = Rx_4[:,:,:,0:1]
        c_x2 = tf.roll(Rx_4[:,:,:,1:2], [1], [1])

        c_y1 = Ry_4[:,:,:,0:1]
        c_y2 = tf.roll(Ry_4[:,:,:,1:2], [1], [2])

        Rx_4 = c_x1 + c_x2
        Ry_4 = c_y1 + c_y2


    if dof == 1 :
        #----------------- first version of implementation ----------- 
        # works fine for large dataset problem because of mask contains both x- and y- direction component. However, 
        # the 2nd implementation is less error-prone, in case channel has value, channel 1 remains empty, then
        # mask can mask out everything.
        # R = Rx + Ry
        # R = tf.multiply(R, neumann_mask[:,:,:,0:1])
        #-------------------------------------------------------------

        R = tf.multiply(Rx, neumann_mask[:,:,:,0:1]) + tf.multiply(Ry, neumann_mask[:,:,:,1:2])
        if pflag: print('R: ', np.shape(R), R[0,:,:,0])
    elif dof == 2:
        print("Not fully tested! Exit... Please test this part to enable the code")
        exit(0)
        R_1 = tf.multiply(Rx_1, neumann_mask[:,:,:,0:1]) + tf.multiply(Ry_1, neumann_mask[:,:,:,1:2]) 
        if pflag: print('R_1: ', np.shape(R_1), R_1[0,:,:,0])
        R_2 = tf.multiply(Rx_2, neumann_mask[:,:,:,2:3]) + tf.multiply(Ry_2, neumann_mask[:,:,:,3:4])  
        if pflag: print('R_2: ', np.shape(R_2), R_2[0,:,:,0])
        R = tf.concat([R_1, R_2], axis=3)
        if pflag: print('R: ', np.shape(R), R[0,:,:,0], R[0,:,:,1])
    elif dof == 3:
        print("Not fully tested! Exit... Please test this part to enable the code")
        exit(0)
        R_1 = tf.multiply(Rx_1, neumann_mask[:,:,:,0:1]) + tf.multiply(Ry_1, neumann_mask[:,:,:,1:2]) 
        if pflag: print('R_1: ', np.shape(R_1), R_1[0,:,:,0])
        R_2 = tf.multiply(Rx_2, neumann_mask[:,:,:,2:3]) + tf.multiply(Ry_2, neumann_mask[:,:,:,3:4]) 
        if pflag: print('R_2: ', np.shape(R_2), R_2[0,:,:,0])
        R_3 = tf.multiply(Rx_3, neumann_mask[:,:,:,4:5]) + tf.multiply(Ry_3, neumann_mask[:,:,:,5:6]) 
        if pflag: print('R_3: ', np.shape(R_3), R_3[0,:,:,0])

        R = tf.concat([R_1, R_2, R_3], axis=3)
        if pflag: print('R: ', np.shape(R), R[0,:,:,0], R[0,:,:,1], R[0,:,:,2])
    elif dof == 4:
        print("Not fully tested! Exit... Please test this part to enable the code")
        exit(0)
        R_1 = tf.multiply(Rx_1, neumann_mask[:,:,:,0:1]) + tf.multiply(Ry_1, neumann_mask[:,:,:,1:2]) 
        R_2 = tf.multiply(Rx_2, neumann_mask[:,:,:,2:3]) + tf.multiply(Ry_2, neumann_mask[:,:,:,3:4]) 
        R_3 = tf.multiply(Rx_3, neumann_mask[:,:,:,4:5]) + tf.multiply(Ry_3, neumann_mask[:,:,:,5:6]) 
        R_4 = tf.multiply(Rx_4, neumann_mask[:,:,:,6:7]) + tf.multiply(Ry_4, neumann_mask[:,:,:,7:8]) 
        R = tf.concat([R_1, R_2, R_3, R_4], axis=3)
        # R = tf.multiply(R, neumann_mask) # remove other edge left R due to conv operation: do test edge (1,3) and (2,4)
        if pflag: print('R: ', np.shape(R), R[0,:,:,0], R[0,:,:,1], R[0,:,:,2], R[0,:,:,3])

    # exit(0)

    return R


def Get1DGaussPointInfo(dh=1.0, GPs=2, dof=1):
    """ 
    args:
        dh (float): element size
        GPs (int): total Gauss point number 
        dof (int): dof per node

    return:
        - shape function (numpy array) with size of [GPs, Nodes=2]
        - gradient shape function (numpy array) [None] Not implemented.
        - weight per gauss point (float scalar)

    todo:
        make this function to work with (1S, 1V), 2S, 1V1S, 3S, 2V, etc.
    """

    # print ("For 1D gauss point, the flip of the index for the y-axis was not tested. Be extremely careful if B is used. But for Neumann BCs, it might be just fine as the order of the node does not matter that much.")
    if GPs == 2 :
        #
        if dof == 1:
            # N (gp=2,nodes*dofs=2)
            N = tf.cast(
                    np.array(
                        [
                            # the shape function value should not be changed
                            # . x x .  . 1 2 .
                            # the closer one get value of 0.788, the further one get value of 0.211
                            [0.7886751345948129, 0.2113248654051871,], #GP1, [N1,N2]
                            [0.2113248654051871, 0.7886751345948129,], #GP2, [N1,N2]
                        ]
                        ),
            tf.float32)

            # B is disabled as the y-axis was not tested. And it is not clear how to make one B to work for 
            # both x-axis and y-axis, as it's a 1D GPs rule.
            # B = tf.cast(
                # np.array([
                    # [
                        # # GP1
                        # [0.7886751345948129], # N1, coor 3
                        # [0.2113248654051871], # N2, coor 4
                    # ],
                    # [
                        # # GP2
                        # [0.2113248654051871], # N1, coor 3
                        # [0.7886751345948129], # N2, coor 4
                    # ],
                # ]),
                # tf.float32) / dh 
        elif dof == 2:
            # N (gp=2,nodes*dofs=4)
            raise ValueError("Please disable this Error. Face GPs in is not tested for dof=2, be careful here!")
            N = tf.cast(
                    np.array(
                        [
                            [0.7886751345948129, 0.7886751345948129, 0.2113248654051871, 0.2113248654051871,], #GP1, [N1,N2]
                            [0.2113248654051871, 0.2113248654051871, 0.7886751345948129, 0.7886751345948129,], #GP2, [N1,N2]
                        ]
                        ),
            tf.float32)

            # B = tf.cast(
                # np.array([
                    # [
                        # # GP1
                        # [0.7886751345948129, 0], # N1, coor 3
                        # [0, 0.7886751345948129], # N1, coor 3
                        # [0.2113248654051871, 0], # N2, coor 4
                        # [0, 0.2113248654051871], # N2, coor 4
                    # ],
                    # [
                        # # GP2
                        # [0.2113248654051871, 0], # N1, coor 3
                        # [0, 0.2113248654051871], # N1, coor 3
                        # [0.7886751345948129, 0], # N2, coor 4
                        # [0, 0.7886751345948129], # N2, coor 4
                    # ],
                # ]),
                # tf.float32) / dh 
        else :
            raise ValueError("dof = ", dof, " is not implemented!")

        B = None

        jxw = dh*0.5
        # print('N: ', np.shape(N), '(q,n)')
        # print('q=0, N: ', N[0,:])
        # print('q=1, N: ', N[1,:])
        # print('q=2, N: ', N[2,:])
        # print('q=3, N: ', N[3,:])
        # print('B: ', np.shape(B), '(q,n,x)' )
        # print('q=0 B: ', B[0,:,:])
        # print('q=1 B: ', B[1,:,:])
        # print('q=2 B: ', B[2,:,:])
        # print('q=3 B: ', B[3,:,:])
        # print('jxw: ', jxw)

        return N, B, jxw
    else:
        raise ValueError("Only GPs == 2 is implemented, please choose a different GPs!", GPs)


def Get2DGaussPointInfo(dh=1.0, GPs=4, dof=1):
    """ 
    args:
        dh (float): element size
        GPs (int): total Gauss point number 
        dof (int): dof per node

    return:
        - shape function (numpy array) with size of [GPs, Nodes=4*dof]
        - gradient shape function (numpy array) [GPs, Nodes=4*dof, dim=2*dof] last dim: dof=1: [dc/dx, dc/dy] dof=2: [dx/dx, dx/dy, dy/dx, dy/dy]
        - weight per gauss point (float scalar)

    todo:
        make this function to work with (1S, 1V), 2S, 1V1S, 3S, 2V, etc.
    """
    if GPs == 4 :
        #
        if dof == 1:
            # N (gp=4,nodes*dofs=4)

            # check reshape [1,2,3,4] to (2,2)
            # check reshape [[1,2],[3,4]] to (4)
            N = tf.cast(
                    np.array(
                        [
                            # the shape function value should not be changed
                            # .      . .      .
                            #   x  x     1  2
                            #   x  x     3  4
                            # .      . .      .
                            # the closer one get value of 0.622, the further one get value of 0.044
                            #GP1, [N3,N4,N1,N2]
                            [0.1666666666666667, 0.04465819873852046, 0.6220084679281462, 0.1666666666666667,], 
                            # [0.6220084679281462, 0.1666666666666667, 0.1666666666666667, 0.04465819873852046,], #GP1, [N1,N2,N3,N4]

                            #GP2, [N3,N4,N1,N2]
                            [0.04465819873852046, 0.1666666666666667, 0.1666666666666667, 0.6220084679281462,], 
                            # [0.1666666666666667, 0.6220084679281462, 0.04465819873852046, 0.1666666666666667,], #GP2, [N1,N2,N3,N4]

                            #GP3, [N3,N4,N1,N2]
                            [0.6220084679281462, 0.1666666666666667, 0.1666666666666667, 0.04465819873852046,], 
                            # [0.1666666666666667, 0.04465819873852046, 0.6220084679281462, 0.1666666666666667,], #GP3, [N1,N2,N3,N4]

                            #GP4, [N3,N4,N1,N2]
                            [0.1666666666666667, 0.6220084679281462, 0.04465819873852046, 0.1666666666666667,],
                            # [0.04465819873852046, 0.1666666666666667, 0.1666666666666667, 0.6220084679281462,], #GP4, [N1,N2,N3,N4]

                        ]
                        ),
            tf.float32)

            B = tf.cast(
                np.array([
                    [
                        # GP1
                        [-0.2113248654051871, 0.7886751345948129],  # N3, coor 1
                        [0.2113248654051871, 0.2113248654051871],   # N4, coor 2
                        [-0.7886751345948129, -0.7886751345948129], # N1, coor 3
                        [0.7886751345948129, -0.2113248654051871],  # N2, coor 4
                    ],
                    [
                        # GP2
                        [-0.2113248654051871, 0.2113248654051871],  # N3, coor 1
                        [0.2113248654051871, 0.7886751345948129],   # N4, coor 2
                        [-0.7886751345948129, -0.2113248654051871], # N1, coor 3
                        [0.7886751345948129, -0.7886751345948129],  # N2, coor 4
                    ],
                    [
                        # GP3
                        [-0.7886751345948129, 0.7886751345948129],  # N3, coor 1
                        [0.7886751345948129, 0.2113248654051871],   # N4, coor 2
                        [-0.2113248654051871, -0.7886751345948129], # N1, coor 3
                        [0.2113248654051871, -0.2113248654051871],  # N2, coor 4
                    ],
                    [
                        # GP4
                        [-0.7886751345948129, 0.2113248654051871],  # N3, coor 1
                        [0.7886751345948129, 0.7886751345948129],   # N4, coor 2
                        [-0.2113248654051871, -0.2113248654051871], # N1, coor 3
                        [0.2113248654051871, -0.7886751345948129],  # N2, coor 4
                    ]
                ]),
                tf.float32) / dh 
        elif dof == 2:
            # N (gp=4,nodes*dofs=8)
            N = tf.cast(
                    np.array(
                        [
                            #GP1, [N3,N4,N1,N2]
                            [0.1666666666666667, 0.1666666666666667,  0.04465819873852046,0.04465819873852046, 0.6220084679281462,  0.6220084679281462,  0.1666666666666667,  0.1666666666666667, ], 
                            #[0.6220084679281462, 0.6220084679281462,  0.1666666666666667, 0.1666666666666667,  0.1666666666666667,  0.1666666666666667,  0.04465819873852046, 0.04465819873852046,], #GP1, [N1,N2,N3,N4]

                            #GP2, [N3,N4,N1,N2]
                            [0.04465819873852046,0.04465819873852046, 0.1666666666666667, 0.1666666666666667,  0.1666666666666667,  0.1666666666666667,  0.6220084679281462,  0.6220084679281462, ], 
                            #[0.1666666666666667, 0.1666666666666667,  0.6220084679281462, 0.6220084679281462,  0.04465819873852046, 0.04465819873852046, 0.1666666666666667,  0.1666666666666667, ], #GP2, [N1,N2,N3,N4]

                            #GP3, [N3,N4,N1,N2]
                            [0.6220084679281462, 0.6220084679281462,  0.1666666666666667, 0.1666666666666667,  0.1666666666666667,  0.1666666666666667,  0.04465819873852046, 0.04465819873852046,], 
                            #[0.1666666666666667, 0.1666666666666667,  0.04465819873852046,0.04465819873852046, 0.6220084679281462,  0.6220084679281462,  0.1666666666666667,  0.1666666666666667, ], #GP3, [N1,N2,N3,N4]

                            #GP4, [N3,N4,N1,N2]
                            [0.1666666666666667, 0.1666666666666667,  0.6220084679281462, 0.6220084679281462,  0.04465819873852046, 0.04465819873852046, 0.1666666666666667,  0.1666666666666667, ], 
                            #[0.04465819873852046,0.04465819873852046, 0.1666666666666667, 0.1666666666666667,  0.1666666666666667,  0.1666666666666667,  0.6220084679281462,  0.6220084679281462, ], #GP4, [N1,N2,N3,N4]
                        ]
                        ),
            tf.float32)

            B = tf.cast(
                np.array([
                    [
                        # GP1
                        [-0.2113248654051871, 0.7886751345948129, 0, 0],
                        [0, 0, -0.2113248654051871, 0.7886751345948129],  # N3, coor 1
                        [0.2113248654051871, 0.2113248654051871, 0, 0],
                        [0, 0, 0.2113248654051871, 0.2113248654051871],   # N4, coor 2
                        [-0.7886751345948129, -0.7886751345948129, 0, 0],
                        [0, 0, -0.7886751345948129, -0.7886751345948129], # N1, coor 3
                        [0.7886751345948129, -0.2113248654051871, 0, 0],
                        [0, 0, 0.7886751345948129, -0.2113248654051871],  # N2, coor 4
                    ],
                    [
                        # GP2
                        [-0.2113248654051871, 0.2113248654051871, 0, 0],
                        [0, 0, -0.2113248654051871, 0.2113248654051871],  # N3, coor 1
                        [0.2113248654051871, 0.7886751345948129, 0, 0],
                        [0, 0, 0.2113248654051871, 0.7886751345948129],   # N4, coor 2
                        [-0.7886751345948129, -0.2113248654051871, 0, 0],
                        [0, 0, -0.7886751345948129, -0.2113248654051871], # N1, coor 3
                        [0.7886751345948129, -0.7886751345948129, 0, 0],
                        [0, 0, 0.7886751345948129, -0.7886751345948129],  # N2, coor 4
                    ],
                    [
                        # GP3
                        [-0.7886751345948129, 0.7886751345948129, 0, 0],
                        [0, 0, -0.7886751345948129, 0.7886751345948129],  # N3, coor 1
                        [0.7886751345948129, 0.2113248654051871, 0, 0],
                        [0, 0, 0.7886751345948129, 0.2113248654051871],   # N4, coor 2
                        [-0.2113248654051871, -0.7886751345948129, 0, 0],
                        [0, 0, -0.2113248654051871, -0.7886751345948129], # N1, coor 3
                        [0.2113248654051871, -0.2113248654051871, 0, 0],
                        [0, 0, 0.2113248654051871, -0.2113248654051871],  # N2, coor 4
                    ],
                    [
                        # GP4
                        [-0.7886751345948129, 0.2113248654051871, 0, 0],
                        [0, 0, -0.7886751345948129, 0.2113248654051871],  # N3, coor 1
                        [0.7886751345948129, 0.7886751345948129, 0, 0],
                        [0, 0, 0.7886751345948129, 0.7886751345948129],   # N4, coor 2
                        [-0.2113248654051871, -0.2113248654051871, 0, 0],
                        [0, 0, -0.2113248654051871, -0.2113248654051871], # N1, coor 3
                        [0.2113248654051871, -0.7886751345948129, 0, 0],
                        [0, 0, 0.2113248654051871, -0.7886751345948129],  # N2, coor 4
                    ]
                ]),
                tf.float32) / dh 
        else :
            raise ValueError("dof = ", dof, " is not implemented!")

        jxw = dh*dh*0.25
        # print('N: ', np.shape(N), '(q,n)')
        # print('q=0, N: ', N[0,:])
        # print('q=1, N: ', N[1,:])
        # print('q=2, N: ', N[2,:])
        # print('q=3, N: ', N[3,:])
        # print('B: ', np.shape(B), '(q,n,x)' )
        # print('q=0 B: ', B[0,:,:])
        # print('q=1 B: ', B[1,:,:])
        # print('q=2 B: ', B[2,:,:])
        # print('q=3 B: ', B[3,:,:])
        # print('jxw: ', jxw)

        return N, B, jxw
    else:
        raise ValueError("Only GPs == 4 is implemented, please choose a different GPs!", GPs)


def GetNodalInfoFromElementInfo(data, residual_mask, dof, padding='SAME'):
    """ 
    reorganize data from a matrix form with 4 nodal values of elements to nodal values

    Args:
        data (numpy array/tensor): [None, elem_height, elem_width, 4*dof] (4 nodal values for 1 dof)
        residual_mask (numpy_array):  [None, elem_height, elem_width, 1] 
        dof (int): dof per node

    return:
        numpy array: output with size of [None, node_height, node_width, dof]

    todo:
        make this function to work with (1S, 1V), 2S, 1V1S, 3S, 2V, etc.
    """
    # tf.roll( input, shift, axis, name=None)
    # 't' is [0, 1, 2, 3, 4]
    # roll(t, shift=2, axis=0) ==> [3, 4, 0, 1, 2]
    
    # shifting along multiple dimensions
    # 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    # roll(t, shift=[1, -2], axis=[0, 1]) ==> [[7, 8, 9, 5, 6], [2, 3, 4, 0, 1]]
    
    # shifting along the same axis multiple times
    # 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    # roll(t, shift=[2, -3], axis=[1, 1]) ==> [[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]]

    pflag = False
    if dof == 1:
        # data = tf.convert_to_tensor(data, dtype=tf.float32)
        data = tf.multiply(data, residual_mask)

        if pflag: print('data', np.shape(data))

        c_n1 = data[:,:,:,0:1]
        c_n2 = tf.roll(data[:,:,:,1:2], [1], [2])
        c_n3 = tf.roll(data[:,:,:,2:3], [1], [1])
        c_n4 = tf.roll(data[:,:,:,3:4], [1,1], [1,2])
        # print('data 1 (before): ', data[0,:,:,0])
        # print('data 1 (after ): ', c_n1[0,:,:,0])
        # print('data 2 (before): ', data[0,:,:,1])
        # print('data 2 (after ): ', c_n2[0,:,:,0])
        # print('data 3 (before): ', data[0,:,:,2])
        # print('data 3 (after ): ', c_n3[0,:,:,0])
        # print('data 4 (before): ', data[0,:,:,3])
        # print('data 4 (after ): ', c_n4[0,:,:,0])

        nodal_c = tf.concat([c_n1, c_n2, c_n3, c_n4], 3)

        nodal_val = tf.reduce_sum(nodal_c, axis=3, keepdims=True )
    elif dof == 2:
        # data = tf.convert_to_tensor(data, dtype=tf.float32)
        data = tf.multiply(data, residual_mask)
        if pflag: print('data', np.shape(data))

        x_n1 = data[:,:,:,0:1]
        y_n1 = data[:,:,:,1:2]
        x_n2 = tf.roll(data[:,:,:,2:3], [1], [2])
        y_n2 = tf.roll(data[:,:,:,3:4], [1], [2])
        x_n3 = tf.roll(data[:,:,:,4:5], [1], [1])
        y_n3 = tf.roll(data[:,:,:,5:6], [1], [1])
        x_n4 = tf.roll(data[:,:,:,6:7], [1,1], [1,2])
        y_n4 = tf.roll(data[:,:,:,7:8], [1,1], [1,2])

        if pflag: print('data 1 (before): ', data[0,:,:,0])
        if pflag: print('data 1 (after ): ', x_n1[0,:,:,0])
        if pflag: print('data 2 (before): ', data[0,:,:,1])
        if pflag: print('data 2 (after ): ', y_n1[0,:,:,0])
        if pflag: print('data 3 (before): ', data[0,:,:,2])
        if pflag: print('data 3 (after ): ', x_n2[0,:,:,0])
        if pflag: print('data 4 (before): ', data[0,:,:,3])
        if pflag: print('data 4 (after ): ', y_n2[0,:,:,0])


        nodal_x = tf.concat([x_n1, x_n2, x_n3, x_n4], 3)
        nodal_y = tf.concat([y_n1, y_n2, y_n3, y_n4], 3)
        if pflag: print('nodal_x ', np.shape(nodal_x))
        if pflag: print('nodal_y ', np.shape(nodal_y))
        # nodal_x = tf.expand_dims(nodal_x,3)
        # nodal_y = tf.expand_dims(nodal_y,3)

        nodal_x = tf.reduce_sum(nodal_x, axis=3, keepdims=True )
        nodal_y = tf.reduce_sum(nodal_y, axis=3, keepdims=True )
        nodal_val = tf.concat([nodal_x, nodal_y], 3)
    else:
        # data = tf.convert_to_tensor(data, dtype=tf.float32)
        data = tf.multiply(data, residual_mask)
        if pflag: print('data', np.shape(data))
        # use the above dof=1/2 as example to understand the following
        R_dof = []
        for i0 in range(0, dof):
            x_n1 = data[:,:,:,i0:i0+1]
            x_n2 = tf.roll(data[:,:,:,i0+dof:i0+1+dof], [1], [2])
            x_n3 = tf.roll(data[:,:,:,i0+dof*2:i0+1+2*dof], [1], [1])
            x_n4 = tf.roll(data[:,:,:,i0+dof*3:i0+1+3*dof], [1,1], [1,2])
            nodal_x = tf.concat([x_n1, x_n2, x_n3, x_n4], 3)
            nodal_x = tf.reduce_sum(nodal_x, axis=3, keepdims=True )
            R_dof.append(nodal_x)
        nodal_val = tf.concat(R_dof, 3)
        print ('Nodal value for dof = ', dof, ' is not fully tested yet!')
    if pflag: print('nodal_val ', np.shape(nodal_val))

    return nodal_val



class LayerFillRandomToBCs(tf.keras.layers.Layer):
    """ 
    A customized Keras layer to add random noise to BCs with :math:`\epsilon~\sim` N(0, stddev=0.005).

    Args:
        stddev (float): default = 0.005
    """

    def __init__(self, stddev=0.005, name='fill-random-num'):
        super(LayerFillRandomToBCs, self).__init__(name=name)
        self.stddev = stddev

    def call(self, input):
        output = input + tf.where(input > 0.0, tf.random.normal(tf.shape(input), 0, self.stddev, tf.float32), tf.fill(tf.shape(input), 0.0))
        return output

class LayerFillZeros(tf.keras.layers.Layer):
    """ 
    A customized Keras layer to generate zeros if value == -2.0
    """

    def __init__(self, name='fill-zeros'):
        super(LayerFillZeros, self).__init__(name=name)

    def call(self, input):
        output = input + tf.where( input > -1.5, tf.fill(tf.shape(input), 0.0), tf.fill(tf.shape(input), 2.0))
        return output


class LayerFillRandomNumber(tf.keras.layers.Layer):
    """ 
    A customized Keras layer to generate uniform random data (0, 1) if value == -2.0
    """

    def __init__(self, name='fill-random-num'):
        super(LayerFillRandomNumber, self).__init__(name=name)

    def call(self, input):
        output = input + tf.where(
            input > -1.5, tf.fill(tf.shape(input), 0.0),
            tf.random.uniform(tf.shape(input), minval=0.0, maxval=1.0)) + tf.where(
                input > -1.5, tf.fill(tf.shape(input), 0.0),
                tf.fill(tf.shape(input), 2.0))
        return output


class LayerBulkResidual(tf.keras.layers.Layer):
    """
    General bulk residual
    """
    # data: [batch, in_height, in_width, in_channels]
    # filter: [filter_height, filter_width, in_channels, out_channels]
    # dh is needed.

    def __init__(self, name='R_bulk_general'):
        super(LayerBulkResidual, self).__init__(name=name)

    def initialize_arrays(self):
        """
        Initialize the kernel array to transform nodal arrangement to element arrangement. Get the Gauss Point information.
        """
        self.n1 = np.array([[1, 0], [0, 0]])
        self.n1 = np.expand_dims(self.n1, axis=2)
        self.n1 = np.expand_dims(self.n1, axis=3)

        self.n2 = np.array([[0, 1], [0, 0]])
        self.n2 = np.expand_dims(self.n2, axis=2)
        self.n2 = np.expand_dims(self.n2, axis=3)

        self.n3 = np.array([[0, 0], [1, 0]])
        self.n3 = np.expand_dims(self.n3, axis=2)
        self.n3 = np.expand_dims(self.n3, axis=3)

        self.n4 = np.array([[0, 0], [0, 1]])
        self.n4 = np.expand_dims(self.n4, axis=2)
        self.n4 = np.expand_dims(self.n4, axis=3)

        self.N, self.B, self.jxw = Get2DGaussPointInfo(dh=self.dh, dof=self.dof)

    def GetElementInfo(self, input):
        """ 
        Reorganize data from nodal value to a matrix form with 4*dof nodal values 
        args:
            inputs (tensor): [batch, node_height, node_width, dof]
        return:
            tensor: data with size of [batch, elem_height, elem_width, dof*4] 


        note:
            - filter n1, n2, n3, n4: [filter_height, filter_width, in_channels, out_channels]
        """
        # It is better to stick with the 2x2 or 2x2x2 format, because the matrix form might be
        # much easier for calling the linear algebra operations in tensorflow.

        if self.dof == 1:
            c_n1 = tf.nn.conv2d(input, self.n1, [1,1,1,1], 'SAME' )
            c_n2 = tf.nn.conv2d(input, self.n2, [1,1,1,1], 'SAME' )
            c_n3 = tf.nn.conv2d(input, self.n3, [1,1,1,1], 'SAME' )
            c_n4 = tf.nn.conv2d(input, self.n4, [1,1,1,1], 'SAME' )
            # elem_c
            data = tf.concat([c_n1, c_n2, c_n3, c_n4], 3)

        elif self.dof == 2:
            c_n1x = tf.nn.conv2d(input[:,:,:,0:1], self.n1, [1,1,1,1], 'SAME' )
            c_n2x = tf.nn.conv2d(input[:,:,:,0:1], self.n2, [1,1,1,1], 'SAME' )
            c_n3x = tf.nn.conv2d(input[:,:,:,0:1], self.n3, [1,1,1,1], 'SAME' )
            c_n4x = tf.nn.conv2d(input[:,:,:,0:1], self.n4, [1,1,1,1], 'SAME' )

            c_n1y = tf.nn.conv2d(input[:,:,:,1:2], self.n1, [1,1,1,1], 'SAME' )
            c_n2y = tf.nn.conv2d(input[:,:,:,1:2], self.n2, [1,1,1,1], 'SAME' )
            c_n3y = tf.nn.conv2d(input[:,:,:,1:2], self.n3, [1,1,1,1], 'SAME' )
            c_n4y = tf.nn.conv2d(input[:,:,:,1:2], self.n4, [1,1,1,1], 'SAME' )

            data = tf.concat([c_n1x, c_n1y, c_n2x, c_n2y, c_n3x, c_n3y, c_n4x, c_n4y], 3)
        else:
            raise ValueError('dof = ', self.dof, ' is not implemented')
        return data

    def ComputeValuAtGPs(self, data):
        """
        Reshape data[:, :, :, 4*dof] to [:, 4*dof] and compute the u(unknown) at each GPs.

        args:
            data (tensor): size of [-1, 4*dof]
        return:
            tensor: valu at each GPs with size of [-1, 1*dof]
        """
        data = tf.reshape(data,[-1, 4*self.dof])
        # print(np.shape(data))
        # print(np.shape(self.N[0,:]))

        valu1 = tf.linalg.matvec(data, self.N[0,:])
        valu2 = tf.linalg.matvec(data, self.N[1,:])
        valu3 = tf.linalg.matvec(data, self.N[2,:])
        valu4 = tf.linalg.matvec(data, self.N[3,:])

        valu1 = tf.expand_dims(valu1,1)
        valu2 = tf.expand_dims(valu2,1)
        valu3 = tf.expand_dims(valu3,1)
        valu4 = tf.expand_dims(valu4,1)
        # print(np.shape(valu1))
        return valu1, valu2, valu3, valu4

    def ComputeGraduAtGPs(self, data):
        """
        Reshape data[:, :, :, 4*dof] to [:, 4*dof] and compute the Grad of u(unknown) at each GPs.

        args:
            data (tensor): size of [-1, 4*dof]
        return:
            tensor: gradu at each GPs with size of [-1, 2*dof]
        """
        data = tf.reshape(data,[-1, 4*self.dof])

        # this is du/dX at each GP
        gradu1 = tf.matmul(data, self.B[0,:,:]) 
        gradu2 = tf.matmul(data, self.B[1,:,:]) 
        gradu3 = tf.matmul(data, self.B[2,:,:]) 
        gradu4 = tf.matmul(data, self.B[3,:,:]) 
        return gradu1, gradu2, gradu3, gradu4

    def Get2ndOrderIdentityTensor(self, gradu1, domain_shape):
        """
        Get the second order identity tensor in the format of I_4[-1, 4] and I_2x2[-1, :, :, 4GPs, 2, 2]
        """

        # create 2nd order tensor
        I = np.array([1.0000001,0.0,0.0,0.99999999])
        I = tf.constant(I, tf.float32)
        I = tf.expand_dims(I,0)
        ones = tf.ones_like(gradu1)
        I4 = tf.multiply(ones, I)

        I2x2_1 = tf.reshape(I4, [-1, domain_shape[0], domain_shape[1], 1, 2, 2])
        I2x2 = tf.concat([I2x2_1, I2x2_1, I2x2_1, I2x2_1], 3) # 4 GPs, are the same.

        return I4, I2x2

    def GetFe(self, gradu1, gradu2, gradu3, gradu4, I4, domain_shape, value1, value2, value3, value4):
        """
        Compute Fe for large deformation
        """
        # this is  Fe at each GP: gradu, I4 = [None, 4=2*dof], value1 = [None, 1]
        gradu1 = (gradu1 + I4) / tf.math.pow( (value1+1.0), 1.0/3.0)
        gradu2 = (gradu2 + I4) / tf.math.pow( (value2+1.0), 1.0/3.0)
        gradu3 = (gradu3 + I4) / tf.math.pow( (value3+1.0), 1.0/3.0)
        gradu4 = (gradu4 + I4) / tf.math.pow( (value4+1.0), 1.0/3.0)

        # this is F2x2 at each GP
        gradu1 = tf.reshape(gradu1, [-1, domain_shape[0], domain_shape[1], 1, 2, 2])
        gradu2 = tf.reshape(gradu2, [-1, domain_shape[0], domain_shape[1], 1, 2, 2])
        gradu3 = tf.reshape(gradu3, [-1, domain_shape[0], domain_shape[1], 1, 2, 2])
        gradu4 = tf.reshape(gradu4, [-1, domain_shape[0], domain_shape[1], 1, 2, 2])
        # tensor/matrix form of F
        F2x2 = tf.concat([gradu1, gradu2, gradu3, gradu4], 3)

        return F2x2

    def GetF(self, gradu1, gradu2, gradu3, gradu4, I4, domain_shape):
        """
        Compute F for large deformation
        """
        # this is  F at each GP
        gradu1 = gradu1 + I4
        gradu2 = gradu2 + I4
        gradu3 = gradu3 + I4
        gradu4 = gradu4 + I4

        # this is F2x2 at each GP
        gradu1 = tf.reshape(gradu1, [-1, domain_shape[0], domain_shape[1], 1, 2, 2])
        gradu2 = tf.reshape(gradu2, [-1, domain_shape[0], domain_shape[1], 1, 2, 2])
        gradu3 = tf.reshape(gradu3, [-1, domain_shape[0], domain_shape[1], 1, 2, 2])
        gradu4 = tf.reshape(gradu4, [-1, domain_shape[0], domain_shape[1], 1, 2, 2])
        # tensor/matrix form of F
        F2x2 = tf.concat([gradu1, gradu2, gradu3, gradu4], 3)

        return F2x2

    def GetEpsilon(self, gradu1, gradu2, gradu3, gradu4, domain_shape):
        """
        Compute epsilon for small deformation
        """
        # this is epsilon at each GP
        gradu1 = tf.reshape(gradu1, [-1, domain_shape[0], domain_shape[1], 1, 2, 2])
        gradu2 = tf.reshape(gradu2, [-1, domain_shape[0], domain_shape[1], 1, 2, 2])
        gradu3 = tf.reshape(gradu3, [-1, domain_shape[0], domain_shape[1], 1, 2, 2])
        gradu4 = tf.reshape(gradu4, [-1, domain_shape[0], domain_shape[1], 1, 2, 2])

        # symmetric epsilon
        gradu1 = 0.5 * (gradu1 + tf.transpose(gradu1, perm=[0,1,2,3,5,4])) # + tf.random.uniform(tf.shape(gradu1), maxval=1e-9)
        gradu2 = 0.5 * (gradu2 + tf.transpose(gradu2, perm=[0,1,2,3,5,4])) # + tf.random.uniform(tf.shape(gradu2), maxval=1e-9)
        gradu3 = 0.5 * (gradu3 + tf.transpose(gradu3, perm=[0,1,2,3,5,4])) # + tf.random.uniform(tf.shape(gradu3), maxval=1e-9)
        gradu4 = 0.5 * (gradu4 + tf.transpose(gradu4, perm=[0,1,2,3,5,4])) # + tf.random.uniform(tf.shape(gradu4), maxval=1e-9)

        # # tensor/matrix form of epsilon
        epsilon = tf.concat([gradu1, gradu2, gradu3, gradu4], 3)
        return epsilon

    def ComputeIntTranBxP(self, P1, P2, P3, P4, domain_shape):
        """
        compute int ( B^T * P) dV

        args:
            P# (tensor): with size of [:, 4]
        """

        # B (q, n, x)
        # TransB (q, x, n)
        TransB = tf.transpose(self.B, perm=[0,2,1])

        R1 = tf.matmul(P1, TransB[0,:,:])
        R2 = tf.matmul(P2, TransB[1,:,:])
        R3 = tf.matmul(P3, TransB[2,:,:])
        R4 = tf.matmul(P4, TransB[3,:,:])

        # int ( B^T * P) dV
        R = self.jxw * (R1 + R2 + R3 + R4)

        R = tf.reshape(R, [-1, domain_shape[0], domain_shape[1], 4*self.dof])

        return R

    def ComputeIntTranNxU(self, valu1, valu2, valu3, valu4, domain_shape):
        """
        compute int ( N^T * valu) dV

        args:
            valu# (tensor): with size of [:, 1]
        """

        # N (q, n)
        R1 = tf.matmul(valu1, self.N[0:1,:])
        R2 = tf.matmul(valu2, self.N[1:2,:])
        R3 = tf.matmul(valu3, self.N[2:3,:])
        R4 = tf.matmul(valu4, self.N[3:4,:])

        # int ( N^T * valu) dV
        R = self.jxw * (R1 + R2 + R3 + R4)

        R = tf.reshape(R, [-1, domain_shape[0], domain_shape[1], 4*self.dof])

        return R

    def E_nu_to_lambda_mu(self, E, nu):
        lambda0 = (E*nu)/(1.0+nu)/(1.0-2.0*nu)
        mu0 = E/2.0/(1.0+nu)
        return lambda0, mu0


if __name__ == '__main__' :
    print('testing the main')
    # test the results for matrix that is not invertible
    # F1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    # InvF = tf.linalg.inv(F1)
    # print('F', F1, 'InvF', InvF)

    # # lead to tensorflow.python.framework.errors_impl.InvalidArgumentError: Input is not invertible. [Op:MatrixInverse]
    # F2 = tf.constant([[1.0, 0], [0, 0.0]])
    # InvF = tf.linalg.inv(F2)
    # print('F', F2, 'InvF', InvF)

    # F3 = tf.constant([[1.0, 1.0], [1.0, 1.0]])
    # InvF = tf.linalg.inv(F3)
    # print('F', F3, 'InvF', InvF)


    # x = tf.constant([5.0, 4.8, 6.8, np.inf, np.nan])
    # print(tf.math.is_finite(x))

        # # detF_mask_finite = tf.where(detF != detF, tf.fill(tf.shape(detF), 0.0), tf.fill(tf.shape(detF), 1.0))
    # print(tf.where(x != x, tf.fill(tf.shape(x), 0.0), tf.fill(tf.shape(x), 1.0)))
    # print(tf.where(tf.math.is_finite(x), tf.fill(tf.shape(x), 1.0), tf.fill(tf.shape(x), 0.0)))

    # dh = 1.0/4
    # dof = 4
    # features = 0.0 * np.random.rand(1,5,5,8)
    # features[:,:,:,0:1] = -2
    # features[:,:,:,1:2] = -2
    # features[:,:,:,2:3] = -2
    # features[:,:,:,3:4] = -2
    
    # features[:,-1:,:,1:2] = 0.5
    # features[:,-1:,:,2:3] = 0.5
    
    # features[:,0:1,:,4:5] = 0.55
    # features[:,0:1,:,5:6] = 0.70
    # features[:,0:1,:,6:7] = 0.60
    # features[:,0:1,:,7:8] = 0.60

    # ComputeNeumannBoundaryResidualNodalData(features, dh, dof, padding='SAME')
    
    dof = 1
    dh = 1.0/7.0
    features = np.load("np-features-e4-s0-constant-0.npy")
    # print(features)
    ComputeNeumannBoundaryResidualNodalDataNew(features, dh, dof, padding='SAME')

