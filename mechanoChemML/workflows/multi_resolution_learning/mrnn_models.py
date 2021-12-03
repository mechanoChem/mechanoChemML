# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import sys, os, datetime
import mechanoChemML.workflows.multi_resolution_learning.mrnn_utility as mrnn_utility

############################### learning rate #####################################
def build_learningrate(config):
    LR = mrnn_utility.getlist_str(config['MODEL']['LearningRate'])
    # print('LR str: ', LR)
    if (len(LR) == 1):
        LearningRate = float(LR[0])
        print('--Decay in mono rate: rate = ', LearningRate)
    elif (len(LR) > 1):
        LR_type = LR[0]
        if (LR_type == 'mono'):
            LearningRate = float(LR[1])
        elif (LR_type == 'decay_exp'):
            initial_learning_rate = 0.001
            decay_steps = 1000
            decay_rate = 0.96
            initial_learning_rate = float(LR[1])
            if len(LR) > 2:
                decay_steps = int(LR[2])
            if len(LR) > 3:
                decay_rate = float(LR[3])
            print('--Decay in exponential rate: initial rate = ', initial_learning_rate, ', decay steps = ', decay_steps, ', decay_rate = ', decay_rate)

            if (mrnn_utility.get_package_version(tf.__version__)[0] == 1 and mrnn_utility.get_package_version(tf.__version__)[1] <= 13):
                global_step = tf.Variable(0, name='global_step', trainable=False)
                global_step = tf.train.get_global_step()
                LearningRate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
            else:
                print("!!!! Caution: use Learning rate with care, there were occasions that tf1.13 should better performance on training. !!!")
                global_step = tf.Variable(0, name='global_step', trainable=False)
                LearningRate = tf.compat.v1.train.exponential_decay(initial_learning_rate, global_step, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
        else:
            raise ValueError('unknown choice for learning rate (mono, decay_exp): ', LR)

    return LearningRate


###################################################################################

############################### optimizer #########################################

def build_optimizer(config):

    ModelOptimizer = config['MODEL']['Optimizer']
    LearningRate = build_learningrate(config)

    print('Avail Optimizer: ', ['adam', 'sgd', 'adadelta', 'gradientdescentoptimizer'])

    if (ModelOptimizer.lower() == "adam".lower()):
        optimizer = tf.keras.optimizers.Adam(LearningRate)
        return optimizer
    elif (ModelOptimizer.lower() == "sgd".lower()):
        optimizer = tf.keras.optimizers.SGD(LearningRate)
        return optimizer
    elif (ModelOptimizer.lower() == "adadelta".lower()):
        optimizer = tf.keras.optimizers.Adadelta(LearningRate)
        return optimizer
    elif (ModelOptimizer.lower() == "gradientdescentoptimizer".lower()):
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(LearningRate)
        return optimizer
    elif (ModelOptimizer.lower() == "user".lower()):
        raise ValueError('Model optimizer = ', ModelOptimizer, ' is chosen, but is not implemented!')
    else:
        raise ValueError('Model optimizer = ', ModelOptimizer, ' is chosen, but is not implemented!')


###################################################################################


############################# call back ####################################
# Display training progress by printing a single dot for each completed epoch
class callback_PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('epoch: ', epoch, 'loss: ', logs['loss'], 'val_loss: ', logs['val_loss'])

def check_point_callback(config):
    checkpoint_dir = config['RESTART']['CheckPointDir'] + config['MODEL']['ParameterID']
    checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}.ckpt"  
    period = int(config['RESTART']['CheckPointPeriod'])
    verbose = int(config['MODEL']['Verbose'])
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        verbose=verbose,
        save_weights_only=True,
    # Save weights, every 5-epochs.
        period=period)
    return cp_callback


def tensor_board_callback(config):
    tensorboard_dir = config['OUTPUT']['TensorBoardDir'] + config['MODEL']['ParameterID']
    log_dir = tensorboard_dir + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if (mrnn_utility.get_package_version(tf.__version__)[0] == 1):
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    elif (mrnn_utility.get_package_version(tf.__version__)[0] == 2):
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=2)
    else:
        raise ValueError("unknown tf version for tensor board callback support")
    return tb_callback


def build_callbacks(config):
    callbacks = []
    callback_names = mrnn_utility.getlist_str(config['MODEL']['CallBacks'])

    if 'checkpoint' in callback_names:
        callbacks.append(check_point_callback(config))

    if 'tensorboard' in callback_names:
        callbacks.append(tensor_board_callback(config))

    if 'printdot' in callback_names:
        callbacks.append(callback_PrintDot())

    return callbacks


###############################################################################

################################## loss ############################################
def my_mse_loss():
    # Create a mse loss function
    def loss(y_true, y_pred):
        # raise ValueError(tf.shape(y_true), tf.shape(y_pred))
        return tf.reduce_mean(tf.square(y_pred - y_true))
    # Return a function
    return loss


def my_mse_loss_with_grad(BetaP=1000.0):
    def loss(y_true, y_pred):
        # compute P based on S
        S_NN = y_pred[:, 1:5]
        S_NN = tf.reshape(S_NN, [-1, 2, 2])

        P_DNS = y_true[:, 1:5]
        P_DNS = tf.reshape(P_DNS, [-1, 2, 2])

        F_DNS = y_true[:, 5:9]
        F_DNS = tf.reshape(F_DNS, [-1, 2, 2])

        P_NN = tf.linalg.matmul(F_DNS, S_NN)

        P_NN = tf.reshape(P_NN, [-1, 4])
        P_DNS = tf.reshape(P_DNS, [-1, 4])

        return tf.reduce_mean(tf.square(y_pred[:, 0] - y_true[:, 0])) + BetaP * tf.reduce_mean(tf.square(P_NN - P_DNS))
    # Return a function
    return loss


def build_loss(config, loss_model=None):

    ModelLoss = config['MODEL']['Loss']

    if (ModelLoss.lower() == "mse".lower()):
        if (tf.__version__[0:1] == '1'):
            loss = tf.keras.losses.MSE()
        else:
            loss = tf.keras.losses.MeanSquaredError()
        return loss
    elif (ModelLoss.lower() == "my_mse_loss".lower()):
        loss = my_mse_loss()
        return loss
    elif (ModelLoss.lower() == "my_mse_loss_with_grad".lower()):
        loss = my_mse_loss_with_grad()
        return loss
    else:
        raise ValueError('Model loss = ', ModelLoss, ' is chosen, but is not implemented!')
###################################################################################


##################################### model ##########################################
def shift_labels(config, dataset, dataset_index, dataset_frame, data_file):
    print("---!!!!--- reach shift_labels!!!!")
    print("---!!!!--- Remember to modify old 'std', old 'mean' for DNN based KBNN")
    trained_model_lists = mrnn_utility.getlist_str(config['KBNN']['LabelShiftingModels'])
    if len(trained_model_lists) > 0:
        # all_fields = mrnn_utility.getlist_str(config['TEST']['AllFields'])
        label_fields = mrnn_utility.getlist_str(config['TEST']['LabelFields'])

        if len(label_fields) > 1:
            # raise ValueError(
            # 'Shift labels is not working for two labels shifting yet!')
            print('Shift labels is not working for two labels shifting yet!')

        # if label_fields[0] != all_fields[-1]:
        # raise ValueError('the single label for KBNN should put at the end of all label fields!')

        print("---!!!!---  load trained model!!!!")
        old_models = load_trained_model(trained_model_lists)
        print("---!!!!---  after load trained model!!!!")
        key0 = label_fields[0]
        old_label_scale = mrnn_utility.getlist_float(config['KBNN']['OldShiftLabelScale'])

        print('old shift features: ', config['KBNN']['OldShiftFeatures'])

        # to switch between vtk and other features
        if (config['KBNN']['OldShiftFeatures'].find('.vtk') >= 0):
            """ """
            print("--- here: vtk for label shift")
            # index should not be used anymore.
            # use base frame info to do the base free energy shifting
            dataset_old = mrnn_utility.load_data_from_vtk_for_label_shift_frame(config, dataset_frame, normalization_flag=True, verbose=0)
        elif (config['KBNN']['OldShiftFeatures'].find('.npy') >= 0):
            """ """
            print("--- here: npy for label shift")
            # index should not be used anymore.
            # use base frame info to do the base free energy shifting
            dataset_old = mrnn_utility.load_data_from_npy_for_label_shift_frame(config, dataset_frame, normalization_flag=True, verbose=0)
        else:
            old_feature_fields = mrnn_utility.getlist_str(config['KBNN']['OldShiftFeatures'])
            raw_dataset_old = mrnn_utility.read_csv_fields(data_file, old_feature_fields)
            dataset_old = raw_dataset_old.copy()

            if len(old_feature_fields) > 0:
                old_mean = mrnn_utility.getlist_float(config['KBNN']['OldShiftMean'])
                old_std = mrnn_utility.getlist_float(config['KBNN']['OldShiftStd'])
                old_data_norm = int(config['KBNN']['OldShiftDataNormOption'])
                if (old_data_norm != 2):
                    dataset_old = (dataset_old - old_mean) / old_std
                else:
                    dataset_old = (dataset_old - old_mean) / old_std + 0.5
                    raise ValueError("This part is not carefully checked. Please check it before you disable it.")

        if (len(old_models) > 0):
            # convert dataset_old to numpy data array in case it is not
            try:
                dataset_old = dataset_old.to_numpy()
            except:
                try:
                    dataset_old = dataset_old.numpy()
                except:
                    pass
                pass

            for model0 in old_models:
                label_shift_amount = []
                batch_size = int(config['MODEL']['BatchSize'])
                print('run...', model0)

                # use model.predict() will run it in the eager mode and evaluate the tensor properly.
                for i0 in range(0, len(dataset_old), batch_size):
                    tmp_shift = model0.predict(mrnn_utility.special_input_case(dataset_old[i0:i0 + batch_size])) / old_label_scale    # numpy type
                    label_shift_amount.extend(tmp_shift)

        for i0 in range(0, len(dataset[key0])):
            a = dataset[key0][i0] - label_shift_amount[i0]
            # tf1.13
            if (i0 % 200 == 0):
                print('--i0--', i0, 'DNS:', dataset[key0][i0], '\t', 'NN:', label_shift_amount[i0], ' key0 = ', key0, ' dataset size = ', len(dataset[key0]),
                      ' label shift size = ', len(label_shift_amount))
            # for tf2.0
            # print('--i0--',i0, 'DNS:', dataset[key0][i0],'\t', 'NN:',label_shift_amount[i0].numpy()[0], '\t', a.numpy()[0], '\t', abs(a.numpy()[0]/dataset[key0][i0])*100, 'new label', new_label[key0][i0])
            dataset[key0][i0] = dataset[key0][i0] - label_shift_amount[i0]




def load_one_model(old_config_file):
    """ Based on the config file name to load pre-saved model info"""
    print('old_config_file:', old_config_file)
    old_config = mrnn_utility.read_config_file(old_config_file, False)
    old_data_file = old_config['TEST']['DataFile']
    if old_data_file.find('.csv') >= 0:
        dummy_train_dataset, dummy_train_labels = mrnn_utility.generate_dummy_dataset(old_config)
        print('dummy_train_dataset:', dummy_train_dataset, 'dummy label: ', dummy_train_labels)
    elif old_data_file.find('.vtk') >= 0:
        dummy_train_dataset, dummy_train_labels, _, _, _, _, _, _ = mrnn_utility.load_data_from_vtk_database(old_config, normalization_flag=True, verbose=0)
    else:
        print('***WARNING***: unknown datafile in old_config file for KBNN, old_datafile = ', old_data_file)
        print('               could potentially lead to errors!!!!')

    old_model = build_model(old_config, dummy_train_dataset, dummy_train_labels, set_non_trainable=True)

    if (old_config['RESTART']['SavedCheckPoint'] != ''):
        saved_check_point = old_config_file[0:old_config_file.rfind('/')] + '/' + old_config['RESTART']['SavedCheckPoint']
        print('saved_check_point: ', saved_check_point)
        old_model.load_weights(saved_check_point)
        print("...loading saved model ...:", old_config['RESTART']['SavedCheckPoint'], old_config_file.rfind('/'), saved_check_point)
        return old_model
    else:
        print("You have to provided the saved check point location in the old config file. Exiting...")
        exit(0)
    return old_model


def load_trained_model(trained_model_lists):
    old_models = []
    if (len(trained_model_lists) > 0):
        for m0 in trained_model_lists:
            model0 = load_one_model(m0)
            print('Pre-trained Model summary: (before) ', m0, model0)
            model0.summary()
            print('Pre-trained Model summary: (after) ', m0)
            old_models.append(model0)
    return old_models




class user_DNN_kregl1l2_gauss_grad(tf.keras.Model):
    def __init__(self, config, NodesList, Activation, train_dataset, train_labels, label_scale, train_stats):
        super(user_DNN_kregl1l2_gauss_grad, self).__init__()
        self.label_scale = label_scale
        self.train_stats_std = train_stats['std'].to_numpy()[0:3]    # E11, E12, E22

        kreg_l1 = 0.0
        kreg_l2 = 0.0
        gauss_noise = 0.0

        try:
            kreg_l2 = float(config['MODEL']['KRegL2'])
            print('l2 regularize could potential cause the loss != mse')
        except:
            pass

        try:
            kreg_l1 = float(config['MODEL']['KRegL1'])
            print('l1 regularize could potential cause the loss != mse')
        except:
            pass

        try:
            gauss_noise = float(config['MODEL']['GaussNoise'])
            print('gauss noise could potential cause the loss != mse')
        except:
            pass

        if (kreg_l1 < 0 or kreg_l2 < 0 or gauss_noise < 0):
            raise ValueError('***ERR***: regularizer or guass noise are < 0! They should be > 0. kreg_l1 = ', kreg_l1, 'kreg_l2 = ', kreg_l2, 'gauss_noise = ', gauss_noise)

        self.all_layers = []

        if kreg_l2 > 0 and kreg_l1 == 0:
            self.all_layers.append(
                layers.Dense(
                    NodesList[0],
                    activation=Activation[0],
                    input_shape=[len(train_dataset.keys())],
                    name='input',
                    kernel_regularizer=regularizers.l2(kreg_l2),
                    kernel_initializer='random_uniform'))
        elif kreg_l1 > 0 and kreg_l2 == 0:
            self.all_layers.append(
                layers.Dense(
                    NodesList[0],
                    activation=Activation[0],
                    input_shape=[len(train_dataset.keys())],
                    name='input',
                    kernel_regularizer=regularizers.l1(kreg_l1),
                    kernel_initializer='random_uniform'))
        elif kreg_l1 > 0 and kreg_l2 > 0:
            raise ValueError('you can not use both l1 and l2 kernel regularizer: try just use l2')
        else:
            self.all_layers.append(layers.Dense(NodesList[0], activation=Activation[0], input_shape=[len(train_dataset.keys())], name='input', kernel_initializer='random_uniform'))

        # first hidden layer
        if gauss_noise > 0.0:
            self.all_layers.append(layers.GaussianNoise(gauss_noise))

        # remaining hidden layer
        for i0 in range(1, len(NodesList)):
            name_str = 'dense-' + str(i0)
            self.all_layers.append(layers.Dense(NodesList[i0], activation=Activation[i0], name=name_str, kernel_initializer='random_uniform'))

        # output layer
        self.all_layers.append(layers.Dense(1, name='output'))

    @tf.function(autograph=False)
    def call(self, inputs):
        with tf.GradientTape() as g:
            g.watch(inputs)
            y1 = self.all_layers[0](inputs)    #,
            for hd in self.all_layers[1:]:
                y2 = hd(y1)
                y1 = y2
        dy_dx = g.gradient(y2, inputs) / self.label_scale / self.train_stats_std

        # for penalize P
        return tf.concat([y2, dy_dx[:, 0:1], dy_dx[:, 1:2], dy_dx[:, 1:2], dy_dx[:, 2:3]], 1)



class CNN_user_supervise(tf.keras.Model):
    """ similar as CNN supervise, but with the check_layer() function  """
    def __init__(self, input_shape, num_output, LayerName, NodesList, Activation, Padding, OutAct):
        super(CNN_user_supervise, self).__init__()

        #-------------------------------------------input layer-----------------------------------------
        Name = LayerName[0]
        Act = Activation[0]
        padding = Padding[0]
        Node = NodesList[0]
        kernel = Name.split('_')[1:]
        kernel = [int(x) for x in kernel]
        if (not (Name.lower().find('conv2d') >= 0)):
            raise ValueError('The first layer should be a conv2D layer!!!')

        self.input_layer = layers.Conv2D(Node, kernel, activation=Act, input_shape=input_shape, padding=padding)

        num_encoder = len(NodesList)

        self.encoder_layer = []
        for i0 in range(1, num_encoder):
            Name = LayerName[i0]
            Act = Activation[i0]
            padding = Padding[i0]
            Node = NodesList[i0]

            if (Name.lower().find('conv2d') >= 0):
                kernel = Name.split('_')[1:]
                kernel = [int(x) for x in kernel]
                self.encoder_layer.append(layers.Conv2D(Node, kernel, activation=Act, padding=padding))
            elif (Name.lower().find('maxpooling2d') >= 0):
                kernel = Name.split('_')[1:]
                kernel = [int(x) for x in kernel]
                self.encoder_layer.append(layers.MaxPooling2D(kernel, padding=padding))
            elif (Name.lower().find('flatten') >= 0):
                self.encoder_layer.append(layers.Flatten())
            elif (Name.lower().find('dense') >= 0):
                self.encoder_layer.append(layers.Dense(Node, activation=Act))
            else:
                raise ValueError('The layer name: ', Name, ' is not programmed!')

        self.output_layer = layers.Dense(num_output, activation=OutAct, name='output')

    def call(self, inputs, training=False):
        x = self.input_layer(inputs)
        for hl in self.encoder_layer:
            x = hl(x)
        x = self.output_layer(x)
        return x

    def check_layer(self, inputs, layer_number=100):
        count = 1

        if layer_number == 0:
            return inputs

        x = self.input_layer(inputs)
        if layer_number == count:
            return x
        count += 1

        for hl in self.encoder_layer:
            x = hl(x)
            if layer_number == count:
                return x
            count += 1

        x = self.output_layer(x)
        return x



def user_DNN_kregl1l2_gauss_grad_setup(config, train_dataset, train_labels, NodesList, Activation, train_stats):
    print('Warning Msg: user_DNN_kregl1l2_gauss_grad is hard coded with fixed label numbers = 1!')

    label_scale = float(config['TEST']['LabelScale'])
    model = user_DNN_kregl1l2_gauss_grad(config, NodesList, Activation, train_dataset, train_labels, label_scale, train_stats)
    return model


def DNN_kregl1l2_gauss(config, train_dataset, train_labels, NodesList, Activation):
    model = keras.Sequential()
    # activity_regularizer
    # bias_regularizer
    # kernel_regularizer
    kreg_l1 = 0.0
    kreg_l2 = 0.0
    gauss_noise = 0.0

    try:
        kreg_l2 = float(config['MODEL']['KRegL2'])
        print('l2 regularize could potential cause the loss != mse')
    except:
        pass

    try:
        kreg_l1 = float(config['MODEL']['KRegL1'])
        print('l1 regularize could potential cause the loss != mse')
    except:
        pass

    try:
        gauss_noise = float(config['MODEL']['GaussNoise'])
        print('gauss noise could potential cause the loss != mse')
    except:
        pass

    if (kreg_l1 < 0 or kreg_l2 < 0 or gauss_noise < 0):
        raise ValueError('***ERR***: regularizer or guass noise are < 0! They should be > 0. kreg_l1 = ', kreg_l1, 'kreg_l2 = ', kreg_l2, 'gauss_noise = ', gauss_noise)

    if kreg_l2 > 0 and kreg_l1 == 0:
        model.add(
            layers.Dense(
                NodesList[0],
                activation=Activation[0],
                input_shape=[len(train_dataset.keys())],
                name='input',
                kernel_regularizer=regularizers.l2(kreg_l2),
                kernel_initializer='random_uniform'))
    elif kreg_l1 > 0 and kreg_l2 == 0:
        model.add(
            layers.Dense(
                NodesList[0],
                activation=Activation[0],
                input_shape=[len(train_dataset.keys())],
                name='input',
                kernel_regularizer=regularizers.l1(kreg_l1),
                kernel_initializer='random_uniform'))
    elif kreg_l1 > 0 and kreg_l2 > 0:
        raise ValueError('you can not use both l1 and l2 kernel regularizer: try just use l2')
    else:
        model.add(layers.Dense(NodesList[0], activation=Activation[0], input_shape=[len(train_dataset.keys())], name='input', kernel_initializer='random_uniform'))

    # first hidden layer
    if gauss_noise > 0.0:
        model.add(layers.GaussianNoise(gauss_noise))

    # remaining hidden layer
    for i0 in range(1, len(NodesList)):
        name_str = 'dense-' + str(i0)

        model.add(layers.Dense(NodesList[i0], activation=Activation[i0], name=name_str, kernel_initializer='random_uniform'))

    # output layer
    model.add(layers.Dense(len(train_labels.keys()), name='output'))

    return model


def add_input_layer(config, model, train_dataset, Name, Node, Act, padding='valid'):
    # [1000, 28, 28, 1] -> [28, 28, 1]
    # print(train_dataset)
    # print(tf.shape(train_dataset.to_numpy()))

    if (tf.__version__[0:1] == '1'):
        # input_shape = train_dataset.get_shape().as_list()[1:]
        input_shape = train_dataset.shape[1:]
    elif (tf.__version__[0:1] == '2'):
        try:
            input_shape = tf.shape(train_dataset).numpy()[1:]
        except:
            input_shape = tf.shape(train_dataset.to_numpy()).numpy()[1:]
    else:
        raise ValueError("Unknown tensorflow version: ", tf.__version__)

    print('input_shape:', input_shape)

    if (Name.lower().find('conv2d') >= 0):
        # _3_3 -> [3,3]
        kernel = Name.split('_')[1:]
        kernel = [int(x) for x in kernel]
        strides = [1, 1]
        if len(kernel) == 4:
            strides = kernel[2:4]
        model.add(layers.Conv2D(Node, kernel[0:2], strides=strides, activation=Act, input_shape=input_shape, padding=padding, name='input'))
    elif (Name.lower().find('conv3d') >= 0):
        # _3_3 -> [3,3]
        kernel = Name.split('_')[1:]
        kernel = [int(x) for x in kernel]
        strides = [1, 1, 1]
        if len(kernel) == 6:
            strides = kernel[3:6]
        model.add(layers.Conv3D(Node, kernel[0:3], strides=strides, activation=Act, input_shape=input_shape, padding=padding, name='input'))

    elif (Name.lower().find('dense') >= 0):
        model.add(layers.Dense(Node, activation=Act, input_shape=input_shape, name='input'))
    elif (Name.lower().find('lstm') >= 0):
        print('!!!!input_shape:', input_shape, ' should be [1, 1]!!!!!')
        model.add(layers.LSTM(Node, input_shape=(1, 1), return_sequences=True))
    elif (Name.lower().find('gru') >= 0):
        print('!!!!input_shape:', input_shape, ' should be [1, 1]!!!!!')
        model.add(layers.GRU(Node, input_shape=(1, 1), return_sequences=True))
    else:
        raise ValueError('The first layer can only be conv2d, your input is: ', Name)


def add_one_layer(config, model, Name, Node, Act, padding='valid', tf_name=''):
    if (Act == 'None'):
        Act = None
    # print('Name:', tf_name)
    if (Name.lower().find('maxpooling2d') >= 0):
        kernel = Name.split('_')[1:]
        kernel = [int(x) for x in kernel]
        print('kernel:', kernel, 'padding:', padding)
        model.add(layers.MaxPooling2D(kernel, padding=padding, name=tf_name))
    elif (Name.lower().find('upsampling2d') >= 0):
        kernel = Name.split('_')[1:]
        kernel = [int(x) for x in kernel]
        model.add(layers.UpSampling2D(kernel, name=tf_name))
        # print('kernel:', kernel)
    elif (Name.lower().find('conv2d') >= 0):
        kernel = Name.split('_')[1:]
        kernel = [int(x) for x in kernel]
        strides = [1, 1]
        if len(kernel) == 4:
            strides = kernel[2:4]
        model.add(layers.Conv2D(Node, kernel[0:2], strides=strides, activation=Act, padding=padding, name=tf_name))
        print('conv2d: ', Node, kernel, strides, Act, padding, tf_name)
    elif (Name.lower().find('conv3d') >= 0):
        kernel = Name.split('_')[1:]
        kernel = [int(x) for x in kernel]
        strides = [1, 1, 1]
        if len(kernel) == 6:
            strides = kernel[3:6]
        model.add(layers.Conv3D(Node, kernel[0:3], strides=strides, activation=Act, padding=padding, name=tf_name))
        # print('kernel:', kernel)
    elif (Name.lower().find('flatten') >= 0):
        model.add(layers.Flatten(name=tf_name))
    elif (Name.lower().find('dense') >= 0):
        model.add(layers.Dense(Node, activation=Act, name=tf_name))
    elif (Name.lower().find('lstm') >= 0):
        # model.add(layers.LSTM(Node,return_sequences = True))
        model.add(layers.LSTM(Node))
    elif (Name.lower().find('gru') >= 0):
        model.add(layers.GRU(Node))
    else:
        raise ValueError('The layer name: ', Name, ' is not programmed!')


def add_output_layer(config, model, train_labels):
    print('****activation:---')
    Act = None
    try:
        Act = config['MODEL']['OutputLayerActivation']
        print('****activation:---', Act)
        if (Act == ''):
            Act = None
    except:
        pass

    num_output = 1
    try:
        num_output = len(train_labels.keys())
    except:
        try:
            num_output = len(train_labels[0])
        except:
            pass
        pass
    print('----activation:---', Act)

    output_layer = 'Dense'
    try:
        output_layer = config['MODEL']['OutputLayer']
    except:
        pass
    if (output_layer == 'Dense'):
        model.add(layers.Dense(num_output, activation=Act, name='output'))
    elif (output_layer == 'No'):
        return
    elif (output_layer == 'Conv3D'):
        """ not checked """
        # model.add(layers.Conv3D(Node, kernel[0:3], strides=strides, activation=Act, input_shape=input_shape, padding=padding,  name='input'))



def CNN_supervise(config, train_dataset, train_labels, NodesList, Activation, LayerName, Padding):
    model = keras.Sequential()

    add_input_layer(config, model, train_dataset, LayerName[0], NodesList[0], Activation[0], Padding[0])
    for i0 in range(1, len(NodesList)):
        add_one_layer(config, model, LayerName[i0], NodesList[i0], Activation[i0], Padding[i0], tf_name=LayerName[i0] + '-' + str(i0))
    add_output_layer(config, model, train_labels)
    print('!!!supervise!!!')

    return model


def CNN_user_supervise_setup(config, train_dataset, train_labels, NodesList, Activation, LayerName, Padding):
    # input_shape=tf.shape(train_dataset).numpy()[1:]
    if (tf.__version__[0:1] == '1'):
        # input_shape = train_dataset.get_shape().as_list()[1:]
        input_shape = train_dataset.shape[1:]
    elif (tf.__version__[0:1] == '2'):
        try:
            input_shape = tf.shape(train_dataset).numpy()[1:]
        except:
            input_shape = tf.shape(train_dataset.to_numpy()).numpy()[1:]
    else:
        raise ValueError("Unknown tensorflow version: ", tf.__version__)

    print('input_shape:', input_shape)

    num_output = 1
    try:
        num_output = len(train_labels.keys())
    except:
        pass

    OutAct = None
    try:
        OutAct = config['MODEL']['OutputLayerActivation']
        if (OutAct == ''):
            OutAct = None
    except:
        pass

    model = CNN_user_supervise(input_shape, num_output, LayerName, NodesList, Activation, Padding, OutAct)
    return model




def build_model(config, train_dataset, train_labels, set_non_trainable=False, train_stats=None):
    ModelArchitect = config['MODEL']['ModelArchitect']

    NodesList = mrnn_utility.getlist_int(config['MODEL']['NodesList'])
    Activation = mrnn_utility.getlist_str(config['MODEL']['Activation'])

    if (len(NodesList) != len(Activation)):
        raise ValueError('In the config file, number of NodesList != Activation list with NodesList = ', NodesList, ' and Activation = ', Activation)

    if (ModelArchitect.lower() == "dnn_kregl1l2_gauss".lower()):
        model = DNN_kregl1l2_gauss(config, train_dataset, train_labels, NodesList, Activation)
    elif (ModelArchitect.lower() == "user_dnn_kregl1l2_gauss_grad".lower()):
        model = user_DNN_kregl1l2_gauss_grad_setup(config, train_dataset, train_labels, NodesList, Activation, train_stats)
    elif (ModelArchitect.lower().find('cnn') >= 0):
        LayerName = mrnn_utility.getlist_str(config['MODEL']['LayerName'])
        Padding = mrnn_utility.getlist_str(config['MODEL']['Padding'])
        if (len(NodesList) != len(LayerName) or len(NodesList) != len(Padding)):
            raise ValueError('In the config file, number of NodesList != LayerName with NodesList = ', NodesList, len(NodesList), ' and LayerName = ', LayerName, len(LayerName),
                             'and Padding = ', Padding, len(Padding))
        elif (ModelArchitect.lower() == "CNN_user_supervise".lower()):
            model = CNN_user_supervise_setup(config, train_dataset, train_labels, NodesList, Activation, LayerName, Padding)
        elif (ModelArchitect.lower() == "CNN_supervise".lower()):
            model = CNN_supervise(config, train_dataset, train_labels, NodesList, Activation, LayerName, Padding)
        else:
            raise ValueError('Model architect = ', ModelArchitect, ' is chosen, but is not implemented!')
    else:
        raise ValueError('Model architect = ', ModelArchitect, ' is chosen, but is not implemented!')

    if set_non_trainable:
        model.trainable = False

    return model
