import numpy as np
import datetime
import pandas as pd
import argparse, sys, os
import tensorflow as tf
from numpy import array
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from configparser import ConfigParser, ExtendedInterpolation
import vtk
import glob
from natsort import natsorted, ns
import fractions



def parse_sys_args():

    # check: https://docs.python.org/3.6/library/argparse.html#nargs
    parser = argparse.ArgumentParser(description='Run ML study for effective properties study', prog="'" + (sys.argv[0]) + "'")
    parser.add_argument('configfile', help="configuration file for the study [*.config]")

    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    #
    parser.add_argument('-p', '--platform', choices=['cpu', 'gpu'], type=str, default='gpu', help='choose either use gpu or cpu platform (default: gpu)')

    #
    parser.add_argument('-o', '--output_dir', type=str, help='folder name to store output data')
    parser.add_argument('-r', '--restart_dir', type=str, help='folder name to store restart data')
    parser.add_argument('-t', '--tensorboard_dir', type=str, help='folder name to store tensor board data')
    parser.add_argument('-i', '--inspect', type=int, default=0, choices=[0, 1], help='pre-inspect the data (default: 0)')
    parser.add_argument('-s', '--show', type=int, default=0, choices=[0, 1], help='show the final plot (default: 0)')

    #
    parser.add_argument('-D', '--debug', type=bool, default=False, help="switch on/off the debug flag")
    parser.add_argument('-V', '--verbose', type=int, default=0, choices=[0, 1, 2, 3], help='verbose level of the code (default: 0)')
    parser.add_argument('-P', '--profile', type=bool, default=False, help='switch on/off the profiling output')
    # parser.add_argument('--integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, help='sum the integers (default: find the max)') # for future references
    args = parser.parse_args()

    if (not (args.verbose == 3 and args.debug)):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    # suppress info output
        # tf.logging.set_verbosity(tf.logging.ERROR) # suppress deprecation warning
        #0 = all messages are logged (default behavior)
        #1 = INFO messages are not printed
        #2 = INFO and WARNING messages are not printed
        #3 = INFO, WARNING, and ERROR messages are not printed

    if (args.verbose == 3):
        print(parser.print_help())

    if (args.platform == 'cpu'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if (args.verbose == 3):
        ml_todos()

    return args


class sys_args:
    def __init__(self):
        self.configfile = ''
        self.platform = 'gpu'
        self.inspect = 0
        self.show = 0
        self.debug = False
        self.verbose = 0


def notebook_args(args):

    if (not (args.verbose == 3 and args.debug)):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    # suppress info output
        # tf.logging.set_verbosity(tf.logging.ERROR) # suppress deprecation warning
        #0 = all messages are logged (default behavior)
        #1 = INFO messages are not printed
        #2 = INFO and WARNING messages are not printed
        #3 = INFO, WARNING, and ERROR messages are not printed

    if (args.platform == 'cpu'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if (args.verbose == 3):
        ml_todos()



def split_data(datax, datay, split_ratio=['0.6', '0.25', '0.15']):
    tr_ratio = float(split_ratio[0])
    cv_ratio = float(split_ratio[1])
    tt_ratio = float(split_ratio[2])

    number_examples = datax.shape[0]
    idx = np.arange(0, number_examples)
    np.random.shuffle(idx)
    datax = [datax[i] for i in idx]    # get list of `num` random samples
    datay = [datay[i] for i in idx]    # get list of `num` random samples

    start = 0
    end_tr = int(tr_ratio * number_examples)
    end_cv = int((tr_ratio + cv_ratio) * number_examples)
    end_tt = number_examples
    tr_datax = np.array(datax[start:end_tr])
    tr_datay = np.array(datay[start:end_tr])
    cv_datax = np.array(datax[end_tr:end_cv])
    cv_datay = np.array(datay[end_tr:end_cv])
    tt_datax = np.array(datax[end_cv:end_tt])
    tt_datay = np.array(datay[end_cv:end_tt])

    return tr_datax, tr_datay, cv_datax, cv_datay, tt_datax, tt_datay


def get_package_version(tf_version):
    """ get the major and minor version of tensor flow """
    versions = tf_version.split('.')[0:2]
    versions = [int(x) for x in versions]
    # print(versions)
    return versions


def getlist_str(option, sep=',', chars=None):
    """Return a list from a ConfigParser option. By default, 
     split on a comma and strip whitespaces."""
    list0 = [(chunk.strip(chars)) for chunk in option.split(sep)]
    list0 = [x for x in list0 if x]
    return list0


def getlist_int(option, sep=',', chars=None):
    """Return a list from a ConfigParser option. By default, 
     split on a comma and strip whitespaces."""
    list0 = option.split(sep)
    list0 = [x for x in list0 if x]
    if (len(list0)) > 0:
        return [int(chunk.strip(chars)) for chunk in list0]
    else:
        return []


def getlist_float(option, sep=',', chars=None):
    """Return a list from a ConfigParser option. By default, 
     split on a comma and strip whitespaces."""
    list0 = option.split(sep)
    list0 = [x for x in list0 if x]
    if (len(list0)) > 0:
        return [float(chunk.strip(chars)) for chunk in list0]
    else:
        return []


def get_now():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def exe_cmd(cmd):
    """ execute shell cmd """
    output_info = os.popen(cmd).read()


def get_dummy_data(num):
    """ get dummy_data for num of fields """
    one_I = [1] * (num + 1)
    data2D = [one_I, one_I]
    I = csvDf(data2D)
    return I


def csvDf(dat, **kwargs):
    data = array(dat)
    if data is None or len(data) == 0 or len(data[0]) == 0:
        return None
    else:
        return pd.DataFrame(data[1:, 1:], index=data[1:, 0], columns=data[0, 1:], **kwargs)


def read_config_file(configfile, print_keys=False):
    """ 
    read configuration file and modify the related path
    """
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(configfile)
    print('... read ... configfile = ', configfile)
    print('old root is:', type(config['TEST']['root']), [config['TEST']['root']])
    modify_root_flag = False
    if config['TEST']['root'] == '':
        config['TEST']['root'] = os.path.dirname(os.path.abspath(configfile)) + '/'
        modify_root_flag = True
    else:
        modify_root_flag = False
        print('root is: ', config['TEST']['root'])

    # note: check if data files is given with absolute path (start with '/') or relative path, will add the root path to it.
    if (config['TEST']['DataFile'][0] != '/'):
        data_file_list = getlist_str(config['TEST']['DataFile'])
        print('...modifying.. DataFile from: ', config['TEST']['DataFile'])
        for i0 in range(0, len(data_file_list)):
            data_file_list[i0] = os.path.dirname(os.path.abspath(configfile)) + '/' + data_file_list[i0]
        config['TEST']['DataFile'] = ', '.join(data_file_list)
        print('...modifying.. DataFile to: ', config['TEST']['DataFile'])

    # if the following values are not given as the absolute path, then, it will be modified to the absolute value
    # In KBNN, use config['TEST']['root'] to provide the relative path.
    # in the training procedure, the following needs to be modified to store data in the scratch folder
    config['RESTART']['CheckPointDir'] = config['TEST']['root'] + '/' + config['RESTART']['CheckPointDir']
    config['OUTPUT']['TensorBoardDir'] = config['TEST']['root'] + '/' + config['OUTPUT']['TensorBoardDir']
    config['OUTPUT']['FinalModelSummary'] = config['TEST']['root'] + '/' + config['OUTPUT']['FinalModelSummary']

    # note: if the given folder name does not end with "/", the following will add "/" to it.
    if (config['RESTART']['CheckPointDir'][-1] != '/'):
        config['RESTART']['CheckPointDir'] = config['RESTART']['CheckPointDir'] + '/'
        print(' ... add ... / to the CheckPointDir, with a new value of ', config['RESTART']['CheckPointDir'])

    cmd = 'mkdir -p ' + config['RESTART']['CheckPointDir']
    exe_cmd(cmd)
    cmd = 'mkdir -p ' + config['OUTPUT']['TensorBoardDir']
    exe_cmd(cmd)

    if (print_keys):
        for sec in config.items():
            sec_name = sec[0]
            print("--SECTION NAME--: ", sec_name)
            for key in config[sec_name]:
                print('         --key--: {:>25s}:'.format(key), '  ', config[sec_name][key])

    print('new test root: ', config['TEST']['root'])
    return config


def read_one_vtk(filepath, scalar='', vector=''):
    # print('read_one_vtk')

    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(filepath)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()

    griddata = reader.GetOutput()

    #TensorFlow’s convolutional conv2d operation expects a 4-dimensional tensor with dimensions corresponding to
    # batch, width, height and channel.
    #[batch, in_height, in_width, in_channels]

    if scalar == 'e2':
        e2 = []
        for i in range(griddata.GetPointData().GetScalars('e2').GetNumberOfTuples()):
            a = griddata.GetPointData().GetScalars('e2').GetTuple(i)[0]
            e2.append(a)

        e2 = np.array(e2)
        e2 = e2 + 0.1
        n_x = int(np.sqrt(len(e2)))
        n_y = n_x
        e2 = np.reshape(e2, (n_x, n_y, 1)) / 0.2
        e2[e2 < 0] = 0
        e2[e2 > 1] = 1
        # return e2[:n_x-1, :n_y-1]
        return e2[:n_x, :n_y]


def read_psi_me_from_mechanical_data(file_path):
    """ this function should be a standalone script to prepare the label and features for vtk files. """
    print('read_psi_me_from_mechanical_data: for temporary label of vtk datatype')

    # delete the leading '=' for the index field
    cmd = "sed -i 's/^.*=//' " + file_path
    exe_cmd(cmd)

    selected_cols = pd.read_csv(file_path, index_col=False, skipinitialspace=True)
    print(selected_cols)
    label = [None] * (len(selected_cols) + 1)
    print("Att: read_psi_me_from_mechanical_data: 'index' is used to index frames")
    print("ERR could occur if some of the frames have psi_me but not vtk, or have vtk but not psi_me [index out of range error]")
    for i in range(0, len(selected_cols)):
        # print(i, selected_cols['index'][i], selected_cols['Psi_me'][i])
        label[selected_cols['index'][i]] = selected_cols['Psi_me'][i]
        # label[selected_cols['index'][i]] = selected_cols['Psi_me_total'][i]
    # print (selected_cols, label)
    label[len(selected_cols)] = label[len(selected_cols) - 1]
    return label


def load_data_from_npy_for_label_shift_frame(config, dataset_frame, normalization_flag=True, verbose=0):
    """ don't mess up the input filename with fcn """
    print('load_data_from_npy_for_label_shift_frame')
    data_file = config['KBNN']['OldShiftFeatures']
    print("numpy_base_frame_file_name: ", data_file)
    all_data = np.load(data_file)
    print('load saved numpy base frame for vtk folder')
    # print('load saved numpy base frame for vtk folder', tf.shape(all_data))
    # all_data = all_data.astype(np.float32)
    # print('all data after cast: ', tf.shape(all_data))
    return all_data


def load_data_from_vtk_for_label_shift_frame(config, dataset_frame, normalization_flag=True, verbose=0):
    """ don't mess up the input filename with fcn """
    print('load_data_from_vtk_for_label_shift_frame')
    data_file = config['KBNN']['OldShiftFeatures']
    all_data = []
    load_numpy_flag = False

    numpy_base_frame_file_name = "numpy_base_frame_" + config['KBNN']['OldShiftCNNSavedBaseFrameNumpyName'] + ".vtk"
    print("numpy_base_frame_file_name: ", numpy_base_frame_file_name)

    for file1 in glob.glob(data_file[0:data_file.rfind('/') + 1] + '*'):
        if file1.find(numpy_base_frame_file_name) >= 0:
            all_data = np.load(file1)
            load_numpy_flag = True
            print('load saved numpy base frame for vtk folder', tf.shape(all_data))

    if (not load_numpy_flag):
        all_the_vtk_files = glob.glob(data_file)
        all_the_vtk_files = natsorted(all_the_vtk_files, alg=ns.IGNORECASE)

        frame_index = [None] * 10000000
        for file1 in all_the_vtk_files:
            # print ('working on:', file1, file1.split('/out'))
            if len(file1.split('/out')) > 1:
                # file1:  out1117.vtk
                framenumber = int(file1.split('/out')[1].split('.vtk')[0])
                frame_index[framenumber] = file1
                # print('framenumber: ', framenumber, " file1: ", file1)

        # print('dataset_frame: ', dataset_frame)
        count = len(dataset_frame)
        for i1 in dataset_frame['frame']:
            count -= 1
            file1 = frame_index[i1]
            # print('i1=: ', i1, file1, ' ', count, ' files left to process!!')
            all_data.append(read_one_vtk(file1, scalar='e2'))

        numpy_file = file1[0:file1.rfind('/')] + '/' + numpy_base_frame_file_name
        all_data = np.array(all_data)
        print('save data to numpy_file: ', numpy_file)
        np.save(numpy_file, all_data)

    all_data = all_data.astype(np.float32)

    # all_data = tf.cast(all_data, tf.float32)
    # all_data = tf.convert_to_tensor(all_data, dtype=tf.float32)
    print('all data after cast: ', tf.shape(all_data))
    return all_data


def load_all_data_from_vtk_database(config, normalization_flag=True, verbose=0):
    print('load_all_data_from_vtk_database')
    data_file = config['TEST']['DataFile']
    # print(data_file)
    data_file_list = getlist_str(config['TEST']['DataFile'])
    # print(data_file_list)

    all_data = []
    the_label = []
    all_data_one = []
    the_label_one = []

    for data_file in data_file_list:
        load_numpy_flag = False
        print(data_file)
        print(data_file[0:data_file.rfind('/') + 1] + '*')
        for file1 in glob.glob(data_file[0:data_file.rfind('/') + 1] + '*'):
            if file1.find('numpy.vtk') >= 0:
                all_data.append(np.load(file1))
                load_numpy_flag = True
                print('load saved numpy for vtk folder')
                # print('all_data', all_data)

            if file1.find('numpy_label.vtk') >= 0:
                the_label.append(np.load(file1))
                print('load saved numpy for the label folder')

        if (not load_numpy_flag):
            # if(len(data_file_list) > 1):
            # raise ValueError ('This subroutine is not checked with multiple folders! Check Carefully! Do not mess up the labels!')
            all_the_vtk_files = glob.glob(data_file)
            print(data_file[0:data_file.rfind('/') + 1] + 'mechanical_data.txt')
            tmp_label = read_psi_me_from_mechanical_data(data_file[0:data_file.rfind('/') + 1] + 'mechanical_data.txt')
            # print(tmp_label)

            all_the_vtk_files = natsorted(all_the_vtk_files, alg=ns.IGNORECASE)
            for file1 in all_the_vtk_files:
                framenumber = int(file1.split('/out')[1].split('.vtk')[0])
                the_label_one.append(tmp_label[framenumber])
                # print (framenumber, tmp_label[framenumber])
                all_data_one.append(read_one_vtk(file1, scalar='e2'))

            numpy_file = file1[0:file1.rfind('/')] + '/numpy.vtk'
            all_data.append(np.array(all_data_one))
            print('save data to numpy_file: ', numpy_file, np.shape(all_data_one))
            np.save(numpy_file, all_data[-1])

            numpy_file = file1[0:file1.rfind('/')] + '/numpy_label.vtk'
            the_label.append(np.array(the_label_one))
            print('save data to numpy_file: ', numpy_file, np.shape(the_label_one))
            np.save(numpy_file, the_label[-1])
            print('all_data: ', np.shape(all_data), len(all_data))
            print('the_label: ', np.shape(the_label), len(the_label))

    _all_data = all_data[0]
    # print(np.shape(_all_data))
    _the_label = the_label[0]
    for a1 in all_data[1:]:
        # print(np.shape(a1))
        _all_data = np.concatenate((_all_data, a1), axis=0)
        # print(np.shape(_all_data))
    for t1 in the_label[1:]:
        # print(np.shape(_the_label))
        # print(np.shape(t1))
        _the_label = np.concatenate((_the_label, t1), axis=0)
        # print(np.shape(_the_label))
    print('all data : ', np.shape(_all_data))
    print('the label: ', np.shape(_the_label))

    all_data = _all_data.astype(np.float32)
    the_label = _the_label.astype(np.float32)

    print('all data : ', tf.shape(all_data))
    print('the label: ', tf.shape(the_label))

    label_scale = float(config['TEST']['LabelScale'])
    the_label = the_label * label_scale

    test_derivative = []
    train_stats = []

    if (tf.__version__[0:1] == '1'):
        return all_data, the_label, test_derivative, train_stats
    elif (tf.__version__[0:1] == '2'):
        all_data = tf.convert_to_tensor(all_data, dtype=tf.float32)
        the_label = tf.convert_to_tensor(the_label, dtype=tf.float32)
        return all_data, the_label, test_derivative, train_stats


def load_all_data_from_npy_database(config, normalization_flag=True, verbose=0):
    print('load_all_data_from_npy_database')
    data_file = config['TEST']['DataFile']
    data_file_list = getlist_str(config['TEST']['DataFile'])

    label_scale = float(config['TEST']['LabelScale'])
    if (label_scale != 1.0):
        raise ValueError('LabelScale != 1.0 for npy database is not supported now!!!')

    all_data = None
    the_label = None
    load_numpy_flag = False
    for data_file in data_file_list:
        # print('data_file: ', data_file)
        for file1 in glob.glob(data_file):
            feature_file = file1
            label_file = feature_file.replace('features', 'labels')

            if all_data is None:
                all_data = np.load(feature_file)
            else:
                tmp_data = np.load(feature_file)
                all_data = np.concatenate((all_data, tmp_data), axis=0)

            if the_label is None:
                the_label = np.load(label_file)
            else:
                tmp_label = np.load(label_file)
                the_label = np.concatenate((the_label, tmp_label), axis=0)

            print(' feature file: ', feature_file, ' label file: ', label_file)
        # print('all data shape: ', np.shape(all_data), ' all label shape: ', np.shape(the_label))
        # if(np.shape(all_data) != np.shape(the_label)):
        # raise ValueError('features shape does not match the label shape. Check if you really want this to happen. So far, the code is for elasticity BVP full field map!!!')

    all_data = all_data.astype(np.float32)
    the_label = the_label.astype(np.float32)

    print('all data : ', tf.shape(all_data))
    print('the label: ', tf.shape(the_label))

    label_scale = float(config['TEST']['LabelScale'])
    the_label = the_label * label_scale

    test_derivative = []
    train_stats = []
    # exit(0)

    if (tf.__version__[0:1] == '1'):
        return all_data, the_label, test_derivative, train_stats
    elif (tf.__version__[0:1] == '2'):
        all_data = tf.convert_to_tensor(all_data, dtype=tf.float32)
        the_label = tf.convert_to_tensor(the_label, dtype=tf.float32)
        return all_data, the_label, test_derivative, train_stats


def load_data_from_vtk_database(config, normalization_flag=True, verbose=0):
    print('load_data_from_vtk_database: hard coded label: mechanical_data.txt', )
    data_file = config['TEST']['DataFile']
    # print(data_file)
    data_file_list = getlist_str(config['TEST']['DataFile'])
    # print(data_file_list)

    all_data = []
    the_label = []
    all_data_one = []
    the_label_one = []

    for data_file in data_file_list:
        load_numpy_flag = False
        print(data_file)
        print(data_file[0:data_file.rfind('/') + 1] + '*')
        for file1 in glob.glob(data_file[0:data_file.rfind('/') + 1] + '*'):
            if file1.find('numpy.vtk') >= 0:
                all_data.append(np.load(file1))
                load_numpy_flag = True
                print('load saved numpy for vtk folder')
                # print('all_data', all_data)

            if file1.find('numpy_label.vtk') >= 0:
                the_label.append(np.load(file1))
                print('load saved numpy for the label folder')

        if (not load_numpy_flag):
            # if(len(data_file_list) > 1):
            # raise ValueError ('This subroutine is not checked with multiple folders! Check Carefully! Do not mess up the labels!')
            all_the_vtk_files = glob.glob(data_file)
            print(data_file[0:data_file.rfind('/') + 1] + 'mechanical_data.txt')
            tmp_label = read_psi_me_from_mechanical_data(data_file[0:data_file.rfind('/') + 1] + 'mechanical_data.txt')
            # print(tmp_label)

            all_the_vtk_files = natsorted(all_the_vtk_files, alg=ns.IGNORECASE)
            for file1 in all_the_vtk_files:
                framenumber = int(file1.split('/out')[1].split('.vtk')[0])
                the_label_one.append(tmp_label[framenumber])
                # print (framenumber, tmp_label[framenumber])
                all_data_one.append(read_one_vtk(file1, scalar='e2'))

            numpy_file = file1[0:file1.rfind('/')] + '/numpy.vtk'
            all_data.append(np.array(all_data_one))
            print('save data to numpy_file: ', numpy_file, np.shape(all_data_one))
            np.save(numpy_file, all_data[-1])

            numpy_file = file1[0:file1.rfind('/')] + '/numpy_label.vtk'
            the_label.append(np.array(the_label_one))
            print('save data to numpy_file: ', numpy_file, np.shape(the_label_one))
            np.save(numpy_file, the_label[-1])
            print('all_data: ', np.shape(all_data), len(all_data))
            print('the_label: ', np.shape(the_label), len(the_label))

    _all_data = all_data[0]
    # print(np.shape(_all_data))
    _the_label = the_label[0]
    for a1 in all_data[1:]:
        # print(np.shape(a1))
        _all_data = np.concatenate((_all_data, a1), axis=0)
        # print(np.shape(_all_data))
    for t1 in the_label[1:]:
        # print(np.shape(_the_label))
        # print(np.shape(t1))
        _the_label = np.concatenate((_the_label, t1), axis=0)
        # print(np.shape(_the_label))
    print('all data : ', np.shape(_all_data))
    print('the label: ', np.shape(_the_label))

    all_data = _all_data.astype(np.float32)
    the_label = _the_label.astype(np.float32)

    print('all data : ', tf.shape(all_data))
    print('the label: ', tf.shape(the_label))
    # exit(0)

    split_ratio = getlist_float(config['TEST']['SplitRatio'])
    if (len(split_ratio) != 3 or abs(sum(split_ratio) - 1.0) > 1.0e-5):
        raise ValueError('split ratio should be a list containing three float values with sum() == 1.0!!! Your current split_ratio = ', split_ratio, ' with sum = ',
                         sum(split_ratio))


    label_scale = float(config['TEST']['LabelScale'])
    the_label = the_label * label_scale

    train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels = split_data(all_data, the_label, split_ratio)

    test_derivative = []
    train_stats = []

    if (tf.__version__[0:1] == '2'):
        train_dataset = tf.convert_to_tensor(train_dataset, dtype=tf.float32)
        train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
        test_dataset = tf.convert_to_tensor(test_dataset, dtype=tf.float32)
        test_labels = tf.convert_to_tensor(test_labels, dtype=tf.float32)
        val_dataset = tf.convert_to_tensor(val_dataset, dtype=tf.float32)
        val_labels = tf.convert_to_tensor(val_labels, dtype=tf.float32)

    ModelArchitect = config['MODEL']['ModelArchitect']
    if (ModelArchitect.lower() == "CNN_autoencoder".lower() or ModelArchitect.lower().find("_unsupervise") >= 0):
        print('unsupervised learning, features = label')
        return train_dataset, train_dataset, val_dataset, val_dataset, test_dataset, test_dataset, test_derivative, train_stats
    else:
        return train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels, test_derivative, train_stats


def load_all_data(config, args):    # for K-fold validation
    """
    load csv, image, url etc data to the main code
    """
    print('load_all_data')
    verbose = args.verbose

    # load / pre-process data / split data
    if (int(config['TEST']['DataNormalization']) == 0):
        normalization_flag = False
    else:
        normalization_flag = True

    data_file = config['TEST']['DataFile']

    if (data_file.find('csv') > 0):
        dataset, labels, derivative, train_stats = load_all_data_from_csv(config, verbose=args.verbose, normalization_flag=normalization_flag)
    elif (data_file.find('.vtk') > 0):
        print("*****************WARNING**********************:")
        print("if have multiple VTK folder and it's the first time to load vtk and save numpy array. There is a")
        print("potential bug, that after the 1st vtk folder, the following numpy array file is getting bigger ")
        print("and bigger, try to fix this bug next time!!!!!")
        print("***********************************************")
        dataset, labels, derivative, train_stats = load_all_data_from_vtk_database(config, verbose=args.verbose, normalization_flag=normalization_flag)
    elif (data_file.find('.npy') > 0):
        dataset, labels, derivative, train_stats = load_all_data_from_npy_database(config, verbose=args.verbose, normalization_flag=normalization_flag)
    else:
        raise ValueError('unknown options for the DataFile:', data_file)

    print("...done with data loading")    #,len(dataset), len(labels)) // len(tensor) is not available for tf1.13

    if (args.inspect == 1):
        print('enter pre-inspection')
        print('exit after pre-inspection')
        exit(0)

    return dataset, labels, derivative, train_stats


# the default data file is in csv format with ',' as the delimiter, and the header to describe the field info
def read_csv_fields(file_path, fields, sep=','):
    # will read the csv file and load the fields according to the new order
    list_of_csv_files = getlist_str(file_path)
    selected_cols = pd.read_csv(list_of_csv_files[0], index_col=False, sep=sep, usecols=fields, skipinitialspace=True)[fields]
    print('read_csv_fields: ', list_of_csv_files[0], len(selected_cols))

    for f1 in list_of_csv_files[1:]:
        new_selected_cols = pd.read_csv(f1, index_col=False, sep=sep, usecols=fields, skipinitialspace=True)[fields]
        print('read_csv_fields: ', f1, len(new_selected_cols))
        selected_cols = selected_cols.append(new_selected_cols, ignore_index=True)
    print('total df datasize: ', len(selected_cols))

    # print (selected_cols)
    # print (type(selected_cols))
    # print (selected_cols.values)
    # print (type(selected_cols.values))
    # return selected_cols.values # return numpy types
    return selected_cols


def dataset_pop_list(data_set, pop_list):
    # print ('before pop: ', data_set.keys())
    df2 = pd.concat([data_set.pop(x) for x in pop_list], 1)
    # print ('after pop: ', data_set.keys())
    return df2


def norm(x, train_stats, DataNormOption=0):
    if DataNormOption == 0:
        print('...mean:', train_stats['mean'])
        print('...std:', train_stats['std'])
        return (x - train_stats['mean']) / train_stats['std']    # ATT > float64
    elif DataNormOption == 1:
        return (x - train_stats['mean']) / train_stats['std']    # ATT > float64
    elif DataNormOption == 2:
        return (x - train_stats['mean']) / train_stats['std'] + 0.5    # ATT > float64
    elif DataNormOption == 3:
        return (x - train_stats['mean']) / train_stats['std']    # ATT > float64


def prepare_data_from_csv_file(config, normalization_flag=True, verbose=0):
    """
  load the desired fields from the csv file, not full list
  split the data based on the label fields
  split the data to three different set [train, validation, test] 
  """
    print('prepare_data_from_csv_file')

    split_ratio = [0.6, 0.25, 0.15],
    data_file = config['TEST']['DataFile']
    print('data_file', data_file)

    all_fields = getlist_str(config['TEST']['AllFields'])
    label_fields = getlist_str(config['TEST']['LabelFields'])
    derivative_fields = getlist_str(config['TEST']['DerivativeFields'])

    try:
        KBNN_flag = (config['KBNN']['LabelShiftingModels'] != '')
    except:
        KBNN_flag = False
        pass

    for l1 in label_fields:
        try:
            all_fields.index(l1)
        except:
            raise ValueError("label_fields = ", label_fields, " is not in all_fields = ", all_fields, " Error: all_fields should contain label_fields!!!")

    try:
        split_ratio = getlist_float(config['TEST']['SplitRatio'])
    except:
        pass

    if (verbose == 3):
        print('Data split ratio [train, validation, test] = ', split_ratio)
        print('Data file: ', config['TEST']['datafile'])

    raw_dataset = read_csv_fields(data_file, all_fields)
    dataset = raw_dataset.copy()

    #-----------------following is not a good feature or needed feature, as data normalization normally handle it well------------------------
    #####  if len(feature_shift) > 0:
    #####    print("""   You have enabled feature shift in config file. The number and sequence of shift is in the same order
    #####    of the features you specified in the label_fields. Now you are shifting: """, all_fields[0:len(feature_shift)], ' with ', feature_shift, '.')
    #####    for i0 in range(0, len(feature_shift)):
    #####      key0 = all_fields[i0]
    #####      dataset[key0] = dataset[key0] - feature_shift[i0]
    #####      # print (i0, key0, dataset.keys(), dataset[key0])

    if (KBNN_flag):
        ## index and frames are used to match the CNN training info
        # index was the first try, but it turns out that we should use the base vtu file to predict the base free energy function.
        try:
            raw_dataset_index = read_csv_fields(data_file, ['index'])
            dataset_index = raw_dataset_index.copy()
        except:
            print("***ERR** in loading the index data. Will be neglected!!!")
            dataset_index = None
            pass

        # frame is the final choice. Rerun the collect data script to get new dataset if needed.
        try:
            raw_dataset_frame = read_csv_fields(data_file, ['frame'])
            dataset_frame = raw_dataset_frame.copy()
        except:
            print("***ERR** in loading the frame data. Will be neglected!!!")
            dataset_frame = None
            pass
    #-------------NN label shift------------------
        import mechanoChemML.workflows.multi_resolution_learning.mrnn_models as mrnn_models
        mrnn_models.shift_labels(config, dataset, dataset_index, dataset_frame, data_file)
    #------------------following is a little bit non-modulated, easy for bugs ---------------------------

# check data
    if (len(split_ratio) != 3 or abs(sum(split_ratio) - 1.0) > 1.0e-5):
        raise ValueError('split ratio should be a list containing three float values with sum() == 1.0!!! Your current split_ratio = ', split_ratio, ' with sum = ',
                         sum(split_ratio))

    # split data for LSTM and GRU without randomly shuffle
    ModelArchitect = config['MODEL']['ModelArchitect']
    if ModelArchitect.lower().find('lstm') >= 0 or ModelArchitect.lower().find('gru') >= 0:
        print('dataset for LSTM or GRU')

        feature_index = list(range(0, len(dataset) - 1))
        label_index = list(range(1, len(dataset)))

        train_num = int(len(feature_index) * split_ratio[0])
        val_num = int(len(feature_index) * split_ratio[1])
        test_num = len(feature_index) - train_num - val_num

        dataset_stats = dataset.describe()
        dataset_stats = dataset_stats.transpose()
        dataset = (dataset_stats['max'] - dataset) / (dataset_stats['max'] - dataset_stats['min'])

        features = dataset.to_numpy()
        labels = np.squeeze(dataset.to_numpy())

        # print(tf.shape(features))
        features = np.expand_dims(features, axis=-1)
        print(np.shape(features), np.shape(labels), type(features))

        train_dataset = features[feature_index[0:train_num]]
        train_labels = labels[label_index[0:train_num]]
        # print(tf.shape(train_dataset), tf.shape(train_labels))

        val_dataset = features[feature_index[train_num:val_num + train_num]]
        val_labels = labels[label_index[train_num:val_num + train_num]]

        test_dataset = features[feature_index[train_num + val_num:val_num + train_num + test_num]]
        test_labels = labels[label_index[train_num + val_num:val_num + train_num + test_num]]

        train_stats = dataset_stats
        test_derivative = []

        # print(train_dataset, train_labels)
        # print(val_dataset, val_labels)
        # print(test_dataset, test_labels)
        # exit(0)
        return train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels, test_derivative, train_stats

    print('len of total dataset: ', len(dataset))

    # split data
    train_dataset = dataset.sample(frac=split_ratio[0], random_state=0)
    tmp_dataset = dataset.drop(train_dataset.index)
    print('len of each dataset (train, tmp: ', len(train_dataset), len(tmp_dataset), 'split_ratio: ', split_ratio)

    val_dataset = tmp_dataset.sample(frac=(split_ratio[1] / (split_ratio[1] + split_ratio[2])), random_state=0)
    test_dataset = tmp_dataset.drop(val_dataset.index)
    print('len of each dataset (train, val, test): ', len(train_dataset), len(val_dataset), len(test_dataset))

    batch_size = int(config['MODEL']['BatchSize'])
    if (batch_size > len(train_dataset) or (batch_size > len(val_dataset) and len(val_dataset) != 0) or batch_size > len(test_dataset)):
        raise ValueError('batch_size is larger than one of your data set, reduce it!', 'batch_size:', batch_size, 'train, validation, test size:', len(train_dataset),
                         len(val_dataset), len(test_dataset), 'Please choose a common factor for your training data!')

    drop_data_flag = 0
    try:
        drop_data_flag = int(config['TEST']['DropData'])
    except:
        pass

    print('data_set info:', len(train_dataset), len(val_dataset), len(test_dataset), 'default batch size:', batch_size)
    if (drop_data_flag):
        train_data_to_drop = len(train_dataset) % batch_size
        val_data_to_drop = len(val_dataset) % batch_size
        test_data_to_drop = len(test_dataset) % batch_size
        print('to_drop:', train_data_to_drop, val_data_to_drop, test_data_to_drop)
        if (train_data_to_drop == 0 and val_data_to_drop == 0 and test_data_to_drop == 0):
            print('the pre-set batch-size is good!')
        else:
            tmp_batch_size = fractions.gcd(fractions.gcd(len(train_dataset), len(val_dataset)), len(test_dataset))

            if (tmp_batch_size >= 32 and tmp_batch_size <= 1024):
                print('use updated batch_size: ', tmp_batch_size)
                config['MODEL']['BatchSize'] = str(tmp_batch_size)
                train_data_to_drop = 0
                val_data_to_drop = 0
                test_data_to_drop = 0
            else:
                print('Please re-split data as good as possible! use default batch_size:', batch_size)
                train_data_to_drop = len(train_dataset) % batch_size
                val_data_to_drop = len(val_dataset) % batch_size
                test_data_to_drop = len(test_dataset) % batch_size
    else:
        print("drop_data_flag is False, no data drop is allowed even the size of data is not a multiple of batch size.")

    test_derivative = []
    # print('---derivative fields: ', derivative_fields)
    if (len(derivative_fields) > 0):
        raw_dataset_derivative = read_csv_fields(data_file, derivative_fields)
        dataset_derivative = raw_dataset_derivative.copy()
        dataset_derivative = dataset_derivative.drop(train_dataset.index)
        test_derivative = dataset_derivative.drop(val_dataset.index)
        # print ('check test_derivative before:', test_derivative)
        if (drop_data_flag):
            if test_data_to_drop > 0:
                test_derivative = test_derivative.drop(test_derivative.index[-test_data_to_drop:])
        # print ('check test_derivative after:', test_derivative)
        test_derivative = test_derivative.to_numpy()

        # print ('----- test---   :', test_dataset)
    if (drop_data_flag):
        if train_data_to_drop > 0:
            train_dataset = train_dataset.drop(train_dataset.index[-train_data_to_drop:])
        if val_data_to_drop > 0:
            val_dataset = val_dataset.drop(val_dataset.index[-val_data_to_drop:])
        if test_data_to_drop > 0:
            test_dataset = test_dataset.drop(test_dataset.index[-test_data_to_drop:])

    # # print(len(train_dataset))
    # # print(train_data_to_drop, val_data_to_drop, test_data_to_drop)
    # # print('tmp_batch_size:', tmp_batch_size)
    # print(len(train_dataset), len(val_dataset), len(test_dataset))
    # exit(0)

    # get mean, std, etc
    train_stats = train_dataset.describe()
    dataset_pop_list(train_stats, label_fields)
    train_stats = train_stats.transpose()

    DataNormOption = 0
    try:
        DataNormOption = int(config['TEST']['DataNormOption'])
    except:
        pass

    if DataNormOption == 0:
        # do nothing
        print("---norm---: use 'mean' and 'std ' do the normalization (-1.7, 1.7)")
    elif DataNormOption == 1:
        print("---norm---: use 0.5*(min+max) and 'max-min' do the normalization (-0.5, 0.5)")
        train_stats['mean'] = 0.5 * (train_stats['min'] + train_stats['max'])
        train_stats['std'] = (train_stats['max'] - train_stats['min'])
    elif DataNormOption == 2:
        print("---norm---: use 0.5*(min+max) and 'max-min' do the normalization (0, 1)")
        train_stats['mean'] = 0.5 * (train_stats['min'] + train_stats['max'])
        train_stats['std'] = (train_stats['max'] - train_stats['min'])
    elif DataNormOption == 3:
        print("---norm---: use 0.5*(min+max) and 'max-min' do the normalization (-1, 1)")
        train_stats['mean'] = 0.5 * (train_stats['min'] + train_stats['max'])
        train_stats['std'] = 0.5 * (train_stats['max'] - train_stats['min'])

    # new_std = 0.5 * ( train_stats['std']['F12'] + train_stats['std']['F21'] )
    # train_stats['std']['F12'] = new_std
    # train_stats['std']['F21'] = new_std
    print('std:', train_stats['std'])
    print('mean:', train_stats['mean'])

    if (KBNN_flag):
        print('replace old mean and old std')
        old_features = getlist_str(config['KBNN']['OldEmbedFeatures'])
        if len(old_features) > 0:
            old_mean = getlist_float(config['KBNN']['OldEmbedMean'])
            old_std = getlist_float(config['KBNN']['OldEmbedStd'])
            for i0 in range(0, len(old_features)):
                key0 = old_features[i0]
                if any(key0 in s for s in all_fields):
                    print('update std, mean of key0=', key0)
                    train_stats['std'][key0] = old_std[i0]
                    train_stats['mean'][key0] = old_mean[i0]
            print('(after)std:', train_stats['std'])
            print('(after)mean:', train_stats['mean'])
    # exit(0)

    label_scale = float(config['TEST']['LabelScale'])
    label_shift = float(config['TEST']['LabelShift'])
    print('Label shift: ', label_shift)

    # get labels
    train_labels = dataset_pop_list(train_dataset, label_fields)
    # print('out_side ', train_dataset.keys(), type(train_dataset))
    val_labels = dataset_pop_list(val_dataset, label_fields)
    test_labels = dataset_pop_list(test_dataset, label_fields)

    train_labels = (train_labels - label_shift) * label_scale
    val_labels = (val_labels - label_shift) * label_scale
    test_labels = (test_labels - label_shift) * label_scale

    # print ('train_labels:', train_labels)

    if (verbose == 3):
        print("Train_dataset tail(5): ")
        print(train_dataset.tail(5))
        print("Train_labels tail(5): ")
        print(train_labels.tail(5))

    # print('  std of train_stats: ', train_stats['std'])
    if (len(test_derivative) > 0):
        test_derivative = test_derivative * label_scale

    normed_test_derivative = []
    if (len(test_derivative) > 0 and normalization_flag):
        std = train_stats['std'].to_numpy()    # pay attention to the types.
        # print('test_derivative * label_scale: ', test_derivative)
        normed_test_derivative = test_derivative * std[0:len(test_derivative[0])]
        if (len(std) > len(test_derivative[0])):
            print("!!!Warning: features number in std is larger than the test derivative field. The first several features std is used to scale test_derivative!!!")
        print('std:', std, ' label scale: ', label_scale)
        # print('test_derivative: ', test_derivative)
        # print('normed_test_derivative: ', normed_test_derivative)

    # print('train_stats: ', train_stats)
    # print('train_labels: ', train_labels)

    if (normalization_flag):
        # normalize data based on train_means

        # print('dataset_old before', train_dataset)
        normed_train_data = norm(train_dataset, train_stats, DataNormOption)
        # print('dataset_old after', normed_train_data)
        # exit(0)
        normed_val_data = norm(val_dataset, train_stats, DataNormOption)
        normed_test_data = norm(test_dataset, train_stats, DataNormOption)

        # print('---aaa--- ', train_dataset.keys(), type(train_dataset))
        return normed_train_data, train_labels, normed_val_data, val_labels, normed_test_data, test_labels, normed_test_derivative, train_stats
    else:
        return train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels, test_derivative, train_stats


def load_all_data_from_csv(config, normalization_flag=True, verbose=0):
    data_file = config['TEST']['DataFile']

    all_fields = getlist_str(config['TEST']['AllFields'])
    label_fields = getlist_str(config['TEST']['LabelFields'])
    derivative_fields = getlist_str(config['TEST']['DerivativeFields'])

    try:
        KBNN_flag = (config['KBNN']['LabelShiftingModels'] != '')
    except:
        KBNN_flag = False
        pass
    print(data_file)
    # if (KBNN_flag): # is enabled
    # raise ValueError("KBNN is not enabled for K-fold validation")

    # delete this from future version
    # feature_shift = getlist_float(config['TEST']['FeatureShift'])

    for l1 in label_fields:
        try:
            all_fields.index(l1)
        except:
            raise ValueError("label_fields = ", label_fields, " is not in all_fields = ", all_fields, " Error: all_fields should contain label_fields!!!")

    raw_dataset = read_csv_fields(data_file, all_fields)
    dataset = raw_dataset.copy()

    if (KBNN_flag):
        ## index and frames are used to match the CNN training info
        # index was the first try, but it turns out that we should use the base vtu file to predict the base free energy function.
        try:
            raw_dataset_index = read_csv_fields(data_file, ['index'])
            dataset_index = raw_dataset_index.copy()
        except:
            print("***ERR** in loading the index data. Will be neglected!!!")
            dataset_index = None
            pass

        # # frame is the final choice. Rerun the collect data script to get new dataset if needed.
        try:
            raw_dataset_frame = read_csv_fields(data_file, ['frame'])
            dataset_frame = raw_dataset_frame.copy()
        except:
            print("***ERR** in loading the frame data. Will be neglected!!!")
            dataset_frame = None
            pass
    #-------------NN label shift------------------
        import mechanoChemML.workflows.multi_resolution_learning.mrnn_models as mrnn_models
        mrnn_models.shift_labels(config, dataset, dataset_index, dataset_frame, data_file)

    # print('---derivative fields: ', derivative_fields)

    test_derivative = []
    if (len(derivative_fields) > 0):
        raw_dataset_derivative = read_csv_fields(data_file, derivative_fields)
        dataset_derivative = raw_dataset_derivative.copy()
        test_derivative = dataset_derivative.to_numpy()

    # get mean, std, etc
    train_stats = dataset.describe()
    dataset_pop_list(train_stats, label_fields)
    train_stats = train_stats.transpose()

    DataNormOption = 0
    try:
        DataNormOption = int(config['TEST']['DataNormOption'])
    except:
        pass

    if DataNormOption == 0:
        # do nothing
        print("---norm---: use 'mean' and 'std ' do the normalization (-1.7, 1.7)")
    elif DataNormOption == 1:
        print("---norm---: use 0.5*(min+max) and 'max-min' do the normalization (-0.5, 0.5)")
        train_stats['mean'] = 0.5 * (train_stats['min'] + train_stats['max'])
        train_stats['std'] = (train_stats['max'] - train_stats['min'])
    elif DataNormOption == 2:
        print("---norm---: use 0.5*(min+max) and 'max-min' do the normalization (0, 1)")
        train_stats['mean'] = 0.5 * (train_stats['min'] + train_stats['max'])
        train_stats['std'] = (train_stats['max'] - train_stats['min'])
    elif DataNormOption == 3:
        print("---norm---: use 0.5*(min+max) and 'max-min' do the normalization (-1, 1)")
        train_stats['mean'] = 0.5 * (train_stats['min'] + train_stats['max'])
        train_stats['std'] = 0.5 * (train_stats['max'] - train_stats['min'])

    print('std:', train_stats['std'])
    print('mean:', train_stats['mean'])

    if (KBNN_flag):
        print('replace old mean and old std')
        old_features = getlist_str(config['KBNN']['OldEmbedFeatures'])
        if len(old_features) > 0:
            old_mean = getlist_float(config['KBNN']['OldEmbedMean'])
            old_std = getlist_float(config['KBNN']['OldEmbedStd'])
            for i0 in range(0, len(old_features)):
                key0 = old_features[i0]
                if any(key0 in s for s in all_fields):
                    print('update std, mean of key0=', key0)
                    train_stats['std'][key0] = old_std[i0]
                    train_stats['mean'][key0] = old_mean[i0]
            print('(after)std:', train_stats['std'])
            print('(after)mean:', train_stats['mean'])

    label_scale = float(config['TEST']['LabelScale'])
    label_shift = float(config['TEST']['LabelShift'])
    print('Label shift: ', label_shift)

    # get labels
    labels = dataset_pop_list(dataset, label_fields)
    labels = (labels - label_shift) * label_scale

    if (verbose == 3):
        print("Train_dataset tail(5): ")
        print(train_dataset.tail(5))
        print("Train_labels tail(5): ")
        print(train_labels.tail(5))

    # print('  std of train_stats: ', train_stats['std'])
    if (len(test_derivative) > 0):
        test_derivative = test_derivative * label_scale

    normed_test_derivative = []
    if (len(test_derivative) > 0 and normalization_flag):
        std = train_stats['std'].to_numpy()    # pay attention to the types.
        # print('test_derivative * label_scale: ', test_derivative)
        normed_test_derivative = test_derivative * std[0:len(test_derivative[0])]
        if (len(std) > len(test_derivative[0])):
            print("!!!Warning: features number in std is larger than the test derivative field. The first several features std is used to scale test_derivative!!!")
        print('std:', std, ' label scale: ', label_scale)

    if (normalization_flag):
        normed_dataset = norm(dataset, train_stats, DataNormOption)
        normed_derivative = normed_test_derivative
        return normed_dataset, labels, normed_derivative, train_stats
    else:
        derivative = test_derivative
        return dataset, labels, derivative, train_stats


def inspect_cnn_features(model, config, test_dataset, savefig=False):
    num_images = int(config['OUTPUT']['NumImages'])
    inspect_layers = getlist_int(config['OUTPUT']['InspectLayers'])

    total_images = 0
    for l0 in inspect_layers:
        out1 = model.check_layer(test_dataset[0:1], l0)
        total_images += tf.shape(out1[0]).numpy()[2]
        print('total_images:', total_images)

    if (int(np.sqrt(total_images)) * int(np.sqrt(total_images)) >= total_images):
        num_col = int(np.sqrt(total_images))
    else:
        num_col = int(np.sqrt(total_images)) + 1
    num_row = num_col

    for i0 in range(0, num_images):
        plt.figure()
        count = 0
        for l0 in inspect_layers:
            out1 = model.check_layer(test_dataset[i0:i0 + 1], l0)
            img0 = out1[0]
            shape0 = tf.shape(img0).numpy()
            for i in range(1, shape0[2] + 1):    # 2nd index is the feature numbers
                count += 1
                ax = plt.subplot(num_col, num_row, count)
                plt.imshow(out1[0, :, :, i - 1])    # tensor
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        if savefig:
            plt.savefig(str(i0) + '.pdf', bbox_inches='tight', format='pdf')
        plt.show()



def generate_dummy_dataset(old_config):
    """ based on the label list, generate dummy dataset """
    all_fields = getlist_str(old_config['TEST']['AllFields'])
    label_fields = getlist_str(old_config['TEST']['LabelFields'])
    train_dataset = get_dummy_data(len(all_fields) - len(label_fields))
    train_label = get_dummy_data(len(label_fields))
    return train_dataset, train_label

def special_input_case(inputs, input_case=''):
    """ deal with special case for old inputs:
      for example, another DNN is for the frame 800, which has input only F11, F12, F21, F22, 
      whereas, the current model might have additional microstructure features, thus, the input
      needs to be sliced to fit for the old DNN. 
  """
    if input_case != '':

        input_ind = getlist_int(input_case)
        if (len(input_ind) == 1):
            s_ind = 0
            e_ind = input_ind[0]
        elif (len(input_ind) == 2):
            s_ind = input_ind[0]
            e_ind = input_ind[1]
        else:
            s_ind = 0
            e_ind = -1
        return tf.slice(inputs, [0, s_ind], [-1, e_ind])
    else:
        return inputs

