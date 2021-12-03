import itertools
from itertools import combinations
from SALib.sample import saltelli
import numpy as np
import pickle
import matplotlib.pyplot as plt
import statistics
from os import path

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


class HyperParametersCNN:
    """
  This class is created to perform hyper-parameters search for CNNs.
  """

    def __init__(self,
                 config,
                 input_shape=1,
                 output_shape=1,
                 filter_size=3,
                 pool_size=1,
                 padding='same',
                 dim=2,
                 uniform_sample_number=10,
                 neighbor_sample_number=5,
                 iteration_time=3,
                 sample_ratio=0.3,
                 best_model_number=10,
                 max_total_parameter=1e5,
                 repeat_train=3,
                 debug=False):

        self.config = config
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.padding = padding
        self.dim = dim
        self.uniform_sample_number = uniform_sample_number
        self.neighbor_sample_number = neighbor_sample_number
        self.iteration_time = iteration_time
        self.sample_ratio = sample_ratio
        self.best_model_number = best_model_number
        # set a maximum parameter number to avoid gpu running out of memory
        self.max_total_parameter = max_total_parameter
        self.repeat_train = repeat_train
        self.debug = debug

        if (self.dim == 3):
            raise ValueError("Dim = 3 for Conv3D is not programmed!")

        self.all_hyperparameter_info = {
            "current_iteration": -1,
            "current_index": 0,
            "sampling_range": [],
            "search_hl_range": [],
            "search_filter_range": [],
            "index_list": [],
            "best_model": [],
            "all_model_structure": [],
            "model_short_summary": [],    # save the averaged loss, val_loss
            "model_long_summary": [],    # save the loss, val_loss history
            "model_loss_summary": [],    # save min(loss, val_loss) for each training
            "model_train_count": []
        }

        self.hyperparameter_filename = "saved_hyperparameter_search.pickle"

        self.prepare_parameter_search()

    def prepare_parameter_search(self):
        if (not self.load_saved_parameter_search()):
            self.generate_all_CNNs()
            self.initialize_info()
            self.do_the_sampling()

        print("Total CNN structures: ", len(self.all_hyperparameter_info["all_model_structure"]))
        if len(self.all_hyperparameter_info["all_model_structure"]) < self.best_model_number:
            print("...ERR...: Total CNN structure = ", len(self.all_hyperparameter_info["all_model_structure"]), " is smaller than the best model number!!!",
                  self.best_model_number)
            print("Please modify the hyperparameter search setting or delete previously saved file: ", self.hyperparameter_filename)
            print(self.all_hyperparameter_info)
            exit(0)
        # print(self.all_hyperparameter_info["index_list"])
        # print(self.all_hyperparameter_info)

    def initialize_info(self):
        """ """
        count = 0
        for _ in self.all_hyperparameter_info["all_model_structure"]:
            self.all_hyperparameter_info["model_train_count"].append(0)
            self.all_hyperparameter_info["model_short_summary"].append([count, 1e9, 1e9])
            self.all_hyperparameter_info["model_long_summary"].append([])
            self.all_hyperparameter_info["model_loss_summary"].append({'index': count, 'loss': [], 'val_loss': []})
            count += 1
        self.all_hyperparameter_info["sampling_range"] = [0, count]

    def load_saved_parameter_search(self):
        """ """
        # print(self.all_hyperparameter_info)
        if (path.exists(self.hyperparameter_filename)):
            self.all_hyperparameter_info = pickle.load(open(self.hyperparameter_filename, "rb"))
            print("successfully load saved hyperparameter files!")
            return True
        else:
            return False

    def save_current_parameter_search(self):
        """ """
        pickle_out = open(self.hyperparameter_filename, "wb")
        pickle.dump(self.all_hyperparameter_info, pickle_out)
        pickle_out.close()

    def update_model_info(self, index0, history):
        """ update the summary for a particular CNN """
        self.all_hyperparameter_info['model_train_count'][index0] += 1

        # ... get min(training loss): obsolete
        # train_loss_min = min(history['loss'])
        # train_loss_min_index = history['loss'].index(train_loss_min)
        # val_loss = history['val_loss'][train_loss_min_index]

        # ... get min(validation loss)
        val_loss_min = min(history['val_loss'])
        val_loss_min_index = history['val_loss'].index(val_loss_min)
        train_loss = history['loss'][val_loss_min_index]
        self.all_hyperparameter_info['model_loss_summary'][index0]['loss'].append(float(train_loss))    # convert to float from either float32 or float64 to avoid problems
        self.all_hyperparameter_info['model_loss_summary'][index0]['val_loss'].append(float(val_loss_min))

        # ... only save the min loss, but not the averaged value: obsolete
        # if (self.all_hyperparameter_info['model_short_summary'][index0][1] > train_loss_min):
        # self.all_hyperparameter_info['model_short_summary'][index0][1] = train_loss_min
        # self.all_hyperparameter_info['model_short_summary'][index0][2] = val_loss

        # ... get the averaged value for different loss, more reasonable.
        self.all_hyperparameter_info['model_short_summary'][index0][1] = statistics.mean(self.all_hyperparameter_info['model_loss_summary'][index0]['loss'])
        self.all_hyperparameter_info['model_short_summary'][index0][2] = statistics.mean(self.all_hyperparameter_info['model_loss_summary'][index0]['val_loss'])
        print('mean(loss):', self.all_hyperparameter_info['model_short_summary'][index0][1], 'loss:', self.all_hyperparameter_info['model_loss_summary'][index0]['loss'])
        print('mean(val_loss):', self.all_hyperparameter_info['model_short_summary'][index0][2], 'val_loss:',
              self.all_hyperparameter_info['model_loss_summary'][index0]['val_loss'])

        self.all_hyperparameter_info['model_long_summary'][index0].append(history)

        all_model_summary = self.all_hyperparameter_info['model_short_summary'][:]
        # sort with validation loss.
        all_model_summary.sort(key=lambda x: x[2])

        self.all_hyperparameter_info['best_model'] = all_model_summary
        print("current best model with least val_loss:")
        print("index \t train_loss \t\t val_loss")
        for i0 in range(0, self.best_model_number):
            m0 = self.all_hyperparameter_info['best_model'][i0]
            print(m0[0], '\t', m0[1], '\t', m0[2])
        print('model index:', [x[0] for x in self.all_hyperparameter_info['best_model'][0:self.best_model_number]])

        self.save_current_parameter_search()

        # print(self.all_hyperparameter_info)

    def compute_total_parameters_per_CNN(self, filters):
        """
    if pool_size == 1, no maxpooling layer, if pool_size > 1, pooling layer will be inserted
    if outputlayer = dense, a flatten layer will be inserted
    """

        cnn_filters = []
        cnn_layername = []
        total_parameters = 0
        conv2D_name = 'Conv2D_' + str(self.filter_size) + '_' + str(self.filter_size)
        pool2D_name = 'MaxPooling2D_' + str(self.pool_size) + '_' + str(self.pool_size)

        # initial x_dim
        x_dim = self.input_shape

        # after first pooling layer
        f0 = filters[0]
        total_parameters += (self.filter_size * self.filter_size) * f0 + f0
        # print(total_parameters)

        cnn_filters.append(f0)
        cnn_layername.append(conv2D_name)

        if (self.pool_size > 1):
            x_dim = int((self.input_shape + (self.pool_size - 1)) / self.pool_size)
            cnn_filters.append(0)
            cnn_layername.append(pool2D_name)

        for n1 in range(1, len(filters)):
            f_n0 = filters[n1 - 1]
            f_n1 = filters[n1]
            total_parameters += (self.filter_size * self.filter_size) * f_n1 * f_n0 + f_n1
            # print('this layer: ', (self.filter_size * self.filter_size) * f_n1 * f_n0 + f_n1, f_n0, f_n1)

            cnn_filters.append(f_n1)
            cnn_layername.append(conv2D_name)

            if (self.pool_size > 1):
                x_dim = int((x_dim + (self.pool_size - 1)) / self.pool_size)
                cnn_filters.append(0)
                cnn_layername.append(pool2D_name)

                # avoid this structure by setting a very layer total parameter numbers
                # because of too many maxpooling layers.
                if (x_dim == 1):
                    total_parameters += 1e9

                # print('x_dim = ', x_dim)
            # print(total_parameters)

        # add a flatten layer if needed
        if (self.config['MODEL']['OutputLayer'].lower() == 'Dense'.lower()):
            cnn_filters.append(0)
            cnn_layername.append('Flatten')
            # final dense layer
            total_parameters += x_dim * x_dim * filters[-1] + 1

            # print(total_parameters)
        # print(cnn_filters, cnn_layername, total_parameters)

        return cnn_filters, cnn_layername, total_parameters

    def generate_all_CNNs(self):

        # [1, 20]
        hidden_layer_num = getlist_int(self.config['HYPERPARAMETERS']['HiddenLayerNumber'])
        # print(hidden_layer_num)
        # [2, 512, 2]
        filter_num = getlist_int(self.config['HYPERPARAMETERS']['NodesList'])
        # print(filter_num)

        self.all_hyperparameter_info["search_hl_range"] = hidden_layer_num[:]
        self.all_hyperparameter_info["search_filter_range"] = filter_num[:]

        if (len(hidden_layer_num) != 2):
            raise ValueError("Please provide HiddenLayerNumber: min, max in the input for HYPERPARAMETERS")
        if (len(filter_num) != 3):
            raise ValueError("Please provide NodesList: min, max, step in the input for HYPERPARAMETERS")

        # create all combinations of layers and neurons
        _tmp_parameters = [[], []]
        _tmp_parameters[0] = list(range(hidden_layer_num[0], hidden_layer_num[1]))
        _tmp_parameters[1] = list(range(filter_num[0], filter_num[1], filter_num[2]))    # reverse the NodeList for decreasing filters, not common
        _tmp_studies = []

        if len(_tmp_parameters[1]) >= 18:
            print("... WARNING .. : combination n=", len(_tmp_parameters[1]), " choose r=", _tmp_parameters[0][-1], " could be huge!!!")
            print("                 try to keep n<18 !!!!")
            exit(0)

        for p1 in _tmp_parameters[0]:
            comb = combinations(_tmp_parameters[1], p1)
            # print('combine: ', comb, p1)
            count = 0
            for i1 in comb:
                count += 1
                # print(p1, i1, count)
                _tmp_studies.append(list(i1))

        # get total_parameters for each CNN
        self.all_hyperparameter_info["all_model_structure"] = []
        for _CNN in _tmp_studies:
            # print("check _CNN: ", _CNN)
            _CNN_filter, _CNN_layername, _total_parameters = self.compute_total_parameters_per_CNN(_CNN)

            # layer info, and total parameters
            self.all_hyperparameter_info["all_model_structure"].append([_CNN_filter, _CNN_layername, _total_parameters])
            # print([_CNN_filter, _CNN_layername, _total_parameters])
        # print(len(_tmp_studies))

        # sort by total parameters and remove CNNs with total_parameters exceed a predefined maximum value
        self.all_hyperparameter_info["all_model_structure"].sort(key=lambda x: x[2])
        count = 0
        for _CNN in self.all_hyperparameter_info["all_model_structure"]:
            # print(_CNN)
            if _CNN[2] > self.max_total_parameter:
                del self.all_hyperparameter_info["all_model_structure"][count:]
                return
            count += 1

    def do_the_sampling(self):
        # uniform sampling
        index_list = np.linspace(self.all_hyperparameter_info["sampling_range"][0], self.all_hyperparameter_info["sampling_range"][1], self.uniform_sample_number + 2, dtype=int)
        # exclude the 1st and last index
        extended_index_list = [x for x in index_list[1:-1]]

        # exclude the 1st index
        index_list = index_list[1:]

        # add the neighbor_sampling
        for i0 in range(0, len(index_list) - 1):
            ind0 = index_list[i0 + 0]
            ind1 = index_list[i0 + 1]

            count = 1
            for j0 in range(ind0, ind1):
                if (count >= self.neighbor_sample_number):
                    continue
                extended_index_list.append(j0)
                count += 1
            # store all info
        self.all_hyperparameter_info["index_list"] = sorted(extended_index_list)
        self.all_hyperparameter_info["current_iteration"] += 1
        self.all_hyperparameter_info["current_index"] = 0
        print('new index_list: len=', len(self.all_hyperparameter_info["index_list"]), self.all_hyperparameter_info["index_list"])
        self.save_current_parameter_search()

    def update_index_list(self):
        """ """
        print("update_index_list")

        # get all the trained models
        all_model_summary = self.all_hyperparameter_info['model_short_summary'][:]
        all_model_summary.sort(key=lambda x: x[2])
        all_model_summary = [x for x in all_model_summary if x[1] < 1e6]
        # print('all model summary: ', all_model_summary)

        # choose the ratio
        total_select_model = int(len(all_model_summary) * self.sample_ratio)
        model_index = [x[0] for x in all_model_summary[0:total_select_model]]
        model_index.sort()
        # print(total_select_model, model_index)
        self.all_hyperparameter_info["sampling_range"] = [model_index[0], model_index[-1]]
        self.do_the_sampling()

    def get_next_model(self):
        """ """

        if (self.all_hyperparameter_info["current_iteration"] == self.iteration_time
                and self.all_hyperparameter_info["current_index"] == len(self.all_hyperparameter_info["index_list"])):
            print(
                "You reach to the end of the searching criteria. You can test other models by restart the simulation and increase the iteration_time. (By the meantime, you can also increase the uniform_sample_number and neighbor_sample_number.)"
            )
            exit(0)
        elif (self.all_hyperparameter_info["current_iteration"] < self.iteration_time
              and self.all_hyperparameter_info["current_index"] == len(self.all_hyperparameter_info["index_list"])):
            print("come to next iteration", self.all_hyperparameter_info["current_index"], len(self.all_hyperparameter_info["index_list"]),
                  self.all_hyperparameter_info["current_iteration"], self.iteration_time)
            self.update_index_list()

        index0 = self.all_hyperparameter_info["index_list"][self.all_hyperparameter_info["current_index"]]

        if (self.all_hyperparameter_info["model_train_count"][index0] >= self.repeat_train):
            self.all_hyperparameter_info["current_index"] += 1

        this_CNN = self.all_hyperparameter_info["all_model_structure"][index0]
        print("index0: ", index0, this_CNN)
        ##[16, 16, 16, 16, 16, 16]
        self.config['MODEL']['NodesList'] = ','.join([str(x) for x in this_CNN[0]])
        self.config['MODEL']['LayerName'] = ','.join(this_CNN[1])

        act_fcn = getlist_str(self.config['HYPERPARAMETERS']['Activation'])
        if (len(act_fcn) > 1):
            raise ValueError("Please provide only one Activation for HYPERPARAMETERS")
        else:
            self.act_fcn = act_fcn[0]

        self.config['MODEL']['Activation'] = ','.join([self.act_fcn] * len(this_CNN[0]))
        self.config['MODEL']['Padding'] = ','.join([self.padding] * len(this_CNN[0]))

        self.para_str = 'V' + str(this_CNN[2]) + '-' + 'model-' + 'H' + str(len(getlist_int(self.config['MODEL']['NodesList']))) \
                                 + 'N' + str(getlist_int(self.config['MODEL']['NodesList'])[0]) \
                                 + 'A' + getlist_str(self.config['MODEL']['Activation'])[0] \
                                 + 'L' + self.config['MODEL']['LearningRate'].replace(", ", '-') \
                                 + 'O' + self.config['MODEL']['Optimizer']

        if (self.all_hyperparameter_info["model_train_count"][index0] >= self.repeat_train):
            print("model: ", index0, " has been trained, move to next model!")
            return index0, self.para_str, False
        else:
            return index0, self.para_str, True

    def plot_best_models(self, model_number_to_plot=5):
        """ """
        print('[layer, neuron, total variable]', '[model index, loss, validation loss]')
        for m0 in range(0, model_number_to_plot):
            index0 = self.all_hyperparameter_info["best_model"][m0][0]
            # print(index0)
            summary = self.all_hyperparameter_info["model_long_summary"][index0][0]
            print(self.all_hyperparameter_info["all_model_structure"][index0], self.all_hyperparameter_info["model_short_summary"][index0])
            dnn_info = "-".join([str(x) for x in self.all_hyperparameter_info["all_model_structure"][index0]])
            val = plt.plot(summary["loss"], '-', label=str(index0) + ': loss ' + dnn_info)
            plt.plot(summary["val_loss"], '--', label=str(index0) + ': val_loss ' + dnn_info, color=val[0].get_color())
            plt.yscale('log')
        plt.legend()
        plt.show()

    def plot_all_models(self):
        """ """
        _short_summary = self.all_hyperparameter_info["model_short_summary"][:]
        short_summary = [x for x in _short_summary if x[1] < 1e6]

        parameter_number = []
        train_loss = []
        val_loss = []
        model_info = []
        for s0 in short_summary:
            index0 = s0[0]
            parameter_number.append(self.all_hyperparameter_info["all_model_structure"][index0][2])
            train_loss.append(s0[1])
            val_loss.append(s0[2])
            model_info.append('L' + str(self.all_hyperparameter_info["all_model_structure"][index0][0]) + 'N' + str(self.all_hyperparameter_info["all_model_structure"][index0][1]))
            # print(self.all_hyperparameter_info["all_model_structure"][index0], '\t', s0[1], '\t', s0[2])

        # print(short_summary)
        # print(parameter_number)
        # print(train_loss)
        # print(val_loss)
        # print(model_info)

        fig, ax = plt.subplots()
        ax.scatter(parameter_number, train_loss, label='train_loss')
        ax.scatter(parameter_number, val_loss, label='val_loss')
        for i, txt in enumerate(model_info):
            ax.annotate(txt, (parameter_number[i], train_loss[i]))
            # ax.annotate(txt, (parameter_number[i], val_loss[i]))
        # plt.plot(parameter_number, train_loss, marker='o', label='train loss')
        # plt.plot(parameter_number, val_loss, marker='o', label='val loss')
        plt.yscale('log')
        ax.set_ylim([1e-7, 1e-0])
        plt.legend()
        plt.show()


if __name__ == '__main__':
    import sys
    print("...testing... : ", sys.argv[0])
    config = ""
    parameter = HyperParametersCNN(config, input_shape=61, output_shape=1)
    # print (parameter.all_hyperparameter_info["all_model_structure"], len(parameter.all_hyperparameter_info["all_model_structure"]))
    # print(type(index_list), index_list)
    # parameter.save_current_parameter_search()
    # parameter.load_saved_parameter_search()

    # l2 norm: not to large, avoid singularity
    # train very slow
    # train very long
    # small neuron networks

    try:
        parameter.plot_best_models(10)
        parameter.plot_all_models()
    except:
        parameter.plot_best_models(3)
        parameter.plot_all_models()
