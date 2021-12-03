from sklearn.model_selection import KFold
import numpy as np
import pickle
from os import path
import tensorflow as tf
import matplotlib.pyplot as plt


class MLKFold:
    """ 
    this class is created to speed up the k-fold training process 
    by default, after an initial shuffle, 10% data will be saved for testing
    the 90% dataset will be used for k-fold train
    to train the best NN structure, 90% dataset will be split as 80%, 10% for train and validation
    the held 10% testing dataset will be used for final model evaluation 
    """

    def __init__(self, total_folds, dataset, split_ratio=[0.8, 0.1, 0.1]):
        """ 
        init the class 
        """

        size_of_data = len(dataset)

        self.kfold_filename = "saved_kfold_status.pickle"
        self.kfold_info = {
            "total_folds": total_folds,
            "size_of_data": size_of_data,
            "current_fold": 0,
            "train_index_list": [],
            "validation_index_list": [],
            "final_test_index_list": [],
            "final_train_index_list": [],
            "final_validation_index_list": [],
        }

        print("***WARNING***: TF1.x only support label with 1 variable.")
        self.prepare_kfold()

    def prepare_kfold(self):
        """ 
        prepare the kfold dataset split 
        """

        if not self.load_status():
            self.init_kfold()
            self.save_status()

    def any_left_fold(self):
        if self.kfold_info["current_fold"] < self.kfold_info["total_folds"]:
            return True
        else:
            self.kfold_info["current_fold"] = 0
            self.save_status()
            return False

    def plot_all_folds(self):
        """ 
        plot all the folds in one figure 
        """

        for i0 in range(0, self.kfold_info["total_folds"]):
            plt.plot(self.kfold_info["train_index_list"][i0], np.ones(len(self.kfold_info["train_index_list"][i0])) * (i0 + 1), 'k.')
            plt.plot(self.kfold_info["validation_index_list"][i0], np.ones(len(self.kfold_info["validation_index_list"][i0])) * (i0 + 1), 'r.')

        # final data
        plt.plot(self.kfold_info["final_train_index_list"], np.ones(len(self.kfold_info["final_train_index_list"])) * (self.kfold_info["total_folds"] + 1), 'k.')
        plt.plot(self.kfold_info["final_validation_index_list"], np.ones(len(self.kfold_info["final_validation_index_list"])) * (self.kfold_info["total_folds"] + 1), 'r.')
        plt.plot(self.kfold_info["final_test_index_list"], np.ones(len(self.kfold_info["final_test_index_list"])) * (self.kfold_info["total_folds"] + 1), 'b.')
        plt.ylabel('folds')
        plt.xlabel('data points')
        plt.title('black: train, red: validation, blue: final test data')

        plt.show()

    def get_next_fold(self, dataset, labels, derivative=[], fold_id=-1, final_data=False):
        """ 
        get next fold data for training/validation 

        input: features, labels, fold_id
               fold_id is only used when a specific number is given

        output: current fold of features and labels

        """

        print("In the fold: ", self.kfold_info["current_fold"], " with ", "train size: ", len(self.kfold_info["train_index_list"][self.kfold_info["current_fold"]]),
              "validation size: ", len(self.kfold_info["validation_index_list"][self.kfold_info["current_fold"]]))

        if (fold_id >= 0 and fold_id < self.kfold_info["total_folds"]):
            train_index = self.kfold_info["train_index_list"][fold_id]
            validation_index = self.kfold_info["validation_index_list"][fold_id]
            test_index = validation_index
        else:
            train_index = self.kfold_info["train_index_list"][self.kfold_info["current_fold"]]
            validation_index = self.kfold_info["validation_index_list"][self.kfold_info["current_fold"]]
            test_index = validation_index

        if final_data:
            train_index = self.kfold_info["final_train_index_list"]
            validation_index = self.kfold_info["final_validation_index_list"]
            test_index = self.kfold_info["final_test_index_list"]

            # fix the messed up index [array([])]
            if (len(train_index) == 1):
                train_index = [x for x in train_index[0]]
                validation_index = [x for x in validation_index[0]]
                test_index = [x for x in test_index[0]]
            # print('train_index:', train_index)

        if (tf.__version__[0:1] == '1'):
            # has to convert
            np_data = np.array([1, 2])
            # print(len(dataset), dataset.shape)
            # print(dataset[1:5,0:].shape)

            # print("TF1.x only support label with 1 variable.")
            if (type(dataset) == type(np_data)):
                # print(type(dataset[0]), type(np.float32(1.0)))
                # if feature only has one variables and is numpy array, use the following
                if type(dataset[0]) == type(np.float32(1.0)) or type(dataset[0]) == type(np.float64(1.0)):
                    train_dataset = dataset[train_index]
                    val_dataset = dataset[validation_index]
                    test_dataset = dataset[test_index]
                else:
                    train_dataset = dataset[train_index, 0:]
                    val_dataset = dataset[validation_index, 0:]
                    test_dataset = dataset[test_index, 0:]

                train_labels = labels[train_index]
                val_labels = labels[validation_index]
                test_labels = labels[test_index]
                # dataset = tf.convert_to_tensor(dataset, dtype=tf.float32) # for tensorflow 1.14
                # labels  = tf.convert_to_tensor(labels, dtype=tf.float32)
                # # tensor use tf.gather(index)
                # # else:
                # # pandas frame will use take(index)
            else:
                try:
                    train_dataset = dataset.take(train_index)
                    train_labels = labels.take(train_index)
                    val_dataset = dataset.take(validation_index)
                    val_labels = labels.take(validation_index)
                    test_dataset = dataset.take(test_index)    # pandas framedata, select the index
                    test_labels = labels.take(test_index)
                    # print("---", train_dataset)
                except:
                    ## but not for tensor
                    train_dataset = tf.gather(dataset, train_index)    ### hopefully, it will work for tensor
                    # print(train_dataset)
                    train_labels = tf.gather(labels, train_index)
                    val_dataset = tf.gather(dataset, validation_index)
                    val_labels = tf.gather(labels, validation_index)
                    test_dataset = tf.gather(dataset, test_index)    # tensor
                    test_labels = tf.gather(labels, test_index)
                    pass
        elif (tf.__version__[0:1] == '2'):
            np_data = np.array([1, 2])
            # print(type(dataset), type(labels))
            if (type(dataset) == type(np_data)):
                dataset = tf.convert_to_tensor(dataset, dtype=tf.float32)
                # print(type(dataset))
            if (type(labels) == type(np_data)):
                labels = tf.convert_to_tensor(labels, dtype=tf.float32)
                # print(type(labels))
            # print('-----------------------')

            try:
                train_dataset = dataset.take(train_index)    # pandas for take
                train_labels = labels.take(train_index)
                val_dataset = dataset.take(validation_index)
                val_labels = labels.take(validation_index)
                test_dataset = dataset.take(test_index)    # pandas framedata, select the index
                test_labels = labels.take(test_index)
            except:
                train_dataset = tf.gather(dataset, train_index)    # tensor use gather
                train_labels = tf.gather(labels, train_index)
                val_dataset = tf.gather(dataset, validation_index)
                val_labels = tf.gather(labels, validation_index)
                test_dataset = tf.gather(dataset, test_index)    # tensor
                test_labels = tf.gather(labels, test_index)
                pass

        if (len(derivative) > 0):
            self.train_derivative = derivative[train_index]
            self.val_derivative = derivative[validation_index]
            self.test_derivative = derivative[test_index]
        else:
            self.train_derivative = []
            self.val_derivative = []
            self.test_derivative = []

        #---- the following is to fix tf.13. etc for residual dataset after one model. Code will crash if this line of code is not here.
        tf.keras.backend.clear_session()
        return train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels, self.test_derivative

    def get_current_fold_derivative_data(self):
        return self.train_derivative, self.val_derivative, self.test_derivative

    def update_kfold_status(self):
        """ 
        wait until the training is done before save the status 
        """

        self.kfold_info["current_fold"] += 1
        self.save_status()

    def save_status(self):
        """ 
        save the k_fold split information such that it will not change index of each fold after restart 
        """

        pickle_out = open(self.kfold_filename, "wb")
        pickle.dump(self.kfold_info, pickle_out)
        pickle_out.close()

    def load_status(self):
        """ 
        load previously saved k_fold split information for consistency 
        """

        if (path.exists(self.kfold_filename)):
            self.kfold_info = pickle.load(open(self.kfold_filename, "rb"))
            print("successfully load saved kfold status files!")
            return True
        else:
            return False

    def init_kfold(self):
        """ 
        initialize kfold split ratio, etc. 
        """

        print("init the kfold with n_splits = ", self.kfold_info["total_folds"], " size_of_dataset = ", self.kfold_info["size_of_data"], " current_fold = ",
              self.kfold_info["current_fold"])

        # prepare the final testing data
        kf = KFold(n_splits=10, shuffle=True)    #, random_state = 1 # provide a positional number to fix the random state
        # index_list is just a dummy dataset for kf.split() to generate the index
        index_list = np.ones(self.kfold_info["size_of_data"])
        count = 0
        for train_index, validation_index in kf.split(index_list):
            training_data = train_index
            self.kfold_info["final_test_index_list"] = validation_index
            # print('in k-fold (test) : ', count)
            # print('  training_data: ', training_data)
            # print('final test data: ', self.kfold_info["final_test_index_list"])
            count += 1
            break

        # prepare the final validation data
        kf = KFold(n_splits=9, shuffle=True)    #, random_state = 1 # provide a positional number to fix the random state
        count = 0
        for train_index, validation_index in kf.split(training_data):
            self.kfold_info["final_train_index_list"] = training_data[train_index]
            self.kfold_info["final_validation_index_list"] = training_data[validation_index]

            # print('in k-fold (train): ', count)
            # print('final train data: ', self.kfold_info["final_train_index_list"])
            # print('final valid data: ', self.kfold_info["final_validation_index_list"])
            count += 1
            break

        kf = KFold(n_splits=self.kfold_info["total_folds"], shuffle=True)    #, random_state = 1 # provide a positional number to fix the random state
        # print('index_list:', index_list)
        count = 0
        for train_index, validation_index in kf.split(training_data):
            self.kfold_info["train_index_list"].append(training_data[train_index])
            self.kfold_info["validation_index_list"].append(training_data[validation_index])
            # print('in k-fold (train): ', count)
            # print('train data: ', self.kfold_info["train_index_list"])
            # print('valid data: ', self.kfold_info["validation_index_list"])
            count += 1

if __name__ == "__main__":
    print("... testing ... MLKFold class")

    size_of_data = 100
    total_folds = 5
    dummy_features = np.linspace(1, 100, size_of_data)
    dummy_labels = np.linspace(101, 200, size_of_data)
    #print(dummy_features, dummy_labels)

    # the point for this test is that the code will run smoothly for a 5-fold split
    print("test 1 is started!")
    test1 = MLKFold(total_folds, dummy_features)
    test1.plot_all_folds()
    a, b, c, d, e, f, g = test1.get_next_fold(dummy_features, dummy_labels, fold_id=1)
    print(a, c, e)
    a, b, c, d, e, f, g = test1.get_next_fold(dummy_features, dummy_labels, final_data=True)
    print(a, c, e)
    exit(0)
    while test1.any_left_fold():
        test1.get_next_fold(dummy_features, dummy_labels)
        test1.update_kfold_status()
    print("test 1 is completed!")

    # create a case where simulation crash in the middle with fold = 3
    print("test 2 is started!")
    test2 = MLKFold(total_folds, dummy_features)
    count = 0
    while test2.any_left_fold():
        count += 1
        if count > 3:
            break
        test2.get_next_fold(dummy_features, dummy_labels)
        test2.update_kfold_status()
    print("test 2 is completed!")

    # create a case where simulation successfully restored and finish this fold
    print("test 3 is started!")
    test3 = MLKFold(total_folds, dummy_features)
    while test3.any_left_fold():
        test3.get_next_fold(dummy_features, dummy_labels)
        test3.update_kfold_status()
    print("test 3 is completed!")

    # create a case where simulation run 2-fold
    print("test 4 is started!")
    test4 = MLKFold(total_folds, dummy_features)
    count = 0
    while test4.any_left_fold():
        count += 1
        if count > 2:
            break
        test4.get_next_fold(dummy_features, dummy_labels)
        test4.update_kfold_status()
    print("test 4 is completed!")

    # create a case where simulation restore and run 2-fold more
    print("test 5 is started!")
    test5 = MLKFold(total_folds, dummy_features)
    count = 0
    while test5.any_left_fold():
        count += 1
        if count > 2:
            break
        test5.get_next_fold(dummy_features, dummy_labels)
        test5.update_kfold_status()
    print("test 5 is completed!")

    # create a case where simulation all simulations are finished
    print("test 6 is started!")
    test6 = MLKFold(total_folds, dummy_features)
    while test6.any_left_fold():
        a, b, c, d, e, f, g = test6.get_next_fold(dummy_features, dummy_labels)
        print('a: ', len(a), a)
        print('b: ', len(b), b)
        print('c: ', len(c), c)
        print('d: ', len(d), d)
        test6.update_kfold_status()
    print("test 6 is completed!")

    print("test 7 is started!")
    test7 = MLKFold(total_folds, dummy_features)
    while test7.any_left_fold():
        test7.get_next_fold(dummy_features, dummy_labels)
        test7.update_kfold_status()
    print("test 7 is completed!")
