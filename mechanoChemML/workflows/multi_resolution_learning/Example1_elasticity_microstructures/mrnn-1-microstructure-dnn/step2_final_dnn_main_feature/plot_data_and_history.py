import sys
import pandas as pd
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    if (len(sys.argv) == 1):
        print('please provide all_data_20200601081735.pickle or history_20200601081735.pickle file')
        exit(0)

    timemark = sys.argv[1].split('_')[-1].split('.pickle')[0]
    print('timemark:', timemark)

    history_file = 'history_' + timemark + '.pickle'
    all_data_file = 'all_data_' + timemark + '.pickle'
    print('loading data:', history_file, all_data_file)

    all_data = pickle.load(open(all_data_file, "rb"))
    history = pickle.load(open(history_file, "rb"))

    epoches = range(0, len(history['loss']))

    #----------------------plot 1---------------------------------------
    plt.clf()
    plt.semilogy(epoches, history['loss'], 'b', lw=1.0, label='Training')
    plt.semilogy(epoches, history['val_loss'], 'k', lw=1.0, label='Validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    # plt.axis('equal')
    plt.savefig('1dns-free-energy-learning-dnn.pdf', bbox_inches='tight', format='pdf')
    plt.show()

    #----------------------plot 2---------------------------------------
    plt.clf()
    plt.plot(all_data['test_label'], all_data['test_nn'], 'k.')
    xmin = min(min(all_data['test_label']), min(all_data['test_nn']))
    xmax = max(max(all_data['test_label']), max(all_data['test_nn']))
    plt.plot([xmin, xmax], [xmin, xmax], 'k-', lw=1.0)

    plt.axes().set_aspect('equal', 'box')
    plt.xlim([xmin, xmax])
    plt.ylim([xmin, xmax])
    plt.xlabel('$\Psi^0_{\mathrm{mech,DNS}}$')
    plt.ylabel('$\Psi^0_{\mathrm{mech,DNN}}$')
    plt.savefig('1dns-free-energy-test-dnn.pdf', bbox_inches='tight', format='pdf')
    plt.show()

    all_dns_data = all_data['train_label']
    all_dns_data.extend(all_data['val_label'])
    all_dns_data.extend(all_data['test_label'])

    all_nn_data = all_data['train_nn']
    all_nn_data.extend(all_data['val_nn'])
    all_nn_data.extend(all_data['test_nn'])

    reference_data_file = '../data/_home_xiaoxuan_comet_quench-20191012-60x60-c_large_8data.csv'
    sep = ','
    fields = ['index', 'Psi_me']
    selected_cols = pd.read_csv(reference_data_file, index_col=False, sep=sep, usecols=fields, skipinitialspace=True)[fields]

    ordered_all_dns_data = []
    ordered_all_nn_data = []

    # sorted all the data based on the frame number
    for ref0 in selected_cols['Psi_me']:
        # print(ref0)
        for j0 in range(0, len(all_dns_data)):
            # print(j0, len(all_dns_data))
            val0 = all_dns_data[j0]
            if (abs(ref0 - val0) < 1e-9):
                ordered_all_dns_data.append(all_dns_data[j0])
                ordered_all_nn_data.append(all_nn_data[j0])
                del all_dns_data[j0]
                del all_nn_data[j0]
                break

    #----------------------plot 3---------------------------------------
    plt.clf()
    plt.plot(selected_cols['index'], ordered_all_dns_data, 'k', linewidth=4, label='DNS')
    plt.plot(selected_cols['index'], ordered_all_nn_data, 'r', label='DNN')
    # plt.plot( selected_cols['index'], selected_cols['Psi_me'], 'b')
    plt.xlabel('frame number')
    plt.ylabel('$\Psi^0_{\mathrm{mech}}$')
    plt.xlim([0, 900])
    # plt.axis('equal')
    plt.legend()
    plt.savefig('1dns-free-energy-predict-all-dnn.pdf', bbox_inches='tight', format='pdf')
    plt.show()
