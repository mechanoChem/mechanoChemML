import numpy as np
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import mechanoChemML.src.pde_layers as pde_layers
import tensorflow as tf
import datetime
import pandas as pd

"""
l2 error is computed in the main_test_min_max.py 
this script is only for plotting purpose.
"""

nn_pred = None
for f1 in sys.argv[1:] :
    _tmp = pd.read_csv(f1)
    if nn_pred is None :
        nn_pred = _tmp 
    else:
        nn_pred = pd.concat([nn_pred, _tmp], ignore_index=True)
        # nn_pred.append(_tmp, ignore_index=True)
    # print(nn_pred)

# nn_pred.dropna(subset = ["l2_error"], inplace=True)
print('all:', nn_pred.describe().transpose()[['count','mean','std']])
# print(nn_pred)
nn_pred_with_neumann = nn_pred[nn_pred['Neumann_min_1'] != '--']

nn_pred_with_neumann['Neumann_min_1'] = nn_pred_with_neumann['Neumann_min_1'].astype(float)
nn_pred_with_neumann['Neumann_min_2'] = nn_pred_with_neumann['Neumann_min_2'].astype(float)
nn_pred_with_neumann['Neumann_max_1'] = nn_pred_with_neumann['Neumann_max_1'].astype(float)
nn_pred_with_neumann['Neumann_max_2'] = nn_pred_with_neumann['Neumann_max_2'].astype(float)

# nn_pred_with_neumann.reset_index(drop=True)
# print(nn_pred_with_neumann)
print('with Neumann: ', nn_pred_with_neumann.describe().transpose()[['count','mean','std']])
nn_pred_without_neumann = nn_pred[nn_pred['Neumann_min_1'] == '--']
# nn_pred_without_neumann.reset_index(drop=True)
# print(nn_pred_without_neumann)
print('without Neumann:', nn_pred_without_neumann.describe().transpose()[['count','mean','std']])

print('all:', nn_pred['l2_error'].describe()[['count', 'mean','std']])
print('with Neumann: ', nn_pred_with_neumann['l2_error'].describe()[['count', 'mean','std']])
print('without Neumann:', nn_pred_without_neumann['l2_error'].describe()[['count','mean','std']])
