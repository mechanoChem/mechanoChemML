#!/usr/bin/env python
import os
import tensorflow as tf
from mechanoChemML.workflows.active_learning.active_learning import Active_learning

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

input_path = os.path.dirname(__file__)
if input_path: # when it is not ''
    input_path += '/NiAl_test.ini'
else:
    input_path = 'NiAl_test.ini' 

workflow = Active_learning(input_path,test=True)
workflow.main_workflow()
