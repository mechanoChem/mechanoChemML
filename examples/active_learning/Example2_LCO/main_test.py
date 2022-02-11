#!/usr/bin/env python
import os
from active_learningLCO import Active_learning

workflow = Active_learning(os.path.dirname(__file__)+'/LCO_test.ini',test=True)
workflow.main_workflow()
