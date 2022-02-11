#!/usr/bin/env python
import os
from active_learningLCO import Active_learning

workflow = Active_learning(os.path.dirname(__file__)+'/LCO_free_energy.ini')
workflow.main_workflow()
