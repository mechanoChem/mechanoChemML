#!/usr/bin/env python
import os
from active_learning import Active_learning

workflow = Active_learning(os.path.dirname(__file__)+'/NiAl_free_energy.ini')
workflow.main_workflow()
