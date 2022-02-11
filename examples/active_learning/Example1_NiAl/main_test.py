#!/usr/bin/env python
import os
from active_learning import Active_learning

workflow = Active_learning(os.path.dirname(__file__)+'/NiAl_test.ini',test=True)
workflow.main_workflow()
