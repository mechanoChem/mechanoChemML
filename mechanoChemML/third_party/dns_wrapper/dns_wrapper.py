import numpy as np
import argparse, sys, os
import ctypes
import importlib
ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)
from available_dns_modules import DNS_MODULES_PATH

class DNS_Wrapper:
    """
    Example of a DNS wrapper, which provides a python interface to running direct numerical simulations written with other programming languages. 
    For example, one can use the pybind11 library to create a python module file based on the C++ code. The module will be loaded via
    self.load_example() function.
    """

    def __init__(self):
        """ initialize the class"""
        self.parse_sys_args()
        self.load_example()

    def parse_sys_args(self):
        """ parse arguments """
        parser = argparse.ArgumentParser(description='DNS python wrapper to run simulations', prog="'" + (sys.argv[0]) + "'")
        parser.add_argument('-e', '--example', type=str, default='diffusion_dynamic', choices=DNS_MODULES_PATH.keys(), help='examples to run')
        args = parser.parse_args()
        self.args = args

    def load_example(self):
        """ add the path to the python path and load the module """
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/' + DNS_MODULES_PATH[self.args.example])
        # load the python module 
        self.DNS_module = importlib.import_module(self.args.example)
        # related to the specific DNS software
        self.DNS_example = self.DNS_module.PYmechanoChem(['./main', 'parameters.prm'])


    def setup_problem(self):
        if self.args.example.find('IGA') >=0 :
            self.DNS_example.setup_mechanoChemIGA()
        elif self.args.example.find('FEM') >=0 :
            self.DNS_example.setup_mechanoChemFEM()
        else :
            print("***WARNING***: I cannot determine the specified physics based library.")
            exit(0)

    def simulate(self):
        """
        run simulation
        """
        self.DNS_example.simulate()

    def set_parameter(self, variable_name, variable_value):
        """
        update parameter values
        """
        self.DNS_example.setParameter(variable_name, variable_value)


if __name__ == "__main__" :
    problem = DNS_Wrapper()
    problem.setup_problem()
    problem.simulate()
    problem.set_parameter(["Diffusion reaction: one species", "dDirichletUc"], "0.2")
    problem.simulate() 
