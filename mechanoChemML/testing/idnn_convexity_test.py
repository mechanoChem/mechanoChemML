#!/usr/bin/env python

import numpy as np
import sys, os

from mechanoChemML.src.gradient_layer import Gradient
from mechanoChemML.src.idnn import IDNN, find_wells

def idnn_convexity_test():
    # Set up simple IDNN with known weights
    w1 = np.array([[10,-10,11,-11,0,0,0,0],
                   [0,0,0,0,10,-10,11,-11]])
    b1 = np.array([0,0,0,0,0,0,0,0])
    w2 = np.array([[-1.15],[-1.15],[1],[1],[-1.15],[-1.15],[1],[1]])
    b2 = np.array([0])

    idnn = IDNN(2,[8],final_bias=True)
    idnn.layers[1].set_weights([w1,b1])
    idnn.layers[2].set_weights([w2,b2])

    # Create random selection of test points within [0,0.25]x[0,0.25]
    # Function will return points within a well
    points = 0.25*np.random.rand(100,2)
    wells = find_wells(idnn,points,rereference=False)
    
    # Points within [0,~0.10303402]x[0,~0.10303402] should be in a well,
    # all other points should not (the cutoff is not to machine precision,
    # so we cut it a little slack)
    success = True
    for well in wells:
        if well[0] > 0.10303403 or well[1] > 0.10303403:
            success = False
            
    for point in points:
        if point not in wells and point[0] < 0.10303402 and point[1] < 0.10303402:
            success = False
        
    if success:
        print('Find wells test: passed')
    else:
        print('Find wells test: failed')


if __name__ == "__main__":
    idnn_convexity_test()
