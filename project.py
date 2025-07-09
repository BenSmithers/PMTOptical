import numpy as np 
from utils import Irregular2DInterpolator
import os 
from math import pi 
import json 

PMT_X = 0.417
PMT_Y = 0.297
PMT_Z = 0.3 # what is it...

A = 0.254564663
B = 0.254110205
E = 0.5*(A+B)
C = 0.186002389 

x0 = np.array([
    PMT_X, 
    PMT_Y, 
    PMT_Z 
])

def load(ptf_datafile):
    if not os.path.exists(ptf_datafile):
        raise IOError("No such file {}".format(ptf_datafile))
    
    _obj = open(ptf_datafile, 'r')
    data = json.load(_obj)
    _obj.close()

    # need to shift these to PMT-centered coordinates 
    edge_xs = np.array(data["pmt0"]["xs"])
    edge_ys = np.array(data["pmt0"]["ys"])

    # grab the centers of these points
    center_xs = 0.5*(edge_xs[1:] + edge_xs[:-1])
    center_ys = 0.5*(edge_ys[1:] + edge_ys[:-1])
    xs, ys = np.meshgrid(center_xs, center_ys)

    zs = np.array(data["pmt0"]["zs"])
    # we have the **GANTRY** X/Y/Z 
    # we need to project those values to the 

    tilt = np.array(data["pmt0"]["tilt"])*pi/180
    
    rot = np.array(data["pmt0"]["rot"]) # weird freaking units... 
    aziumth = (rot + 75)*90/(11 + 75)



    det_eff = np.array(data["pmt0"]["det_eff"]).T

    saw_pmt = det_eff>0.05