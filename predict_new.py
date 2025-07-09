import os 
import h5py as h5 
import numpy as np 
import matplotlib.pyplot as plt 
from math import pi 
from utils import ipv_wrapper

def predict_result(amat:str, f_factor:str):
    a_data = h5.File(amat)
    a_thetas =np.array(a_data["theta_center"][:])
    a_phis = np.array(a_data["phi_center"][:])
    a_matrix = np.array(a_data["a_matrix"][:])

    f_data = h5.File(f_factor) 
    f_thetas = np.array(f_data["theta_pos"][:])
    f_phis = np.array(f_data["azi"][:])
    f_vector = np.array(f_data["f_vector"])

    if len(a_thetas)!=len(f_thetas):
        raise ValueError("Differing number of zenith angles in matrices: {} in A, {} in F".format(len(a_thetas), len(f_thetas)))
    
    if len(a_phis)!=len(f_phis):
        raise ValueError("Differing number of zimuth angles in matrices: {} in A, {} in F".format(len(a_phis), len(f_phis)))
    
    if len(a_matrix)!=len(f_vector):
        raise ValueError("Incorrect shapes for A-Matrix and F-Vector ({} and {})".format(np.shape(a_matrix), np.shape(f_vector)))
    
    predicted = np.matmul(a_matrix, f_vector)

    theta_bins = np.array(a_data["theta_edge"])
    phi_bins = np.array(a_data["phi_edge"])
    th_center = 0.5*(theta_bins[1:] + theta_bins[:-1])
    ph_center =0.5*(phi_bins[1:] + phi_bins[:-1])

    reshaped = predicted.reshape(19, 29).T
    print(np.shape(theta_bins))
    visible = ipv_wrapper(th_center, ph_center, -pi/4, pi/2)
    print(np.shape(visible))
    #reshaped[visible<0]=0

    plt.pcolormesh(theta_bins, phi_bins, reshaped, vmin=0, vmax=0.5, cmap='inferno')
    plt.colorbar()
    plt.xlabel("cos(zenith)",size=14)
    plt.ylabel("Azimuth [deg]", size=14)
    #plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig("./plots/prediction.png", dpi=400)
    plt.show()

if __name__=="__main__":
    if True:
        import argparse
        parser = argparse.ArgumentParser(
                        prog='A-Matrix Builder')
        
        parser.add_argument('--amat')
        parser.add_argument('--f_factor')

        args = parser.parse_args()

        if not os.path.exists(args.amat):
            raise IOError("A-Matrix file not found: {}".format(args.amat))
        if not os.path.exists(args.f_factor):
            raise IOError("F-Factor File not found: {}".format(args.f_factor))
        

        predict_result(args.amat, args.f_factor)

