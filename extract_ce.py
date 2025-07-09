

"""
This code is used to extract the collection efficiency information given...
    + a PTF measurement
    + a similarly-performed optical scan 

Functionality
    1. files for the above are loaded in
    2. An interpolator based on the PTF data is built
"""

import h5py as h5 
import os 
import json 
import numpy as np 
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import griddata
import matplotlib.pyplot as plt 
from math import pi

from utils import Irregular2DInterpolator

DEBUG = True

PMT_X = 0.417
PMT_Y = 0.297

A = 0.254564663
B = 0.254110205
C = 0.186002389 

def build_interpolator(ptf_data):
    if not os.path.exists(ptf_data):
        raise IOError("No such file {}".format(ptf_data))

    _obj = open(ptf_data, 'r')
    data = json.load(_obj)
    _obj.close()

    # transform these into PMT-centered coordinates 
    edge_xs = np.array(data["pmt0"]["xs"]) - PMT_X
    edge_ys = np.array(data["pmt0"]["ys"]) - PMT_Y

    # grab the centers of these points
    center_xs = 0.5*(edge_xs[1:] + edge_xs[:-1])
    center_ys = 0.5*(edge_ys[1:] + edge_ys[:-1])

    # now move into a theta/azimuth coordinate system 
    xs, ys = np.meshgrid(center_xs, center_ys)
    zs = C*np.sqrt( 1 - (xs/A)**2 - (ys/B)**2)
    rad = np.sqrt(xs**2 + ys**2 + zs**2)
    
    # cut out the NaNs (outside the PMT region)
    trim = np.logical_not(np.isnan(zs))
    theta = (zs/rad)[trim].flatten()
    azi = np.arctan2(ys, xs)[trim].flatten()

    

    # and get the detection efficiency at the relevant points
    det_eff = np.array(data["pmt0"]["det_eff"]).T
    bad_mask = np.isnan(det_eff)
    det_eff[bad_mask] = 0.0 
    det_eff = det_eff[trim].flatten()   

    plt.scatter(theta, azi*180/pi, c=det_eff, vmin=0, vmax=0.5, cmap='inferno')
    plt.colorbar()
    plt.xlabel("cos(zenith)",size=14)
    plt.ylabel("Azimuth [deg]", size=14)
    plt.savefig("./plots/scattered_points.png", dpi=400)
    plt.show()
    alt = Irregular2DInterpolator(
        theta, azi, det_eff, interpolation='nearest'
    )

    if DEBUG:

            # since we're no longer on a regular grid, we have to setup an irregular grid interpolator 
        theta_interp = np.linspace(0, 1, 100)
        azi_interp = np.linspace(-pi, pi, 100)
        alt_values = alt(theta_interp, azi_interp, grid=True)
        plt.pcolormesh(theta_interp, azi_interp*180/pi, alt_values.T, vmin=0, vmax=0.5, cmap='inferno')
        plt.colorbar()
        plt.xlabel("cos(zenith)",size=14)
        plt.ylabel("Azimuth [deg]", size=14)
        plt.savefig("./plots/unwrapped_PDE.png", dpi=400)
        plt.show()
    
    return alt


def extract_factor(spline:Irregular2DInterpolator, amat_file):
    if not os.path.exists(amat_file):
        raise IOError("No AMAT File {}".format(amat_file))
    
    data = h5.File(amat_file)
    # now, we evaluate the detector efficiencies 
    a_matrix = np.array(data["a_matrix"][:])
    print("a shape", np.shape(a_matrix))
    
    theta_edges = np.array(data["theta_edge"])
    phi_edges = np.array(data["phi_edge"])

    basic_a = np.diag(np.ones(len(a_matrix)))
    a_matrix[np.isnan(a_matrix)]= 1.0 # somehow there are diagonal elements that are nan

    toplot = np.log(1+a_matrix)
    #toplot = a_matrix
    print("{} - {}".format(np.min(toplot), np.max(toplot)))
    plt.pcolormesh(range(len(a_matrix)+1), range(len(a_matrix)+1), a_matrix,vmin=0, vmax=0.2, cmap='inferno')
    cbar = plt.colorbar()
    plt.title("Transmission Matrix", size=16)
    plt.xlabel("Injected Position [ID]",size=14)
    plt.ylabel("Hit Position [ID]", size=14)
    plt.savefig("./plots/a_matrix.png", dpi=400)
    #plt.hist(toplot.flatten(), np.linspace(0, 3, 1000))
    #plt.scatter(theta_pos, phi_pos,c=f_factor, vmin=0, vmax=1)
    plt.show()
    inverted_a = np.linalg.inv(a_matrix)

    plt.pcolormesh(range(len(a_matrix)+1), range(len(a_matrix)+1), inverted_a,vmin=-0.5, vmax=0.5, cmap='RdBu')
    plt.colorbar()
    plt.title("Inverted A-Matrix", size=16)
    plt.xlabel("Injected Position [ID]",size=14)
    plt.ylabel("Hit Position [ID]", size=14)
    plt.savefig("./plots/inv_a_matrix.png", dpi=400)
    plt.show()

    # we invert it, but it's messy 
    # so we clean up the eigenvalues and rebuild the matrix
    if False:
        eigen, inv_t = np.linalg.eig(inverted_a)
        eps = 1e-6
        mask = eigen>eps 
        not_mask = np.logical_not(mask)
        trans_vec = np.zeros_like(eigen)
        trans_vec[mask] = eigen
        trans_vec[not_mask] = np.min(eigen[mask])*10


        rebuilt_a = np.matmul(inv_t, np.matmul(np.diag(eigen), np.linalg.inv(inv_t)))
    
    if False:
        plt.pcolormesh(range(len(a_matrix)+1), range(len(a_matrix)+1), np.log(inverted_a+1), vmin=0, vmax=0.05, cmap='inferno')
        plt.colorbar()
        plt.title("log(1+A)", size=16)
        plt.xlabel("cos(zenith)",size=14)
        plt.ylabel("Azimuth [deg]", size=14)
        plt.savefig("./plots/inverted_and_rebuilt.png", dpi=400)
        #plt.hist(toplot.flatten(), np.linspace(0, 3, 1000))
        #plt.scatter(theta_pos, phi_pos,c=f_factor, vmin=0, vmax=1)
        plt.show()

    theta_pos = np.array(data["theta_center"][:])
    phi_pos = np.array(data["phi_center"][:])

    print("{} - {}".format(np.min(theta_pos), np.max(theta_pos)))
    print("{} - {}".format(np.min(phi_pos), np.max(phi_pos)))

    print("{} - {}".format(np.shape(theta_pos), np.shape(phi_pos)))
    evaluate_de = spline(theta_pos, phi_pos, grid=False)

    plt.scatter(theta_pos, phi_pos*180/pi, c=evaluate_de, vmin=0, vmax=0.5, cmap='inferno')
    plt.title("Interpolated PDE")
    plt.xlabel("cos(zenith)",size=14)
    plt.ylabel("Azimuth [deg]", size=14)
    plt.tight_layout()
    plt.savefig("./plots/interpolated_pde.png", dpi=400)
    plt.show()

    f_factor = np.matmul(inverted_a, evaluate_de)

    outname = os.path.join(os.path.dirname(__file__), "processed_data",
                           "f_factor.h5")
    out = h5.File(outname, 'w')
    out.create_dataset("f_vector", data=f_factor)
    out.create_dataset("theta_pos", data=theta_pos)
    out.create_dataset("azi",data=phi_pos)
    out.close()

    boring_f = np.matmul(basic_a, evaluate_de)

    print(np.min(f_factor), np.max(f_factor))

    print(len(theta_edges), len(phi_edges), np.shape(f_factor))    
    #plt.scatter(theta_pos*180/pi , phi_pos*180/pi, c=f_factor, vmin=0, vmax=0.5, cmap="inferno", s=40)
    plt.pcolormesh(theta_edges, phi_edges*180/pi, f_factor.reshape(19,29).T, vmin=0, vmax=0.5, cmap='inferno')
    plt.colorbar()
    plt.xlabel("cos(zenith)",size=14)
    plt.ylabel("Azimuth [deg]", size=14)
    #plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig("./plots/f_factor.png", dpi=400)
    plt.show()

    plt.pcolormesh(theta_edges, phi_edges*180/pi, 1- boring_f.reshape(19,29).T/f_factor.reshape(19,29).T,vmin=-0.1, vmax=0.1, cmap='RdBu_r')
    plt.colorbar()
    plt.xlabel("cos(zenith)",size=14)
    plt.ylabel("Azimuth [deg]", size=14)
    plt.title("Comparison", size=16)
    #plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig("./plots/fractional.png", dpi=400)
    plt.show()

if __name__=="__main__":
    if True:
        import argparse
        parser = argparse.ArgumentParser(
                        prog='A-Matrix Builder')
        
        parser.add_argument('--amat')
        parser.add_argument('--ptf_file')

        args = parser.parse_args()

        spline = build_interpolator(args.ptf_file)
        extract_factor(spline, args.amat)
