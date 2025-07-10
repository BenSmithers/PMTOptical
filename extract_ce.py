

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

DEBUG = False

PMT_X = 0.417-0.0265
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
    
    # cut out the NaNs (outside the PMT region)
    trim = np.logical_not(np.isnan(zs))

    # and get the detection efficiency at the relevant points
    det_eff = np.array(data["pmt0"]["det_eff"]).T #/np.array(data["monitor"]["det_eff"]).T
    bad_mask = np.isnan(det_eff)
    det_eff[bad_mask] = 0.0 

    plt.scatter(xs.flatten(), ys.flatten(), c=det_eff.flatten(), cmap='inferno')
    plt.colorbar()
    plt.xlabel("X [m]",size=14)
    plt.ylabel("Y [m]", size=14)
    plt.savefig("./plots/scattered_points.png", dpi=400)
    plt.show()
    print(np.shape(center_xs), np.shape(center_ys), np.shape(det_eff))
    alt = RectBivariateSpline(
        center_xs, center_ys, det_eff.T,
    )

    if DEBUG:

            # since we're no longer on a regular grid, we have to setup an irregular grid interpolator 
            
        x_interp = np.linspace(-0.255, 0.255, 100)
        y_interp = np.linspace(-0.255, 0.255, 101)
        alt_values = alt(x_interp, y_interp, grid=True)
        plt.pcolormesh(x_interp, y_interp, alt_values.T, vmin=0, vmax=0.5, cmap='inferno')
        plt.colorbar()
        plt.xlabel("x[m]",size=14)
        plt.ylabel("y[m]", size=14)
        plt.savefig("./plots/unwrapped_PDE.png", dpi=400)
        plt.show()
    
    return alt


def extract_factor(spline:Irregular2DInterpolator, amat_file):
    if not os.path.exists(amat_file):
        raise IOError("No AMAT File {}".format(amat_file))
    
    data = h5.File(amat_file)
    # now, we evaluate the detector efficiencies 
    a_matrix = np.array(data["a_matrix"][:])
    a_matrix = a_matrix.T
    for i in range(len(a_matrix)):
        if np.sum(a_matrix)<1e-3:
            a_matrix[i][i]=1 
    a_matrix = a_matrix.T
    
    x_edges = np.array(data["x_edge"])
    y_edges = np.array(data["y_edge"])

    basic_a = np.diag(np.ones(len(a_matrix)))
    a_matrix[np.isnan(a_matrix)]= 1.0 # somehow there are diagonal elements that are nan

    toplot = np.log(1+a_matrix)
    print("{} - {}".format(np.min(toplot), np.max(toplot)))
    plt.pcolormesh(range(len(a_matrix)+1), range(len(a_matrix)+1), a_matrix.T,vmin=0, vmax=0.2, cmap='inferno')
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

    x_pos = np.array(data["x_center"][:])
    y_pos = np.array(data["y_center"][:])

    print("{} - {}".format(np.min(x_pos), np.max(x_pos)))
    print("{} - {}".format(np.min(y_pos), np.max(y_pos)))

    print("{} - {}".format(np.shape(x_pos), np.shape(y_pos)))
    evaluate_de = spline(x_pos, y_pos, grid=False)

    plt.scatter(x_pos, y_pos, c=evaluate_de, vmin=0, vmax=0.5, cmap='inferno')
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
    out.create_dataset("x_pos", data=x_pos)
    out.create_dataset("y_pos",data=y_pos)
    out.close()

    boring_f = np.matmul(basic_a, evaluate_de)

    print(np.min(f_factor), np.max(f_factor))

    print(len(x_edges), len(y_edges), np.shape(f_factor))    
    #plt.scatter(theta_pos*180/pi , phi_pos*180/pi, c=f_factor, vmin=0, vmax=0.5, cmap="inferno", s=40)
    plt.pcolormesh(x_edges, y_edges, f_factor.reshape(39,40).T, vmin=0, vmax=0.5, cmap='inferno')
    plt.colorbar()
    plt.xlabel("cos(zenith)",size=14)
    plt.ylabel("Azimuth [deg]", size=14)
    #plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig("./plots/f_factor.png", dpi=400)
    plt.show()

    plt.pcolormesh(x_edges, y_edges, 1- (boring_f.reshape(39,40).T/f_factor.reshape(39,40).T),vmin=-0.1, vmax=0.1, cmap='RdBu_r')
    plt.colorbar()
    plt.xlabel("cos(zenith)",size=14)
    plt.ylabel("Azimuth [deg]", size=14)
    plt.title("Comparison", size=16)
    #plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig("./plots/fractional.png", dpi=400)
    plt.show()

    re_converted_pde = np.matmul(a_matrix, f_factor)
    plt.pcolormesh(x_edges, y_edges,re_converted_pde.reshape(39,40).T, cmap='inferno')
    plt.colorbar()
    plt.xlabel("x [m]",size=14)
    plt.ylabel("y [m]", size=14)
    #plt.title("Comparison", size=16)
    #plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig("./plots/re_fold_f_factor.png", dpi=400)
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
