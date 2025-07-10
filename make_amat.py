import numpy as np
import uproot 
import h5py as h5 
import os 
from tqdm import tqdm 
from enum import Enum
from math import pi 
import numpy as np 
from utils import is_point_visible

import matplotlib.pyplot as plt 


from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d


# reimplement that thing from earlier 
DEBUG = False
SCALER = 10000.0

class PhotoStatus(Enum):
    transmission = 0
    reflection=1
    abs_photocathode = 2
    abs_glass = 3
    abs_dynode =4 
    abs_other = 5
    departure = 6
    other = 7

A = 0.254564663
B = 0.254110205
C = 0.186002389 

def diff_area(theta, phi):
    SCALER = 0.05 # puts this roughly between 0 and 1
    temp = np.sqrt(
        (A*B*np.cos(theta)*np.sin(theta))**2 
        + (C**2)*(np.sin(theta)**4)*( (B*np.cos(phi))**2 +(A*np.sin(phi))**2 )
    )/SCALER

    temp[np.isnan(temp)] = 0
    return temp

def get_h5file(filename):
    """
        Takes a root file name and renames it to an hdf5 file 
    """

    fullpath, extension = os.path.splitext(filename)
    new_name = fullpath + ".h5"
    return new_name

def cache_numat(filename, ignore_existing=False):
    newname = get_h5file(filename)
    print(os.path.exists(newname))
    print(ignore_existing)
    print( (not os.path.exists(newname)) and (not ignore_existing) )
    if (not os.path.exists(newname)) or ignore_existing:
        
                
        init_x = []
        init_y = []
        init_z = []

        final_x = []
        final_y = []
        final_z = []
        ids = []
        meta_data = []

        data = uproot.open(filename)
        
        all_photons = data["Photons_0;1"]

        photonID = all_photons["Photon_ID"].array(library="numpy")
        steps = all_photons["Step_Status"].array(library="numpy")
        posx = all_photons["PosX"].array(library="numpy")
        posy = all_photons["PosY"].array(library="numpy")
        posz = all_photons["PosZ"].array(library="numpy")
        last_id = -1 

        start_x = -1 
        start_y = -1 
        start_z = -1 
        for index in tqdm(range(len(photonID))):
            if photonID[index]!=last_id:
                start_x = posx[index]
                start_y = posy[index]
                start_z = posz[index]
                last_id = photonID[index]
            if steps[index]==PhotoStatus.abs_photocathode.value:
                init_x.append(start_x)
                init_y.append(start_y)
                init_z.append(start_z)
                final_x.append(posx[index])
                final_y.append(posy[index])
                final_z.append(posz[index])
                ids.append(photonID[index])
        meta_data.append( 0*pi/180)

        all_data = {
            "init_x":init_x,
            "init_y":init_y,
            "init_z":init_z,
            "final_x":final_x,
            "final_y":final_y,
            "final_z":final_z,
            "ids":ids,
            "thetas":meta_data
        }

        datafile = h5.File(newname, 'w')
        for key in all_data:
            datafile.create_dataset(key, data=all_data[key])
        datafile.close()

        return all_data
    else:
        datafile = h5.File(newname, 'r')
        return {
            key: np.array(datafile[key]) for key in datafile.keys()
        }
                
 

def cache_matrix(filename, ignore_existing=False):
    """
        Loads the photonics matrix if it exists 
        Otherwise, it'll rebuild it. Will always rebuild it if `ignore_existing` is true 

    """
    newname = get_h5file(filename)
    if (not os.path.exists(newname)) and (not ignore_existing):
        
                
        init_x = []
        init_y = []
        init_z = []

        final_x = []
        final_y = []
        final_z = []
        ids = []
        meta_data = []

        data = uproot.open(filename)
        

        # we don't actually want these ones though... can we step these forward?
        """
            Options 
                + use external knowledge of the angle, manually step the photons forward using the initial positions (bad - easy to lose that info). Actually - we'll need the angle anyways 
                + use the _second_ position in the position array (might be slow -  yeah way too slow)
                + modify the root files? 

                SOLUTION - use the first few or so to get the angular information. Then we apply the little step once from the initial values 

                Aaah lets extract the angle from the first and second entries 
        """
        initial_x = data["Photons_Master;1"]["PosX_Initial"].array()
        initial_y = data["Photons_Master;1"]["PosY_Initial"].array()
        initial_z = data["Photons_Master;1"]["PosZ_Initial"].array()
        

        for ip, photon_key in tqdm(enumerate(data.keys())):  
                        
            if ip%200==0:
                print("collected {} photons".format(len(init_x)))
            if ip==0:
                print("Skipping ", photon_key)
                continue 
            photons = data[photon_key]["Photon_ID"].array()
            px =  data[photon_key]["PosX"].array()
            py =  data[photon_key]["PosY"].array()
            pz =  data[photon_key]["PosZ"].array()
            status = data[photon_key]["Step_Status"].array()

            init_pos = np.array([
                initial_x[ip-1], initial_y[ip-1], initial_z[ip - 1]
            ])
            # it doesn't matter what happens here, this is the first step after photon initialization. 
            # This is the _new_ initial location on the photocathode, and this *must* be shared by all of the photons! 
            # we can now calculate the injection angle, which we will save as some metadata
            very_first = np.array([
                px[0], py[0], pz[0]
            ])

            # theta was the angle relative to the Z-axis 
            step = very_first - init_pos
            meta_data.append(np.arctan2(np.sqrt(step[0]**2 + step[1]**2), step[2]))

            mask = status==PhotoStatus.abs_photocathode.value

            final_x += px[mask].tolist()
            final_y += py[mask].tolist()
            final_z += pz[mask].tolist()

            if False:
                fig = plt.figure()
                ax = plt.axes(projection="3d")
                ax.scatter(px[mask],py[mask],pz[mask])
                ax.set_xlim([-0.250, 0.250])
                ax.set_xlim([-0.250, 0.250])
                plt.show()

            ids += photons[mask].tolist()

            init_x  += (very_first[0]*np.ones_like(px[mask].tolist())).tolist()
            init_y  += (very_first[1]*np.ones_like(py[mask].tolist())).tolist()
            init_z  += (very_first[2]*np.ones_like(pz[mask].tolist())).tolist()            

        all_data = {
            "init_x":init_x,
            "init_y":init_y,
            "init_z":init_z,
            "final_x":final_x,
            "final_y":final_y,
            "final_z":final_z,
            "ids":ids,
            "thetas":meta_data
        }

        datafile = h5.File(newname, 'w')
        for key in all_data:
            datafile.create_dataset(key, data=all_data[key])
        datafile.close()

        return all_data
    else:
        datafile = h5.File(newname, 'r')
        return {
            key: np.array(datafile[key]) for key in datafile.keys()
        }

def encode_data(a_tensor, thetas, phis):
    """
        Encodes the A-tensor (4D) into a 2D matrix. 
    """
    n_theta = len(thetas)
    n_phi = len(phis)

    # let's say its theta*phi
    encoded = np.zeros((n_theta*n_phi, n_theta*n_phi))
    encoded_thetas = np.zeros(n_theta*n_phi)
    encoded_phis = np.zeros(n_theta*n_phi)
    hit = np.ones(len(encoded_thetas)).astype(bool)
    hit2 = np.ones(len(encoded_thetas)).astype(bool)
    for i in range(n_theta):
        for j in range(n_phi):
            encoded_thetas[i*n_phi + j] = thetas[i]
            encoded_phis[i*n_phi + j] = phis[j]
            for k in range(n_theta):
                for l in range(n_phi):
                    hit[i*n_phi + j] = False
                    hit2[k*n_phi + l] = False
                    encoded[i*n_phi + j][k*n_phi + l] = a_tensor[i][j][k][l]

    print("{} missed".format(np.sum(hit.astype(int))))   
    print("{} missed".format(np.sum(hit2.astype(int))))                    
    return encoded, encoded_phis, encoded_thetas

def build_amat(filename):
    """
        Load the data in, fill in an A-matrix, and weight it

        Returns the A-matrix (which is more like an A-tensor, 4D)
    """
    
    # Take the raw generation values and extrapolate out some bin edges
    raw_thetas =np.linspace(0, 1, 20)    
    raw_phis = np.linspace(-pi, pi, 30)
    xbin = np.linspace(-0.255, 0.255, 60)
    ybin = np.linspace(-0.255, 0.255, 61)

    xcenter = 0.5*(xbin[1:] + xbin[:-1])
    ycenter = 0.5*(ybin[1:] + ybin[:-1])

    xmesh, ymesh = np.meshgrid(xcenter, ycenter)
    zmesh = C*np.sqrt(1 - (xmesh/A)**2 - (ymesh/B)**2)
    
    bin_area_weight = np.sqrt(xmesh**2 + ymesh**2 + zmesh**2)/(zmesh)
    bin_area_weight= bin_area_weight.T

    #theta_bins = np.arange(raw_thetas[0] - th_step*0.5, raw_thetas[-1]+th_step*0.51, th_step)
    #phi_bins = np.arange(raw_phis[0]-ph_step*0.5, raw_phis[-1] +ph_step*0.51, ph_step)

    

    data = cache_numat(filename, False)

    injected_th = np.mean(data["thetas"])
    print("Injected angle: {}".format(injected_th*180/pi))
    
    initial_x = np.array(data["init_x"])/1000
    initial_y =  np.array(data["init_y"])/1000
    initial_z = np.array(data["init_z"])/1000
    init_r = np.sqrt(initial_x**2 + initial_y**2 + initial_z**2)
    

    # z = r*cos(theta)
    init_theta = initial_z / init_r
    init_azi = np.arctan2(initial_y, initial_x)

    hit_x =  np.array(data["final_x"])/1000
    hit_y =  np.array(data["final_y"])/1000

    print("{} - {}".format(np.min(hit_x), np.max(hit_x)))


    hit_z =  np.array(data["final_z"])/1000
    hit_r = np.sqrt(hit_x**2 + hit_y**2 + hit_z**2)

    hit_theta = hit_z / hit_r
    hit_azi = np.arctan2(hit_y, hit_x)
    #hit_weights = diff_area(np.arccos(hit_theta), hit_azi)
    hit_weights =  is_point_visible(initial_x, initial_y, -90*pi/180 , pi/2, True)
    hit_weights[hit_weights<0]= 0
    hit_weights[hit_weights>0] = 1.0



    print("{} - {}".format(np.min(hit_weights), np.max(hit_weights)))
    #print("{} - {}".format(np.nanmin(hit_weights), np.nanmax(hit_weights)))
    histo = np.histogram2d(hit_x, hit_y, bins=(xbin, ybin), weights=hit_weights)[0]*bin_area_weight
    plt.pcolormesh(xbin ,ybin ,np.log10(histo.T))
    plt.xlabel("X [m]", size=14)
    plt.ylabel("Y [m]", size=14)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("./plots/hitmap.png", dpi=400)
    plt.show()

    # other hitmap 

    histo = np.histogram2d(hit_x, hit_y, bins=(xbin, ybin), weights=hit_weights)[0]*bin_area_weight
    plt.pcolormesh(xbin, ybin,np.log10(histo.T))
    plt.xlabel("x [m]", size=14)
    plt.ylabel("y [m]", size=14)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


    photons = np.histogramdd(
        sample = (
            initial_x, initial_y, hit_x, hit_y
        ),
        bins = (
            xbin, ybin, xbin, ybin
        ),
        weights=hit_weights
    )[0]
    prenorm = np.zeros_like(photons)

    for ix in range(len(photons)):
        for iy in range(len(photons[0])):

            if False : # np.nansum(photons[ix][iy])==0 or np.isnan(np.sum(photons[ix][iy])):
                #prenorm[ix][iy] = np.ones_like(photons[ix][iy])
                prenorm[ix][iy][ix][iy] = 1.0
            else:
                prenorm[ix][iy] += bin_area_weight[ix][iy]*bin_area_weight*photons[ix][iy]/np.nansum(photons[ix][iy])
            # if there were none detected at this angle, aah... well we still need this to be invertible down the line
            

            if DEBUG : #and ix%11==0 and iy%11==0 :
                if ix%2==1 or iy%2==1:
                    continue

                if np.sum(photons[ix][iy])<1:
                    continue
                
                # count good! 
                plt.clf()
                plt.pcolormesh(xbin, ybin,np.log10(prenorm[ix][iy].T))
                plt.plot(xbin[ix], ybin[iy], 'rd')
                #plt.title("{} - {}".format(ix, iy))
                plt.xlabel(r"x [m]")
                plt.ylabel("y [m]")
                plt.colorbar()
                plt.show()

    # now we need to encode this data into a simpler matrix

    
    print("{} - {} - {}".format(np.shape(xcenter), np.shape(ycenter), np.shape(prenorm)))
    encoded_a,encodey, encodex = encode_data(prenorm, xcenter, ycenter)

    out_file = os.path.join(
        os.path.dirname(__file__),
        "processed_data",
        "a_matrix_{:.2f}deg.h5".format(injected_th*180/pi)
    )
    dataset = h5.File(out_file, 'w')
    dataset.create_dataset("a_matrix", data=encoded_a)
    dataset.create_dataset("x_center", data=encodex)
    dataset.create_dataset("y_center", data=encodey)
    dataset.create_dataset("x_edge", data=xbin)
    dataset.create_dataset("y_edge", data=ybin)

    return prenorm

if __name__=="__main__":
    if True:
        import argparse
        parser = argparse.ArgumentParser(
                        prog='A-Matrix Builder')
        
        parser.add_argument('filename')
        args = parser.parse_args()
        build_amat(args.filename)

    else:
        phis = np.linspace(0, 2*pi, 200)
        thetas = np.linspace(0, pi, 201)

        mphi, mtheta = np.meshgrid(phis, thetas)
        areas = diff_area(mtheta, mphi)
        print(np.max(areas))

        
        plt.pcolormesh(phis*180/pi, thetas*180/pi, areas, vmin=0, vmax=1, cmap='inferno')
        plt.colorbar()
        plt.show()


