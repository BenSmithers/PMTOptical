from scipy.interpolate import griddata, RectBivariateSpline
import numpy as np 
from math import log10,sqrt 


A = 0.254564663
B = 0.254110205
E = 0.5*(A+B)
a = E
b = E
c= 0.186002389 
C = c
def analytic_norm(theta, phi):
    norm_y = 1.0/np.sqrt(
        1 + ((b/a)*np.cos(phi)/np.sin(phi))**2 + ((b/c)*(np.tan(theta)/np.sin(phi)))**2
    )

    norm_y = np.ones_like(phi)
    
    if True:
        if isinstance(phi, np.ndarray):
            ones = np.ones_like(phi)
            ones[phi<0]*=-1
            norm_y*=-ones
        else:
            if phi<0:
                norm_y*=-1

    norm_z = norm_y*(b/c)*np.tan(theta)/np.sin(phi)
    norm_x = norm_y*(b/a)*(np.cos(phi)/np.sin(phi))

    mag = np.sqrt(norm_x**2 + norm_y**2 + norm_z**2)
    norm_x/=mag
    norm_y/=mag
    norm_z/=mag

    return norm_x, norm_y, norm_z # make sure we always aim up 


def ipv_wrapper(ctheta, phi, zenith_angle, azimuth_angle, meshes=False):
    # r**2 - (rcos(theta))**2 = x**2 + (xtan(phi))**2  
    # r**2  = (x**2 + (xtan(phi))**2)/( 1-cos(theta)**2)
    # rcos(theta) = z
    # xtan(phi) =  y
    """
        1 = (x/a)**2 + (xtan(phi)/b)**2 ) +  ( rcos(theta)/c )**2
        
        1 = (x/a)**2 + (xtan(phi)/b)**2 ) +  ((x**2 + (xtan(phi))**2)/( 1-cos(theta)**2)) ( cos(theta)/c )**2
        x = sqrt[ 1/[ (1/a**2) + (tan(phi)/b)**2 ) + ((1 + tan(phi)**2)/(1-cos(theta)**2))(cos(theta)/c)**2 ] ]
        
        sqrt(x**2 + y**2 + z**2 ) = z/cos(theta)
    """
    if meshes:
        ctheta = ctheta
        phi = phi
    else:    
        ctheta, phi= np.meshgrid(ctheta, phi)

    xs = np.sqrt( 1/( (1/(E**2)) + (np.tan(phi)/E)**2 + ((1 + np.tan(phi)**2)/(1-ctheta**2) )*(ctheta/C)**2 ))
    ys = xs*np.tan(phi)

    return is_point_visible(xs, ys, zenith_angle, azimuth_angle, True)

def is_point_visible(x_pos, y_pos, zenith_angle, azimuth_angle, meshes=False)->np.ndarray:
    """
    
    """
    n_x = np.cos(azimuth_angle)*np.cos(zenith_angle)
    n_y = np.sin(azimuth_angle)*np.cos(zenith_angle)
    n_z = np.sin(zenith_angle)

    if meshes:
        xmesh = x_pos
        ymesh = y_pos
    else:    
        xmesh, ymesh= np.meshgrid(x_pos, y_pos)
    z_pos = c*((1 - (xmesh/a)**2 - (ymesh/b)**2)**0.5)
    phi = np.arctan2(ymesh, xmesh)

    theta = np.arcsin(z_pos/c)

    pmt_nx, pmt_ny, pmt_nz = analytic_norm(theta, phi)

    visible = (n_x*pmt_nx + n_y*pmt_ny + n_z*pmt_nz) 
    visible[np.isnan(visible)]=0
    return visible


class Irregular2DInterpolator:
    """
        This is used to make a 2D interpolator given a set of data that do not lie perfectly on a grid.
        This is done using scipy griddata and scipy RectBivariateSpline 
        interpolation can be `linear` or `cubic` 
        if linear_x/y, then the interpolation is done in linear space. Otherwise, it's done in log space
            setting this to False is helpful if your x/y values span many orders of magnitude 
        if linear_values, then the values are calculated in linear space. Otherwise they'll be evaluated in log space- but returned in linear space 
            setting this to False is helpful if your data values span many orders of magnitude 
        By default, nans are replaced with zeros. 
    """
    def __init__(self, xdata:np.ndarray, 
                 ydata:np.ndarray,
                   values:np.ndarray, linear_x = True, linear_y = True, linear_values=True,
                   replace_nans_with= 0.0, interpolation='linear'):

        self._nomesh_x = xdata
        self._nomesh_y = ydata 
        self._values = values if linear_values else np.log10(values)
        self._linear_values = linear_values
        if linear_x:
            self._xfine = np.linspace(min(self._nomesh_x), 
                                      max(self._nomesh_x), 
                                      int(sqrt(len(self._nomesh_x)))*2, endpoint=True)
        else:
            self._xfine = np.logspace(log10(min(self._nomesh_x)), 
                                      log10(max(self._nomesh_x)), 
                                      int(sqrt(len(self._nomesh_x)))*2, endpoint=True)

        
        if linear_y:
            self._yfine = np.linspace(min(self._nomesh_y), 
                                      max(self._nomesh_y), 
                                      int(sqrt(len(self._nomesh_y)))*2, endpoint=True)
        else:
            self._yfine = np.logspace(log10(min(self._nomesh_y)), 
                                      log10(max(self._nomesh_y)), 
                                      int(sqrt(len(self._nomesh_y)))*2, endpoint=True)


        mesh_x, mesh_y = np.meshgrid(self._xfine, self._yfine)

        # usee grideval to evaluate a grid of points 
        grid_eval = griddata(
            points=np.transpose([self._nomesh_x, self._nomesh_y]),
            values=self._values, 
            xi=(mesh_x, mesh_y),
            method=interpolation
        )
        
        # if there are any nans, scipy 
        if np.any(np.isnan(grid_eval)):
            print("Warning! Nans were found in the evaluation of griddata - we're replacing those with zeros")
        grid_eval[np.isnan(grid_eval)] = replace_nans_with

        # and then prepare an interpolator 
        self._data_int = RectBivariateSpline(
            self._xfine, 
            self._yfine, 
            grid_eval.T
        )

    def __call__(self, xs, ys, grid=False):
        if self._linear_values:
            return self._data_int( xs, ys ,grid=grid)
        else:
            return 10**self._data_int( xs, ys , grid=grid)