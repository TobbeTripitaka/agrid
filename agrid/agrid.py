#!/usr/bin/env python3

# Tobias Staal 2019
# tobias.staal@utas.edu.au
# version = '0.1.0'

# https://doi.org/10.5281/zenodo.2553966
#
#MIT License#

#Copyright (c) 2019 Tobias Stål#

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:#

# The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.#

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.#

import os
import sys
import re
import json
import numpy as np
import xarray as xr
import pyproj as proj
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap

from scipy import interpolate 
import scipy.ndimage

import pandas as pd
import geopandas as gpd

from affine import Affine
import rasterio
from rasterio.crs import *
from rasterio import features
from rasterio.warp import Resampling
from rasterio.windows import Window

from rasterio.plot import reshape_as_image, reshape_as_raster

###
# Mayavi, Bokeh etc are imported in methods, when needed. 

km = 1000

class Grid(object):
    '''
    Methods to set up, save and modify a multidimensional grid.

    '''

    #Switch for print statements
    verbose = False

    def __init__(self,
                 km=km,
                 x1y1x2y2=None,
                 wsen=None,
                 left=-180,
                 up=90,
                 right=-180,
                 down=-90,
                 res=[1, 1],
                 set_frame = True, 
                 center = False,
                 depths=[0. * km, 8. * km, 16. * km, 40. * km, 350. * km],
                 crs=4326,
                 crs_src=4326,
                 band_coord='RGB',
                 use_dask=True,
                 chunk_n=10,
                 coord_d_type=np.float32,
                 *args, **kwargs):
        '''
        Define projection of grid:
        crs : integer
        '''

        # adjust grid to centre points rather than outer extent:
        if set_frame:
            if center:
                left += res[0]/2
                right -= res[0]/2
                up -= res[1]/2
                down += res[1]/2
            else: #lower left corner
                right -= res[0]
                up -= res[1]

        self.res = list(res)

        if isinstance(depths, (int, float)):
            depths = [depths]
        self.depths = list(depths)

        self.x1y1x2y2 = (left, up, right, down)
        self.wsen = (left, down, right, up)
        self.left, self.up, self.right, self.down = left, up, right, down

        self.ul = (self.left, self.up)
        self.ur = (self.right, self.up)
        self.lr = (self.right, self.down)
        self.ll = (self.left, self.down)

        self.nx = int(abs(right - left) // res[0])
        self.ny = int(abs(down - up) // res[1])
        self.nn = (self.ny, self.nx)

        self.transform = rasterio.transform.from_bounds(
            left, down, right, up, self.nx, self.ny)
        self.coord_d_type = coord_d_type

        #rasterio.transform.from_origin(self.left, self.up, *self.res)

        # Make the xarray dataset
        self.ds = xr.Dataset()

        self.ds.coords['X'] = np.linspace(
            left, right, self.nx).astype(self.coord_d_type)
        self.ds.coords['Y'] = np.linspace(
            up, down, self.ny).astype(self.coord_d_type)
        self.nz = len(depths)
        
        self.ds.coords['Z'] = np.array(depths).astype(coord_d_type)
        

        self.ds.coords[band_coord] = list(band_coord)

        # Numpy arrays are indexed rows and columns (y,x)
        self.shape2 = (self.ny, self.nx)
        self.shape3 = (self.ny, self.nx, self.nz)

        self.xv, self.yv = np.meshgrid(
            self.ds.coords['X'], self.ds.coords['Y'])
        self.ds.coords['XV'] = (('Y', 'X'), self.xv.astype(coord_d_type))
        self.ds.coords['YV'] = (('Y', 'X'), self.yv.astype(coord_d_type))

        #Define projections as proj4 strings
        if isinstance(crs, int):
            crs = '+init=epsg:' + str(crs)

        if isinstance(crs_src, int):
            crs_src = '+init=epsg:' + str(crs_src)

        self.crs_src = crs_src
        self.crs = crs

        self.lon, self.lat = proj.transform(proj.Proj(self.crs),
                                            proj.Proj(init='epsg:4326'), self.xv, self.yv)


        self.ds.coords['lat'] = (('Y', 'X'), self.lat.astype(coord_d_type))
        self.ds.coords['lon'] = (('Y', 'X'), self.lon.astype(coord_d_type))

        # Using dask for coordinates
        if use_dask:
            self.ds = self.ds.chunk(
                {'X': self.nx // chunk_n, 'Y': self.ny // chunk_n})

        if self.verbose:
            print('Created model:', self.name)
            print('Class corners', x1y1x2y2)
            print('Class nx ny', self.nx, self.ny, )
            print('Depths:', depths)

    # Accessories
    def _check_if_in(self, xx, yy, margin=2):
        '''
        Generate an array of the condition that coordinates are within the 
        model or not. 
        xx = list or array of x values
        yy = list or array of y values
        margin = extra cells added to mitigate extrapolation of 
            interpolated values along the frame

        returns boolean array True for points within the frame.
        '''
        x_min = self.left - margin * self.res[0]
        x_max = self.right + margin * self.res[0]
        y_min = self.down - margin * self.res[1]
        y_max = self.up + margin * self.res[1]
        return (xx > x_min) & (xx < x_max) & (yy > y_min) & (yy < y_max)

    def _set_meridian(self, x_array, center_at_0=True):
        '''
        Sloppy function to change longitude values from [0..360] to [-180..180]
        x_array :   Numpy array with longitude values (X)
        center_at_0 : Bool select direction of conversion. 
        '''
        if center_at_0:
            x_array[x_array > 180] = x_array[x_array > 180] - 360
        else:
            x_array[x_array < 0] = x_array[x_array < 0] + 360
        return x_array


    def _user_to_array(self, im_data):
        '''Reads user input to numpy array. 
        '''
        if isinstance(im_data, str):
            im_data = self.ds[im_data].values
        elif isinstance(im_data, (np.ndarray, np.generic) ):
            pass # if numpy array
        else:
            im_data = im_data.values # if data frame
        return im_data


    def meta_to_dict(self, 
            f_name = '', 
            meta_dict = {}, 
            get_meta_data=True, 
            meta_file = None):
        '''Open JSON to dict
        f_name : string file to import, suffix is not used

        meta_dict : Dictionary with meta data
        get_meta_data : reads metadata from json file
        meta_file : path to JSAOn file, if not included same name as data file 
        returns : dict
        '''
        if get_meta_data:
            if meta_file == None:
                meta_name = os.path.splitext(f_name)[0] + '.json'
            if os.path.isfile(meta_name):     
                with open(meta_name, 'r') as fp:
                    meta_dict = {**meta_dict, **json.loads(fp.read()) }
            #else:
            #    print('No json file.') # Add support for other formats

        return meta_dict

    def data_to_grid(self, 
            data, 
            dims_order = ['Y', 'X', 'Z', 'T'],
            **kwargs):
        '''Convenience function 
        data : numpy array in the right size
        dims_order: list of order to fit dims of array with grid model
        kwargs sent to meta_to_dict:
            meta_dict dict with meta data
        '''
        dims = dims_order[:data.ndim]

        #Look for meta data and write to attrs
        meta_data = self.meta_to_dict(**kwargs)

        return xr.DataArray(data, dims=dims, attrs = meta_data) 


    def save(self, data=None, file_name='grid.nc'):
        '''
        Saves dataset to netCDF. 
        file_name string
        returns size of file.
        '''
        if data == None:
            data = self.ds
        data.to_netcdf(file_name)
        return os.path.getsize(file_name)

    def save_info(self, ds=None, file_name='info.txt', write_coords = False, 
        **kwargs):
        '''Save json file with instance parameters
        Keyword arguments:
        write_coords -- writes complete list of coordinates '''
        if ds == None:
            ds = self.ds

        if file_name == None:
            file_name = 'info.txt'

        info = self.__dict__.copy()
        info['ds'] = 'xarray dataset'
        info['coord_d_type'] = str(info['coord_d_type'])
        for array in ['xv', 'yv', 'lon', 'lat']:
            if write_coords:
                info[array] = info[array].tolist()
            else:
                info[array] = info[array][[0,0,-1,-1],[0,-1,0,-1]].tolist()

        with open(file_name, 'w') as outfile:
            json.dump(info, outfile, indent=4, ensure_ascii=False, **kwargs)

        return info

    def land_mask(self, polygon_frame=None, polygon_res=None, all_touched=True, land_true=True):
        '''Create a 2D array with only land '''

        if polygon_frame == None:
            pass
            # Download global vector file in with the resolution option of
            # polygon_res=None)

        mask = 1  # rasterize map for section
        return mask

    def change_coord(self,
                     array,
                     old,
                     new,
                     fill_value=np.nan,
                     interpol='linear',
                     axis=2,
                     bounds_error=False, 
                     **kwargs):
        '''Interpolate dimension into new defined depth from coord or list. 
        
        Keyword arguments:
        array -- np.array, list or dataset to be interpolated at new points
                    if array is a string, it will be converted to data frame in self
        old -- coord to interpolate
        new -- coord to interpolate to
        interpol -- interpolation method, e.g. nearest, linear or cubic
        fill_value -- extrapolation value
        '''
        for label in [array, old, new]:
            label = self._user_to_array(label)

        return interpolate.interp1d(old,
                                    array,
                                    axis=axis,
                                    bounds_error=bounds_error,
                                    kind=interpol,
                                    fill_value=fill_value, 
                                    **kwargs)(new)

    def fold_to_low_res(self, large, small):
        '''Takes high resolution 2D array (large) and places subarrays in additional dimensions. 

        The output array have the same resolution as the second array (small) and can be computed together. 
        nx and nn of large must be a multiple of nx, and ny of small.

        Keyword arguments:
        large -- is high res array
        small -- low res array

        Returns folded high res with shape[:2] same as small.
        '''
        res = (np.shape(large)[0] // np.shape(small)[0],
               np.shape(large)[1] // np.shape(small)[1])
        return large.values.reshape(np.shape(small.values)[0], res[0],
                                    np.shape(small.values)[1], res[1]).transpose(0, 2, 1, 3)

    def flatten_to_high_res(self, folded, large):
        '''Flatten a processed array back to high dimension. Reverse of fold_to_low_res. 
        
        Returns a high resolution array. 
        '''
        return folded.transpose(0, 2, 1, 3).reshape(np.shape(large.values)[0],
                                                    np.shape(large.values)[1])

    # Import data
    def assign_shape(self, f_name, attribute=None,
                     z_dim=False, z_max='z_max', z_min='z_min',
                     all_touched=True, 
                     burn_val=None,
                     map_to_int = True, 
                     save_map_to_text = None, 
                     return_map = False,
                     fill_value = np.nan, 
                     **kwargs):
        '''Rasterize vector polygons to grid 
        
        Keyword arguments:
        attribute -- Attribute values to be burned to raster
        z_dim -- Make 3D raster with attributes assigned to layers
        z_min -- Label for attribute that defines min... 
        z_max -- ...and max depths
        all_touched -- Burn value if cell touches or only if crossing centre
        burn_val -- Replaces attribute value
        str_to_int -- Converts string attribute to integer classes. 
                            If False, integer will result in error
        save_map_to_text -- Save map to text file. E.g. attr_to_value.csv
        return_map -- Set if function returns dict of integer map

        Returns numpy array.


        Thanks: 
        https://gis.stackexchange.com/questions/216745/get-polygon-shapefile-in-python-shapely-by-clipping-linearring-with-linestring/216762
        '''
        
        shape = gpd.read_file(f_name).to_crs(self.crs)

        if burn_val != None:
            shape[attribute] = [burn_val] * len(shape)

        # Convert strings to integers
        if map_to_int:
            x = sorted(list(set(shape[attribute])), key=str.lower)
            moby_dict = dict(zip(x, list(range(1,len(x)+1))))
            print(moby_dict)
            if save_map_to_text != None:
                pd.DataFrame(list(moby_dict.items() )).to_csv(save_map_to_text)
            shape[attribute] = [moby_dict[v] for v in shape[attribute]]

        # With z_dim, a 3D grid can be formed where attributes are written to layers between z_min and Z_max
        if z_dim:
            data = np.empty(self.shape3)

            z_select = np.empty([len(shape)]).astype('bool')
            for i, zi in enumerate(self.depths):
                z_select = [z_min <= zi and z_max >= zi for
                            z_min, z_max in zip(shape[z_min], shape[z_max])]

                shape_select = shape[z_select]
                to_burn = ((geom, value) for geom, value in zip(
                    shape_select.geometry, shape_select[attribute]))
                data[:, :, i] = features.rasterize(
                    shapes=to_burn,
                    out_shape=self.shape2,
                    transform=self.transform,
                    fill=fill_value,
                    all_touched=all_touched,
                    **kwargs)

        else:
            data = np.empty(self.nn)

            to_burn = ((geom, value)
                       for geom, value in zip(shape.geometry, shape[attribute]))


            data = features.rasterize(
                shapes=to_burn,
                out_shape=self.shape2,
                transform=self.transform,
                fill=fill_value,
                all_touched=all_touched,
                **kwargs)
    
        if (map_to_int and return_map):
            return data, moby_dict
        else:
            return data

    def read_grid(self, f_name,
                  xyz=('x', 'y', 'z'),
                  interpol='linear',
                  crs_src=None,
                  crs=None,
                  only_frame=True,
                  deep_copy=False,
                  set_center=False, 
                  **kwargs):

        '''Read irregular (or regular) grid. Resampling and interpolating. 

        Keyword arguments:
        xyz --- Sequence with x, y and data labels
        interpol --- Interpolation method, e.g cubic, nearest
        only_frame --- Speeds up interpolation by only 
                regard points within the grid extent (+ margins)

        Returns numpy array'''

        if crs_src == None:
            crs_src = self.crs_src

        if crs == None:
            crs = self.crs

        if isinstance(f_name, str):
            array = xr.open_dataset(f_name, ).copy(deep=deep_copy)
        else:
            array = f_name

        x = array[xyz[0]].values
        y = array[xyz[1]].values

        if set_center:
            x = self._set_meridian(x)

        xx, yy = np.meshgrid(x, y)
        xv, yv = proj.transform(proj.Proj(crs_src),
                                proj.Proj(crs), xx, yy)

        zv = array[xyz[2]].values
        n = zv.size

        zi = np.reshape(zv, (n))
        xi = np.reshape(xv, (n))
        yi = np.reshape(yv, (n))

        # Check and interpolate only elements in the frame
        if only_frame:
            is_in = self._check_if_in(xi, yi)
            xi = xi[is_in]
            yi = yi[is_in]
            zi = zi[is_in]

        return interpolate.griddata((xi, yi),
                                    zi, 
                                    (self.ds.coords['XV'], 
                                    self.ds.coords['YV']), 
                                    method=interpol, 
                                    **kwargs)

    def read_ascii(self,
                   f_name,
                   x_col=0,
                   y_col=1,
                   data_col=2,
                   interpol='linear',
                   no_data = None,
                   only_frame=True,
                   crs_src=None,
                   crs=None,
                   coord_factor=1,
                   skiprows = 0,
                   **kwargs):
        '''Read ascii table to grid

        Textfile, e.g. csv, to grid. 

        Keyword arguments:
        f_name -- String, name of file to import
        x_col -- index for column holding x values in given crs
        y_col --index for column holding y values in given crs
        data_col -- index for column with data values

        '''
        if crs == None:
            crs = self.crs
        if crs_src == None:
            crs_src = self.crs_src

        table = np.loadtxt(f_name, skiprows= skiprows)  # Add kwargs

        if self.verbose:
            print(table[:5, :])


        table[:, x_col] *= coord_factor
        table[:, y_col] *= coord_factor

        xx, yy = proj.transform(proj.Proj(crs_src),
                                proj.Proj(crs), table[:, x_col], table[:, y_col])

        if only_frame:
            is_in = self._check_if_in(xx, yy)
        else:
            is_in = (xx,yy)


        return interpolate.griddata((xx[is_in], yy[is_in]),
                                    table[:, data_col][is_in],
                                    (self.ds.coords['XV'],
                                     self.ds.coords['YV']),
                                    method=interpol,
                                    **kwargs)

    def read_raster(self,
                    f_name,
                    src_crs=None,
                    source_extra=500,
                    resampling=None,
                    sub_sampling=None,
                    sub_window=None,
                    num_threads=4,
                    no_data=None,
                    rgb_convert=True,
                    bit_norm=255, 
                    **kwargs):
        '''Imports raster in geotiff format to grid. 

        Using gdal/rasterio warp to transform raster to right crs and extent. 

        sub_sampling  -- integer to decrease size of input raster and speed up warp

        Resampling -- Interpolation method
                Options for resampling:
                    Resampling.nearest, 
                    Resampling.bilinear, 
                    Resampling.cubic, 
                    Resampling.cubic_spline, 
                    Resampling.lanczos, 
                    Resampling.average    

        A window is a view onto a rectangular subset of a raster 
        dataset and is described in rasterio by column and row offsets 
        and width and height in pixels. These may be ints or floats.
        Window(col_off, row_off, width, height)



        Returns numpy array. 
        '''

        #import rasterio.windows

        in_raster = rasterio.open(f_name)

        if src_crs == None:
            src_crs = in_raster.crs
            if self.verbose:
                print(src_crs)

        if resampling == None:
            resampling = Resampling.nearest

        if self.verbose:
            print('Raster bounds:', in_raster.bounds, in_raster.shape)

        dst_crs = self.crs

        if sub_sampling in (None, 0, 1):
            sub_sampling = 1


        raster_shape = (in_raster.count, in_raster.height //
                        sub_sampling, in_raster.width // sub_sampling)
        source = in_raster.read(out_shape=raster_shape) #window=Window.from_slices(sub_window)

        if sub_window == None:
            pass
        else:
            print('Window not implimented yet.')

        src_transform = rasterio.transform.from_bounds(*in_raster.bounds,
                                                       raster_shape[2], raster_shape[1])

        dst_transform = self.transform
        dst_array = np.zeros((in_raster.count, *self.shape2))

        rasterio.warp.reproject(
            source,
            dst_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            source_extra=source_extra,
            num_threads=num_threads,
            **kwargs)

        if (rgb_convert and in_raster.count > 2):
            dst_array = reshape_as_image(dst_array / bit_norm).astype(float)

        if in_raster.count == 1:
            dst_array = dst_array[0, :, :]

        if no_data != None:
            dst_array[dst_array == no_data] = np.nan

        return dst_array


# Exports
    def grid_to_grd(self, data, save_name='grid.nc'):
        '''Save data array as netCDF

        Keyword arguments:
        data --- string or data array
        '''
        if isinstance(data, str):
            data = self.ds[data]

        save_grid = data.to_netcdf(save_name)
        return save_grid


    def grid_to_raster(self, data,
                       save_name='raster_export.tif', 
                       raster_dtype = np.float64,
                       raster_factor = 1):
        '''Save as geoTIFF

        Keyword arguments: Save to file name

        '''
        data = self._user_to_array(data)


        # If 2D array, define 3rd dimention as 1
        if data.ndim == 2:
            data.shape += 1,

        n_bands = data.shape[2]

        #data = reshape_as_raster(data)

        with rasterio.open(save_name, 'w', driver='GTiff',
                                   height=data.shape[0], width=data.shape[1],
                                   count=n_bands, dtype=raster_dtype,
                                   crs=self.crs,
                                   transform=self.transform) as dst:

            for k in range(n_bands):
                dst.write(data[:,:,k]*raster_factor, indexes=k+1)

        return os.path.getsize(save_name)

    # def frame_to_polygon(self,
    #                      file_name,
    #                      grid_driver='ESRI Shapefile',
    #                      frame_crs=None,
    #                      coordinates=None,
    #                      proj_crs=None,
    #                      id_attribute=1):
    #     '''
    #     Save outer frame as polygon. 
    #     '''

    #     from shapely.geometry import Polygon

    #     if proj_crs == None:
    #         proj_crs = 3031

    #     if frame_crs == None:
    #         frame_crs = 4326

    #     if coordinates == None:
    #         coordinates = [(self.left, self.up),
    #                        (self.right, self.upp),
    #                        (self.right, self.down),
    #                        (self.left, self.down)]

    #     ds = gpd.GeoDataFrame()
    #     ds.crs = '+init=epsg:%s' % proj_crs
    #     ds.loc[0, 'geometry'] = Polygon(coordinates)

    #     ds = ds.to_crs('+init=epsg:%s' % frame_crs)
    #     ds.to_file(file_name, driver='ESRI Shapefile')
    #     return ds

    # def frame_to_point():
    #     '''
    #     Save points in center of each cell. 

    #     '''
    #     from shapely.geometry import Point

    #     return None

    def grid_to_ascii(self, 
                    data, 
                    asc_file_name = 'corners.txt',
                    center = True,
                    fmt = '%6.2f', 
                    no_data = -9999):
        '''Save to asc format

        Keyword arguments:
        corner  --   Coordinates of corner, else centre

        https://gis.stackexchange.com/questions/37238/writing-numpy-array-to-raster-file?rq=1
        http://resources.esri.com/help/9.3/ArcGISengine/java/Gp_ToolRef/Spatial_Analyst_Tools/esri_ascii_raster_format.htm
        '''

        data = self._user_to_array(data)

        header_labels = ['NCOLS', 'NROWS', 'XLLCORNER', 'YLLCORNER', 'CELLSIZE', 'NODATA_VALUE']
        header_values = [self.nx, self.ny, self.left, self.down, self.res[0], no_data]

        if center:
            header_labels[2:4] = ['XLLCENTER', 'YLLCENTER']
            header_values[2:4] = header_values[2:3] + [self.res[0]/2, self.res[1]/2]    
 
        # The wunder of Python: 
        header = ''.join([''.join(h) for h in zip(header_labels, [' ']*6, [str(val) for val in header_values], ['\n']*6)])
        
        np.savetxt(asc_file_name, data, 
           delimiter=' ', 
           header = header, 
           newline='', 
           comments = '', 
           fmt=fmt)

        return os.path.getsize(asc_file_name)


# Vizualisations
    def extract_profile():
        '''
        To be implemented from Staal et al 2019: extract profiles. 
        '''
        return 0

    def oblique_view(self, data, 
        save_name=None, 
        show_oblique=False, 
        azimuth=0, 
        elevation=7500, 
        distance=1100, 
        roll=90, 
        bgcolor = (1., 1., 1.),
        warp_scale=0.015, 
        vmin = None, 
        vmax = None, 
        cmap = 'terrain'):

        '''3D oblique view

        Keyword arguments:
        azimut -- Camera direction
        elevation -- Camera height
        distance -- Camera distance
        roll -- Camera rotation
        bgcolor -- Tuple of lenght 3, values from 0 to 1 RGB
        warp_scale -- Enhance z, lower value increase the distortion
        vmin and vmax -- Set color range

        Function exemplifies the use of mayavi and VTK for visualizing multidimensional data
        '''

        # Import mlab
        from mayavi import mlab
        
        data = self._user_to_array(data)

        if vmin == None:
            vmin = np.nanpercentile(data, 0.1)
        if vmax == None:
            vmax = np.nanpercentile(data, 99.9)

        if show_oblique:
            mlab.options.offscreen = False
        else:
            mlab.options.offscreen = True

        mlab.figure(size=(1000, 1000), bgcolor= bgcolor)
        mlab.clf()

        mlab.surf(data, warp_scale=warp_scale, colormap=cmap, vmin=vmin, vmax=vmax)

        mlab.view(azimuth=azimuth, elevation=elevation, distance=distance, roll=roll)

        if save_name != None:
            mlab.savefig(save_name, size=(1000, 1000))

        if show_oblique:
            mlab.show()

        mlab.close(all=True)

        # Return obj 
        return None


    def volume_slice(self, data,
            save_name=None,  
            cmap = 'viridis',
            vmin = None, 
            vmax = None, 
            show_slice = False, 
            bgcolor = (1., 1., 1.)):
        '''Open Mayavi scene

        New function
        '''

        # Import mlab
        from mayavi import mlab

        #try:
        #    engine = mayavi.engine
        #except NameError:
        #    from mayavi.api import Engine
        #engine = Engine()
        #engine.start()

        if vmin == None:
            vmin = np.nanpercentile(data, 0.1)
        if vmax == None:
            vmax = np.nanpercentile(data, 99.9)

        #if len(engine.scenes) == 0:
        #    engine.new_scene()
        
        mlab.figure(size=(1000, 1000), bgcolor= bgcolor)
        mlab.clf()

        mlab.volume_slice(data.values, plane_orientation='x_axes')

        mlab.view(azimuth=azimuth, elevation=elevation, distance=distance, roll=roll)
    
        #module_manager = engine.scenes[0].children[0].children[0]

        #module_manager.scalar_lut_manager.lut_mode = cmap
        #scene = engine.scenes[0]
        #scene.scene.x_minus_view()
        
        if save_name != None:
            mlab.savefig(save_name, size=(1000, 1000))

        if show_slice:
            mlab.show()

        return 




    def map_grid(self, im_data,
                 vmin=None,
                 vmax=None,
                 cmap='gray',
                 cbar=False,
                 cbar_label = '',
                 extent= None,
                 line_c='gray',
                 line_grid_c='gray',
                 line_w=0.9,
                 circ_map=False,
                 figsize=None,
                 land_only=True,
                 ocean_color='white',
                 no_land_fill=np.nan,
                 title=None,
                 save_name=None,
                 show_map=True,
                 ax = None,
                 map_res = 'i',
                 draw_coast=True,
                 draw_grid=True,
                 par=None,
                 mer=None, 
                 *args, **kwargs):

        '''Make map view for print or display. 

        Keyword arguments:

        vmin, vmax -- Set range oc colormap. If not set 0.1 percentille is used
        cmap -- Select colormap
        cbar -- Boolean colorbar or not
        extent -- Select a different extent than the object (left, right, down, up)
        line_c -- Color of lines
        line_grid_c -- Color of gridlines
        line_w --  Whidth of lines 
        circ_map -- If selected, map is cropped to a circle around the center as 
                        a hub. Sometimes visually appealing, this is a leftover from early 
                        use of the code for only Antarctic continental scale maps. 
        figsize  -- Size of figure in cm Default is 12cm high
        land_only -- Crop oceans (In furure versions)
        ocean_color -- Colour of oceans
        no_land_fill -- Value for no land
        title -- String for title
        save_name -- Name of file to save to. E.g. png, pdf, or jpg 
        show_map -- Off if only saving, good for scripting
        map_res -- 'c'is fast but not detailed, 'i' in between, 
                            'f' is slow and detaile. For 'i' and finer hires data is needed
        draw_coast -- If True, Basemab coastline is drawn. 
        draw_grid -- Draw parallells and meridieans
        par -- List of Parallels
        mer -- List of Meridians        
    
        This function might eventually be amended to use cartopy or GMT. 
        '''

        def create_circular_mask(h, w, center=None, radius=None):
            center = [int(w / 2), int(h / 2)]
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            mask = dist_from_center <= radius
            return mask

        im_data = self._user_to_array(im_data)

        if figsize == None:
            figsize = (12, 12 * self.nx / self.ny)

        if par == None:
            par = np.arange(-90, 90, 10)

        if mer == None:
            mer = np.arange(-180, 180, 45)

        #New pyproj version to be implimented:
        #pyproj.CRS 
        #basemap_epsg = to_epsg(self.crs, min_confidence=90)

        basemap_epsg = int(re.findall("\d+.\d+", self.crs)[0] )

        m = Basemap(llcrnrlon=self.lon[-1, 0],
                    llcrnrlat=self.lat[-1, 0],
                    urcrnrlon=self.lon[0, -1],
                    urcrnrlat=self.lat[0, -1],
                    resolution=map_res,
                    epsg=basemap_epsg, **kwargs)

        if extent == None:
            extent = [self.left, self.right, self.down, self.up]


        fig = plt.figure(figsize=figsize)
        ax = ax or plt.axes()
        ax.axis('off')

        if im_data is not None:
            if vmin == None:
                vmin = np.nanpercentile(im_data, 0.1)
            if vmax == None:
                vmax = np.nanpercentile(im_data, 99.9)

            if circ_map:
                h, w = im_data.shape[:2]
                mask = create_circular_mask(h, w, radius=h / 2)
                im_data[~mask] = np.nan

            im = m.imshow(im_data,
                          extent=extent,
                          origin='upper',
                          vmin=vmin,
                          vmax=vmax,
                          cmap=cmap,
                          zorder=5)

            if cbar:
                cbar = fig.colorbar(im, orientation='vertical',
                             fraction=0.046, pad=0.01)
                cbar.set_label(cbar_label)

        if land_only:
            #
            pass

        if draw_coast:
            m.drawcoastlines(color=line_c, linewidth=line_w, zorder=10)
            #ax.coastlines(resolution='50m', color=line_c, linewidth=1)
        if draw_grid:
            m.drawparallels(par, color=line_grid_c, alpha=0.9, zorder=15)
            m.drawmeridians(mer, color=line_grid_c,
                            alpha=0.9, latmax=88, zorder=15)

        if title != None:
            fig.suptitle(title)

        fig.tight_layout(pad=0)
        if save_name != None:
            plt.savefig(save_name, transparent=True,
                        bbox_inches='tight', pad_inches=0)
            print('Saved to:', save_name)

        if show_map:
            plt.show()

        return m

    def look(self, rasters, save_name='look.png',
             interp_method=None,
             color_map='viridis',
             show=True,
             save=False,
             max_n_plots=16,
             x_unit='(km)',
             y_unit='(km)',
             ref_range=None,
             **kwargs):
        """Quick look of 2D slices. 

        Keyword arguments:
        interp_method -- imshow interpolation methods
        
        Rather undeveloped function. 
        """

        if len(rasters) > max_n_plots:
            print('To much to see!')
            return 0
        else:
            n_sq = int(np.ceil(np.sqrt(len(rasters))))
            fig, ax = plt.subplots(n_sq, n_sq, figsize=(10, 10))

            for k, raster in enumerate(rasters):
                x_coords = raster.coords[sorted(raster.dims)[0]]
                y_coords = raster.coords[sorted(raster.dims)[1]]

                if ref_range is None:
                    plot_range = (np.nanmin(raster.values),
                                  np.nanmax(raster.values))
                else:
                    plot_range = ref_range

                ax[k // n_sq, k % n_sq].imshow(raster.values,
                                               interpolation=interp_method,
                                               cmap=color_map,
                                               extent=self.x1y1x2y2,
                                               vmin=plot_range[0], vmax=plot_range[1],
                                               **kwargs)
                ax[k // n_sq, k % n_sq].title.set_text('%s' % (raster.name,))
                ax[k // n_sq, k % n_sq].set_aspect('auto')
                ax[k // n_sq, k % n_sq].set_ylabel(y_coords.name + x_unit)
                ax[k // n_sq, k % n_sq].set_xlabel(x_coords.name + y_unit)

        fig.tight_layout(pad=0)
        if save:
            fig.savefig(save_name, transparent=True,
                        bbox_inches='tight', pad_inches=0)
        if show:
            plt.show()
        return ax

    def slider(self, data,
               kdims=['z', 'x', 'y'],
               vdims=['v'],
               flat=['x', 'y'],
               slider='z',
               invert_yaxis=False,
               sub_sample=4):
        '''Test for interactive display 3D grid with slider. 

        '''
        import holoviews as hv
        hv.extension('bokeh', logo=False)

        data = self._user_to_array(data)

        ds = hv.Dataset((np.arange(np.shape(data)[2]),
                         np.linspace(0., 1., np.shape(data)[1]),
                         np.linspace(0., 1., np.shape(data)[0])[::-1],
                         data),
                        kdims=kdims,
                        vdims=vdims)
        return ds.to(hv.Image, flat).redim(slider).options(colorbar=True,
                                                           invert_yaxis=invert_yaxis).hist()
