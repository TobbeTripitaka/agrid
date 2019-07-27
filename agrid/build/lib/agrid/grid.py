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

# Standard library imports
import os
import sys
import re
import json
import glob

# Array packages and scipy etc 
import numpy as np
import xarray as xr
import pyproj as proj
import pandas as pd
from scipy import interpolate
from scipy import stats
import scipy.ndimage
import imageio

# Vector packages
import geopandas as gpd
import fiona

# Raster packages
from affine import Affine
import rasterio
from rasterio.crs import * #Fix! 
from rasterio import features
from rasterio.warp import Resampling
from rasterio.windows import Window
from rasterio.plot import reshape_as_image, reshape_as_raster


# Matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from mpl_toolkits.basemap import Basemap  # To be replaced by cartopy
except:
    print('Can not install Basemap, map_grid method will not work in this release.')

###
# Mayavi is imported in methods, when needed.

km = 1000


class Grid(object):
    '''
    Methods to set up, save and modify a multidimensional grid.
    '''

    # Switch for print statements
    verbose = False

    def __init__(self,
                 km=km,
                 x1y1x2y2=None,
                 wsen=None,
                 left=-180,
                 up=90,
                 right=180,
                 down=-90,
                 res=[1, 1],
                 center=False,
                 depths=[0. * km, 8. * km, 16. * km, 40. * km, 350. * km],
                 is_huge=1e5,
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

        # res given as [x,y]
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

        if self.nx > is_huge or self.ny > is_huge:
            raise Exception('The array is too large:', self.nn)

        self.transform = rasterio.transform.from_bounds(
            left, up, right, down, self.nx, self.ny)
        self.coord_d_type = coord_d_type

        #rasterio.transform.from_origin(self.left, self.up, *self.res)

        # Make the xarray dataset
        self.ds = xr.Dataset()

        self.ds.coords['X'] = np.linspace(
            left, right, self.nx).astype(self.coord_d_type)
        self.ds.coords['Y'] = np.linspace(
            down, up, self.ny).astype(self.coord_d_type)
        self.nz = len(depths)
        self.ds.coords['Z'] = np.array(depths).astype(coord_d_type)

        assert (np.all(self.ds.coords['X'][1:] >= self.ds.coords['X'][:-1])), \
            'X coords must be strictly increasing.'
        assert (np.all(self.ds.coords['Y'][1:] >= self.ds.coords['Y'][:-1])), \
            'Y coords must be strictly increasing.'
        assert (np.all(self.ds.coords['Z'][1:] >= self.ds.coords['Z'][:-1])), \
            'Z coords must be strictly increasing.'

        # Define edges
        self.ds.coords['X_edge'] = np.linspace(
            left-res[0]/2, right+res[0]/2, self.nx+1).astype(self.coord_d_type)
        self.ds.coords['Y_edge'] = np.linspace(
            down-res[1]/2, up+res[1]/2, self.ny+1).astype(self.coord_d_type)

        # Edges of depths slices are inter- and extrapolated around chosen depths
        depth_e = [_-0.5 for _ in range(len(depths)+1)]
        inter_d = interpolate.InterpolatedUnivariateSpline(range(len(depths)),
                                                           depths, k=1)
        self.ds.coords['Z_edge'] = inter_d(depth_e)

        # Bands are named after string, e.g. R, G and B bands
        self.ds.coords[band_coord] = list(band_coord)

        # Numpy arrays are indexed rows and columns (y,x)
        self.shape2 = (self.ny, self.nx)
        self.shape3 = (self.ny, self.nx, self.nz)

        self.xv, self.yv = np.meshgrid(
            self.ds.coords['X'], self.ds.coords['Y'])
        self.ds.coords['XV'] = (('Y', 'X'), self.xv.astype(coord_d_type))
        self.ds.coords['YV'] = (('Y', 'X'), self.yv.astype(coord_d_type))

        # Define projections as proj4 strings. Integer is epsg code
        if isinstance(crs, int):
            crs = '+init=epsg:' + str(crs)

        # Default crs of data to import
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
        Returns copy of data.
        '''
        if isinstance(im_data, str):
            im_data = self.ds[im_data].values
        elif isinstance(im_data, list):
            im_data = np.array(im_data)
        elif isinstance(im_data, (np.ndarray, np.generic)):
            pass  # if numpy array
        else:
            im_data = im_data.values  # if data frame
        return np.copy(im_data)

    def _set_v_range(vmin, vmax, im_data,
                     first_pile=0.1,
                     last_pile=99.9):
        '''
        Set color range from percentilles, to be implimented. 
        '''

        if vmin is None:
            vmin = np.nanpercentile(im_data, first_pile)
        if vmax is None:
            vmax = np.nanpercentile(im_data, last_pile)
        return vmin, vmax

    def _meta_to_dict(self,
                 f_name='',
                 meta_dict={},
                 get_meta_data=True,
                 meta_file=None):
        '''Open JSON ot textfile to dict, for use as attribute data
        f_name : string file to import, suffix is not used
        meta_dict : Dictionary with meta data
        get_meta_data : reads metadata from json file
        meta_file: path to JSAOn file, if not included same name as data file
        returns: dict
        '''
        if get_meta_data:
            if meta_file is None:
                meta_name = os.path.splitext(f_name)[0]
            if os.path.isfile(meta_name + '.json'):
                with open(meta_name + '.json', 'r') as fp:
                    meta_dict = {**meta_dict, **json.loads(fp.read())}
            elif os.path.isfile(meta_name + '.txt'):
                with open(meta_name + '.txt', 'r') as fp:
                    meta_dict = {**meta_dict, **{meta_name: fp.read()} } 
            else:
                print('Need json  or text file.')
        return meta_dict

    def data_to_grid(self, data,
                 dims_order=['Y', 'X', 'Z', 't'],
                 **kwargs):
        '''Convenience function

        data : numpy array in the right size
        dims_order: list of order to fit dims of array with grid model
        kwargs sent to _meta_to_dict:
            meta_dict dict with meta data
        '''
        dims = dims_order[:data.ndim]

        # Look for meta data and write to attrs
        meta_data = self._meta_to_dict(**kwargs)

        return xr.DataArray(data, dims=dims, attrs=meta_data)

    def save(self, data=None, file_name='grid.nc'):
        '''
        Saves dataset to netCDF.
        file_name string
        returns size of file.
        '''
        if data is None:
            data = self.ds
        data.to_netcdf(file_name)
        return os.path.getsize(file_name)

    def save_info(self, ds=None, file_name='info.txt', write_coords=False,
                  **kwargs):
        '''Save json file with instance parameters
        Keyword arguments:
        write_coords -- writes complete list of coordinates '''
        if ds is None:
            ds = self.ds

        if file_name is None:
            file_name = 'info.txt'

        info = self.__dict__.copy()
        info['ds'] = 'xarray dataset'
        info['coord_d_type'] = str(info['coord_d_type'])
        for array in ['xv', 'yv', 'lon', 'lat']:
            if write_coords:
                info[array] = info[array].tolist()
            else:
                info[array] = info[array][[0, 0, -1, -1], [0, -1, 0, -1]].tolist()

        with open(file_name, 'w') as outfile:
            json.dump(info, outfile, indent=4, ensure_ascii=False, **kwargs)

        return info

    def land_mask(self, 
        polygon_frame=None, 
        polygon_res=None, 
        all_touched=True, 
        oceans=True):
        '''Create a 2D array with only land '''

        if polygon_frame is None:
            pass
            # Download global vector file in with the resolution option of
            # polygon_res=None)

        mask = 1  # rasterize map for section

        if oceans:
            mask = np.invert(mask)

        return mask

    def change_coord(self,
                     array,
                     old,
                     new,
                     fill_value=np.nan,
                     interpol='linear',
                     axis=None,
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
        array = self._user_to_array(array)
        old = self._user_to_array(old)
        new = self._user_to_array(new)

        # If none, use last dim!
        if axis is None:
            axis = 2
        if array.ndim == 1:
            axis = 0

        return interpolate.interp1d(old,
                                    array,
                                    axis=axis,
                                    bounds_error=bounds_error,
                                    kind=interpol,
                                    fill_value=fill_value,
                                    **kwargs)(new)

    def fold_to_low_res(self, large, small):
        '''
        Takes high resolution 2D array (large) and places subarrays in additional dimensions.
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
                     map_to_int=True,
                     sort_alphabetically = False,
                     print_map = False,
                     save_map_to_text=None,
                     return_map=False,
                     fill_value=np.nan,
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

        if burn_val is not None:
            shape[attribute] = [burn_val] * len(shape)

        # Convert strings
        if map_to_int:
            if sort_alphabetically:
                x = sorted(list(set(shape[attribute])), key=str.lower)
            else:
                x = list(set(shape[attribute]))
            moby_dict = dict(zip(x, list(range(1, len(x) + 1))))
            if print_map:
                print(moby_dict)
            if save_map_to_text is not None:
                pd.DataFrame(list(moby_dict.items())).to_csv(save_map_to_text)
            shape[attribute] = [moby_dict[v] for v in shape[attribute]]

        # With z_dim, a 3D grid can be formed where attributes are written to
        # layers between z_min and Z_max
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

    def read_grid(self,
                  f_name,
                  xyz=('x', 'y', 'z'),
                  interpol='linear',
                  crs_src=None,
                  crs=None,
                  bulk=False,
                  extension='',
                  sort=True,
                  only_frame=True,
                  deep_copy=False,
                  set_center=False,
                  regex_index = None, 
                  def_depths = None,
                  **kwargs):
        '''Read irregular (or regular) grid. Resampling and interpolating.

        Keyword arguments:
        f_name : string path to dir or file. Ii list, it is read as list of paths to files. 
        xyz --- Sequence with x, y and data labels
        interpol --- Interpolation method, e.g cubic, nearest
        only_frame --- Speeds up interpolation by only
                regard points within the grid extent (+ margins)

        Returns numpy array'''

        if crs_src is None:
            crs_src = self.crs_src

        if crs is None:
            crs = self.crs

        if bulk:
            if isinstance(f_name, str):
                assert os.path.isdir(f_name), 'Please provide path to directory containing files.'
                f_names = glob.glob(f_name + '*' + extension)
            elif isinstance(f_names, list):
                for f_name in f_names:
                    assert os.path.isfile(f_name), '%s Is not a file.'%f_name
            else:
                f_names = []
        else:
            assert os.path.isfile(f_name), 'Please provide path to a file, not directory. Or set bulk=True'
            f_names = [f_name]

        if sort:
            f_names.sort(key=str.lower)

        i_grid = np.empty(self.nn + (len(f_names),) )
        for i, f in enumerate(f_names):
            # Read input from file or as data array variable
            array = xr.open_dataset(f, ).copy(deep=deep_copy)
            x = array[xyz[0]].values
            y = array[xyz[1]].values

            # Set longitude 
            if set_center:
                x = self._set_meridian(x)

            xx, yy = np.meshgrid(x, y)  # x, y
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

            i_grid[...,i] = interpolate.griddata((xi, yi),
                              zi,
                              (self.ds.coords['XV'],
                               self.ds.coords['YV']),
                              method=interpol,
                              **kwargs)
                              

            if len(f_names) is 1:
                i_grid = np.squeeze(i_grid, axis=2)

        return i_grid



    def read_ascii(self,
                   f_name,
                   x_col=0,
                   y_col=1,
                   data_col=2,
                   interpol='linear',
                   no_data=None,
                   only_frame=True,
                   crs_src=None,
                   crs=None,
                   coord_factor=1,
                   skiprows=0,
                   **kwargs):
        '''Read ascii table to grid

        Textfile, e.g. csv, to grid.

        Keyword arguments:
        f_name -- String, name of file to import
        x_col -- index for column holding x values in given crs
        y_col --index for column holding y values in given crs
        data_col -- index for column with data values

        '''
        if crs is None:
            crs = self.crs
        if crs_src is None:
            crs_src = self.crs_src

        table = np.loadtxt(f_name, skiprows=skiprows, **kwargs)  # Add kwargs

        if self.verbose:
            print(table[:5, :])

        table[:, x_col] *= coord_factor
        table[:, y_col] *= coord_factor

        xx, yy = proj.transform(proj.Proj(crs_src),
                                proj.Proj(crs), table[:, x_col], table[:, y_col])

        if only_frame:
            is_in = self._check_if_in(xx, yy)
        else:
            is_in = (xx, yy)

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

        if src_crs is None:
            src_crs = in_raster.crs
            if self.verbose:
                print(src_crs)

        if resampling is None:
            resampling = Resampling.nearest

        if self.verbose:
            print('Raster bounds:', in_raster.bounds, in_raster.shape)

        dst_crs = self.crs

        if sub_sampling in (None, 0, 1):
            sub_sampling = 1

        raster_shape = (in_raster.count, in_raster.height //
                        sub_sampling, in_raster.width // sub_sampling)
        # window=Window.from_slices(sub_window)
        source = in_raster.read(out_shape=raster_shape)

        if sub_window is None:
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
                       raster_dtype=np.float64,
                       raster_factor=1):
        '''Save as geoTIFF
        data : array or label
        svae_name : string save as tif name
        raster_dtype : dtype to save to, e.g. bit depth
        raster_factor : factor to multiply value
        '''
        data = (self._user_to_array(data)* raster_factor).astype(raster_dtype) 

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
                dst.write(data[:, :, k], indexes=k + 1)

        return None


    def grid_to_ascii(self,
                      data,
                      asc_file_name='corners.txt',
                      center=True,
                      fmt='%6.2f',
                      no_data=-9999):
        '''Save to asc format

        Keyword arguments:
        corner  --   Coordinates of corner, else centre

        https://gis.stackexchange.com/questions/37238/writing-numpy-array-to-raster-file?rq=1
        http://resources.esri.com/help/9.3/ArcGISengine/java/Gp_ToolRef/Spatial_Analyst_Tools/esri_ascii_raster_format.htm
        '''

        data = self._user_to_array(data)

        header_labels = ['NCOLS', 'NROWS', 'XLLCORNER',
                         'YLLCORNER', 'CELLSIZE', 'NODATA_VALUE']
        header_values = [self.nx, self.ny, self.left,
                         self.down, self.res[0], no_data]

        if center:
            header_labels[2:4] = ['XLLCENTER', 'YLLCENTER']
            header_values[2:4] = header_values[2:3] + \
                [self.res[0] / 2, self.res[1] / 2]

        # The wunder of Python:
        header = ''.join([''.join(h) for h in zip(
            header_labels, [' '] * 6, [str(val) for val in header_values], ['\n'] * 6)])

        np.savetxt(asc_file_name, data,
                   delimiter=' ',
                   header=header,
                   newline='',
                   comments='',
                   fmt=fmt)

        return os.path.getsize(asc_file_name)

    def bins_to_grid(self,
                     values,
                     samples=None,
                     xi=None,
                     yi=None,
                     zi=None,
                     sample_src=None,
                     function='mean',
                     return_only_statistic=True,
                     ):
        '''Reads lists of data values, and coordinates and generate
        bins and apply function to each bin. E.g. to generate geographical histagrams. 

        values : list or array of data values
        samples : array with coordnates. Must be the shape D arrays of length N, or as an (N,D) 
        xi : if samples is none, coordinates rae read xi, yi, zi
        yi :
        zi : If no zi is given, 2D bins are generated
        sample_src : If not None, data points are reprojected from this CSR. 
        function to apply to bins. A string eg: ‘median’, 'std', 'median',
            or a defined function that takes a 1D array an returnsrn a scalar.
        return_only_statistic : boolean. If False, method returns statistics, edges, binnumbers
        '''
        def _build_samples(di, values):
            di = np.array(di)
            assert (np.shape(values) == np.shape(di)
                    ), 'Samples and values must have same shape.'
            return di

        values = np.array(values)
        if samples is not None:
            samples_dim = samples.ndim + 1
        else:
            if zi is None:
                samples_dim = 2
            else:
                samples_dim = 3

        assert (samples_dim in [2, 3]), 'Samples must be 2 or 3D arrays of same lenght'

        if samples_dim is 2:
            if samples is None:
                samples = np.array((
                    _build_samples(yi, values),
                    _build_samples(xi, values),))
            bins = (self.ds['Y_edge'], self.ds['X_edge'])
        elif samples_dim is 3:
            if samples is None:
                samples = np.array((
                    _build_samples(yi, values),
                    _build_samples(xi, values),
                    _build_samples(zi, values),))
            bins = (self.ds['Y_edge'],
                        self.ds['X_edge'], self.ds['Z_edge'])

        if sample_src is not None:
            if isinstance(sample_src, int):
                sample_src = '+init=epsg:' + str(sample_src)


            xi, yi = proj.transform(proj.Proj(sample_src),
                                    proj.Proj(self.crs), samples[0], samples[1])
            samples[0] = xi
            samples[1] = yi

        bin_data = scipy.stats.binned_statistic_dd(samples.T,
                                                   values,
                                                   statistic=function,
                                                   bins=bins,
                                                   expand_binnumbers=False)

        if return_only_statistic:
            return bin_data.statistic
        else:
            return bin_data

# Vizualisations
    def extract_profile(self):
        '''
        To be implemented from Staal et al 2019: extract profiles.
        '''
        pass
        return 0

    def oblique_view(self, data,
                     save_name=None,
                     show=False,
                     azimuth=0,
                     elevation=45,
                     distance=1100,
                     roll=90,
                     figsize=(1800, 1800),
                     bgcolor=(1., 1., 1.),
                     warp_scale=0.015,
                     lut=None,
                     vmin=None,
                     vmax=None,
                     cmap='terrain'):
        '''3D oblique view

        Keyword arguments:
        azimut -- Camera direction
        elevation -- Camera height
        distance -- Camera distance
        roll -- Camera rotation
        bgcolor -- Tuple of lenght 3, values from 0 to 1 RGB
        warp_scale -- Enhance z, lower value increase the distortion
        vmin and vmax -- Set color range
        lut 
        cmap

        Function exemplifies the use of mayavi and VTK for visualizing multidimensional data
        '''

        # Import mlab
        from mayavi import mlab
        # mlab.clf()

        data = self._user_to_array(data)

        if vmin is None:
            vmin = np.nanpercentile(data, 0.1)
        if vmax is None:
            vmax = np.nanpercentile(data, 99.9)

        if show:
            mlab.options.offscreen = False
        else:
            mlab.options.offscreen = True

        if cmap is None:
            set_lut = True
            cmap = 'viridis'
        else:
            set_lut = False

        fig = mlab.figure(size=figsize, bgcolor=bgcolor)

        surf = mlab.surf(data, warp_scale=warp_scale,
                         colormap=cmap, vmin=vmin, vmax=vmax, figure=fig)

        mlab.view(azimuth=azimuth, elevation=elevation,
                  distance=distance, roll=roll)

        if set_lut:
            surf.module_manager.scalar_lut_manager.lut.table = lut
            mlab.draw()

        if save_name != None:
            save_array = mlab.screenshot(
                figure=fig, mode='rgba', antialiased=True)*255
            imageio.imwrite(save_name, save_array.astype(np.uint8))
            #mlab.savefig(save_name, size=figsize)

        if show:
            mlab.show()

        mlab.close(all=True)

        # Return obj
        return None

    def volume_slice(self, data,
                     save_name=None,
                     cmap='viridis',
                     vmin=None,
                     vmax=None,
                     show=False,
                     bgcolor=(1., 1., 1.)):
        '''Open Mayavi scene

        New function
        '''

        # Import mlab
        from mayavi import mlab

        # try:
        #    engine = mayavi.engine
        # except NameError:
        #    from mayavi.api import Engine
        #engine = Engine()
        # engine.start()

        if vmin is None:
            vmin = np.nanpercentile(data, 0.1)
        if vmax is None:
            vmax = np.nanpercentile(data, 99.9)

        # if len(engine.scenes) == 0:
        #    engine.new_scene()

        mlab.figure(size=(1000, 1000), bgcolor=bgcolor)
        mlab.clf()

        mlab.volume_slice(data.values, plane_orientation='x_axes')

        mlab.view(azimuth=azimuth, elevation=elevation,
                  distance=distance, roll=roll)

        #module_manager = engine.scenes[0].children[0].children[0]

        #module_manager.scalar_lut_manager.lut_mode = cmap
        #scene = engine.scenes[0]
        # scene.scene.x_minus_view()

        if save_name != None:
            mlab.savefig(save_name, size=(1000, 1000))

        if show_slice:
            mlab.show()

        return None

    def map_grid(self,
                 im_data,
                 vectors=[],
                 v_col=[],
                 v_alpha=1,
                 v_lw=1,
                 v_x_offset=0,
                 v_y_offset=0,
                 vmin=None,
                 vmax=None,
                 cmap='gray',
                 cbar=False,
                 cbar_label='',
                 extent=None,
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
                 show=True,
                 map_res='i',
                 draw_coast=True,
                 draw_grid=True,
                 par=None,
                 mer=None,
                 *args, **kwargs):
        '''Make map view for print or display.

        Keyword arguments:
        vmin, vmax - - Set range oc colormap. If not set 0.1 percentille is used
        cmap - - Select colormap
        cbar - - Boolean colorbar or not
        extent - - Select a different extent than the object(left, right, down, up)
        line_c - - Color of lines
        line_grid_c - - Color of gridlines
        line_w - -  Whidth of lines
        circ_map - - If selected, map is cropped to a circle around the center as
                        a hub. Sometimes visually appealing, this is a leftover from early
                        use of the code for only Antarctic continental scale maps.
        figsize - - Size of figure in cm Default is 12cm high
        land_only - - Crop oceans(In furure versions)
        ocean_color - - Colour of oceans
        no_land_fill - - Value for no land
        title - - String for title
        save_name - - Name of file to save to. E.g. png, pdf, or jpg
        show_map - - Off if only saving, good for scripting
        map_res - - 'c'is fast but not detailed, 'i' in between,
                            'f' is slow and detaile. For 'i' and finer hires data is needed
        draw_coast - - If True, Basemab coastline is drawn.
        draw_grid - - Draw parallells and meridieans
        par - - List of Parallels
        mer - - List of Meridians

        This function will in a near future be amended to use cartopy or GMT.

        '''


        def create_circular_mask(h, w, center=None, radius=None):
            center = [int(w / 2), int(h / 2)]
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            mask = dist_from_center <= radius
            return mask

        im_data = self._user_to_array(im_data)

        if figsize is None:
            figsize = (12, 12 * self.nx / self.ny)

        if par is None:
            par = np.arange(-90, 90, 10)

        if mer is None:
            mer = np.arange(-180, 180, 45)

        # New pyproj version to be implimented:
        # pyproj.CRS
        # basemap_epsg = to_epsg(self.crs, min_confidence=90)

        basemap_epsg = int(re.findall("\d+.\d+", self.crs)[0])

        m = Basemap(llcrnrlon=self.lon[0, 0],
                    llcrnrlat=self.lat[0, 0],
                    urcrnrlon=self.lon[-1, -1],
                    urcrnrlat=self.lat[-1, -1],
                    resolution=map_res,
                    epsg=basemap_epsg, **kwargs)

        if extent is None:
            extent = [self.left, self.right, self.up, self.down]

        fig, ax = plt.subplots(figsize=figsize)

        if im_data is not None:
            # vmin, vmax = self._set_v_range(vmin, vmax, im_data)

            if vmin is None:
                vmin = np.nanpercentile(im_data, 0.1)
            if vmax is None:
                vmax = np.nanpercentile(im_data, 99.9)

            if circ_map:
                h, w = im_data.shape[:2]
                mask = create_circular_mask(h, w, radius=h / 2)
                im_data[~mask] = np.nan

            im = m.imshow(im_data,
                          extent=extent,
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
        if draw_grid:
            m.drawparallels(par, color=line_grid_c, alpha=0.9, zorder=15)
            m.drawmeridians(mer, color=line_grid_c,
                            alpha=0.9, latmax=88, zorder=15)

        for j, vector in enumerate(vectors):
            with fiona.open(str(vector), 'r') as src:
                for i, geom in enumerate(src):
                    x = [i[0] for i in geom['geometry']['coordinates']]
                    y = [i[1] for i in geom['geometry']['coordinates']]

                # x, y = proj.transform(proj.Proj(init='epsg:%s'%env['shape_proj']),
                #                    proj.Proj(init='epsg:%s,
                #                    x, y)

                    x = [v_x_offset + _ for _ in x]
                    y = [v_y_offset + _ for _ in y]
                    m.plot(x, y, c=v_col[j], alpha=v_alpha, lw=v_lw, zorder=20)

        if title != None:
            fig.suptitle(title)

        fig.tight_layout(pad=0)
        if save_name != None:
            plt.savefig(save_name, transparent=True,
                        bbox_inches='tight', pad_inches=0)
            print('Saved to:', save_name)

        if show:
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
        '''
        Quick look of 2D slices. 

        Keyword arguments:
        interp_method -- imshow interpolation methods

        Rather undeveloped function. 
        '''

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
                         np.linspace(0., 1., np.shape(data)[0]),
                         data),
                        kdims=kdims,
                        vdims=vdims)
        return ds.to(hv.Image, flat).redim(slider).options(colorbar=True, invert_yaxis=invert_yaxis).hist()

    def layer_cake(self,
                       data,
                       figsize=None,
                       save_name=None,
                       show_map=True,
                       make_wireframe=True,
                       d_alpha=0.6,
                       d_levels=100,
                       g_alpha=0.3,
                       g_lw=0.4,
                       scale_x=1,
                       scale_y=1,
                       scale_z=0.5,
                       vmin=None,
                       vmax=None,
                       cmap='viridis',
                       cbar=True,
                       layers=None,
                       dims=['X', 'Y', 'Z'],
                       global_vm = True,
                       reduce_dims=[5, 5, 5],
                       wireframe_cell_size = (10,10),
                       ax_grid=False,
                       azim=250,
                       elev=10,
                       dist=10,
                       x_lim=None,
                       y_lim=None,
                       z_lim=None,
                       outer_frame=False,
                       xlabel='$X$',
                       ylabel='$Y$',
                       zlabel='$Z$'):
        '''Method to display 3D data by using only matplotlib
        data : data to display
        save_name : Name to save file to
        make_wireframe : Display wireframe
        data : data to plot, 2D or 3D
        figsize ; figuee outout size in cm
        save:name : filename as string to save file, don't save if None
        wireframe_cel_size is a multiple of resolution in x and y.
        '''
        data = self._user_to_array(data)

        if figsize is None:
            figsize = (12, 12)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')


        xve, yve = np.meshgrid(
            self.ds.coords['X_edge'], self.ds.coords['Y_edge'])
        xv = self.ds.coords['XV'].values
        yv = self.ds.coords['YV'].values


        if layers is None:
            layers = self.ds.coords['Z'].values

        if global_vm:
            if vmin is None:
                vmin = np.nanpercentile(data, 0.1)
            if vmax is None:
                vmax = np.nanpercentile(data, 99.9)

        for i, z in enumerate(layers):
            if data.ndim == 2:
                layer_data = np.copy(data)
            elif data.ndim == 3:
                layer_data = np.copy(data[:, :, i])
            else:
                print('Can not display data in', data.ndim, 'dimensions.')

            if make_wireframe:
                ax.plot_wireframe(
                    xv, yv, z * np.ones(self.nn),
                    color='k',
                    alpha=g_alpha,
                    lw=g_lw)

            if not global_vm:
                vmin = np.nanpercentile(layer_data, 0.1)
                vmax = np.nanpercentile(layer_data, 99.9)

            zv = z * np.ones(self.nn)
            zv[np.isnan(layer_data)] = np.nan

            cube = ax.contourf(xv, yv,
                   layer_data, d_levels,
                   vmin=vmin,
                   vmax=vmax,
                   offset=z,
                   alpha=d_alpha,
                   cmap=cmap)


        if cbar:
               cbar = fig.colorbar(cube, shrink=0.6, aspect=5)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_zlabel(zlabel, fontsize=12)
        ax.yaxis._axinfo['label']['space_factor'] = 3.0

        plt.rcParams['axes3d.grid'] = ax_grid

        if x_lim is None:
            x_lim = (self.ds['X'].min(), self.ds['X'].max())

        if y_lim is None:
            y_lim = (self.ds['Y'].min(), self.ds['Y'].max())

        if z_lim is None:
            z_lim = (self.ds['Z'].max(), self.ds['Z'].min())

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.azim = azim
        ax.elev = elev
        ax.dist = dist

        fig.tight_layout(pad=0)
        if save_name is not None:
            fig.savefig(save_name, transparent=True,
                        bbox_inches='tight', pad_inches=0)

        if show_map:
            plt.show()

        return None

    def slice_3D():
        return None

        
