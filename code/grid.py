#!/usr/bin/env python3

# Tobias Staal 2019
# tobias.staal@utas.edu.au
# version = '0.4.5'

import os, sys #, requests

import numpy as np
import xarray as xr
import pyproj as proj

import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.crs import *
from rasterio import features

from matplotlib import pyplot as plt
#import matplotlib.path as mpath
from mpl_toolkits.basemap import Basemap
#from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import interpolate #, signal #clean up

import scipy.ndimage

#from scipy.stats import norm
#import matplotlib.mlab as mlab
#from skimage.transform import resize 
#from skimage.draw import line_aa, line #clean up


km = 1000
milli = 0.001

class Grid(object):
    """
    Methods to set up, save and modify a multidimensional grid.
    """
    
    verbose = False
    
    def __init__(self, 
                 km=km, 
                 x1y1x2y2 = None,
                 wsen = None,
                 left = -4200*km,
                 up = 4200*km,
                 right = 4200*km,
                 down = -4200*km,
                 res = [10*km, 10*km],
                 depths = [0.*km, 8.*km, 16.*km, 40.*km, 350.*km],
                 crs_tgt = 3031,
                 crs_src = 4326, 
                 sub_skips = [10, 100], 
                 color_coord = 'RGB',
                 use_dask = True,
                 chunk_n = 10,
                 coord_d_type = np.float32):
                


        # default grid
        self.res = list(res)
        self.depths = list(depths)


        self.x1y1x2y2 = (left, up, right, down)
        self.wsen = (left, down, right, up)
        self.left, self.up, self.right, self.down = left, up, right, down

        self.ul = (self.left, self.up)
        self.ur = (self.right, self.up)
        self.lr = (self.right, self.down)
        self.ll = (self.left, self.down)

        self.nx = int(abs(right - left)//res[0])
        self.ny = int(abs(down - up)//res[1])
        self.nn = (self.ny, self.nx)

        self.transform = rasterio.transform.from_bounds(left, down, right, up, self.nx, self.ny)
        self.coord_d_type = coord_d_type

        #rasterio.transform.from_origin(self.left, self.up, *self.res)

        #Make xarray dataset
        self.ds = xr.Dataset()

        self.ds.coords['X'] = np.linspace(left, right, self.nx).astype(self.coord_d_type)
        self.ds.coords['Y'] = np.linspace(up, down, self.ny).astype(self.coord_d_type)
        self.nz = len(depths)
        self.ds.coords['Z'] = np.array(depths).astype(coord_d_type)
        self.depths = depths
        self.ds.coords[color_coord] = list(color_coord)

        self.shape2 = (self.ny, self.nx) # Numpy arrays are indexed rows and columns (y,x)
        self.shape3 = (self.ny, self.nx, self.nz) 

        self.xv, self.yv = np.meshgrid(self.ds.coords['X'], self.ds.coords['Y'])
        self.ds.coords['XV'] = (('Y', 'X'), self.xv.astype(coord_d_type))
        self.ds.coords['YV'] = (('Y', 'X'), self.yv.astype(coord_d_type))
            
        self.crs_src = crs_src
        self.crs_tgt = crs_tgt
        
        self.lon, self.lat = proj.transform(proj.Proj(init='epsg:%s'%crs_tgt), 
                                  proj.Proj(init='epsg:4326'), self.xv, self.yv) 
        
        self.ds.coords['lat'] = (('Y', 'X'), self.lat.astype(coord_d_type))
        self.ds.coords['lon'] = (('Y', 'X'), self.lon.astype(coord_d_type))
        

        #Start using dask
        if use_dask:
            self.ds = self.ds.chunk({'X': self.nx//chunk_n, 'Y': self.ny//chunk_n})


        #Make coarse grid
        

        if self.verbose:
            print('Created model:',self.name)
            print('Class corners',x1y1x2y2)
            print('Class nx ny', self.nx, self.ny, )
            print('Depths:', depths)


    # Accessories
    def _check_if_in(self, xx, yy, margin=2):
        '''Generate an array of the condition that coordinates are whitin the model or not. 
        xx = list or array of x values
        yy = list or array of y values
        margin = extra cells added to mitigate extrapolation of interpolation along the frame

        returns boolean array True for points within the frame.
        '''
        x_min = self.left - margin*self.res[0]
        x_max = self.right + margin*self.res[0]
        y_min = self.down - margin*self.res[1]
        y_max = self.up + margin*self.res[1]
        return (xx > x_min) & (xx < x_max) & (yy > y_min) & (yy < y_max)


    def _set_meridian(self, x_array, center_at_0=True):
        '''
        Sloppy function to change longetude values from [0..360] to [-180..180]
        x_array :   Numpy array with longetude values (X)
        center_at_0 : Bool select direction of conversion. 
        '''

        if center_at_0:
            x_array[x_array>180] = x_array[x_array>180]-360
        else: 
            x_array[x_array<0] = x_array[x_array<0]+360
        return x_array



    def open(file_name=None):
        '''
        Open dataset from netCDF. 
        file_name string
        returns dataset.
        '''
        if file_name == None:
            file_name = '%s.nc' % type(self).__name__
        return xr.open_dataset(file_name)
    
    def save(self,data= None, file_name='grid.nc'):
        '''
        Saves dataset to netCDF. 
        file_name string
        returns size of file.
        '''
        if data == None: 
            data = self.ds
        data.to_netcdf(file_name)
        return os.path.getsize(file_name)
       
    def make_info(self,ds = None,file_name=None):
        if ds == None:
            ds = self.ds        
        if file_name == None:
            file_name = 'info_%s.txt' % self.name
        np.savetxt(file_name, [], header='Info %s:'%str(ds))
        return 0
        
    def land_mask(self, polygon_frame=None, polygon_res=None, all_touched = True, land_true=True):
        '''
        Create a 2D array with 
        '''

        if polygon_frame == None:
            pass
            #Download global vector file in with the resolution option of polygon_res=None)

        mask = 1 # razerize map for section
        return mask


    def change_coord(self, 
                array, 
                old, 
                new, 
                fill_value=np.nan, 
                kind='linear', 
                axis=2, 
                bounds_error=False):
        '''
        Function interpolate dimention into new defined depth from coord or list. 
        '''
        for label in [array, old, new]:
            if isinstance(label, str):
                label = self.ds[label].values
                print(array, old, new)


        return interpolate.interp1d(old, 
                    array, 
                    axis=axis,
                    bounds_error=bounds_error, 
                    kind = kind,
                    fill_value=fill_value)(new)

    def fold_to_low_res(self, large, small):
        '''
        Function takes high resolution 2D array (large) and places subarrays in additional dimentions. 
        The output array have the same resolution as the second array (small) and can be computed together. 
        nx and nn of large must be a multipe of nx, and ny of small.

        '''
        res = (np.shape(large)[0]//np.shape(small)[0], np.shape(large)[1]//np.shape(small)[1])
        return large.values.reshape(np.shape(small.values)[0],res[0],
                                      np.shape(small.values)[1],res[1]).transpose(0,2,1,3)

    def flatten_to_high_res(self, folded, large):
        '''
        Function flatten a processed array back to high dimention. Reverse of fold_to_low_res. 
        Returns a high resolution array. 

        '''
        return folded.transpose(0,2,1,3).reshape(np.shape(large.values)[0],
                                             np.shape(large.values)[1])

# Import data
    def assign_shape(self, shape_file, attribute, 
                     z_dim = False, z_max='z_max', z_min='z_min', 
                     all_touched = True, burn_val=None):
        '''
        https://gis.stackexchange.com/questions/216745/get-polygon-shapefile-in-python-shapely-by-clipping-linearring-with-linestring/216762
        '''

        shape = gpd.read_file(shape_file).to_crs({'init': 'epsg:%s'%self.crs_tgt})

        if burn_val != None:
            shape[attribute] = [burn_val] * len(shape)

        if z_dim:
            data = np.empty(self.shape3)

            z_select = np.empty([len(shape)]).astype('bool')
            for i, zi in enumerate(self.depths):    
                z_select = [z_min <= zi and z_max >= zi for 
                                       z_min, z_max in zip(shape[z_min], shape[z_max] )]

                shape_select = shape[z_select]
                to_burn = ((geom,value) for geom, value in zip(shape_select.geometry, shape_select[attribute]))
                data[:,:,i] = features.rasterize(
                    shapes=to_burn, 
                    out_shape=self.shape2, 
                    transform=self.transform, 
                    fill=np.nan,
                    all_touched = all_touched) 

        else:
            data = np.empty((self.nx, self.ny))
            to_burn = ((geom,value) for geom, value in zip(shape.geometry, shape[attribute]))
            data = features.rasterize(
                    shapes=to_burn, 
                    out_shape=self.shape2, 
                    transform=self.transform, 
                    fill=np.nan,
                    all_touched = True) 
        return data
        


    def read_grid(self, f_name, 
            xyz = ('x','y', 'z'), 
            interpol_method = 'linear',
            crs_src = None,
            crs_tgt = None,
            only_frame = True,
            deep_copy = False,
            set_center = False):

        if crs_src == None:
            crs_src = self.crs_src

        if crs_tgt == None:
            crs_tgt = self.crs_tgt

        array = xr.open_dataset(f_name, autoclose = True, ).copy(deep=deep_copy)

        x = array[xyz[0]].values
        y = array[xyz[1]].values

        if set_center:
            x = self._set_meridian(x)

        xx, yy = np.meshgrid(x, y)

        xv, yv = proj.transform(proj.Proj(init='epsg:%s'%crs_src), 
                proj.Proj(init='epsg:%s'%crs_tgt), xx, yy)

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
            zi, (self.ds.coords['XV'], self.ds.coords['YV']), method = interpol_method)


    def read_ascii(self,
            f_name,
            x_col = 0,
            y_col = 1,
            data_col = 2,
            interpol = 'linear',
            crs_src = None,
            crs_tgt = None,
            coord_factor = 1,
              ):
        '''
            Function reads textfile, e.g. csv, to grid. 
            f_name : String, name of file to import
            x_col : index for column holding x values in given crs
            y_col : index for column holding y values in given crs

        '''
        if crs_tgt == None:
            crs_tgt = self.crs_tgt
        if crs_src == None:
            crs_src = self.crs_src
    
        table = np.loadtxt(f_name) #Add kwargs

        print(table[:5,:])

        table[:,x_col] *= coord_factor
        table[:,y_col] *= coord_factor




        xx, yy = proj.transform(proj.Proj(init='epsg:%s'%crs_src), 
                proj.Proj(init='epsg:%s'%crs_tgt), table[:,x_col], table[:,y_col])

        is_in = self._check_if_in(xx, yy)

        return interpolate.griddata((xx[is_in], yy[is_in]), 
                         table[:,data_col][is_in], 
                         (self.ds.coords['XV'], self.ds.coords['YV']), 
                         method = interpol)
    
    def read_raster(self, 
            raster_name, 
            raster_bounds = None, 
            src_crs= None,
            source_extra = 1000,
            resampling =  None, 
            sub_sampling = None,
            sub_window = None,
            num_threads = 4,
            no_data = None,
            rgb_convert = True, 
            bit_norm = 255):
        '''

        Options for resampling:
            Resampling.nearest, 
            Resampling.bilinear, 
            Resampling.cubic, 
            Resampling.cubic_spline, 
            Resampling.lanczos, 
            Resampling.average    

        '''
        from rasterio.plot import reshape_as_image
        from rasterio.warp import Resampling

        in_raster = rasterio.open(raster_name)

        if src_crs == None:
            src_crs = in_raster.crs
            if self.verbose:
                print(src_crs)

        if resampling == None:
            resampling = Resampling.nearest

        if self.verbose:
            print('Raster bounds:', in_raster.bounds, in_raster.shape)


        dst_crs = CRS.from_epsg(self.crs_tgt) 


        if sub_sampling in (None, 0, 1):
            sub_sampling = 1

        raster_shape = (in_raster.count,in_raster.height//sub_sampling, in_raster.width//sub_sampling)
        source = in_raster.read(out_shape=raster_shape, window = sub_window)

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
            num_threads=num_threads)

        if (rgb_convert and in_raster.count > 2):
            dst_array = reshape_as_image(dst_array/bit_norm).astype(float)
    
        if in_raster.count == 1:
            dst_array = dst_array[0,:,:]
    
        if no_data != None:
            dst_array[dst_array==no_data] = np.nan
        
        return dst_array


# Exports
    def grid_to_grd():
        '''
        Save as netCDF
        '''
        return None


    def grid_to_raster(self, 
                    ds,
                    raster_name = 'raster_export.tif'
                    ):
        '''
        Save as geoTIFF
        https://gis.stackexchange.com/questions/37238/writing-numpy-array-to-raster-file?rq=1
        ''' 
        arr = ds.values
        out_raster = rasterio.open(raster_name, 'w', driver='GTiff',
                            height = arr.shape[0], width = arr.shape[1],
                            count=1, dtype=str(arr.dtype),
                            crs={'init': 'EPSG:%s'%self.crs_tgt},
                            transform=self.transform)
        out_raster.write(arr, 1)
        out_raster.close()
        return None

    def frame_to_polygon(self, 
        file_name, 
        grid_driver = 'ESRI Shapefile', 
        frame_crs = None,
        coordinates = None,
        proj_crs = None,
        id_attribute=1):

        from shapely.geometry import Polygon


        if proj_crs == None:
            proj_crs = 3031 

        if frame_crs == None:
            frame_crs = 4326 
        
        if coordinates == None:
            coordinates = [(self.left, self.up),
                (self.right, self.upp), 
                (self.right, self.down), 
                (self.left, self.down)]
    
        ds = gpd.GeoDataFrame()
        ds.crs = '+init=epsg:%s' %proj_crs
        ds.loc[0, 'geometry'] = Polygon(coordinates)
    
        ds = ds.to_crs('+init=epsg:%s' %frame_crs)
        ds.to_file(file_name, driver='ESRI Shapefile')
        return ds
   
        #coordinates = [(self.left, self.up),
        #        (self.right, self.up), 
        #        (self.right, self.down), 
        #        (self.left, self.down)]

        #poly = Polygon(coordinates)#

#

 #       ds = gpd.GeoDataFrame()
  #      ds.loc[0, 'geometry'] = poly
   #     ds.crs = {'init': 'epsg:3031', 'no_defs': True}




    #    df = gpd.GeoDataFrame(poly, geometry='geometry')
     #   df.loc[0, 'geometry'] = poly
      #  df.crs = {'init': 'EPSG:%s'%self.crs_tgt}
       # df.to_file(file_name, driver=grid_driver)

        #return None


    def frame_to_point():
        '''
        Save points in center of each cell. 

        '''
        from shapely.geometry import Point


        return None


    def frame_to_ascii():
        '''
        Save to asc format
        https://gis.stackexchange.com/questions/37238/writing-numpy-array-to-raster-file?rq=1
        http://resources.esri.com/help/9.3/ArcGISengine/java/Gp_ToolRef/Spatial_Analyst_Tools/esri_ascii_raster_format.htm
        '''

# Vizualisations
    def extract_profile():
        return 0


    def map_grid(self, im_data, 
                 vmin = None, 
                 vmax = None, 
                 cmap ='gray',
                 cbar = False,
                 extent = None,
                 line_c = 'gray',
                 line_grid_c = 'gray',
                 line_w = 0.9, 
                 circ_map = False,
                 figsize=None, 
                 land_only=True,
                 ocean_color = 'w',
                 no_land_fill = np.nan,
                 title = None,
                 save_name = None,
                 show_map = True,
                 draw_coast = True,
                 draw_grid = True,
                 par = None, 
                 mer = None):
        
        def create_circular_mask(h, w, center=None, radius=None):
            center = [int(w/2), int(h/2)]
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
            mask = dist_from_center <= radius
            return mask

        if isinstance(im_data, str):
            im_data = self.ds[im_data].values

        if figsize == None:
            figsize = (12, 12*self.nx/self.ny)

        if par == None:
            par = np.arange(-90,90, 10)

        if mer == None:
            mer = np.arange(-180,180,45)

        m = Basemap(llcrnrlon=self.lon[-1,0],
            llcrnrlat=self.lat[-1,0],
            urcrnrlon=self.lon[0,-1],
            urcrnrlat=self.lat[0,-1], 
            resolution = 'i', 
            epsg=self.crs_tgt)

        if extent ==None:
            extent = [self.left,self.right,self.down,self.up]

        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        ax.axis('off')
        
        if im_data is not None:
            try:
                im_data = im_data.values
            except:
                pass

            if vmin == None:
                vmin = np.nanpercentile(im_data, 0.1)
            if vmax == None:
                vmax = np.nanpercentile(im_data, 99.9)

            if circ_map:
                h, w = im_data.shape[:2]
                mask = create_circular_mask(h, w, radius=h/2)
                im_data[~mask] = np.nan

            im = m.imshow(im_data,
                extent = extent,
                origin='upper', 
                vmin=vmin, 
                vmax=vmax, 
                cmap = cmap,
                zorder=5)

            if cbar:
                fig.colorbar(im, orientation='vertical',fraction=0.046, pad=0.04)

        if land_only:
            pass

        if draw_coast:
            m.drawcoastlines(color=line_c, linewidth=line_w, zorder=10)
            #ax.coastlines(resolution='50m', color=line_c, linewidth=1)
        if draw_grid:
            m.drawparallels(par, color=line_grid_c, alpha = 0.9,zorder=15)
            m.drawmeridians(mer, color=line_grid_c, alpha = 0.9, latmax=88, zorder=15)

        if title != None:    
            fig.suptitle(title)
        
        fig.tight_layout(pad=0)
        if save_name != None:
            plt.savefig(save_name, transparent=True, 
                        bbox_inches='tight', pad_inches = 0)
            print('Saved as', save_name)
        
        if show_map:
            plt.show()

        return None
  
    def look(self, rasters, file_name='look.png', 
             interp_method = None, 
             color_map = 'viridis',
             show=True, 
             save=False, 
             max_n_plots=16,
             x_unit = '(km)', 
             y_unit = '(km)', 
             ref_range = None
            ): 
        """
        Quick look of 2D slices. 
        """
        
        if len(rasters)>max_n_plots:
            print('To much to see!')
            return 0
        else:
            n_sq = int(np.ceil(np.sqrt(len(rasters))))
            fig, ax = plt.subplots(n_sq,n_sq,figsize=(10,10))
            
            for k, raster in enumerate(rasters):      
                x_coords = raster.coords[sorted(raster.dims)[0]]
                y_coords = raster.coords[sorted(raster.dims)[1]]
                
                if ref_range is None:
                    plot_range = (np.nanmin(raster.values), np.nanmax(raster.values))
                else:
                    plot_range = ref_range
                
                ax[k//n_sq, k%n_sq].imshow(raster.values, 
                                interpolation=interp_method, 
                                cmap= color_map, 
                                extent = self.x1y1x2y2,
                                vmin=plot_range[0], vmax=plot_range[1])
                ax[k//n_sq, k%n_sq].title.set_text('%s' %(raster.name,))
                ax[k//n_sq, k%n_sq].set_aspect('auto')
                ax[k//n_sq, k%n_sq].set_ylabel(y_coords.name + x_unit)
                ax[k//n_sq, k%n_sq].set_xlabel(x_coords.name + y_unit)

                # Print depth values
                # Print slice in title
                
        fig.tight_layout(pad=0)
        if save:
            fig.savefig(file_name, transparent=True, bbox_inches='tight', pad_inches=0)

        if show:
            plt.show()
        return 0

    def slider(self, data, 
          kdims=['z', 'x', 'y'], 
          vdims=['v'], 
          flat=['x', 'y'], 
          slider='z',
          invert_yaxis=False,
          sub_sample = 4 ):
        '''
        Function displays 3D grid with slider to 

        '''
        import holoviews as hv
        hv.extension('bokeh',logo=False)

        ds =  hv.Dataset((np.arange(np.shape(data)[2]),
                 np.linspace(0., 1., np.shape(data)[1]),
                 np.linspace(0., 1., np.shape(data)[0])[::-1],
                 data),
                kdims=kdims,
                vdims=vdims)
        return ds.to(hv.Image, flat).redim(slider).options(colorbar=True, 
            invert_yaxis=invert_yaxis).hist()


# Features
    
    def read_model(self, 
            file_name, 
            kernel_size=100, 
            sigma=5000,
            norm = True):
        '''
        Read and colvolve line vectors. 
        See Staal et al 2019
        '''
        import fiona # Replace with geopandas, update model!
        from scipy.ndimage.filters import gaussian_filter

        with fiona.open(file_name,'r') as src:
            blurred = geom_np = np.zeros(self.shape2 + (len(src),) )
            for i, geom in enumerate(src):
                geom_np[:,:,i] = rasterio.features.rasterize(
                        [geom['geometry']], 
                        out_shape=self.shape2, 
                        transform=self.transform) * 2 + geom['properties']['P']/2

                std = (sigma + sigma/geom['properties']['W']) / (self.res[0]+self.res[1])
                blurred[:,:,i] = gaussian_filter(geom_np[:,:,i], std)
        model = np.sum(blurred,axis=2)
        if norm:
            model = (model-np.min(model))/np.ptp(model)
        return model
    


    def export_morse_png(self, 
            v, 
            png_name, 
            v_min = 0., 
            v_max = 14.0, 
            nx=3600, 
            ny=1800, 
            morse_proj = 4326,
            set_geometry = True,
            bit_depth = 8,
            interpol_method = 'nearest', 
            confine_nearest = False, 
            rgb = True, 
            clip = False):
        '''
        This function reads a 2D layer and export as a global png, nx x ny pixels 
        At the moment it only takes RGBA 8 bit png with size 3600x1800 pixels. EPSF 4326. 
        See Morse et al 2019
        '''

        import imageio 

        # String is taken as label
        if isinstance(v, str):
            v = self.ds[v].values
    
        if bit_depth == 16:
            d_type = np.uint16
        else:
            d_type = np.uint8
            if bit_depth != 8:
                print('Bit depth set to 8')

        norm = 2**bit_depth-1

        # If the grid is allready in the right extent, resolution and projection, there is no need to do it again. 
        if set_geometry:
    
            # Reproject grid to Morse image, usually 4326
            xp, yp = proj.transform(proj.Proj(init='epsg:%s'%self.crs_tgt), 
                proj.Proj(init='epsg:%s' % morse_proj), self.xv, self.yv)
    
            # Resshape for interpolation
            vi = np.reshape(v, (v.size))
            xi = np.reshape(xp, (v.size))
            yi = np.reshape(yp, (v.size))
     
            xi = ((xi*nx//360)+nx//2).astype('int') # Making index of coordinates
            yi = ((yi*ny/180)+ny//2).astype('int') # Making index of coordinates

            xxx, yyy = np.meshgrid(range(0,nx), range(ny, 0, -1)) #yyy as array index from top to bottom

            v = interpolate.griddata((xi, yi), vi, (xxx, yyy), 
                               method = interpol_method,
                               fill_value = np.nan)


            # If nearest, interpolate extrapolate voronoi type fields, to remove them, we need to take a detour 
            # and make a mask from a diffrenet interpolation technique, e.g. linear. 
            if interpol_method == 'nearest' and confine_nearest:
                alpha = (norm*np.isfinite(interpolate.griddata((xi, yi), vi, (xxx, yyy), 
                               method = 'linear',
                               fill_value = np.nan))).astype(d_type) 
            else:
                alpha = (norm*np.isfinite(v)).astype(d_type)
        else:
            alpha = norm*np.ones_like(v).astype(d_type)
    
        # alpa is set by alpha array, not nan
        v = np.nan_to_num(v)

        #np.clip values outside interval are clipped:
        if clip:
            v_png = (np.clip(v, v_min, v_max)-v_min)/(v_max-v_min)
        else:
            v_png = (v-v_min)/(v_max-v_min)

        #png saved as uint
        png = (norm*v_png).astype(np.uint8)

        if rgb:
            png_write = np.dstack((png, png, png, alpha))
        else:
            png_write = np.dstack((png, alpha))
    
        imageio.imwrite(png_name, png_write)
        read_file = imageio.imread(png_name)[:,:,:-1]
    
        # Return string with report of convenrtiom. 
        report  = '\n%s \nmin v: %s max v: %s bit depth: %s\n' %(png_name, v_min, v_max, bit_depth)
        report += 'bands: %s interpolation: %s\n' %(np.shape(png_write)[2], interpol_method)
        report += 'v \t  norm \t  to png \t png \n'
        report += '%.3f \t  %.3f \t %s \t  %s \n' % (np.nanmin(v),
                                                  np.nanmin(v_png),
                                                  np.nanmin(png),
                                                  np.nanmin(read_file))
        report += '%.3f \t  %.3f \t %s \t  %s \n' % (np.nanmax(v),
                                                  np.nanmax(v_png),
                                                  np.nanmax(png),
                                                  np.nanmax(read_file))
    
        read_file = None
        return report



