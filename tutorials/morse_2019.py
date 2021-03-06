#!/usr/bin/env python3

# Tobias Staal 2019
# tobias.staal@utas.edu.au

import numpy as np

# Import PyPNG
# https://pythonhosted.org/pypng/index.html
import png


def export_png(self,
               data,
               png_name,
               png_format='LA',
               v_min=0.,
               v_max=14.0,
               png_nx=3600,
               png_ny=1800,
               morse_proj=4326,
               set_geometry=True,
               bit_depth=8,
               flip=True,
               flipping_ax=0,
               interpol_method='nearest',
               confine_data='input',
               confine_mask=None,
               mask_to_value=None,
               clip=False):
    '''Save 2D array as png.file formatted for Morse et al vizualisation software
    Works well with global rasters for e.g. Gplates

    Keyword arguments:
    self -- is an agrid object
    data   --  2D array as string (label) or dataframe (XXX read also numpy XXX)
    png_name  --  Name of file to save 'foo.png'
    png_format -- 'RGB', 'RGBA', 'L' or 'LA' bit depth as ';N' eg 'RGB;16' or 'LA;8'
    v_min   --  Data value to replresent pixel value 0 or (0,0,0) (for 8 bit)
    v_max   --  Data value to replresent pixel value 255 or (255,255,255) (for 8 bit)
    png_nx, png_ny  --  Output size of png_file
    morse_proj  --  Set to epsg_4326, each degree = 10 pixels
    set_geometry   --  False if the agrid is already the same format as Morse 
    bit_depth  --  Bit depth of png file, for now only 8 bit
    interpol_method   --  Interpolation method, 'nearest', 'linear', 'cubic'
    confine_data   --  'estimate', 'mask', 'input' or None. Defines if transparant mask 
                        is made from an interpolated selection, a provided mask (confine_mask), 
                        the data itself or not confined. The later will extrapolate the map for
                        nearest interpolation. 
    confine_mask      --  Mask is used 
    mask_to_value      --  Set masked areas to zero, for formats that doesn't support alpha. 
    clip  --  If true, values will be clipped to set v_min and v_max, 
                                else normalisation. 

    Returns : Log string to print or write to textfile.

    Saves png file (png_name)

    The saved png file:
        3600x1800 pixels
        EPSG 4326

    See Morse et al 2019 (in prep)

    '''

    # String is taken as label
    data = self._user_to_array(data)

    if bit_depth == 16:
        d_type = np.uint16
        norm = 2**16 - 1
    else:
        d_type = np.uint8
        norm = 2**8 - 1
        if bit_depth != 8:
            print('Bit depth set to 8')

    # If the grid is already in the right extent, resolution and
    # projection, there is no need to do it again, and set_geometry can be False
    if set_geometry:
        # Reproject grid to Morse image, usually 4326
        xp, yp = proj.transform(proj.Proj(init='epsg:%s' % self.crs),
                                proj.Proj(init='epsg:%s' % morse_proj), self.xv, self.yv)

        # Resshape for interpolation
        vi = np.reshape(data, (data.size))
        xi = np.reshape(xp, (data.size))
        yi = np.reshape(yp, (data.size))

        # Making index of coordinates
        xi = ((xi * png_nx // 360) + png_nx // 2).astype('int')
        # Making index of coordinates
        yi = ((yi * png_ny / 180) + png_ny // 2).astype('int')

        # yyy as array index from top to bottom, hence -1
        xxx, yyy = np.meshgrid(range(0, png_nx), range(png_ny, 0, -1))
        data = interpolate.griddata((xi, yi), vi, (xxx, yyy),
                                    method=interpol_method,
                                    fill_value=np.nan)

    # Reverse order for png
    if flip:
        data = np.flip(data, axis=flipping_ax)

    # If nearest, interpolate extrapolate voronoi type fields, to remove them, we need to take a detour
    # and make a mask from a different interpolation technique, e.g. linear.
    if confine_data == 'estimate':
        vi = np.reshape(data, (data.size))
        xi = np.reshape(self.xv, (data.size))
        yi = np.reshape(self.yv, (data.size))
        xxx, yyy = np.meshgrid(range(0, png_nx), range(png_ny, 0, -1))
        alpha = np.isfinite(interpolate.griddata((xi, yi), vi, (xxx, yyy),
                                                 method='linear', fill_value=np.nan))
    elif confine_data == 'mask':
        alpha = confine_mask
    elif confine_data == 'input':
        alpha = np.isfinite(data)
    else:
        alpha = np.ones_like(data)

    # Set masked areas to zero
    if mask_to_value != None:
        data[~alpha] = mask_to_value

    # alpha is set by alpha array, not nan
    d_num = np.nan_to_num(data)

    # alpha from boolean to integer channel
    alpha = alpha * norm

    # np.clip values outside interval are clipped:
    if clip:
        n_png = norm * (np.clip(d_num, v_min, v_max) - v_min) / (v_max - v_min)
    else:
        n_png = norm * (d_num - v_min) / (v_max - v_min)

    n_png = np.rint(n_png)

    if png_format == 'L':
        png_write = n_png
    elif png_format == 'LA':
        png_write = np.dstack((n_png, alpha))
    elif png_format == 'RGB':
        png_write = np.dstack((n_png, n_png, n_png))
    elif png_format == 'RGBA':
        png_write = np.dstack((n_png, n_png, n_png, alpha))
    else:
        print('Not supported format, use L, LA, RGB or RGBA.')

    # Save png file
    png.from_array(png_write.astype(d_type), png_format).save(png_name)

    # Read back:
    read_file = png.Reader(png_name).asDirect()
    info = "\n".join("{}: {}".format(k, v)
                          for k, v in read_file[3].items())
    print(info, end='\n\n')


    read_file = np.array(list(read_file[2]))

    report = '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s' % (png_name,
                                                                             v_min, v_max, 
                                                                             bit_depth, interpol_method,
                                                                             np.nanmin(
                                                                                 data), np.nanmax(data),
                                                                             np.nanmin(
                                                                                 n_png) / norm, np.nanmax(n_png) / norm,
                                                                             np.nanmin(n_png), np.nanmax(
                                                                                 n_png),
                                                                             np.nanmin(read_file),  np.nanmax(read_file))

    read_file = None
    return report
