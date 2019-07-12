#!/usr/bin/env python3

# Tobias Staal 2019
# tobias.staal@utas.edu.au

# Features for special use in studies.

import numpy as np
import rasterio

import fiona  # Replace with geopandas, update model!
from scipy.ndimage.filters import gaussian_filter


def read_model(self,
               shape_file_name,
               kernel_size=None,
               std=5000,
               p_att='P',
               w_att='W',
               norm=True):
    '''Read and colvolve line vectors. 

    Keyword arguments:
    file_name  --  Name of shapefile containing lines
    kernel_size --  Size of Gaussian kernel. Larger is slower, but generate a broader 
                        gradient. 
    sigma --  Standard deviation for layer
    norm  --  Normalise distribution to [0,1]
    p_att --  Attribute for vertices P (rating)
    w_att  --  Attribute for vertices W (rating)
    See Staal et al (2019) (submitted)
    '''
    if kernel_size is None:
        kermel_size = (self.nx // 2, self.ny // 2)

    with fiona.open(shape_file_name, 'r') as src:
        convolved = geom_np = np.zeros(self.shape2 + (len(src),))
        for i, geom in enumerate(src):
            geom_np[:, :, i] = rasterio.features.rasterize(
                [geom['geometry']],
                out_shape=self.shape2,
                transform=self.transform) * 2 + geom['properties'][p_att] / 2

            sigma_0 = (std + std / geom['properties']
                       [w_att]) / (2 * self.res[0])
            sigma_1 = (std + std / geom['properties']
                       [w_att]) / (2 * self.res[1])

            convolved[:, :, i] = gaussian_filter(
                geom_np[:, :, i], (sigma_0, sigma_1))
    model = np.sum(convolved, axis=2)
    if norm:
        model[np.isnan(model)] = 0
        model = (model - np.min(model)) / np.ptp(model)
    return model
