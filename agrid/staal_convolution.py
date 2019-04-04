#!/usr/bin/env python3


# Tobias Staal 2019
# tobias.staal@utas.edu.au
# version = '0.5.0'

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





def read_model(self,
               shape_file_name,
               kernel_size=100,
               std=5000,
               p_att = 'P',
               w_att = 'W',
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
    import fiona  # Replace with geopandas, update model!
    import numpy as np
    from scipy.ndimage.filters import gaussian_filter

    with fiona.open(shape_file_name, 'r') as src:
        blurred = geom_np = np.zeros(self.shape2 + (len(src),))
        for i, geom in enumerate(src):
            geom_np[:, :, i] = rasterio.features.rasterize(
                [geom['geometry']],
                out_shape=self.shape2,
                transform=self.transform) * 2 + geom['properties'][p_att] / 2

            sigma_0 = (std + std / geom['properties'][w_att]) / (2 * self.res[0])
            sigma_1 = (std + std / geom['properties'][w_att]) / (2 * self.res[1])

            convolved[:, :, i] = gaussian_filter(geom_np[:, :, i], (sigma_0, sigma_1))
    model = np.sum(convolved, axis=2)
    if norm:
        model[np.isnan(model)] = 0 
        model = (model - np.min(model)) / np.ptp(model)
    return model
