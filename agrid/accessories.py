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



import requests
import os
import zipfile
import tarfile

import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def download(url, f_name, 
             un_pack=True, 
             unpack_path = None,
             check_first=True,
             block_size = 1024):
    '''
    ownload a file and unpack, if you prefer
    url -- url as string
    f_name -- file to download
    un_pack -- Unpack zip or tar if True
    check_first -- Check if file already exist before download
    block_size -- download blocks
    
    Rather sketchy method to simplify workflow and reproducibility of agrid. 
    '''
    f_path = os.path.dirname(f_name)
    
    if not os.path.isfile(f_name) or not check_first:
        if not os.path.exists(f_path):
            if os.path.dirname(f_name) != '':
                os.makedirs(f_path)
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0)); 
        wrote = 0 
        print('Download:', f_name)
        with open(f_name,'wb') as f:
            for data in tqdm(r.iter_content(block_size), 
                    total=total_size//block_size , 
                    unit='KB', 
                    unit_scale=True):
                wrote = wrote  + len(data)
                f.write(data)
        if total_size != 0 and wrote != total_size:
            print("DOWNLOAD ERROR.") 

        print('Done!')
    else:
        print('File %s already exists.'% f_name)
        
    if unpack_path == None:
        unpack_path = f_path

    extension = os.path.splitext(f_name)[1]

    if extension == '.zip' and un_pack:
        zip_ref = zipfile.ZipFile(f_name, 'r')
        zip_ref.extractall(unpack_path)
        zip_ref.close()
    
    elif extension in ['.gz', '.tar'] and un_pack:
        tar = tarfile.open(f_name)
        tar.extractall(path=unpack_path)
        tar.close()

    elif extension == '.bz2' and un_pack:
        tar = tarfile.open("path_to/test/sample.tar.bz2", "r:bz2")  
        tar.extractall(path=unpack_path)
        tar.close()


    return os.path.getsize(f_name)


def quick_look_netcdf(f_name):
    '''
    '''
    data = xr.open_dataset(f_name)
    print(data)

    for array in data.data_vars:
        plt.imshow(data[array].values)
        plt.show()
        info = str(data[array])
        print(info)
    return os.path.getsize(f_name)


#def open(file_name=None):
#    '''
#    Open dataset from netCDF. 
#    file_name string
#    returns dataset.
#    '''
#    if file_name == None:
#        file_name = '%s.nc' % type(self).__name__
#    return xr.open_dataset(file_name)
