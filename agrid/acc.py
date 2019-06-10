#!/usr/bin/env python3


# Tobias Staal 2019
# tobias.staal@utas.edu.au
# version = '0.5.0'

# https://doi.org/10.5281/zenodo.2553966
#

# MIT License#

# Copyright (c) 2019 Tobias Stål#

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions: #

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.#

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


import requests
import os
import zipfile
import tarfile
import json
import datetime

import xarray as xr
from matplotlib import pyplot as plt
from tqdm import tqdm


def download(url,
             f_name,
             un_pack=True,
             data_files=None,
             unpack_path=None,
             check_first=True,
             block_size=1024,
             confirm_data=True,
             make_meta=True,
             meta_dict={},
             _return='files'):
    '''
    Download a file and unpack, if you prefer
    url -- url as string
    f_name -- file to download
    un_pack -- Unpack zip or tar if True
    check_first -- Check if file already exist before download
    block_size -- download blocks
    _return : select 'files' (list of strings, paths to extracted files), 'target', 'meta' or none
    '''
    f_path = os.path.dirname(f_name)

    if data_files is None:
        data_files = [f_name]

    # Check if any file is missing
    get_file = 'True'
    if check_first:
        for data_file in data_files:
            if os.path.isfile(data_file):
                get_file = False

    if get_file:
        if not os.path.exists(f_path):
            if os.path.dirname(f_name) != '':
                os.makedirs(f_path)
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        wrote = 0
        with open(f_name, 'wb') as f:
            for data in tqdm(r.iter_content(block_size),
                             total=total_size // block_size,
                             unit='KB',
                             unit_scale=True):
                wrote = wrote + len(data)
                f.write(data)
        if total_size != 0 and wrote != total_size:
            print("DOWNLOAD ERROR.")
        meta_dict = {**meta_dict, **{
            'url': url,
            'accessed': str(datetime.datetime.now())}
        }
    else:
        print('File %s already exists.' % f_name)

    if unpack_path is None:
        unpack_path = f_path

    extension = os.path.splitext(f_name)[1]

    if extension == '.zip' and un_pack:
        zip_ref = zipfile.ZipFile(f_name, 'r')
        zip_ref.extractall(unpack_path)
        recieved_files = zip_ref.namelist()
        zip_ref.close()

    elif extension in ['.gz', '.tar'] and un_pack:
        tar = tarfile.open(f_name)
        tar.extractall(path=unpack_path)
        recieved_files = tar.getnames()
        tar.close()

    elif extension == '.bz2' and un_pack:
        tar = tarfile.open(f_name, "r:bz2")
        tar.extractall(path=unpack_path)
        recieved_files = tar.getnames()
        tar.close()

    else:
        recieved_files = data_files

    # Check that we got the file(s) and make meta files
    for data_file in data_files:
        if os.path.isfile(data_file):
            if make_meta:
                meta_name = os.path.splitext(data_file)[0] + '.json'
                data_meta_dict = {**meta_dict, **
                                  {'data_size': os.path.getsize(data_file)}}
                if not os.path.isfile(meta_name):
                    with open(meta_name, 'w') as fp:
                        json.dump(data_meta_dict, fp)
        else:
            print('Missing', data_file)

    if _return == 'target':
        return data_files
    elif _return == 'meta':
        return meta_dict
    elif _return == 'none':
        return None
    elif _return == 'files':
        return recieved_files
    else:
        return data_files, meta_dict


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


# def open(file_name=None):
#    '''
#    Open dataset from netCDF.
#    file_name string
#    returns dataset.
#    '''
#    if file_name == None:
#        file_name = '%s.nc' % type(self).__name__
#    return xr.open_dataset(file_name)
