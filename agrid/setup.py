
import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='agrid',
    version='0.3.3',
    author="Tobias Staal",
    author_email="tobbe@tripitaka.se",
    description="A multidimensional grid for scientific computing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/TobbeTripitaka/agrid',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
    install_requires=required)






#[        'numpy',
#        'xarray',
#        'matplotlib',
#        'pyproj',
#        'scipy',
#        'fiona',
#        'geopandas',
#        'rasterio',
#        'imageio', ]