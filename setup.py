import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


requirements = [
'Cartopy>=0.17.0',
'dask>=2.5.1',
'numpy>=1.17.2'
'tqdm>=4.36.1',
'requests>=2.22.0',
'geopandas>=0.6.1',
'imageio>=2.6.0',
'rasterio>=1.0.21',
'scipy>=1.3.1',
'xarray>=0.14.1'
]

setuptools.setup(
    name='agrid', 
    version="0.3.9.6",
    author="Tobias Staal",
    author_email="tobbetripitaka@gmail.com",
    description='A grid for spatial multidimensional processing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/TobbeTripitaka/agrid',
    packages=setuptools.find_packages(),
    install_requires = requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
