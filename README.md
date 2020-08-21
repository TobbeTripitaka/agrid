[![DOI](https://zenodo.org/badge/163904331.svg)](https://zenodo.org/badge/latestdoi/163904331) 
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

# agrid
A grid for modelling, analyse, map and visualise multidimensional and multivariate data. The module contains a class for generating grid objects with variables and functions that defines a multidimensional space with defined extent. 

Main features:
  - Labelled dimensions coordinates using xarray
  - Fast process of high resolution and low resolution data
  - Using dask arrays
  - 2D map plots and cross-sections
  - 3D visualisation
  - Features for modelling, analysis and visualisation

The repository contains: 
 - Module with class for functions and variables. 
 - Jupyter notebook tutorials:
  - 1. The grid object
  - 2. Import data
  - 3. Visualize data
  - 4. Introduction to processing and modelling using grid. 

Software paper availible here: [JORS](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.287/)

---
## Instructions: 

The package can be installed by adding it to your Python path, but can also be an incorporated part of you project. 

**Alternative 1**
Using pip: 
`pip install agrid` or `pip3 install agrid`

Set up an envirinment for the installation. Use e.g. [virtualenv](https://virtualenv.pypa.io/en/latest/) or [venv](https://docs.python.org/3/library/venv.html) to set up your environment. For more complex isnallations, have a look at [pipenv](https://pipenv.readthedocs.io/en/latest/) or [poetry](https://poetry.eustace.io). 

OSX users can set up an environment directely in conda. 

`conda install -c tobbetripitaka agrid `



**Alternative 2**
Download module and import agrid from local path

See example in the Jupyter Notebook agrid. 

---

To get started:

    from agrid.grid import Grid
    world = Grid()
    # The grid is already populated with default coordinates
    print(world.ds) 

Further tutorials are available at [GitHub tutorials](https://github.com/TobbeTripitaka/agrid/tree/master/tutorials) 

## Methods to import data

Data can be imported from grids, vector data files, rasters or numerically. 


## Methods to export data

Data can be exported and saved as netCDF, raster files, text files. 

## Methods to visualise data

Visualisation is not the core use of agrid, but it contains basic functions to plot maps and 3D renderings. 

## Additional functions

Additional functions are included to download data and structure the content.  

\---

If used in publication, please cite:

    @article{Staal2020a,
    abstract = {Researchers use 2D and 3D spatial models of multivariate data of differing resolutions and formats. It can be challenging to work with multiple datasets, and it is time consuming to set up a robust, performant grid to handle such spatial models. We share 'agrid', a Python module which provides a framework for containing multidimensional data and functionality to work with those data. The module provides methods for defining the grid, data import, visualisation, processing capability and export. To facilitate reproducibility, the grid can point to original data sources and provides support for structured metadata. The module is written in an intelligible high level programming language, and uses well documented libraries as numpy, xarray, dask and rasterio.},
    author = {St{\aa}l, Tobias and Reading, Anya M.},
    doi = {10.5334/JORS.287},
    issn = {20499647},
    journal = {Journal of Open Research Software},
    keywords = {Multivariate processing, Python, Regular grid, Spatial model},
    month = {jan},
    number = {1},
    pages = {1--10},
    publisher = {Ubiquity Press, Ltd.},
    title = {{A grid for multidimensional and multivariate spatial representation and data processing}},
    volume = {8},
    year = {2020}
    }