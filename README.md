[![DOI](https://zenodo.org/badge/163904331.svg)](https://zenodo.org/badge/latestdoi/163904331) 
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

# agrid
A grid for modelling, analyse, map and visualise multidimensional and multivariate data. The module contains a class for generating grid objects with variables and functions that deifines a multidimetional space with spatial extent. 

This shared code can be used as a package or just copy and paste snippets into your project. I've tried to keep the dependencies to a minimum and let the functions be rather independent. 

Main features:
  - Named coordinates using xarray
  - Cunning appraoch to process high resolution and low resoultion data quickly
  - Dask arrays
  - 2D map plots and crossections
  - 3D visualisation. 
  - Features for modelling, analysis and visualisation
 
The repo contains: 
 - Module with class for functions and variables. 
 - Jupyter notebook tutorials:
  - 1. The grid object
  - 2. Import data
  - 3. Vizualise data
  - 4. Intriduction to processing and modelling using grid. 
 - TikZ code for generating illustrations.
 
To come soon: 
 - Software Metapaper
 - More code



The code can be uses as a package, or as sniplets to import to any project. 

Soon to come: Paper and code. 

![Antarctica_geo](https://github.com/TobbeTripitaka/grid/blob/master/fig/Antarctica_geo.png)

With some magic functions: 

![Subsampling](https://github.com/TobbeTripitaka/grid/blob/master/fig/Unknown.png)

Future development: 

 - Non regular grids
 - Hexagonal grids
 - More robust import functions
