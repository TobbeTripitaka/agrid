import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


requirements = [
'Cartopy>=0.17.0',
'geopandas>=0.6.1',
'imageio>=2.6.0',
'rasterio>=1.0.21',
'scipy>=1.3.1',
'xarray>=0.14.1'
]

setuptools.setup(
    name='agrid', 
    version="0.3.8.2",
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

# python3 setup.py sdist bdist_wheel   
# twine upload dist/*
# virtualenv agrid      
# pip install agrid
# pip freeze > requirements.txt   
# conda skeleton pypi agrid      
# conda-build agrid  
# anaconda upload /Users/tobias_stal/anaconda3/conda-bld/osx-64/agrid-0.3.8.2-py37_0.tar.bz2
 