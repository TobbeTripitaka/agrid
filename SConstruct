#!/usr/bin/env python3
#SCons 

#This SCons script builds the meta paper that that describes the code. 
#www.scons.org


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

# Install the class and dependencies
from code.agrid import *

# os in installed with the class 
env  = Environment(ENV = os.environ ) # only one environment used

def make_maps_fig_2(target, source, env):    
	ant = Grid(crs=3031, res = [10*km, 10*km], left = -3100*km, up=3100*km, right = 3100*km, down = -3100*km)

	#Read bedmap to grid
	ant.ds['ICE'] = (('Y', 'X'), 
                ant.read_raster(str(source[0])) )

	# Set nodata
	no_data = 32767.
	ant.ds['ICE'] = ant.ds['ICE'].where(ant.ds['ICE'] != no_data)  

	# Read polygons, map string attribute to integer as burn values. Map of integer values saved as dict
	dranage, int_map = ant.assign_shape(str(source[1]),'ID', map_to_int = True, return_map = True)
	ant.ds['DRANAGE'] = (('Y', 'X'), dranage)

	# Use ID from dict to select segments of map. Divide with value of integer class to get 1
	polygons = [int_map[str(x) + 'g'] for x in range(2,18)] # Grounded dranage areas in East Antarctica
	ant.ds['EAST_ICE'] = ant.ds['ICE']*ant.ds['DRANAGE'].isin(polygons) #Select only ice in polygons

	#Make maps
	ant.map_grid(ant.ds['DRANAGE'], 
		cmap='Spectral', 
		save_name=str(target[0]), 
		show_map=False)
	ant.map_grid(ant.ds['ICE'], 
		cmap='viridis', 
		cbar=True, 
		cbar_label = '(m)', 
		save_name=str(target[1]), 
		show_map=False)
	ant.map_grid('EAST_ICE', 
		cmap='viridis', 
		cbar=True, 
		cbar_label = '(m)', 
		save_name=str(target[2]), 
		show_map=False)

	# Compute volume
	print(int(ant.ds['EAST_ICE'].sum()*np.prod(ant.res)/km**3),'km3')
	return None

# Define Python functions as builders
env.Append( BUILDERS = {'Make_Maps' : Builder(action = make_maps_fig_2)})


env.Make_Maps(target = ['fig/dranage.pdf', 'fig/ice.pdf', 'fig/selected.pdf'], 
	source = ['data/bedmap2_tiff/bedmap2_thickness.tif', 'data/GSFC_DrainageSystems.shp'])

download_data = True

# Build Fig 2 Flow chart (TikZ)
fig_1= env.PDF(target = 'fig/fig_1.pdf', 
	source = 'tex/fig_1.tex')


# Build Fig 2
#! mkdir -p ../../data/vector

#! unzip -n ../../data/vector/simplified-land-polygons-complete-3857.zip -d ../../data/vector

#curl http://www.site.org/image.jpg --create-dirs -o /path/to/save/images.jpg


if download_data:
	# Download test raster
	url_dem_data = 'https://secure.antarctica.ac.uk/data/bedmap2/bedmap2_tiff.zip'
	env.Command('data/bedmap2_tiff.zip',None,'curl %s > $TARGET' %url_dem_data)
	env.Command('data/bedmap2_tiff/bedmap2_bed.tif','data/bedmap2_tiff.zip','unzip -n $SOURCE -d data/ ')

	# Download test poygons
	url_dranage_data = 'http://quantarctica.tpac.org.au/Quantarctica3/Glaciology/GSFC%20Drainage/GSFC_DrainageSystems'
	for file_extension in ['.shp', '.prj', '.shx', '.dbf', '.qix']:
		env.Command('data/GSFC_DrainageSystems%s'%file_extension, 
			None,'curl -L %s > $TARGET' %(url_dranage_data+file_extension))







# Make numpy noise

# Toy processing

# Viz examples
# TikZ
fig_1= env.PDF(target = 'fig/fig_2.pdf', 
	source = 'tex/fig_2.tex')



# Download bibfile 

# Compile TeX
# Depends




print('A grid for multidimensional and multivariate spatial modelling and processing. ')

print('This research was supported under Australian Research Council’s Special Research Initiative for Antarctic Gateway Partnership (Project ID SR140300001).')

print('Tobias Stål, Anya M Reading,\nUniversity of Tasmania\ncontact: tobias.staal@utas.edu.au ')
print('tobias.staal@utas.edu.au')