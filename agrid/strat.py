#!/usr/bin/env python3

# Tobias Staal 2019
# tobias.staal@utas.edu.au
# version = '0.0.1'
#

#GNU GENERAL PUBLIC LICENSE#

#Copyright (c) 2019 Tobias Stål#

import numpy as np
import pandas as pd
import glob


class Strat(object):

    def __init__(self,
                 chart=None,
                 rgb=None,
                 cymk=None,
                 format_chart=True,
                 chart_file=None):
        '''Set up
        Read times and colors from excel file
        '''

        if chart is None:
            if chart_file is None:
                # Expand to other formats
                chart_file = glob.glob('./*.xlsx')[0]
            print('reading: ', chart_file)
            xls = pd.ExcelFile(chart_file)

            chart = pd.read_excel(xls, sheet_name='CHART')
            rgb = pd.read_excel(xls, sheet_name='RGB')
            cymk = pd.read_excel(xls, sheet_name='CYMK')

        if format_chart:
            chart = chart.applymap(
                lambda s: s.strip().lower() if isinstance(
                    s, str) else s)

        self.divisions = np.array(chart.iloc[:, 0:7])
        self.ages = np.array(chart.iloc[:, 8])
        self.classes = self.divisions.shape[0]

        self.chart = chart
        self.rgb = rgb
        self.cymk = cymk

    def geo_to_year(self,
                    age_str,
                    gauge=0.5,
                    early=0.75,
                    mid=0.5,
                    late=0.25,
                    latest=0,
                    earliest=1,
                    period=True,
                    late_labels=None,
                    early_labels=None,
                    mid_labels=None,
                    use_descriptive=True,
                    m=True,
                    age_col=8,
                    uncertainty=False,
                    uncertainty_col=10, ):
        '''Function to convert name of startigraphic age to years in million years
        age -- string with geological age
        chart -- The International Chronostratigraphic Chart as pandas data array
        gauge -- a scalafr to set time from 0 (min for period) to 1 (max for period)
        mid, late, latest, early, earliest = predefined gauges
        use_descriptive -- True if prefixes defined in labels should change returned time
        m -- rerturne Ma, else years
        age_col -- colomn on chart that contains age in Ma
        uncertainty -- if uncertainty of period definitions should be included
        False if not, 'min' for min age and 'max' for max age
        '''

        # Clean inphrase and split to list of worlds
        in_phrase = age_str.strip().lower().split()

        # Make empty array for fits
        find = np.zeros((self.classes, len(in_phrase))).astype('bool')

        # Check if string mach
        for i, in_p in enumerate(in_phrase):
            find[:, i] = np.any(self.divisions == in_p, axis=1)

        # for each word:
            # For each column
            # Only populate one column

        # Stack each row for most fits and find max, get index for max
        here = np.sum(find, axis=1)
        max_here = np.max(here)
        ii = np.where(here == max_here)[0]

        if late_labels is None:
            late_labels = ['upper', 'later', 'late']

        if early_labels is None:
            early_labels = ['lower', 'early', 'earlier']

        if mid_labels is None:
            mid_labels = ['mid', 'middle']

        if max_here == 1 and len(in_phrase) == 2 and use_descriptive:
            print('No definition for expression.')
            if in_phrase[0] in late_labels:
                gauge = late
            elif in_phrase[0] in ['latest']:
                gauge = latest
            elif in_phrase[0] in early_labels:
                gauge = early
            elif in_phrase[0] in ['earliest']:
                gauge = earliest
            elif in_phrase[0] in mid_labels:
                gauge = mid
            else:
                print('Unknown prefix.')

        if uncertainty == False:
            e_top = e_bottom = 0
        # elif uncertainty=='max':
        #    e_top = np.nan_to_num(- chart[ii-1, uncertainty_col][0])
        #    e_bottom = np.nan_to_num(chart[ii, uncertainty_col][-1])
        # elif uncertainty=='min':
        #    e_top = np.nan_to_num(age[ii-1, uncertainty_col][0])
        #    e_bottom = np.nan_to_num(- age[ii, uncertainty_col][-1])

        # ii-1 because start of next division is end of this
        min_my = self.ages[ii - 1][0] - e_top
        max_my = self.ages[ii][-1] + e_bottom

        #print('MIN:', min_my, 'MAX:', max_my, 'GAUGE:', gauge, sep='\n')
        my = min_my + gauge * (max_my - min_my)


        if period:
            # Option to return years instead of Ma
            if not m:
                min_my *= 1e6
                max_my *= 1e6

            return min_my, max_my
        else:
            if not m:
                m *= 1e6
            return my

    def year_to_geo(
            years,
            chart,
            m=False):
        '''
        Convert year to stratigraphic age.
        '''
        if m:
            years /= 1e6

        A = chart.ix[(chart['Ma'] - years).abs().argsort()[:2]]
        return A

    def age_to_color(data, rgb=True):
        '''Returns array or single tuple with colors representing ages.
        If data is numbers, it is assumed to be in Ma
        If data are strings, attempts to match with names
        if rgb is false, CMYK i sreturned
        '''
        #my_list = my_string.split("/")
        #Series.str.split(pat=None, n=-1, expand=False)

    def make_tuple(s, sep='/'):
        if isinstance(s, str):
            x = tuple(re.findall(r'\d+', s))
            if len(x) == 0:
                x = s
        else:
            x = np.nan
        return x

    # df.applymap(make_tuple)


#	yr_to_geo
        #	input year (in My or year)
#		return dict with {'Eon:', 'Jurrasic'} etc. Mind Capitalation

#	geo_to_yr:
#		help function:
#			input string -> list of strings
#			returns tuple start, stop, e
#
#	yr_to_mean:
#		input tuple:
#			oitput mean

#	yr_to_rgb:
#		input year
#		return color

#	yr_to_cymk:
#		input year
#		return color


#	nonstandard:
#		return False
#		return evaluation
