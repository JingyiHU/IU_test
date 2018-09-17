#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 16:14:01 2018

@author: jingyi
"""

import pandas as pd
from pandas import datetime as dt

from dataforge.environment import Device
from dataforge.baseschemas import DeviceData
from dataforge.environment import Account

d = Device['88:4A:EA:69:E3:09']

q = DeviceData.bigtable.query('accel_energy_512',
                              mac = d.mac,
                              timestamp = ('2018-9-4 12:00', '2018-9-4 13:00'))

s = q.sequence()

data = s.data

data1 = s.data.head()

# we set the threshold as 105

#data[data['accel_energy_512'] > 110]['timestamp']

# or maybe : the same functionnality
#data[['timestamp']][(data.accel_energy_512 > 110)]

res1 = data[['timestamp','accel_energy_512']][(data.accel_energy_512 > 105)]

#########################
# use difference between lines in pandas

#res1['gaps'] = res1['timestamp'].diff().fillna('0 days 00:00:00.000000')

res1['gaps'] = res1['timestamp'].diff()

res1.dtypes

res1['gapTotalSeconds'] = res1.gaps.dt.total_seconds() * 1000 # ms

# maxGap = 600ms

res_2Peaks = res1[res1['gapTotalSeconds'] < 600]
res_onePeak = res1[res1['gapTotalSeconds'] >= 600]


onePeak = res_onePeak.timestamp
twoPeaks = res_2Peaks.timestamp

# add the y value

# add one more column named categories

# use boxplot
import seaborn as sns

# plot boxplot with seabon
bplot = sns.boxplot(data = res1['gapTotalSeconds'], width = 0.5, palette = "colorblind")

# add swarmplot
bplot = sns.swarmplot(data = res1['gapTotalSeconds'], color = 'red', alpha = 0.75)

res1['gapTotalSeconds'].max()
res1['gapTotalSeconds'].min()

q1 = res1['gapTotalSeconds'].quantile(0.25)


res1['gapTotalSeconds'].quantile(0.5)


q3 = res1['gapTotalSeconds'].quantile(0.75)

QR = q3 - q1

upper = 1.5 * QR + q3

lower = q1 - 1.5 * QR

# stroke stop
stokeStop = tmp[['timestamp','accel_energy_512']][(tmp.deltaSeconds > upper)]

# normal stroke: one peak === one stroke
stokeNormal = tmp[['timestamp','accel_energy_512']][(tmp.deltaSeconds >= lower and tmp.deltaSeconds <= upper)]

# several peaks === one stroke  max_Gap????
stokeComposed = tmp[['timestamp','accel_energy_512']][(tmp.deltaSeconds < lower)]

# add a new columns to contain the categories
res_f['categories'] = e

pd.cut(res1['gapTotalSeconds'], bins = [500, 5381, 21723, 500000], include_lowest = True, labels = ['low', 'mid', 'high'])

res_Cat = pd.cut(res1['gapTotalSeconds'], bins = [500, 5381, 21723, 500000], include_lowest = True, labels = ['low', 'mid', 'high'])

res1['categories'] = res1['gapTotalSeconds']

res1['categories'] = pd.cut(res1['categories'], bins = [500, 5381, 21723, 500000], include_lowest = True, labels = ['low', 'mid', 'high'])






# we deplace the line
res1['timestamp_1'] = res1['timestamp'].shift(1)

res1['ecart'] = res1['timestamp'] - res1['timestamp_1']

# find all the ecart with 00:00:00 (00seconds)

res1[['timestamp','accel_energy_512']]['ecart' == '0 days 00:00:00.*']
# we convert all into int
res1['gap_second'] = res1['ecart'].astype(int)

# try to find the ones related to 00:00:00 ==> 2 as 1 peak
res2 = res1[res1['gap_second'] < 600000000]['timestamp_1']

res_f = res2.drop(res2.index[0]) # drop the first line'

# change name for the first columns

#res_f.rename(columns = {res_f.columns[0]:"starttime"}, inplace = True)
resf = pd.DataFrame({'endtime':res_f.index, 'starttime':res_f.values})

# change the columns order

twoPeakToOne = resf[['starttime', 'endtime']]






# =============================================================================
#         data = DeviceData.bigtable.query('accel_energy_512',
#                                          mac = device.mac,
#                                          timestamp = ('2018-9-4 12:00', '2018-9-4 13:00')).sequence().data
#
#         tmp = data[['timestamp', 'accel_energy_512']][(data.accel_energy_512 > threshold)]
#         # difference between lines
#         tmp['gaps'] = tmp['timestamp'].diff()
#         # convert time gap into ms
#         tmp['deltaSeconds'] = tmp.gaps.dt.total_seconds() * 1000
#
# =============================================================================
# =============================================================================
#         minValue = tmp['deltaSeconds'].min()
#         maxValue = tmp['deltaSeconds'].max()
#         q1 = tmp['deltaSeconds'].quantile(0.25)
#         #q2 = tmp['deltaSeconds'].quantile(0.5)
#         q3 = tmp['deltaSeconds'].quantile(0.75)
#
#         QR = q3 - q1
#
#         upper = 1.5 * QR + q3
#
#         lower = q1 - 1.5 * QR
# =============================================================================


# =============================================================================
#  def findPeaks(tmp, minValue, maxValue, lower, upper):
#         """
#         Categorize all the peaks in a given time series to 3 categories;
#         Add all the categories in the given dataframe.
#         Return dataframe with categories.
#
#         Params:
#
#             maxGap: an integer to determine whether several peak-like
#                     pikes should be considered as only one peak.
#
#         """
# # =============================================================================
# #         # trace 1: to seperate all differents categories to single dataframe.
# #         stoppedStroke = tmp[['timestamp', 'accel_energy_512']][(tmp.deltaSeconds > upper)]
# #         normalStroke = tmp[['timestamp', 'accel_energy_512']][(tmp.deltaSeconds >= lower and tmp.deltaSeconds <= upper)]
# #         composedStroke = tmp[['timestamp', 'accel_energy_512']][(tmp.deltaSeconds < lower)]
# # =============================================================================
#         # trace 2:
#         # add categories to dataframe
#         tmp['categories'] = tmp['deltaSeconds']
#         tmp['categories'] = pd.cut(tmp['categories'], bins = [minValue-1, lower-1, upper+1, maxValue+1], include_lowest = True, labels = ['low', 'mid', 'high'])
#
#         # return stokeNormal, stokeComposed
#         return tmp
#
# =============================================================================
