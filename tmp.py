#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:29:38 2018

@author: jingyi
"""


# =============================================================================
# =============================================================================
# # # so in total:
# =============================================================================
# =============================================================================
d = Device['88:4A:EA:69:E3:09']
#q = DeviceData.bigtable.query('accel_energy_512',
                                         mac = d.mac,
                                         timestamp = ('2018-9-4 12:00', '2018-9-4 13:00'))
# s = q.sequence()

s = DeviceData.bigtable.query('accel_energy_512',
                                         mac = d.mac,
                                         timestamp = ('2018-9-4 12:00', '2018-9-4 13:00')).sequence()
data = s.data

data1 = data

data1.index = range(len(data1))
data1.reset_index(drop = True, inplace = True)
# the second has the same result as the first one
# ===> change index into 0,1,2...

# we collect only 'timestamp' and 'accel_energy_512' columns
res_removeMacAddress = data1[['timestamp','accel_energy_512']][(data1.accel_energy_512 > 105)]
# add one column called 'gaps' and turn it directly into millonseconds
res_removeMacAddress['deltaSeconds'] = res_removeMacAddress['timestamp'].diff().dt.total_seconds() * 1000
# set the maxGap value
maxGap = 600
# res_removeMacAddress['deltaSeconds'] > maxGap
# del column
del res_removeMacAddress['gaps']
# never need to do this!!! :
#res_removeMacAddress[res_removeMacAddress['deltaSeconds'] > maxGap]
res_removeMacAddress['Boolean'] = res_removeMacAddress['deltaSeconds'] > maxGap
res_removeMacAddress['CumGap'] = res_removeMacAddress['Boolean'].cumsum()

# create a new df contains all stokes number, timestamps associated with
res_groupedStroke = res_removeMacAddress[['timestamp', 'CumGap']]

# or we can try to create a tuple contains each stroke time
boundaries = [(list(res_groupedStroke.groupby('CumGap'))[i][1].index[0] - 1,
               list(res_groupedStroke.groupby('CumGap'))[i][1].index[-1] + 1) for i in range(len(list(res_groupedStroke.groupby('CumGap'))))]

starttimes = [data.ix[boundaries[i][0]].timestamp for i in range(len(boundaries))]
endtimes = [data.ix[boundaries[i][1]].timestamp for i in range(len(boundaries))]

# plot
s.plot()
for i in range(len(starttimes)):
    plt.axvspan(xmin = starttimes[i], xmax = endtimes[i], color = 'red')
plt.savefig('/home/pat/Documents/plot_v3.png')

# much simple one: from JD:
groups = list(res_groupedStroke.groupby('CumGap'))
boundaries = [(g[0] - 1, g[-1] + 1) for g in gb.groups.values()]
starttimes = [data.ix[boundaries[i][0]].timestamp for i in range(len(boundaries))]
endtimes = [data.ix[boundaries[i][1]].timestamp for i in range(len(boundaries))]
# plot
s.plot()
for i in range(len(starttimes)):
    plt.axvspan(xmin = starttimes[i], xmax = endtimes[i], color = 'pink')
plt.savefig('/home/pat/Documents/plot_v4.png')
