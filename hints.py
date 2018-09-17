#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:54:35 2018

@author: jingyi
"""

# =============================================================================
# # ===========================================================================
# # # all hints here::
# # ===========================================================================
# =============================================================================

###############################################################################

# =============================================================================
# # get data sequence and data corresponded
# =============================================================================

d = Device['88:4A:EA:69:E3:09']
q = DeviceData.bigtable.query('accel_energy_512',
                                         mac = d.mac,
                                         timestamp = ('2018-9-4 12:00', '2018-9-4 13:00'))
# s = q.sequence()

s = DeviceData.bigtable.query('accel_energy_512', mac = d.mac, timestamp = ('2018-9-4 12:00', '2018-9-4 13:00')).sequence()
data = s.data

# =============================================================================
# # reset dataframe index
# =============================================================================
data1 = data
data1.index = range(len(data1))

# =============================================================================
# # changes on the original data
# =============================================================================

res1 = data[['timestamp','accel_energy_512']][(data.accel_energy_512 > 105)]
res1['gaps'] = res1['timestamp'].diff()
res1['deltaSeconds'] = res1.gaps.dt.total_seconds() * 1000
maxGap = 600
res1['deltaSeconds'] > maxGap
res1['Boolean'] = res1['deltaSeconds']
res1[res1['deltaSeconds'] > maxGap]
res1['Boolean'] = res1['deltaSeconds'] > maxGap
res2 = res1['Boolean'].cumsum() # same one represent the same stroke
res1['CumGap'] = res1['Boolean']
res1['CumGap'] = res1['Boolean'].cumsum()
# a new df
res3 = res1[['timestamp', 'CumGap']]
res3.index = range(len(res3))

# =============================================================================
# # differences between normal peaks and the others
# =============================================================================
# if normal peak
peak = data.loc[i]
startloc = data.loc[i-1]
endloc = data.loc[i+1]

# if double or more peaks
# then, if the nb of the result > 1
data['timestamp'] == res3['timestamp'].loc[res3['CumGap']==i]
c=res3['timestamp'].loc[res3['CumGap']==i]
c[0] # the first result


peak = data.loc[i]
startloc = data.loc[i-1]
endloc = data.loc[i+2]


# =============================================================================
# # select the ith Peak
# =============================================================================
res3['timestamp'].loc[res3['CumGap']==i]
# res3['timestamp'].loc[res3['CumGap']==1].get(0)
# Out[155]: Timestamp('2018-09-04 12:00:26.490000128+0000', tz='UTC')


# print all lines

for row in res3.itertuples(index = True, name = 'Pandas'):
    print(getattr(row, "CumGap"))


# =============================================================================
# # different ways to plot
# =============================================================================
s.plot()
for i in range(len(starttimes)):
# draw lines in the plot, green ones are starttimes, red ones are endtimes for the strokes
plt.axvline(x = starttimes[i], color = 'green', linestyle = 'dashed', linewidth = 2)
plt.axvline(endtimes[i], color = 'red', linestyle = 'dashed', linewidth = 2)

# draw a zone
ax = s.plot()
ymin, ymax = ax.get_ylim()
ax.axvspan(xmin = starttimes[0], xmax = endtimes[0], ymin = ymin, ymax = ymax-1, color = 'r')

# draw 2 lines
ax = s.plot()
ymin, ymax = ax.get_ylim()
ax.vlines(x = [starttimes[0], endtimes[0]], ymin = ymin, ymax = ymax-1, color = 'r', linewidth = 2)

# or
s.plot()
plt.axvline(starttimes[0], color = 'green', linestyle = 'dashed', linewidth = 2)
plt.axvline(endtimes[0], color = 'red', linestyle = 'dashed', linewidth = 2)

# we final use:
s.plot()
for i in range(len(starttimes)):
    plt.axvline(x = starttimes[i], color = 'green', linestyle = 'dashed', linewidth = 2)
    plt.axvline(endtimes[i], color = 'red', linestyle = 'dashed', linewidth = 2)


# find the value

for indexs in data.index:
    for i in range(len(data.loc[indexs].values)):
        if(data.loc[indexs].values[i] == 'p'):
            print(indexs,i)
            print(data.loc[indexs].values[i])

# print each line
 for index, row in df.iterrows():
     print(row.value)

# progress bar:

# =============================================================================
# # version3 : with progressbar
# =============================================================================
### error: no module named progressbar
import time
from progressbar import *

progress = ProgressBar()

for i in progress(range(res3['CumGap'].loc[len(res3)-1]+1)):
    time.sleep(0.01)

# =============================================================================
# # plot all the peaks (like-peaks)
# =============================================================================

input1 = np.array([ 1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1.1,  1. ,  0.8,  0.9,
    1. ,  1.2,  0.9,  1. ,  1. ,  1.1,  1.2,  1. ,  1.5,  1. ,  3. ,
    2. ,  5. ,  3. ,  2. ,  1. ,  1. ,  1. ,  0.9,  1. ,  1. ,  3. ,
    2.6,  4. ,  3. ,  3.2,  2. ,  1. ,  1. ,  1. ,  1. ,  1. ])
peak = (input1 > np.roll(input1,1)) & (input1 > np.roll(input1,-1))
plt.plot(input1)
plt.plot(peak.nonzero()[0], input1[peak], 'ro')
plt.show()



# =============================================================================
# # distribution plot
# =============================================================================
# in console
s.data
signal = np.array(data.accel_energy_512)

peaks_boolean = (signal > np.roll(signal, 1)) & (signal > np.roll(signal, -1))
peaks = signal[peaks_boolean]

peaks

mean = peaks.mean()
mean
std = peaks.std()
std
max_Peak = peaks.max()
min_Peak = peaks.min()
def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

x = np.arange(min_Peak, max_Peak, 0.1)
y = normfun(x, mean, std)
plt.plot(x, y, color='red')
plt.hist(peaks, bins=10, color='steelblue',rwidth=0.9, normed=True)

# or using seaborn
#sns.distplot(peaks, rug=True, hist=True)
sns.distplot(peaks, rug=True, hist=False) # biger
sns.distplot(peaks, bins=10, kde=True) # faster



# =============================================================================
# # plot max and min point
# =============================================================================
def demo_test():
    a=np.array([0.15,0.16,0.14,0.17,0.12,0.16,0.1,0.08,0.05,0.07,0.06]);
    max_indx=np.argmax(a)#max value index
    min_indx=np.argmin(a)#min value index
    plt.plot(a,'r-o')
    plt.plot(max_indx,a[max_indx],'ks')
    show_max='['+str(max_indx)+' '+str(a[max_indx])+']'
    plt.annotate(show_max,xytext=(max_indx,a[max_indx]),xy=(max_indx,a[max_indx]))
    plt.plot(min_indx,a[min_indx],'gs')
    plt.show()


# =============================================================================
# # add max and min point in a (x, y) plot
# =============================================================================
x = np.linspace(-2,8, num=301)
y = np.sinc((x-2.21)*3)


fig, ax = plt.subplots()
ax.plot(x,y)

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

annot_max(x,y)


ax.set_ylim(-0.3,1.5)
plt.show()
###############################################################################
# =============================================================================
# # all hints end
# =============================================================================



# =============================================================================
# # ===========================================================================
# # # main idea : correct now ==> to find a better idea?
# # ===========================================================================
# =============================================================================






# try3: using pandas.groupby()
d = Device['88:4A:EA:69:E3:09']
q = DeviceData.bigtable.query('accel_energy_512',
                                         mac = d.mac,
                                         timestamp = ('2018-9-4 12:00', '2018-9-4 13:00'))
# s = q.sequence()

s = DeviceData.bigtable.query('accel_energy_512',
                                         mac = d.mac,
                                         timestamp = ('2018-9-4 12:00', '2018-9-4 13:00')).sequence()
data = s.data

# =============================================================================
# change on original data
# =============================================================================
data1 = data

data1.index = range(len(data1))
data1.reset_index(drop = True, inplace = True)
# the second has the same result as the first one
# ===> change index into 0,1,2...


# we collect only 'timestamp' and 'accel_energy_512' columns
res_removeMacAddress = data1[['timestamp','accel_energy_512']][(data1.accel_energy_512 > 105)]
# add one column called 'gaps' and turn it directly into millonseconds
res_removeMacAddress['deltaSeconds'] = res1['timestamp'].diff().dt.total_seconds() * 1000
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

# using groupby 'CumGap'

grouped = res_groupedStroke.groupby('CumGap')


# =============================================================================
# # hints
# =============================================================================
# res[‘timestamp’].groupby(‘CumGap’)

grouped = res.groupby(‘CumGap’)
# example hints:
grouped = list(res_groupedStroke.groupby('CumGap'))[i][1].index[0]
boundaries_index = [(res_groupedStroke.index[0], res_groupedStroke.index[-1] for stroke in grouped.values()]
# start loc in data
boundaries_index[0] - 1
# end loc in data
boundaries_index[1] + 1


# sub-function : find all the start and end index in original dataset

# =============================================================================
# # First way: myself way, a little heavy!!!
# =============================================================================
# Version 1: with lists
starttimes = []
endtimes = []
for i in tqdm(range(res3['CumGap'].loc[len(res3)-1]+1)):
    time.sleep(0.01)
    # find all the peak starttime
    tmp = res3['timestamp'].loc[res3['CumGap'] == i]
    # tmp[0] # peak starttime(when have several results)
    for j in range(1, len(data1)): # from 1 sinon error
        #starttimes = []
        if(tmp[0] == data1.ix[j, 1]): # find the time in data1
            if len(tmp) > 1:
                starttime = data1.timestamp[j-1]
                endtime = data1.timestamp[j+2]
                starttimes.append(starttime)
                endtimes.append(endtime)
            else: # len(tmp) == 1
                starttime = data1.timestamp[j-1]
                endtime = data1.timestamp[j+1]
                starttimes.append(starttime)
                endtimes.append(endtime)
print(starttimes, endtimes)

# version2: with dataframe
# define a df to stock timestamps
data = pd.DataFrame([])
# this need to be put outside the function or not
# we can only get the last one result
for i in range(res3['CumGap'].loc[len(res3)-1]+1):
    # find all the peak starttime
    tmp = res3['timestamp'].loc[res3['CumGap'] == i]
    # tmp[0] # peak starttime(when have several results)
    for j in range(1, len(data1)): # from 1 sinon error
        #starttimes = []
        if(tmp[0] == data1.ix[j, 1]): # find the time in data1

            if len(tmp) > 1:
                starttime = data1.timestamp[j-1]
                endtime = data1.timestamp[j+2]
                data = data.append(pd.DataFrame({'starttime':starttime, 'endtime': endtime}, index=[0]), ignore_index = True)
                #print(starttimes)

            else: # len(tmp) == 1
                starttime = data1.timestamp[j-1]
                endtime = data1.timestamp[j+1]
                data = data.append(pd.DataFrame({'starttime':starttime, 'endtime': endtime}, index=[0]), ignore_index = True)
print(data)

# whole version2: with list + sys(progress bar)
import sys, time
starttimes = []
endtimes = []
for i in range(res3['CumGap'].loc[len(res3)-1]+1):
    k = i + 1
    str = '>'*(i//2)+''*((100-k)//2)
    sys.stdout.write('\t'+str+'[%s%%]'%(i+1))
    sys.stdout.flush()
    time.sleep(0.1)
    # find all the peak starttime
    tmp = res3['timestamp'].loc[res3['CumGap'] == i]
    # tmp[0] # peak starttime(when have several results)
    for j in range(1, len(data1)): # from 1 sinon error
        #starttimes = []
        if(tmp[0] == data1.ix[j, 1]): # find the time in data1
            if len(tmp) > 1:
                starttime = data1.timestamp[j-1]
                endtime = data1.timestamp[j+2]
                starttimes.append(starttime)
                endtimes.append(endtime)
            else: # len(tmp) == 1
                starttime = data1.timestamp[j-1]
                endtime = data1.timestamp[j+1]
                starttimes.append(starttime)
                endtimes.append(endtime)
print(starttimes, endtimes)
###plot
s.plot()
for i in range(len(starttimes)):
    # draw lines in the plot, green ones are starttimes, red ones are endtimes for the strokes
    plt.axvline(x = starttimes[i], color = 'green', linestyle = 'dashed', linewidth = 2)
    plt.axvline(endtimes[i], color = 'red', linestyle = 'dashed', linewidth = 2)
# save plot
plt.savefig('/home/pat/Documents/result_v1.png')

# =============================================================================
# =============================================================================
# # # ways to find the start and end time for each stroke
# =============================================================================
# =============================================================================

# =============================================================================
# # first way: a big for
# =============================================================================
from tqdm import tqdm
start_indexs = []
end_indexs = []
for i in tqdm(range(len(list(res_groupedStroke.groupby('CumGap'))))):
    time.sleep(0.01)
    start_index = list(res_groupedStroke.groupby('CumGap'))[i][1].index[0] - 1
    end_index = list(res_groupedStroke.groupby('CumGap'))[i][1].index[-1] + 1
    #print(start_index, end_index)
    start_indexs.append(start_index)
    end_indexs.append(end_index)

# find the timestamp associated
starttime = [data.ix[start_indexs[i]].timestamp for i in len(start_indexs)]
endtime = [data.ix[end_indexs[i]].timestamp for i in len(end_indexs)]

# =============================================================================
# # Second way by using tuple:
# =============================================================================
# or we can try to create a tuple contains each stroke time
boundaries = [(list(res_groupedStroke.groupby('CumGap'))[i][1].index[0] - 1,
               list(res_groupedStroke.groupby('CumGap'))[i][1].index[-1] + 1) for i in range(len(list(res_groupedStroke.groupby('CumGap'))))]

starttimes = [data.ix[boundaries[i][0]].timestamp for i in range(len(boundaries))]
endtimes = [data.ix[boundaries[i][1]].timestamp for i in range(len(boundaries))]

# =============================================================================
# # the most simple way:
# =============================================================================
groups = list(res_groupedStroke.groupby('CumGap'))
boundaries = [(g[0] - 1, g[-1] + 1) for g in gb.groups.values()]
starttimes = [data.ix[boundaries[i][0]].timestamp for i in range(len(boundaries))]
endtimes = [data.ix[boundaries[i][1]].timestamp for i in range(len(boundaries))]
# plot
s.plot()
for i in range(len(starttimes)):
    plt.axvspan(xmin = starttimes[i], xmax = endtimes[i], color = 'pink')
plt.savefig('/home/pat/Documents/plot_v4.png')




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
res_removeMacAddress['deltaSeconds'] = res1['timestamp'].diff().dt.total_seconds() * 1000
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


# plot starttime and endtime while starttime = value just before the peak
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:32:40 2018

@author: jingyi
"""

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
#import seaborn as sns

from dataforge.environment import Device
from dataforge.baseschemas import DeviceData
from dataforge.environment import Account


def getDevices(i):
    """
    Get the device monpressprod of KTFLR;
    return a list of device, usecase: getDevices()[i]

    """
    devices = Account['KTFLR'].devices('monpressprod')
    device = devices[i]
    return device


def getSequence(device, starttime, endtime):
    """
    Get original data and sequence.
    Parameters:
        device: the result return from function called getDevice()
                use case: devices = getDevices()
                devices[0]
                ex: Device['88:4A:EA:69:E3:09'];
        startTime: the start time that user want to plot the strokes from
        endTime: the end time of the period that user want to plot the strokes
    Use cases:
        To get data: sequence.data
        To plot sequence: sequence.plot()
        startTime = '2018-9-4 12:00'
        endTime = '2018-9-4 13:00'

    """
    sequence = DeviceData.bigtable.query('accel_energy_512',
                                         mac = device.mac,
                                         timestamp = (starttime, endtime)).sequence()

    return sequence



class DeviceCount:
    #device = Device['88:4A:EA:69:E3:09']

    def __init__(self, sequence, threshold, maxGap):
        """
         threshold: defined by the users after seeing the plot
         maxGap: the max gap to define whether several peaks belong
                    to a same stroke.
        """
        # self is the instance, different for each device
        self.sequence = sequence
        self.threshold = threshold
        self.maxGap = maxGap
        # self.mac_address = device.mac


    def printStrokes(self):
        """
        Plot all the strokes in a given time series.
        Params:
            sequence: sequence related to the device chose


        """
        data = self.sequence.data
        # reset index of dataframe
        data.index = range(len(data))
        # or just drop it: data1.reset_index(drop = True, inplace = True)
        # we collect only 'timestamp' and 'accel_energy_512' columns
        tmp = (data[['timestamp', 'accel_energy_512']]
                   [(data.accel_energy_512 > self.threshold)])
        # add one column called 'gaps' and turn it directly into millonseconds
        tmp['deltaSeconds'] = (tmp['timestamp']
                               .diff()
                               .dt.total_seconds() * 1000)
        # del res_removeMacAddress['gaps']
        tmp['Boolean'] = tmp['deltaSeconds'] > self.maxGap
        tmp['CumGap'] = tmp['Boolean'].cumsum()
        # create a new df contains all stokes number, timestamps associated with
        res_grouped_stroke = tmp[['timestamp', 'CumGap']]

        # initialize 2 lists to contain starttime and endtime for each stroke.
# =============================================================================
#         groups = res_grouped_stroke.groupby('CumGap')
#         boundaries = [(g[0] - 1, g[-1] + 1) for g in groups.groups.values()]
#         starttimes = [data.ix[boundaries[i][0]].timestamp for i in range(len(boundaries))]
#         endtimes = [data.ix[boundaries[i][1]].timestamp for i in range(len(boundaries))]
# =============================================================================
        randomValue = self.maxGap * 1/3
        groups = res_grouped_stroke.groupby('CumGap')
        boundaries = [(g[0] , g[-1]) for g in groups.groups.values()]
        starttimes = [data.ix[boundaries[i][0]].timestamp - randomValue for i in range(len(boundaries))]
        endtimes = [data.ix[boundaries[i][1]].timestamp - randomValue for i in range(len(boundaries))]

        # plot all strokes zones in the s.plot()
        self.sequence.plot()
        for i in range(len(starttimes)):
            plt.axvspan(xmin = starttimes[i], xmax = endtimes[i], color = 'pink')
        plt.savefig('/home/pat/Documents/plot1.png')

    def __repr__(self):
        return "DeviceCount('{}', '{}', '{}')".format(self.sequence, self.threshold, self.maxGap)
# =============================================================================
#     def __str__(self):
#         return '{} \n {}'.format(self.getSequence(starttime, endtime), self.mac_address)
#
# =============================================================================
# =============================================================================
#     @classmethod
#     def getDevices(cls, i):
#         """
#         Get the device monpressprod of KTFLR;
#         Use case:
#             ex: DeviceCount.getDevices(3)
#             DeviceCount.device
#             then get the sequence:
#                 s1 = Device1.getSequence('2018-9-4 12:00', '2018-9-4 13:00')
#
#         """
#         devices = Account['KTFLR'].devices('monpressprod')
#         device = devices[i]
#         return cls(device)
# =============================================================================


# try:
# =============================================================================
# # 2 WITH CLASS METHODE
# =============================================================================
if __name__ == 'main':
    new_device = getDevices(3)
    new_sequence = getSequence(new_device, '2018-9-4 12:00', '2018-9-4 13:00')
    # new_sequence.plot()
    new_deviceCount = DeviceCount(new_sequence, 96, 500)
    new_deviceCount.printStrokes()


# through the graph, we have:
#threshold = 96
#maxGap = 600

################################################################################
################################################################################
################################################################################
# USE CASE

#s = DeviceData.bigtable.query('accel_energy_512', mac = d.mac, timestamp = ('2018-9-4 12:00', '2018-9-4 13:00')).sequence()




# =============================================================================
# # use cases:
# d = Device['88:4A:EA:69:E3:09']
# device = d
# startTime = '2018-9-4 12:00'
# endTime = '2018-9-4 13:00'
# threshold = 105
# maxGap = 60
# sequence = DeviceCount.getSequence(device, startTime, endTime)
#
# =============================================================================

# =============================================================================
# =============================================================================
# # # 1 GETDEVICE OUTSIDE THE CLASS
# =============================================================================
# Device1 = DeviceCount(Device['88:4A:EA:69:E3:09'], 105, 600)
# # Device1.__dict__ can print out the informations associated
# sequence = Device1.getSequence('2018-9-4 12:00', '2018-9-4 13:00')
#
# # plot
# Device1.printStrokes(sequence)
# =============================================================================
















