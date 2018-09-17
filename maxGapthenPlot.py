#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:41:17 2018

@author: jingyi
"""

import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from dataforge.environment import Device
from dataforge.baseschemas import DeviceData
from dataforge.environment import Account


def getDevices():
    """
    Get the device monpressprod of KTFLR;
    return a list of device, usecase: getDevices()[i]

    """
    devices = Account['KTFLR'].devices('monpressprod')
    return devices


# use case:
    devices = getDevices()
    devices[0]
# Out[557]: PressProdVibrationsMonitor
#(machine=Machine(account=Account(name='KTFLR'),
#mtype='Monitored Forging Machine', name='1000 (2)T Line'),
#mac='88:4A:EA:69:DD:C7', label='Forging Press .')





def getSequence(device, starttime, endtime):
    """
    Get original data and sequence

    Parameters:
        device: ex: Device['88:4A:EA:69:E3:09'];
        threshold: a given value
    Use cases:
        To get data: sequence.data
        To plot sequence: sequence.plot()

    """
    sequence = DeviceData.bigtable.query('accel_energy_512',
                                         mac = device.mac,
                                         timestamp = (starttime, endtime)).sequence()

    return sequence
#  use case:
    # here we take d = Device['88:4A:EA:69:E3:09']
    # starttime = '2018-9-4 12:00'
    # endtime = '2018-9-4 13:00'
    # timestamp = ('2018-9-4 12:00', '2018-9-4 13:00')
    # s: sequence; d:device
    s = getSequence(d, '2018-9-4 12:00', '2018-9-4 13:00')
# for plot:
    s.plot()
# data:
    s.data

# =============================================================================
# def convertData(data, threshold):
#     """
#     Input original data and compute time difference and return a dataframe.
#
#     Parameters:
#         device: ex: Device['88:4A:EA:69:E3:09'];
#         threshold: a given value
#         """
#     tmp = data[['timestamp', 'accel_energy_512']][(data.accel_energy_512 > threshold)]
#     # difference between lines
#     tmp['gaps'] = tmp['timestamp'].diff()
#     # convert time gap into ms
#     tmp['deltaSeconds'] = tmp.gaps.dt.total_seconds() * 1000
#
#     return tmp
#
# =============================================================================



def printStrokes(device, startTime, endTime, sequence, threshold, maxGap):
    """
    Params:
        device: the result return from function called getDevice()
                use case: devices = getDevices()
                devices[0]
        startTime: the start time that user want to plot the strokes from
        endTime: the end time of the period that user want to plot the strokes
        sequence: sequence related to the device chose
        threshold: defined by the users after seeing the plot
        maxGap: the max gap to define whether several peaks belong to a same stroke.
    Basically contain 4 parts:
        part1: get device sequence and data associated
        part2: change the origianl dataset
        part3: get the start and end time for each stroke
        part4: plot the timestamps in the timeseries
    Local variables:
        tmp: a temporary dataframe
        res: the dataframe after add columns 'deltaSeconds' & 'CumGap'
        starttimes: a list contains all the starttimes for each stroke
        endtimes: a list contains all the endtimes for each stroke
    """

    ### part1:
    sequence = DeviceData.bigtable.query('accel_energy_512',
                                         mac = device.mac,
                                         timestamp = (startTime, endTime)).sequence()
    data = sequence.data

    ### part2:
    # reset index of dataframe
    data.index = range(len(data))
    tmp = data[['timestamp', 'accel_energy_512']][(data.accel_energy_512 > threshold)]
    # difference between lines
    tmp['gaps'] = tmp['timestamp'].diff()
    # convert time gap into ms
    tmp['deltaSeconds'] = tmp.gaps.dt.total_seconds() * 1000
    # compare the time gap with maxGap
    #tmp['Boolean'] = tmp['deltaSeconds']
    tmp['Boolean'] = tmp['deltaSeconds'] > maxGap
    #tmp['CumGap'] = tmp['Boolean']
    tmp['CumGap'] = tmp['Boolean'].cumsum()
    # make a new dataframe contains the number of strokes called "CumGap"
    res = tmp[['timestamp', 'CumGap']]
    res.index = range(len(res))
    ### part3:
    # initialize 2 lists to contain starttime and endtime for each stroke.
    starttimes = []
    endtimes = []
    # for i in range(res['CumGap'].loc[len(res)-1]+1):
    # add progress bar
    for i in tqdm(range(res['CumGap'].loc[len(res)-1]+1)):
        time.sleep(0.01)
        # find all starttime for each peak
        peak = res['timestamp'].loc[res['CumGap'] == i]
        # tmp[0] # peak starttime(when have several results)
        for j in range(1, len(data)): # from 1 sinon error
            if(peak[0] == data.ix[j, 1]): # find the time in data1
                if len(peak) > 1:
                    starttime = data.timestamp[j-1]
                    endtime = data.timestamp[j+2]
                    starttimes.append(starttime)
                    endtimes.append(endtime)
                else: # len(peak) == 1
                    starttime = data.timestamp[j-1]
                    endtime = data.timestamp[j+1]
                    starttimes.append(starttime)
                    endtimes.append(endtime)
    # print(starttimes, endtimes)

    ### part4:
    # plot all strokes zones in the s.plot()
    sequence.plot()
    for i in range(len(starttimes)):
        # draw lines in the plot, green ones are starttimes, red ones are endtimes for the strokes
        # hint1:
        # ax = s.plot()
        # ymin, ymax = ax.get_ylim()
        plt.axvspan(xmin = starttimes[i], xmax = endtimes[i], color = 'lightgreen')
        # hint2:
        # plt.axvline(x = starttimes[i], color = 'green', linestyle = 'dashed', linewidth = 2)
        # plt.axvline(endtimes[i], color = 'red', linestyle = 'dashed', linewidth = 2)
    plt.savefig('/home/pat/Documents/plot_v2.png')

################################################################################
################################################################################
################################################################################
# USE CASE
d = Device['88:4A:EA:69:E3:09']
q = DeviceData.bigtable.query('accel_energy_512',
                                         mac = d.mac,
                                         timestamp = ('2018-9-4 12:00', '2018-9-4 13:00'))
# s = q.sequence()

s = DeviceData.bigtable.query('accel_energy_512',
                                         mac = d.mac,
                                         timestamp = ('2018-9-4 12:00', '2018-9-4 13:00')).sequence()
data = s.data


# d = Device['88:4A:EA:69:E3:09']
device = d
startTime = '2018-9-4 12:00'
endTime = '2018-9-4 13:00'
sequence = s
threshold = 105
maxGap = 60












# PROGRESS BAR
# =============================================================================
# try:
#     from tqdm import tqdm
# except:
#     import os
#     os.system('sudo pip3 install tqdm')
#     from tqdm import tqdm
#
# pbar = tqdm(res3)
#
# for (idx, ele) in enumerate(pbar):
#     #main()
#     pbar.set_description(' COMPLETE ')
# =============================================================================


# TRY IN CONSOLE
starttimes = []
endtimes = []
# for i in range(res3['CumGap'].loc[len(res3)-1]+1):
# add progress bar
for i in tqdm(range(res3['CumGap'].loc[len(res3)-1]+1)):
   #pass
    time.sleep(0.01)
    # show the progress
# =============================================================================
#     k = i + 1
#     str = '>'*(i//2)+''*((100-k)//2)
#     sys.stdout.write('\t'+str+'[%s%%]'%(i+1))
#     sys.stdout.flush()
#     time.sleep(0.1)
# =============================================================================
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
#print(starttimes, endtimes)
k = i + 1
str = '>'*(i//2)+''*((100-k)//2)
sys.stdout.write('\t'+str+'[%s%%]'%(i+1))
sys.stdout.flush()
time.sleep(0.1)
###plot
s.plot()
for i in range(len(starttimes)):
    # draw lines in the plot, green ones are starttimes, red ones are endtimes for the strokes
    plt.axvline(x = starttimes[i], color = 'green', linestyle = 'dashed', linewidth = 2)
    plt.axvline(endtimes[i], color = 'red', linestyle = 'dashed', linewidth = 2)
# save plot
plt.savefig('/home/pat/Documents/result_v1.png')


