#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:11:38 2018

@author: jingyi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

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


def getSequence(device, starttime='2018-9-4 12:00', endtime='2018-9-4 13:00'):
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
        s = getSequence(Device['88:4A:EA:69:E3:09'], '2018-9-4 12:00', '2018-9-4 13:00')

    """
    sequence = DeviceData.bigtable.query('accel_energy_512',
                                         mac = device.mac,
                                         timestamp = (starttime, endtime)).sequence()

    return sequence

def convert_data(sequence, threshold, maxGap):

    """
    threshold = 105
    maxGap = 600
    """
    data = sequence.data
    # reset index of dataframe
    data.index = range(len(data))
    # or just drop it: data1.reset_index(drop = True, inplace = True)
    # we collect only 'timestamp' and 'accel_energy_512' columns
    tmp = (data[['timestamp', 'accel_energy_512']]
                            [(data.accel_energy_512 > threshold)])
    # add one column called 'gaps' and turn it directly into millonseconds
    tmp['deltaSeconds'] = (tmp['timestamp']
                            .diff()
                            .dt
                            .total_seconds() * 1000)
    # del res_removeMacAddress['gaps']
    tmp['Boolean'] = tmp['deltaSeconds'] > maxGap
    tmp['CumGap'] = tmp['Boolean'].cumsum()
    # create a new df contains all stokes number, timestamps associated with
    # res_groupedStroke = res_removeMacAddress[['timestamp', 'CumGap']]

    return tmp





def boxplot(data):
    """
    Plot boxplot with seabon and add swarmplot.
    Use cases:
        case1: Plot data_cov['deltaSeconds']
        case2: Plot data_ori['accel_energy_512']
    """
    bplot = sns.boxplot(data, width = 0.5, palette = "colorblind")
    # add points on the plot
    bplot = sns.swarmplot(data, color = 'red', alpha = 0.75)


def quantileValues(data, device):
    """
    remarque: data here must be dataframe!!!
    Compute the min, max, quantile values for a given dataset and device.
    Output:
        Out[55]:
            lower                mac       maxValue    minValue            q1  \
        0  5382.50032  88:4A:EA:69:E3:09  490330.000128  529.999872  11510.000128

            q3         upper
        0  15595.0  21722.499808
    Usecases:
        case1: data = data_cov['deltaSeconds']
        case2: data = data_ori['accel_energy_512']
    """
    r = pd.DataFrame([])
    if isinstance(data, pd.DataFrame):
        minValue = data.min()
        maxValue = data.max()
        q1 = data.quantile(0.25)
        # q2 = tmp['deltaSeconds'].quantile(0.5)
        q3 = data.quantile(0.75)
        QR = q3 - q1
        upper = 1.5 * QR + q3
        lower = q1 - 1.5 * QR
    elif isinstance(data, np.ndarray):
        minValue = data.min()
        maxValue = data.max()
        q1 = np.percentile(data, 25)
        # q2 =  np.percentile(data, 50)
        q3 = np.percentile(data, 75)
        QR = q3 - q1
        upper = 1.5 * QR + q3
        lower = q1 - 1.5 * QR
    r = (r.append(pd.DataFrame({'mac': device.mac, 'minValue':minValue,
                                'maxValue': maxValue,'lower':lower,
                                'upper':upper, 'q1':q1, 'q3':q3},
                                index=[0]), ignore_index = True))
    return r


def categorizePeaks(data, minValue, maxValue, lower, upper, q1, q3):
    """
    Categorize all the peaks in a given time series to 3 categories;
    Add one column named categories in the given dataframe.
    Params:
        usecase:
            r = quantileValues(data_cov, device)
            categorizePeaks(data_cov, r.minValue, r.maxValue, r.lower,
            r.upper, r.q1, r.q3)
    """
# =============================================================================
#         # trace 1: to seperate all differents categories to single dataframe.
#         stoppedStroke = tmp[['timestamp', 'accel_energy_512']][(tmp.deltaSeconds > upper)]
#         normalStroke = tmp[['timestamp', 'accel_energy_512']][(tmp.deltaSeconds >= lower and tmp.deltaSeconds <= upper)]
#         composedStroke = tmp[['timestamp', 'accel_energy_512']][(tmp.deltaSeconds < lower)]
# =============================================================================
    # trace 2:
    # add categories to dataframe
    data['categories'] = data['deltaSeconds']
    if lower > 0:
        data['categories'] = (pd.cut(data['categories'],
                              bins = [minValue-1, lower-1, upper+1, maxValue+1],
                              include_lowest = True,
                              labels = ['SeveralStroke', 'NormalStroke', 'StopStroke']))
    else:
        data['categories'] = (pd.cut(data['categories'],
                              bins = [minValue-1, q1, upper+1, maxValue+1],
                              include_lowest = True,
                              labels = ['low', 'mid', 'high']))

    return data

def getThreshold(data, device, threshold, plotAfterThreshold):

    """
    To plot pictures to define wherther un threshold is correct or not.
    Use case:
        getThreshold(s.data, d, 105, 1) or getThreshold(s.data, d, 105, 0)
        where d = Device['88:4A:EA:69:E3:09']
        s = getSequence(d, '2018-9-4 12:00', '2018-9-4 13:00')
    """
    # s = getSequence(device, '2018-9-4 12:00', '2018-9-4 13:00')
    signal = np.array(data.accel_energy_512)


    peaks_boolean = (signal > np.roll(signal,1)) & (signal > np.roll(signal,-1))
    peaks = signal[peaks_boolean]
    res = quantileValues(peaks, device)
    # plot all the peaks after filter by threshold
    # way2 try threshold===> need to plot many times
    peaks_after_threshold_boolean = (signal > threshold)
    peaks_after_threshold = signal[signal > threshold]
    res = quantileValues(peaks_after_threshold, device)
        # boxplot(peaks)  # this will take long time
    if plotAfterThreshold == 0:
        # plot all the peaks ==> BLUE points in the picture
        plt.plot(signal) # same with s.plot()
        plt.plot(peaks_boolean.nonzero()[0], signal[peaks_boolean], 'ro', color = 'blue')

    elif plotAfterThreshold == 1:
        plt.plot(signal)
        plt.plot(peaks_after_threshold_boolean.nonzero()[0], signal[peaks_after_threshold_boolean], 'ro', color = 'yellow')
    else:
        boxplot(peaks_after_threshold)
    return res









# =============================================================================
# Main program : test all the machine of 'KTFLR'
# =============================================================================


if __name__ == '__main__':
    # get all device in KTFLR, must run sessionrunner.py before
    i = 1
    device = getDevices(i)
    startTime = '2018-9-4 12:00'
    endTime = '2018-9-4 13:00'
    sequence = getSequence(device, startTime, endTime)
    # we can define a threshold from the picture
    data_cov = convert_data(sequence, threshold, maxGap)
    # plot data
    boxplot(data_cov['deltaSeconds'])
    # convert operation is done before boxplot, that's ok cuz we only use
    # 'deltaSeconds' in the df

    # find quantiles values
    r = quantileValues(data_cov['deltaSeconds'], device)
    # categorize the peaks
    categorizePeaks(data_cov, r.minValue, r.maxValue, r.lower,
                    r.upper, r.q1, r.q3)


# to define threshold:














# device[0] ==== > 110

# =============================================================================
# def findPeaks(device, threshold, maxGap):
#     data = DeviceData.bigtable.query('accel_energy_512',
#                                      mac = device.mac,
#                                      timestamp = ('2018-9-4 12:00', '2018-9-4 13:00')).sequence().data
#     tmp = data[['timestamp', 'accel_energy_512']][(data.accel_energy_512 > threshold)]
#     # difference between lines
#     tmp['gaps'] = tmp['timestamp'].diff()
#     # convert time gap into ms
#     tmp['deltaSeconds'] = tmp.gaps.dt.total_seconds() * 1000
#     q1 = tmp['deltaSeconds'].quantile(0.25)
#     q2 = tmp['deltaSeconds'].quantile(0.5)
#     q3 = tmp['deltaSeconds'].quantile(0.75)
#     QR = q3 - q1
#     upper = 1.5 * QR + q3
#     lower = q1 - 1.5 * QR
#
#     # stroke stop
#     stokeStop = tmp[['timestamp','accel_energy_512']][(tmp.deltaSeconds > upper)]
#
#     # normal stroke: one peak === one stroke
#     stokeNormal = tmp[['timestamp','accel_energy_512']][(tmp.deltaSeconds >= lower and tmp.deltaSeconds <= upper)]
#
#     # several peaks === one stroke
#     stokeComposed = tmp[['timestamp','accel_energy_512']][(tmp.deltaSeconds < lower)]
#
#     return stokeNormal, stokeComposed
# =============================================================================

# =============================================================================
#
#
# # To find all lists of machines
#
# # Account['KTFLR'].devices('monpressprod') # list of machines
#
# d = Device['88:4A:EA:69:E3:09']
# # devices
# device = Account['KTFLR'].devices('monpressprod')[i]
# # mac @
# device.mac
#
# =============================================================================
# =============================================================================
#         for i in range(len(Account['KTFLR'].devices('monpressprod'))):
#             device = Account['KTFLR'].devices('monpressprod')[i]
#             print(device)
# =============================================================================

def convert_plot_data(sequence, threshold, maxGap):

    data = sequence.data
    # reset index of dataframe
    data.index = range(len(data))
    # or just drop it: data1.reset_index(drop = True, inplace = True)
    # we collect only 'timestamp' and 'accel_energy_512' columns
    res_removeMacAddress = (data[['timestamp', 'accel_energy_512']]
                            [(data.accel_energy_512 > threshold)])
    # add one column called 'gaps' and turn it directly into millonseconds
    res_removeMacAddress['deltaSeconds'] = (res_removeMacAddress['timestamp'].
                                            diff().dt.total_seconds() * 1000)
    # del res_removeMacAddress['gaps']
    res_removeMacAddress['Boolean'] = res_removeMacAddress['deltaSeconds'] > maxGap
    res_removeMacAddress['CumGap'] = res_removeMacAddress['Boolean'].cumsum()
    # create a new df contains all stokes number, timestamps associated with
    res_groupedStroke = res_removeMacAddress[['timestamp', 'CumGap']]

    # initialize 2 lists to contain starttime and endtime for each stroke.
    groups = res_groupedStroke.groupby('CumGap')
    boundaries = [(g[0] - 1, g[-1] + 1) for g in groups.groups.values()]
    starttimes = [data.ix[boundaries[i][0]].timestamp for i in range(len(boundaries))]
    endtimes = [data.ix[boundaries[i][1]].timestamp for i in range(len(boundaries))]

    # plot all strokes zones in the s.plot()
    sequence.plot()
    for i in range(len(starttimes)):
        plt.axvspan(xmin = starttimes[i], xmax = endtimes[i], color = 'pink')
    plt.savefig('/home/pat/Documents/plot1.png')

