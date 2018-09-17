#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 09:26:40 2018

@author: jingyi
"""
# if (y(t) - y(t - dt) > m) && (y(t) + y(t + dt) > m)
# then t is the timestamp that peaks associated wuth

import numpy as np
import matplotlib.pyplot as plt
import sessionrunner
import boxplotFindmaxGap
import seaborn as sns
import pandas as pd

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


# =============================================================================
# # Find threshold
# =============================================================================
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
# # try in console to find the threshold
# =============================================================================

d = sessionrunner.Device['88:4A:EA:69:E3:09']
s = boxplotFindmaxGap.getSequence(d, '2018-9-4 12:00', '2018-9-4 13:00')

input2 = np.array(s.data.accel_energy_512)

peaks_boolean = (input2 > np.roll(input2,1)) & (input2 > np.roll(input2,-1))

peaks = input2[peaks_boolean]
# this will take long time
# boxplot(peaks)

plt.plot(input2) # same with s.plot()
plt.plot(peaks_boolean.nonzero()[0], input2[peaks_boolean], 'ro', color = 'blue')


# way2 try threshold===> need to plot many times
threshold = 120
peaks_after_threshold_boolean = (input2>threshold)
peaks_after_threshold = input2[peaks_after_threshold_boolean]
plt.plot(input2)
index_threshold_in_input2 = peaks_after_threshold_boolean.nonzero()[0]
plt.plot(index_threshold_in_input2, input2[peaks_after_threshold_boolean], 'ro', color = 'yellow')

