#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:11:38 2018

@author: jingyi
"""

import pandas as pd
import sessionrunner
from dataforge.environment import Device
from dataforge.baseschemas import DeviceData
from dataforge.environment import Account
# use boxplot
import seaborn as sns

from datetime import datetime, timedelta

def getDevices():
    """
    Get the device monpressprod of KTFLR;
    return a list of device, usecase: getDevices()[i]

    """
    devices = Account['KTFLR'].devices('monpressprod')
    return devices

def getDeviceData(device, starttime, endtime):
    """
    Get original data and sequence

    Parameters:
        device: ex: Device['88:4A:EA:69:E3:09'];
        threshold: a given value
    """
    data = DeviceData.bigtable.query('accel_energy_512',
                                         mac = device.mac,
                                         timestamp = (starttime, endtime)).sequence().data

    return data



def plotSequence(device, starttime, endtime):
    DeviceData.bigtable.query('accel_energy_512',
                              mac = device.mac,
                              timestamp = (starttime, endtime)).sequence().plot()



class DeviceStroke:

    def __init__(self, device, threshold, starttime, endtime):
        """
        Params_ex:
            device = Account['KTFLR'].devices('monpressprod')[0];
            threshold = 105
            starttime = '2018-9-4 12:00'
            endtime = '2018-9-4 13:00'

        """
        self.device = device
        self.threshold = threshold
        self.starttime = starttime
        self.endtime = endtime



    def convertData(data, threshold):
        """
        Input original data and compute time difference and return a dataframe.

        Parameters:
            device: ex: Device['88:4A:EA:69:E3:09'];
            threshold: a given value
        """
        tmp = data[['timestamp', 'accel_energy_512']][(data.accel_energy_512 > threshold)]
        # difference between lines
        tmp['gaps'] = tmp['timestamp'].diff()
        # convert time gap into ms
        tmp['deltaSeconds'] = tmp.gaps.dt.total_seconds() * 1000

        return tmp

# =============================================================================
#  s = DeviceData.bigtable.query('accel_energy_512',
#                                         mac = device[0].mac,
#                                         timestamp = (starttime, endtime)).sequence()
#
# =============================================================================
    def boxplot(tmp):
        """
        Plot boxplot with seabon and add swarmplot.
        Parameter:
            tmp: a dataframe.
        """
        bplot = sns.boxplot(data = tmp['deltaSeconds'], width = 0.5, palette = "colorblind")
        bplot = sns.swarmplot(data = tmp['deltaSeconds'], color = 'red', alpha = 0.75)


    def histogram(tmp):

        tmp.hist(column = 'deltaSeconds')

    def quantileValues(tmp):
        """
        Compute the min, max, quantile values for a given dataset typed
        dataframe and return them.
        Parameter:
            tmp: a dataframe.
        """
        minValue = tmp['deltaSeconds'].min()
        maxValue = tmp['deltaSeconds'].max()
        q1 = tmp['deltaSeconds'].quantile(0.25)
        # q2 = tmp['deltaSeconds'].quantile(0.5)
        q3 = tmp['deltaSeconds'].quantile(0.75)
        QR = q3 - q1
        upper = 1.5 * QR + q3
        lower = q1 - 1.5 * QR
# =============================================================================
#
#         result = [minValue, maxValue, lower, upper]
#         return result
#     # TODO: to find a more clear way to show result
# =============================================================================
        return minValue, maxValue, lower, upper, q1, q3



    def findPeaks(tmp, minValue, maxValue, lower, upper, q1, q3):
        """
        Categorize all the peaks in a given time series to 3 categories;
        Add all the categories in the given dataframe.
        Return dataframe with categories.

        Params:

            maxGap: an integer to determine whether several peak-like
                    pikes should be considered as only one peak.

        """
# =============================================================================
#         # trace 1: to seperate all differents categories to single dataframe.
#         stoppedStroke = tmp[['timestamp', 'accel_energy_512']][(tmp.deltaSeconds > upper)]
#         normalStroke = tmp[['timestamp', 'accel_energy_512']][(tmp.deltaSeconds >= lower and tmp.deltaSeconds <= upper)]
#         composedStroke = tmp[['timestamp', 'accel_energy_512']][(tmp.deltaSeconds < lower)]
# =============================================================================
        # trace 2:
        # add categories to dataframe
        tmp['categories'] = tmp['deltaSeconds']
        if lower > 0:
            tmp['categories'] = pd.cut(tmp['categories'], bins = [minValue-1, lower-1, upper+1, maxValue+1], include_lowest = True, labels = ['SeveralStroke', 'NormalStroke', 'StopStroke'])
        else:
            tmp['categories'] = pd.cut(tmp['categories'], bins = [minValue-1, q1, upper+1, maxValue+1], include_lowest = True, labels = ['low', 'mid', 'high'])

        # return stokeNormal, stokeComposed
        return tmp

# TODO: think a clever way. each time need to see the boxplot and data to change the value(categories)
        #????????????????

# =============================================================================
# Main program : test all the machine of 'KTFLR'
# =============================================================================


if __name__ == '__main__':
    # get all device in KTFLR, must run sessionrunner.py before
    device = getDevices()


    dataOriginal = getDeviceData(device[1],'2018-9-4 12:00', '2018-9-4 13:00' )
    plotSequence(device[1], '2018-9-4 12:00', '2018-9-4 13:00')
    # we can define a threshold from the picture
    stroke1 = DeviceStroke(device[1], 110, '2018-9-4 12:00', '2018-9-4 13:00')
    #stroke2 = DeviceStroke(device[1], 105, '2018-9-4 12:00', '2018-9-4 13:00')

    # get data
    data = DeviceStroke.convertData(dataOriginal, stroke1.threshold)
    # plot data
    DeviceStroke.boxplot(data)
    # find quantiles values
    minValue, maxValue, lower, upper, q1, q3 = DeviceStroke.quantileValues(data)
    result = DeviceStroke.findPeaks(data, minValue, maxValue, lower, upper, q1, q3)
















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


