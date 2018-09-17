#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:32:40 2018

@author: jingyi
"""

import pandas as pd
import matplotlib.pyplot as plt
# from tqdm import tqdm
import seaborn as sns
import numpy as np
import datetime
from scipy.stats import norm

from dataforge.environment import Device
from dataforge.baseschemas import DeviceData
from dataforge.environment import Account


def getDevices(i):
    """Get the device monpressprod of KTFLR;
    return a list of device, usecase: getDevices()[i]

    """
    devices = Account['KTFLR'].devices('monpressprod')
    device = devices[i]
    return device


def getSequence(device, starttime, endtime):
    """Get original data and sequence.
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
    sequence = (DeviceData
                .bigtable
                .query('accel_energy_512', mac=device.mac,
                       timestamp=(starttime, endtime)).sequence())

    return sequence


def plotThreshold(data, type_plot):
    """Different plot with different type_plot
    Params:
        type_plot == 0: hist + pdf
        type_plot == 1: sns plot
    """
    signal = np.array(data.accel_energy_512)
    p = (signal > np.roll(signal, 1)) & (signal > np.roll(signal, -1))
    peaks = signal[p]

    if type_plot == 0:
        mean = peaks.mean()
        std = peaks.std()
        max_Peak = peaks.max()
        min_Peak = peaks.min()
        x = np.arange(min_Peak, max_Peak, 0.1)
        # y = normfun(x, mean, std)
        y = norm.pdf(x, mean, std)
        plt.plot(x, y, color='red')
        plt.hist(peaks, bins=500, color='steelblue', rwidth=0.9, normed=True)
        sns.distplot(peaks, kde=True, rug=True, rug_kws={"color": "k"},
                     kde_kws={"color": "red", "lw": 3, "label": "KDE"},
                     hist_kws={"histtype": "step", "lw": 3, "alpha": 1,
                               "color": "g"},
                     bins=500)
        plt.title('Vibration Intensity Distribution')
        plt.xlabel('Vibration Intensity')
        plt.ylabel('Probability')
    elif type_plot == 1:
        # or using seaborn
        # sns.distplot(peaks, rug=True, hist=True)
        # ax = sns.distplot(peaks, rug=True, hist=False)  # biger
        ax = sns.distplot(peaks, kde=True, rug=True, rug_kws={"color": "k"},
                          kde_kws={"color": "red", "lw": 3, "label": "KDE"},
                          hist_kws={"histtype": "step", "lw": 3, "alpha": 1,
                                    "color": "g"},
                          bins=500)  # faster
        ax.set(xlabel='Vibration Intensity', ylabel='Probability')


def tryDifferentThreshold(data, device, threshold, isboxplot):
    """To plot pictures to define wherther un threshold is correct or not.
    Params:
        plotAfterThreshold == 0 : plot all peaks
        plotAfterThreshold == 1 : plot rest peaks after cut by threshold
        plotAfterThreshold == else : boxplot rest peaks
    Use case:
        tryDifferentThreshold(s.data, d, 105, 1) or
        tryDifferentThreshold(s.data, d, 105, 0)
        where d = Device['88:4A:EA:69:E3:09']
        s = getSequence(d, '2018-9-4 12:00', '2018-9-4 13:00')
    """

    signal = np.array(data.accel_energy_512)

    intensity_bool = ((signal >
                       np.roll(signal, 1)) & (signal > np.roll(signal, -1)))
    intensity = signal[intensity_bool]
    res = quantileValues(intensity, device)
    nb_intensities = len(intensity)
    # plot all the peaks after filter by threshold
    # way2 try threshold===> need to plot many times
    peaks_boolean = (signal > threshold)
    peaks = signal[peaks_boolean]
    res = quantileValues(peaks, device)
    nb_peaks = len(peaks)
    # boxplot(peaks)  # this will take long time
    if threshold == 0 and isboxplot == 0:
        # plot all the peaks ==> BLUE points in the picture
        plt.plot(signal)  # same with s.plot()
        plt.plot(intensity_bool.nonzero()[0],
                 signal[intensity_bool], 'ro', color='blue')
        nb_peaks = nb_intensities
    elif isboxplot == 0:
        plt.plot(signal)
        plt.plot(peaks_boolean.nonzero()[0],
                 signal[peaks_boolean], 'ro', color='yellow')
    elif isboxplot == 1:
        boxplot(peaks)

    return res, nb_intensities, nb_peaks


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf


def getDelta(sequence, threshold):
    """Get converted dataframe
    """
    data = sequence.data
    data.index = range(len(data))
    tmp = (data
           [['timestamp', 'accel_energy_512']]
           [(data.accel_energy_512 > threshold)])
    tmp['deltaSeconds'] = (tmp['timestamp']
                           .diff()
                           .dt
                           .total_seconds() * 1000)
    return tmp


def annot_max_min(x, y, ax=None):
    """Mark max and min point on the plot.
    """
    xmax = x[np.argmax(y)]
    ymax = y.max()
    xmin = x[np.argmin(y)]
    ymin = y.min()
    textmax = "x={:.2f}, y={:.8f}".format(xmax, ymax)
    textmin = "x={:.2f}, y={:.13f}".format(xmin, ymin)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square, pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->",
                      connectionstyle="angle, angleA=0, angleB=60")
    kw1 = dict(xycoords='data', textcoords="axes fraction",
               arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    kw2 = dict(xycoords='data', textcoords="axes fraction",
               arrowprops=arrowprops, bbox=bbox_props, ha="left", va="bottom")
    ax.annotate(textmin, xy=(xmin, ymin), xytext=(0.94, 0.96), **kw1)
    ax.annotate(textmax, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw2)


def plotTimeDelta(data, type_plot, device):
    """Must give a threshold before using this function.
    Params:
        data: data = getDelta(s, 105)
        type_plot == 0 ===> hist and pdf plot
        type_plot == 1 ===> plot points max and min
        type_plot == 2 ===> boxplot()
    """
    mean = data.mean()
    std = data.std()
    max_data = data.max()
    min_data = data.min()
    max_indx = np.argmax(data)  # max value index
    min_indx = np.argmin(data)  # min value index
    x = np.arange(min_data, max_data, 0.1)
    y = normfun(x, mean, std)
    res_quantile = quantileValues(data, device)
    if type_plot == 0:
        plt.plot(x, y, color='blue')
        annot_max_min(x, y)
        # plt.hist(data.dropna(), bins=500, rwidth=0.9, normed=True)
        plt.title('Time Delta distribution')
        plt.xlabel('Time Delta')
        plt.ylabel('Probability')
        sns.distplot(tmp.deltaSeconds.dropna(),
                     kde=True, rug=True, rug_kws={"color": "k"},
                     kde_kws={"color": "red", "lw": 3, "label": "KDE"},
                     hist_kws={"histtype": "step", "lw": 3, "alpha": 1,
                               "color": "g"},
                     bins=500)
        # ax.set(xlabel='Vibration Intensity', ylabel='Probability')
    elif type_plot == 1:  # plot the max and min point
        plt.plot(data)
        plt.plot(max_indx, data[max_indx], 'ks')
        show_max = '['+str(max_indx)+' '+str(data[max_indx])+']'
        plt.annotate(show_max,
                     xytext=(max_indx, data[max_indx]),
                     xy=(max_indx, data[max_indx]))
        plt.plot(min_indx, data[min_indx], 'gs')
        show_min = '['+str(min_indx)+' '+str(data[min_indx])+']'
        plt.annotate(show_min,
                     xytext=(min_indx, data[min_indx]),
                     xy=(min_indx, data[min_indx]))
        plt.title('Time Delta')
        plt.xlabel('Index')
        plt.ylabel('Vibration Intensity Value')
    elif type_plot == 2:  # boxplot
        boxplot(data.dropna())
    return res_quantile


def printStrokes(sequence, threshold, maxGap):
    """Plot all the strokes in a given time series.
    Params:
        sequence: sequence related to the device chose
        threshold: defined by the users after seeing the plot
        maxGap: the max gap to define whether several peaks belong
                to a same stroke.
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
                           .dt.total_seconds() * 1000)
    # del res_removeMacAddress['gaps']
    tmp['Boolean'] = tmp['deltaSeconds'] > maxGap
    tmp['CumGap'] = tmp['Boolean'].cumsum()
    # create a new df contains all stokes number, timestamps associated with
    res_grouped_stroke = tmp[['timestamp', 'CumGap']]
    shift = datetime.timedelta(milliseconds=2/3 * maxGap)
    groups = res_grouped_stroke.groupby('CumGap')
    boundaries = [(g[0], g[-1]) for g in groups.groups.values()]
    starttimes = [data.ix[boundaries[i][0]].timestamp - shift
                  for i in range(len(boundaries))]
    endtimes = [data.ix[boundaries[i][1]].timestamp + shift
                for i in range(len(boundaries))]
    # plot all strokes zones in the s.plot()
    sequence.plot()
    for i in range(len(starttimes)):
        plt.axvspan(xmin=starttimes[i], xmax=endtimes[i], color='red')
    plt.savefig('/home/pat/Documents/plot_v0.png')


def boxplot(data):
    """Plot boxplot with seabon and add swarmplot.
    Use cases:
        case1: Plot data_cov['deltaSeconds']
        case2: Plot data_ori['accel_energy_512']
    """
    sns.boxplot(data, width=0.5, palette="colorblind")
    # add points on the plot
    sns.swarmplot(data, color='red', alpha=0.75)


def quantileValues(data, device):
    """Compute the min, max, quantile values for a given dataset and device.
    Usecases:
        case1: data = data_cov['deltaSeconds']
        case2: data = data_ori['accel_energy_512']
    """
    r = pd.DataFrame([])
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
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
    r = (r.append(pd.DataFrame({'mac': device.mac, 'minValue': minValue,
                                'maxValue': maxValue, 'lower': lower,
                                'upper': upper, 'q1': q1, 'q3': q3},
                               index=[0]), ignore_index=True))
    return r


if __name__ == 'main':

    # d = getDevices(0)
    # startTime = '2018-9-4 12:00'
    # endTime = '2018-9-4 13:00'
    # s_d = getSequence(d, startTime, endTime)

    d = Device['88:4A:EA:69:E3:09']
    startTime = '2018-9-4 12:00'
    endTime = '2018-9-4 13:00'
    # threshold = 105
    # maxGap = 600

    s = getSequence(d, startTime, endTime)
    data = s.data

# To define threshold:
    # distribution plot hist+pdf+sns.displot
    plotThreshold(data, 0)
    """With this we can choose our threshold as range of [97:110].
    used library: from scipy.stats import norm
    then norm.pdf
    """
    """From the result, we can see that nb of intensities is 2192,
    so in our plotThreshold(), we can change the bins to 500.
    """
    """sns.distplot(kde, rug, bins=500)
    kde: whether to plot a gaussian kernel density estimate.
    rug: whether to plot a rugplot on the support axis.
    """

    # dictribution plot only with seabon
    plotThreshold(data, 1)

    # PLot peaks
    # original vibration intensities
    res1, nb_intensities1, nb_peaks1 = tryDifferentThreshold(s.data, d, 0, 0)
    """Blues ones
    """
    # peaks without boxplot, threshold == 105, 110, 120, 125
    """We set that our threshold is range from 97 to 110. So we can try
    this function to draw all the rest peaks in order to have a clearer
    visual effect.
    """
    res2, nb_intensities, nb_peaks2 = tryDifferentThreshold(s.data, d, 97, 0)
    res2, nb_intensities, nb_peaks2 = tryDifferentThreshold(s.data, d, 100, 0)
    res2, nb_intensities, nb_peaks2 = tryDifferentThreshold(s.data, d, 105, 0)
    res2, nb_intensities, nb_peaks2 = tryDifferentThreshold(s.data, d, 110, 0)
    res2, nb_intensities, nb_peaks2 = tryDifferentThreshold(s.data, d, 120, 0)
    res2, nb_intensities, nb_peaks2 = tryDifferentThreshold(s.data, d, 125, 0)
    """An idea here: we note that the number of the peaks is acccordingly as
    follow: 279, 259, 231, 196, 121, 65.
    Maybe we can use Human experience combine with the number of the peaks to
    decide which ones to choose as threshold.
    """

    # boxplot of peaks, threshold == 120

    res4, nb_intensities4, nb_peaks4 = tryDifferentThreshold(s.data, d, 105, 1)
    res4, nb_intensities4, nb_peaks4 = tryDifferentThreshold(s.data, d, 100, 1)
    res4, nb_intensities4, nb_peaks4 = tryDifferentThreshold(s.data, d, 110, 1)
    res4, nb_intensities4, nb_peaks4 = tryDifferentThreshold(s.data, d, 115, 1)
    res4, nb_intensities4, nb_peaks4 = tryDifferentThreshold(s.data, d, 120, 1)
    res3, nb_intensities3, nb_peaks3 = tryDifferentThreshold(s.data, d, 125, 1)
    res3, nb_intensities3, nb_peaks3 = tryDifferentThreshold(s.data, d, 127, 1)

    """We can find that the boxplot of the peaks is less and less.
    we can conclude that if the distribution of the peaks is uniform and much
    enough, that might be a good candidate of threshold.
    """
    # boxplot of peaks, threshold == 105
# =============================================================================
#     w = 10
#     h = 10
#     fig = plt.figure()
#     rows = 4
#     columns = 5
#     for i in [100, 105, 110, 115, 120, 127]:
#         # img = np.random.randint(10, size=(h, w))
#         img = tryDifferentThreshold(s.data, d, 120, 1)
#         fig.add_subplot(rows, columns, i)
#         plt.imshow(img)
#     plt.show()
# =============================================================================

# To define maxGap:

    # if we take threshold as 110
    tmp = getDelta(s, 110)
    tmp2 = getDelta(s, 97)

    # plot with probablity density function
    res_quantile1 = plotTimeDelta(tmp.deltaSeconds, 0, d)
    plt.hist(tmp.deltaSeconds.dropna(), bins=500, rwidth=0.9, normed=True)
    """the smallest value is the plot is 579.
    """
    """Conbine pdf+hist+sns.distplot(kde, rug, bins=500)
    kde: whether to plot a gaussian kernel density estimate.(red one)
    rug: whether to plot a rugplot on the support axis.(black one)
    the type of the hist: step (green one)
    We can tell from the plot, [600:] can be candidate of maxGap.
    """
    res_quantile1.minValue
    res_quantile1.maxValue
    res_quantile1.lower
    res_quantile1.upper
    res_quantile1.q1
    res_quantile1.q3

    res_quantile1 = plotTimeDelta(tmp2.deltaSeconds, 0, d)

    # max and min points in indensity plot
    plotTimeDelta(tmp.deltaSeconds, 1, d)

    # boxplot of delta time
    plotTimeDelta(tmp.deltaSeconds, 2, d)
    # by this we can define that maxGap > 530, maybe 540, 550, 600

    # finally plot strokes
    printStrokes(s, 110, 550)


# =============================================================================
#
# # use cases:
# d = Device['88:4A:EA:69:E3:09']
# startTime = '2018-9-4 12:00'
# endTime = '2018-9-4 13:00'
# threshold = 105
# maxGap = 600
# s = getSequence(d, startTime, endTime)
#
#
# =============================================================================















