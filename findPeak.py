#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 13:31:28 2018

@author: pat
"""

import pandas as pd
from anaximander.data.tract.DataTract import DeviceData
import csv

from dataforge.environment import Device

d = Device['88:4A:EA:69:44:E2']

q = DeviceData.bigtable.query('accel_energy_512',
                              mac = d.mac,
                              timestamp = ('2018-9-4 12:00', '2018-9-4 13:00'))

s = q.sequence()

data = s.data.head()
print(data['accel_energy_512'])


df = pd.DataFrame(data)['accel_energy_512']

ress = pd.DataFrame({'timestamp:':df.index,'peak':df.values})

df_array = df.as_matrix() # convert to array

# q.sequence().plot()

def findPeak(A):
    """return the index correspond to the peak."""
    if len(A) == 0 or A is None:
        return 0
    # 2 pointers
    start = 0
    end = len(A) - 2
    # to make sure tha the peak is in the array
    while start + 1 < end:
        mid = int(start + (end - start) / 2)
        if A[mid] < A[mid + 1]:
            start = mid
        else:
            end = mid
    if A[start] < A[end]:
        return end
    else:
        return start

 findPeak(df_array)


def findAllPeaks(B):
    """ return all the index related to the peaks in an array."""
    if len(B) == 0 or B is None:
        return 0
    k = 0
    n = len(B)
    res = []
    peak = []

    for i in range(n):
        if i == 0 and B[i] > B[i+1]:
            print(i)
            k = k + 1
            res.append(i)
            peak.append(B[i])
            # peak.append('%.2f' % B[i])
        elif i == n-1 and B[i] > B[i-1]:
            print(i)
            k = k + 1
            res.append(i)
            peak.append(B[i])
        elif B[i] > B[i-1] and B[i] > B[i+1]:
            print(i)
            k = k + 1
            res.append(i)
            peak.append(B[i])
        elif k == 0:
            print("No peaks.")

    print(k) # 1973
    return res,peak


 res_tuple1 = findAllPeaks(df_array)

 peaks = res_tuple[1]

 # now we select the timestamps correspond to the peak value

 ress[ress['peak'] == peaks[0]]

 ress[ress['peak'] == peaks[0]]['timestamp'].head(3)


def timestampCorrespondPeak(C):
    for i in range(len(C)):
        return ress[ress['peak'] == peaks[i]]['timestamp']

