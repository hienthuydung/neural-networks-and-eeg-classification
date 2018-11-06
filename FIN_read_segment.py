
# coding: utf-8

# In[ ]:


import tensorflow
import pyedflib
import datetime
import numpy as np
import codecs
import csv
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np
import multiprocessing
import pandas as pd
import os
import fnmatch
get_ipython().run_line_magic('matplotlib', 'notebook')

Edf_file = "C:\\Users\\hien\\Desktop\\EEG_annotations\\0-4.edf"
# source = '1519'
with pyedflib.EdfReader(Edf_file) as f:
        Edf_start = f.getStartdatetime() # returns datetime.datetime object
        print("Got Startdatetime for", Edf_file)
        Num_annot = f.annotations_in_file # returns integer
        print("Got number of annotations in", Edf_file)
        print("***", Num_annot, "annotations in", Edf_file,"***")
        Annots = f.readAnnotations() # returns array 
        print("Read in annotations in", Edf_file)


        print("Processing", Edf_file, "with", Num_annot, "annotations...")


        print("Entering Num_annot loop")
        for i in range(Num_annot):
            
            print("Entered Num_annot loop")
            # Get datetime.datetime object with absolute event onset 
            Event_start = Edf_start + datetime.timedelta(seconds=Annots[0][i])
            print("Got Event_start", Event_start)

            #Edf_start = f.getStartdatetime()

            Start = float(Annots[0][i])
            Duration = float(f.readAnnotations()[1][i])
            End = Start + Duration
            Fs = f.getSampleFrequency(0)
            print("Start:", Start, "Type", type(Start))
            print("Duration:", Duration, "Type", type(Duration))
            print("End: ", End, "Type", type(End))
            print("Fs:", Fs)
            print(np.str(Annots[2][i]))

            print(Annots[2][i])
                
            print("Gathering event signal data")
            data = f.readSignal(0)[int( (Start) * Fs) : int( (End) * Fs)]
            print("data has", len(data), "data points")
            np.savetxt("C:\\Users\\hien\\\Desktop\\EEG_annotations"                       + "sep09_anno_{}_".format(i) + np.str(Annots[2][i])+".csv".format(i), data,                         header=np.str(Annots[2][i]) +np.str(Start)+'-dur-'+np.str(Duration), fmt="%10.5f")
#             , delimiter=','




def read_data(filename, header=0, ifnorm=True, mono=True, row_data=True):
    '''read data from .csv
    return:
        data: 2d array [seq_len, channel]'''
    
    data = pd.read_csv(filename, header=header, nrows=None)
    data = data.values   ### get data without row_index
    if ifnorm:   ### 2 * 10240  normalize the data into [0.0, 1.0]]
        data_norm = zscore(data)
        data = data_norm
    if row_data:
        data = data.T
    data = np.squeeze(data)   ## shape from [1, seq_len] --> [seq_len,]
    return data



def split_filter_data_with_long_loss(data, save_name='filename', accept_loss_threshold=10, accept_data_len=2048):
    '''#### start new filter.
    ### 1. discard repeated data if the number of repeatation is below a threshold(10).
        ### it shouldn't make a hug difference since the sampling reate is 512Hz
    ### 2. then segement the recording with long data loss (if the data loss over a threshold, split the data into segments)
    ### 3. discard short recordings
    ### 4. leftover segments with no data repeatation and long enough
    Param:
        data: 1D array with data losses, shape(seq_len,)
        accept_loss_threshold: int, below the threshold, the data can be interpolated
        acccep_data_len: if after segmentation, the data is shorter than this, will be discarded
    return:
        data_segs: shape=[num_seg, variable(seq_len), 1]'''

    error = data[0:-1] - data[1:]
    non_zero_error_ind = np.where(error!=0)[0]   ### those indices where there is no data loss

    loss_intervals = non_zero_error_ind[1:] - non_zero_error_ind[0:-1]

    # ## if smaller than threshold, it means the data loss is short and make sense to squize out the loss points
    long_loss_interval_start = np.where(loss_intervals>accept_loss_threshold)[0]   ##get where there are long(>threshold) data loss
    
    ### either the recording is seperated by long data loss or not
    if long_loss_interval_start.size == 0 and data.size>= accept_data_len: ## the whole sequence is valid
        print("whole sequence is good")
        data_seg_interp = linear_interpolation(data[0:data.size-data.size%2048])
        data_segs = data_seg_interp
        for ii in range(data_seg_interp.size//2048):
            print(data_seg_interp.size//2048, "segments")
            seg = data_seg_interp[ii*2048 : (ii+1)*2048]
            np.savetxt(os.path.dirname(filename) + "/segNorm_{}_No{}.csv"                       .format(os.path.basename(filename[0:-4]), ii),                        seg, delimiter=',', fmt="%10.5f", comments='')
            print(os.path.dirname(filename) + "/segNorm_{}_No{}.csv"                  .format(os.path.basename(filename[0:-4]), ii))
    else:
        print("segmenting data")
        accepted_segs_ind = np.array(np.split( non_zero_error_ind, long_loss_interval_start+1))
        
        data_segs = []
        for ii, ind_seg in enumerate(accepted_segs_ind):
            if ind_seg.size > accept_data_len:
                ### interpolate the missing data
                data_seg_interp = linear_interpolation(data[ind_seg])
                data_segs.append(data_seg_interp[0:data_seg_interp.size-data_seg_interp.size%accept_data_len])
                for ii in range(data_seg_interp.size//2048):
                    seg = data_seg_interp[ii*2048 : (ii+1)*2048]
                    np.savetxt(os.path.dirname(filename) + "/segNorm_{}_No{}.csv".                               format(os.path.basename(filename[0:-4]), ii),                                seg, delimiter=',', fmt="%10.5f", comments='')
            
                  ## discard the data at the end
# #                 np.savetxt(save_name, np.array(data_segs), delimiter=',', fmt="%10.5f", comments='')
                print("file saved")
    return data_segs

def linear_interpolation(data):
    """Helper to handle indices and logical indices of repeated data points(data loss points).

    Input:
        - data, 1d numpy array with possible data loss which appears with repeating previous values
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices

    """

    err = np.insert((data[1:] - data[0:-1]), 0, data[0])   ## the err of the first element is itself
    zeros_ind = err == 0
    x = lambda z:z.nonzero()[0]

    data[zeros_ind] = np.interp(x(zeros_ind) ,x(~zeros_ind), data[~zeros_ind])

    return data

def find_files(directory, pattern='*.csv', withlabel=False):
    '''fine all the files in one directory and assign '1'/'0' to F or N files'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            if withlabel:
                if 'Data_F' in filename:
                    label = '1'
                elif 'Data_N' in filename:
                    label = '0'
                files.append((os.path.join(root, filename), label))
            else:  # only get names
                files.append(os.path.join(root, filename))
#     random.shuffle(files)   # randomly shuffle the files
    return files

