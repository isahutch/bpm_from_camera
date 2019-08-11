# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 09:57:48 2018
# Recording instructions:
1. Use index finger tip
2. Place fingertip exactly on camera, covering the entire lens
3. Limit pressure, tough lens gently so as not to restrict blood flow in the finger
4. 50 seconds is required for reliable measure of HRV; Source: https://www.ncbi.nlm.nih.gov/pubmed/18003044

# Script outputs measures of heart rate variability (sdnn & rmssd)
# Script outputs heart rate in bpm (mhr)
# https://elitehrv.com/normal-heart-rate-variability-age-gender

@author: Isabel
"""

# Import necessary modules
import cv2 as cv # OpenCV for initial video processing
import colorsys # Required for rgb to hsv conversion
import numpy as np
from scipy.signal import butter, lfilter # Required for butterworth filter
from scipy.interpolate import interp1d # Required for cubic spline interpolation to turn 30Hz signal into 180Hz signal
from scipy.signal import find_peaks # Required for peak detection in ECG

# Read in video
video = cv.VideoCapture('finger_video.mov')

# Get sampling frequency and round it up (should be 30 by default, we should use higher frequencies if we can and downsample to 200 Hz ideally)
fs = round(video.get(cv.CAP_PROP_FPS))

# Count number of frames
frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))

# Declare empty hues array 
hues = np.zeros((frame_count,1))

# Create an array hues with average hue per frame for further analysis
framecount = 0
while(1):    
    
    #Read next frame of video
    ret, frame = video.read()

    # end after the last frame is read        
    if frame is None:
        break

    # Extract average r, g and b values from each frame
    av_rgb = frame.mean(axis=0).mean(axis=0)
    
    # Convert to HSV and only save hue in hues array
    hues[framecount,0] = colorsys.rgb_to_hsv(av_rgb[0], av_rgb[1], av_rgb[2])[0]
    framecount += 1
    

# Use 4th order butterworth filter to remove high frequency noise and DC artifact

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def run():
    # Sample rate and desired cutoff frequencies (in Hz).
    # We want to isolate the heart rate rhythm which is between 45 and 180bpm in healthy adults, in Hz that is 0.75 - 3
    lowcut = 0.75
    highcut = 3
    y = butter_bandpass_filter(hues[:,0], lowcut, highcut, fs, order=4)
    return y
   
filtered_hues = run()    


# Smoothing Source: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x, window_len, window): # 5 samples as used by Camera HRV app
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        
    if window == 'flat':
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')     
    
    y = np.convolve(w/w.sum(),s,mode='valid')

    return y

smoothed_hues = smooth(filtered_hues, 5, 'flat')

# Cubic spline interpolation to turn 30Hz signal into 180Hz signal
f = interp1d(np.linspace(0,10, num = len(smoothed_hues), endpoint = True), smoothed_hues, kind = 'cubic')
xnew = np.linspace(0,10, num = int((180/fs)*len(smoothed_hues)), endpoint = True) #we want a new sampling rate of 180, so multiply length of trace by desired frequency divided by current frequency (180/30 = 6) so trace is 6x longer than current
interpolated_hues = f(xnew)

# Remove artifacts 
# Look at 10 seconds at a time, as the fs is 180Hz, 10 seconds is 1800 data points
rri = np.empty((0,)) # This is where the final distances will live

for i in range(int(np.floor(len(interpolated_hues)/1800))):
    
    window = interpolated_hues[i*1800:i*1800+1800]
    
    peaks, _ = find_peaks(window, height=0)
    
    if len(peaks) > 3: # discard window if there are less than 4 intervals in a 10s window
        distances = np.empty((len(peaks),))
        for j in range(len(peaks)-1):
            distances[j] = peaks[j+1] - peaks[j]       
           
    a = 0
    b = 0
    med = np.median(distances)
    length = len(distances)
    for j in range(length):
        if (distances[length-1-j] > (1.2 * med)) or (distances[length-1-j] < (0.8 * med)):
            distances = np.delete(distances, (length-1-j))
        else:
            b += 1
            
    if len(distances) > 0.5 * length: # if the previous step removed more than half of the beats, then discard
        rri = np.concatenate((rri,distances/0.18), axis = 0) # Convert into milliseconds before saving 
            

# Mean interval between RRs        
mrri = np.mean(rri) # Normal range is 785 - 1160ms, healthy mean is 926ms +/- 90

# Mean heart rate
mhr = np.mean(60 / (rri / 1000.0))

# Standard deviation of RR interval differences
sdnn = np.std(rri, ddof=1) # Normal range 32 - 93, healthy mean 50 +/- 16

# The Root Mean Square of Successive RR Intervals    
diff_rri = np.diff(rri)
rmssd = np.sqrt(np.mean(diff_rri ** 2)) # Normal range 19 - 75, mean 42 +/- 15

