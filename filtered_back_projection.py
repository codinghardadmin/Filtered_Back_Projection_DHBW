# result MSE: 289.24

import numpy as np 
from scipy.misc import ascent
from skimage.transform import radon
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fft2, ifft2, fftshift, fftfreq
 
im = ascent()
#im = im[::4,::4]
n = im.shape[0]
padded = np.pad(im, n // 2, mode='constant')
# ANGLES is an array between 0 – 180 degrees, e.g. np.arange(0, 180, 1)    
ANGLES = np.arange(0, 360, 1)
# You need to decide how many angles you need to obtain a good result
sino = radon(padded.astype(np.float), theta=ANGLES, circle=True)

# Your solution here

sino = np.transpose(sino) #matrix transpose

part = (im.shape[0]*2) // 256 # divide image in parts, to generate filter
peak = 0.158 # adjusted value. 0.158 is the best peak value i found

zeros = np.zeros(part*80) # create 80 parts zeros (left and right of the _M_ filter)
filter = np.linspace(-peak,peak,endpoint=True,num=part*96) # 96 create linear function with the rest parts
filter_abs = np.abs(filter) # Make linear function "\"" to "V"

filter_abs = np.concatenate([zeros, filter_abs, zeros]) # concatenate all 

omega = fftshift(filter_abs) # fftshift to get omega

# main logic

n = len(sino[0])
num_theta = len(sino)
y, x = np.mgrid[-n//2:n//2, -n//2:n//2] # create mgrid
result = np.zeros((n,n))

# print(n)
# print(len(sino))
 
for theta in range(num_theta):
    function_temp = np.zeros_like(result)

    for i in range(n): # following from pseudocode in skript
        filtered = omega * fft(sino[theta]) # create filtered of fft
        back_transformed = ifft(filtered) # inverse fourier transform
        real_part = np.real(back_transformed) # only real part via numpy
        back_ratio = np.pi / 360 # from skript, 180 degree because choosed above

        function_temp[i] = function_temp[i] + back_ratio * real_part
 
    # calc backwards approach

    theta_rad = np.deg2rad(theta) # pre-calc theta_rad

    cos_ = np.cos(theta_rad) # pre-calc cos
    sin_ = np.sin(theta_rad) # pr-calc sin

    # from lessons
    new_x   = np.clip(x*cos_ - y*sin_ + n/2, 0, n-1).astype(np.int) # clip to 0 .. 1023
    new_y   = np.clip(x*sin_ + y*cos_ + n/2, 0, n-1).astype(np.int) # clip to 0 .. 1023
 
    # add new calculated values to image.
    # time by time the full image is reconstructed
    result += function_temp[new_y, new_x]


# cut func to image shape
part = im.shape[0] // 2
solution = result[part:part*3, part:part*3]

# MSE of your solution with respect to original
print(np.mean((im - solution) ** 2))