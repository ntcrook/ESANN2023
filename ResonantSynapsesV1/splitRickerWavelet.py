#################################################################################
#   Split Mexican Hat or Ricker Wavelet
#   18/07/2022
#   Based on Wikipedia: https://en.wikipedia.org/wiki/Ricker_wavelet
#   Beta distribution based on : https://vitalflux.com/beta-distribution-explained-with-python-examples/
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as plo

sig = 5

def rickerWavelet(t):
    norm = 2/(np.sqrt(3*sig*np.pi**(1/4)))
    rickerW =  norm * (1 - (t/sig)**2) * np.exp(-(t**2)/(2*sig**2))
    if t > 0:
        rickerW = -rickerW
    return rickerW

###################################################################################################################
# TEST CODE
###################################################################################################################
# min_t = -40
# max_t = 40
# x = np.zeros(100)
# t = np.arange(min_t,max_t,(max_t - min_t)/len(x))
# for i in range(len(x)):
#     x[i] = rickerWavelet(t[i])
#
# lowISI_s = 0                    # lowISI_s = lowest ISI permitted in gBeta distribution- in seconds
# highISI_s = 0.05
# lowRickerWavelet = -40          # lowest value for split Ricker wavelet
# highRickerWavelet = 40          # higest value for split Ricker wavelet
# splitRickerWaveletScaling = (lowRickerWavelet - highRickerWavelet)/(lowISI_s - highISI_s)
# isi = 0.008
# mean = isi
# srwParam = splitRickerWaveletScaling*(isi - mean)
# srwValue = rickerWavelet(srwParam)
# scaledSRWValue = srwValue/splitRickerWaveletScaling
# print("Scaled srvValue = " + str(scaledSRWValue))
#
# plt.plot(t,x)
# plt.show()


