import numpy as np
# import matplotlib.pyplot as plt
from brian2 import *

class TSSTGen():
    minISI = 0
    maxISI = 0

    def __init__(self, minISI, maxISI):
        self.minISI = minISI
        self.maxISI = maxISI

    def getISIs(self,tsst):
        '''
        Calculates the ISIs of a multi-spike spike train - with any number of spikes
        :param tsst: array of spike times
        :return: array of ISIs
        '''
        # get first len(tsst) - 1 spike times
        first = tsst[0:len(tsst)-1]
        last = tsst[1:]
        return np.subtract(last,first)

    def validTSST(self,tsst):
        '''
        Performs three tests on a spike train: \n
        1) none of the ISIs are less than self.minISI \n
        2) none of the ISIs are more than self.maxISI \n
        :param tsst: array of spike times.
        :return:
        valid: \n
        Boolean - True if all conditions are met, false otherise. \n
        reasonStr: string which shows which test(s) passed or failed.
        '''
        valid = True
        reasonStr = ''
        isis = self.getISIs(tsst)
        # Check that all ISIs are > minISI
        valid = valid and not any(isis < self.minISI)
        reasonStr += 'ISIs >= minISI:  ' + str(isis >= self.minISI) + '\n'
        # Check that all ISIs are < maxISI
        valid = valid and not any(isis > self.maxISI)
        reasonStr += 'ISIs >= minISI:  ' + str(isis < self.maxISI) + '\n'
        # Check that all spike times are +ve
        valid = valid and not any(isis <= 0)
        return valid, reasonStr

    # def genRndTSST(self,noSpikes):
    #     #
    #     validTSST = False
    #     tsst = np
    #     while not validTSST:

    def genNoisyVersion(self, origTSST, stDevs):
        noisyTSST = np.random.normal(origTSST, stDevs, len(origTSST))
        valid, _ = self.validTSST(noisyTSST)
        while not valid:
            noisyTSST = np.random.normal(origTSST, stDevs, len(origTSST))
            valid, _ = self.validTSST(noisyTSST)
        return noisyTSST

if __name__ == "__main__":
    minISI_ms = 5.0
    highISI_s = 0.05  # highISI_s = highest ISI permitted in gBeta distribution - in seconds original = 0.05 DON'T CHANGE THIS
    highISI_ms = highISI_s * 1000.0
    M = 3
    stDevs = np.array([3 for i in range(M)])
    patternGenerator = TSSTGen(minISI_ms,highISI_ms)
    isis = [[10, 45], [20, 30]]
    origPatternA = [minISI_ms, minISI_ms + isis[0][0], minISI_ms + isis[0][0] + isis[0][1]]
    origPatternB = [minISI_ms, minISI_ms + isis[1][0], minISI_ms + isis[1][0] + isis[1][1]]
    print(f'noisey versions of {origPatternA}')
    for i in range(10):
        print(patternGenerator.genNoisyVersion(origPatternA,stDevs))
    print(f'noisey versions of {origPatternB}')
    for i in range(10):
        print(patternGenerator.genNoisyVersion(origPatternB, stDevs))

###################################################################################################################
# TEST CODE
###################################################################################################################
# print ('TEST TSSTGenerator.py =====================================================================================')
# tsstGen = TSSTGen(0, 50, 5)
# tsst1 = np.array([5, 7, 12, 45]) # ISI < minISI
# tsst2 = np.array([-5, 12, 20, 45]) # spike time < startTime
# tsst3 = np.array([5, 12, 20, 55]) # spike time > endTime
# tsst4 = np.array([5, 12, 20, 45]) # all conditions met
# print('TSSTGen(0, 50, 5)')
# print('Test TSST: ' + str(tsst1))
#
# print ('Test getISI()')
# print(tsstGen.getISIs(tsst1))
# print ('PASSED ====================================================================================================')
# print ('Test validTSST()')
# print('TEST 1: ISI < minISI')
# valid, reasonStr = tsstGen.validTSST(tsst1)
# print('Result = ' + str(valid))
# print('Reason = ' + reasonStr)
# print('TEST 2: spike time < startTime')
# valid, reasonStr = tsstGen.validTSST(tsst2)
# print('Result = ' + str(valid))
# print('Reason = ' + reasonStr)
# print('TEST 3: spike time > endTime')
# valid, reasonStr = tsstGen.validTSST(tsst3)
# print('Result = ' + str(valid))
# print('Reason = ' + reasonStr)
# print('TEST 4: all conditions met')
# valid, reasonStr = tsstGen.validTSST(tsst4)
# print('Result = ' + str(valid))
# print('Reason = ' + reasonStr)
# print ('PASSED ====================================================================================================')
# print ('Test genNoisyVersion(...)')
# tsstGen = TSSTGen(0, 50, 5)
# stDevs = [1, 1, 1, 1]
# tssts = np.zeros(200).reshape(50,4)
# tssts[0] = [5, 12, 20, 45]
# for i in range(1,len(tssts[:,0])):
#     tssts[i] = tsstGen.genNoisyVersion(tssts[0],stDevs)
# print(tssts)
# plot(tssts, np.arange(0,len(tssts[:,0])), '.k')
# xlabel('Time (ms)')
# ylabel('Neuron index')
# show()
# print ('PASSED ====================================================================================================')