import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
from generalBetaDistribution import gBeta, approxArea, meanGBeta, varianceGBeta, gBetaParamEst_Eleni2
import TSSTGenerator as tsstGen
import welfordsOnlineAlgorithm as woa
from csv import writer
import time
import splitRickerWavelet as srw

# FORWARD DECLARATION
# spikes = SpikeGeneratorGroup(1, [], []*ms)
# group_mon = SpikeMonitor(spikes)

# MODEL PARAMETERS
thresh = 1.0                    # Firing threshold of the neurons
isiRelevanceThresh = 0.1
patternDuration = 200*ms        # The length of time that the model is run for each input pattern - orginally 100
noNeurons = 3                   # noNeurons = number of neurons
noSynapses = 6                  # noSynapses = number of synapses
N = 400                           # N = number of noisy versions of the TSST
M = 3                           # M = number of spikes in each TSST
initialW_high =2.0/(M - 1)               # highest randomised initial value of synaptic weight
initialW_low = initialW_high/2                # lowest randomised initial value of synaptic weight
deltaW = 0.05
transitionW = 0.75              # Transmission weight
minSynapticDelay = 10.0
maxSynapticDelay = 50.0 # original value = 30,0
lowISI_s = 0.0                    # lowISI_s = lowest ISI permitted in gBeta distribution- in seconds
highISI_s = 0.05                # highISI_s = highest ISI permitted in gBeta distribution - in seconds original = 0.05 DON'T CHANGE THIS
lowISI_ms = lowISI_s*1000.0       # lowISI_ms = lowest ISI permitted in gBeta distribution - in milliseconds
highISI_ms = highISI_s*1000.0     # highISI_ms = highest ISI permitted in gBeta distribution - in milliseconds
minISI_ms = 5.0                  # minISI_ms = minimum ISI permitted for the spike generator
gBetaScale = (thresh * 1.8)/(100 * (M - 1))              # gBetaScale = Scaling factor for gBeta original = 1.0/200.0 changed to bring the total bBeta above threshold.
gBetaParamSum = 40.0              # gBetaParamSum = q + r, used to generate random q and r values
gBetaParamMin = 10.0             # gBetaParamMin = the lowest value of q and r to maintain bell-shaped curve
q = [[] for i in range(noNeurons-1)]
r = [[] for i in range(noNeurons-1)]
for neuron in range(noNeurons-1):
    q[neuron] = np.random.uniform(gBetaParamMin,gBetaParamSum - gBetaParamMin,noSynapses//(noNeurons-1)) # Set random q values for each synapse
    r[neuron] = gBetaParamSum - q[neuron]           # Set random r values for each synapse
synapticDelays = np.zeros(noSynapses)

# store transmission times and values for all synapses to each neuron
transmissionRecord = [[[[],[]] for i in range(noSynapses//(noNeurons-1))] for n in range(noNeurons-1)]

class SynapticTransmissions():

    synapticTrans = [False for i in range(noSynapses//(noNeurons-1))]
    isiCounts = [-1 for i in range(noSynapses // (noNeurons - 1))]
    isiTrans = [-1 for i in range(noSynapses // (noNeurons - 1))]

    def update(self,synapseIndex,value):
        self.synapticTrans[synapseIndex] = value
        if value:
            self.isiTrans[synapseIndex] = self.isiCounts[synapseIndex]

    def updateISICount(self,synapseIndex):
        self.isiCounts[synapseIndex] += 1

    def getSynapsticTrans(self):
        return self.synapticTrans

    def ith(self,i):
        return self.synapticTrans[i]

    def ithISINo(self,i):
        return self.isiTrans[i]

    def reset(self):
        self.synapticTrans = [False for i in range(noSynapses//(noNeurons-1))]
        self.isiCounts = [-1 for i in range(noSynapses // (noNeurons - 1))]
        self.isiTrans = [-1 for i in range(noSynapses // (noNeurons - 1))]

synTrans= [SynapticTransmissions() for i in range(noNeurons-1)]

delayAdjustWeight = 0.3         # The amount by which dynaptic delays are moved closer to the mean of the delays of all synapses that transmitted this run - original value 0.2
noResTableCols = 11                # The number of columns in results table

meanBeta = [np.zeros(noSynapses//(noNeurons-1)) for i in range(noNeurons-1)] # meanBeta = The mean of the gBeta distribution for each synapse
for neuron in range(noNeurons-1):
    for synapseIndex in range(noSynapses//(noNeurons-1)):
        meanBeta[neuron][synapseIndex] = meanGBeta(lowISI_s, highISI_s,q[neuron][synapseIndex], r[neuron][synapseIndex])
dtArea = 0.0001                  # dt use to calculate area under curve of gBeta
# lowGBetaArea = np.zeros(noSynapses) # lowGBetaArea = the area of the gBeta distribution below the mean for each synapse
# highGBetaArea = np.zeros(noSynapses) # highGBetaArea = the area of the gBeta distribution above the mean for each synpase
# for synapseIndex in range(noSynapses):
#     lowGBetaArea[synapseIndex] = approxArea(meanBeta[synapseIndex], lowISI_s, dtArea, lowISI_s, highISI_s,
#                                             q[synapseIndex],r[synapseIndex])
#     highGBetaArea[synapseIndex] = approxArea(meanBeta[synapseIndex], highISI_s, dtArea, lowISI_s, highISI_s,
#                                             q[synapseIndex], r[synapseIndex])
existingAggregates = [[] for i in range(noNeurons-1)]
existingAggregates[0] = [(0, 0, 0) for i in range(noSynapses//(noNeurons-1))]  # Stores count, mean, M2 (M2 aggregates the squared distance from the mean)
existingAggregates[1] = [(0, 0, 0) for i in range(noSynapses//(noNeurons-1))]                                # for each synpase for use with the Welfords online mean and variance algorithm

# SPIKE GENERATOR FOR NEURON 0
# Create original TSST with M spike together with N noisy versions of it
tsstGen = tsstGen.TSSTGen(minISI_ms, highISI_ms)
# Set the standard deviation for the randomisation of each spike
stDevs = np.ones(M)
tsstsA = np.zeros(N*M).reshape(N,M)
origPatternA = np.random.randint(low=(lowISI_ms + minISI_ms),high=(highISI_ms - minISI_ms),size=1)
for spike in range(1,M):
    [isi] = np.random.randint(low=(lowISI_ms + minISI_ms),high=(highISI_ms - minISI_ms),size=1)
    origPatternA = np.append(origPatternA, [origPatternA[spike - 1] + isi])
tsstsA[0] = origPatternA
tsstsB = np.zeros(N*M).reshape(N,M)
origPatternB = np.random.randint(low=(lowISI_ms + minISI_ms),high=(highISI_ms - minISI_ms),size=1)
for spike in range(1,M):
    [isi] = np.random.randint(low=(lowISI_ms + minISI_ms),high=(highISI_ms - minISI_ms),size=1)
    origPatternB = np.append(origPatternB, [origPatternB[spike - 1] + isi])
tsstsB[0] = origPatternB

########################################################################################################################
########################################################################################################################
# DEBUG: Replace random patterns with fixed, very different isi sequences.
tsstsA[0] = [10, 55, 65]
tsstsB[0] = [10, 30, 75]
########################################################################################################################
########################################################################################################################

trainSig = [1] # Timing of the training signal this pattern.
# Create random versions of original tssts
neuronIndices = np.zeros(M)
for i in range(1,len(tsstsA[:,0])):
    tsstsA[i] = tsstGen.genNoisyVersion(tsstsA[0],stDevs)
for i in range(1,len(tsstsB[:,0])):
    tsstsB[i] = tsstGen.genNoisyVersion(tsstsB[0],stDevs)

# CREATE LISTS TO STORE THE VALUE OF y (ACTUAL OUTPUT) IN NEURONS 1 AND 2 TOGETHER WITH TARGET OUTPUT TO USE IN MSE CALCULATION
# mseTrace[0][i] = count of spikes (minus training spike) neuron 1 during run i
# mseTrace[1][i] = count of spikes (minus training spike) neuron 2 during run i
# mseTrace[2][i] = target output for neuron 1 at run i
# mseTrace[3][i] = target output for neuron 2 at run i
# mseTrace[4][i] = mse across all neurons after run i
# mseTrace[5][i] = running average mse across the last nMSE runs after run i
mseTrace = [[] for i in range((noNeurons-1)*2 + 2)]
nMSE = 10 # running average MSE over last n runs

# PRINT RESULTS AS A TABLE
# Collect data as dictionary
results = {}
resultIndex = 0
tableFormat = "{:<5} {:.4f} {:^13} {:^13} {: e} {: e} {: e} {: e} {: e} {: e} {:<13}"
def addResult(thisResult):
    global resultIndex
    results[resultIndex] = thisResult
    # print(tableFormat.format(resultIndex, thisResult[0], thisResult[1], thisResult[2], thisResult[3], thisResult[4], thisResult[5], thisResult[6],thisResult[7], thisResult[8], thisResult[9]))
    # printTableHeader()
    resultIndex += 1

def printTableHeader():
    print("{:<5} {:<7} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13}".format('Item', 'Time', 'NeuronId', 'SynpaseId', 'Delay', 'ISI', 'gBeta:q','gBeta:r', 'gBeta:mean','gBeta:var', 'Relevant?'))

# BUILD THE MODEL
# Define and create neurons
eqs = '''
    dv/dt = -v/tau : 1
    tau : second
'''
G = NeuronGroup(noNeurons, eqs, threshold='v>thresh', reset='v = 0', method='exact', name = 'G')
G.tau = [10, 10, 10]*ms
group_mon = SpikeMonitor(G, name='group_mon')

# GET gBeta PARAMETER q AND r AT RUN TIME
@implementation('numpy', discard_units=True)
@check_units(result=1)
def getBetaParam(neuronId, param, synapseNumber):
    # param = 1 signifies 'q'
    # print('Synapse: ' + str(synapseNumber) + ' param: ' + str(param))
    if param == 1:
        value = q[neuronId][synapseNumber];
        # print('q = ' + str(value))
    else:
        value = r[neuronId][synapseNumber]
        # print('r = ' + str(value))
    return value

# CALCULATE ISI AT RUN TIME
@implementation('numpy', discard_units=True)
@check_units(result=1)
def calcISI(neuronId,synapseIndex, t,lastUpdate):
    global synTrans,existingAggregates, q, r, meanBeta, synapticDelays
    count, mu, var = existingAggregates[neuronId][synapseIndex]
    pISI = 0
    isi = 0
    transition = 0
    scaledGBeta = 0
    relevant = False
    if lastUpdate > 0:
        global synTrans
        synTrans[neuronId].updateISICount(synapseIndex)
        isi = t - lastUpdate
        # print(f">>>>>>>>>>> ISI = {isi}")
        # find the mean of the current gBete function
        # meanBeta = meanGBeta(lowISI_s, highISI_s,getBetaParam(1,synapseIndex), getBetaParam(2,synapseIndex))

        ################################################################################################################
        ###################################### Code patch 1.1 - out  #####################################################
        # area = approxArea(meanBeta[neuronId][synapseIndex], isi, dtArea, lowISI_s, highISI_s, getBetaParam(neuronId, 1, synapseIndex),
        #            getBetaParam(neuronId, 2, synapseIndex))
        # # adjust for error in the calculation of the area
        # # area = min(area,0.5)
        # pISI = 1 - 2*area
        # relevant = np.random.uniform(0,1) <= pISI
        ################################################################################################################
        ###################################### Code patch 1.1 - in  #####################################################
        scaledGBeta = gBetaScale * gBeta(isi, lowISI_s, highISI_s, getBetaParam(neuronId,1,synapseIndex), getBetaParam(neuronId,2,synapseIndex),False)
        if scaledGBeta > isiRelevanceThresh:
            relevant = True
        ################################################################################################################
        ################################################################################################################
        # Check that neuronId fired within patternDuration ms
        trainingSignal = False
        neuronIdSpikeTrain = group_mon.spike_trains()
        if neuronId + 1 in neuronIdSpikeTrain:
            neuronIdSpikeTrain = asarray(group_mon.spike_trains()[neuronId+1])
            lenSpikeMon = len(neuronIdSpikeTrain)
            if lenSpikeMon > 0:
                lastPostSynSpikeTime = neuronIdSpikeTrain[lenSpikeMon-1]
                timeUnitless = np.array(defaultclock.t)
                if timeUnitless - lastPostSynSpikeTime < patternDuration:
                    trainingSignal = True
        if relevant:
            transition = transitionW
            # Record that this synapse transmitted this run
            synTrans[neuronId].update(synapseIndex, True)
            ############################################################################################################
            ############################################# DEBUG ########################################################
            # Capture the time and value of the transmission from this synpase
            transmissionRecord[neuronId][synapseIndex][0].append(t)
            transmissionRecord[neuronId][synapseIndex][1].append(transitionW)  #scaledGBeta)
            ############################################################################################################
            ############################################################################################################
            if trainingSignal:
                # If the ISI is relevant and training is on, then adjust the mean of the gBeta
                existingAggregates[neuronId][synapseIndex] = woa.update(existingAggregates[neuronId][synapseIndex], isi)
                count, mu, var = existingAggregates[neuronId][synapseIndex]
                # sigma = np.sqrt(var)
                if count > 20:
                    # synTrans[neuronId].update(synapseIndex,True) # MOVED TO 175
                    qTemp, rTemp = gBetaParamEst_Eleni2(mu, var, lowISI_s, highISI_s)
                    # Check that the new values of q and r will generate a bell shaped curve
                    if (qTemp >= gBetaParamMin) and (rTemp >= gBetaParamMin):
                        q[neuronId][synapseIndex] = qTemp
                        r[neuronId][synapseIndex] = rTemp
                        meanBeta[neuronId][synapseIndex] = mu
        else:
            # If not relevant, then do not transmit the spike.
            isi = lowISI_s
    # Prepare data for results table
    # result = ['' for j in range(noResTableCols)]
    # result[0] = resultIndex
    # gBetaMean = meanGBeta(lowISI_s, highISI_s,q[neuronId][synapseIndex],r[neuronId][synapseIndex])
    # gBetaVar = varianceGBeta(lowISI_s, highISI_s,q[neuronId][synapseIndex],r[neuronId][synapseIndex])
    # call addResult with parameter: 'Time', 'SynpaseId', 'Delay', 'ISI', 'gBeta:q', 'gBeta:r', 'gBeta:mean', 'gBeta:var', 'Relevant?'
    # addResult([t, neuronId, synapseIndex, synapticDelays[synapseIndex], isi, q[neuronId][synapseIndex], r[neuronId][synapseIndex], gBetaMean, gBetaVar,str(relevant)])
                                                                        # TODO: Check that S.delay[i] references the correct synpase
    # print("calcISI(synapseIndex = " + str(synapseIndex) + " t = " + str(t) + "; lastUpdate = " + str(lastUpdate) + ") = " + str(isi) + " ISI (count, mean, var) = " + str(existingAggregates[synapseIndex]) + " p(tau_mu,isi) = " + str(pISI) + " Relevant ISI = " + str(relevant))
    ########################################################################################################################
    ############################################## Code patch 1.2 - out  #####################################################
    # return isi
    ########################################################################################################################
    ############################################## Code patch 1.2 - in  #####################################################
    return transition
    ########################################################################################################################
    ########################################################################################################################

# Create resonant synapses between G[0] and G[1]
synModel = '''
    w :1
    lastupdate : second
'''
########################################################################################################################
############################################## Code patch 1.3 - out  #####################################################
# v_post_Eq = '''
#     isi = calcISI(j-1,synapse_number, t, lastupdate)
#     v_post += gBetaScale * gBeta(isi, lowISI_s, highISI_s, getBetaParam(j-1,1,synapse_number), getBetaParam(j-1,2,synapse_number),False)
#     lastupdate = t
# '''
########################################################################################################################
############################################## Code patch 1.3 - in  #####################################################
v_post_Eq = '''
    v_post += calcISI(j-1,synapse_number, t, lastupdate) 
    lastupdate = t
'''
########################################################################################################################
########################################################################################################################
S = Synapses(G, G, model = synModel, on_pre=v_post_Eq, multisynaptic_index='synapse_number')
S.connect(i=0, j=1, n=noSynapses//(noNeurons-1))
S.connect(i=0, j=2, n=noSynapses//(noNeurons-1))
# S.w = np.random.uniform(initialW_low,initialW_high,noSynapses)
# S.w = '1.75' # Original value '0.02'
# randomly set synpatic delays
for i in range(noSynapses):
    S.delay[i] = np.random.uniform(minISI_ms,highISI_ms)*ms
    synapticDelays[i] = S.delay[i]

# Create SpikeGeneratorGroup and connect to neuron 0 in G
inp = SpikeGeneratorGroup(1,neuronIndices, tsstsA[0]*ms, name='inp', when='before_synapses')
syn = Synapses(inp, G, on_pre='v_post += 2', name='syn')
syn.connect(i=0,j=0)
inpIndices = np.zeros(M)
# Create SpikeGeneratorGroups for the training Signal and connect to neurons in G - assuming neuron index 0 is input neuron
trainSigsA = SpikeGeneratorGroup(1,[0], trainSig*ms, name='trainSigsA', when='before_synapses')
trainSigsB = SpikeGeneratorGroup(1,[0], trainSig*ms, name='trainSigsB', when='before_synapses')
trainSynA = Synapses(trainSigsA, G, on_pre='v_post += 2', name='trainSynA')
trainSynB = Synapses(trainSigsB, G, on_pre='v_post += 2', name='trainSynB')
trainSynA.connect(i=0, j=1)
trainSynB.connect(i=0, j=2)

# STATE MONITORING
stateMonitor = StateMonitor(G, 'v', record=True)
inp_mon = SpikeMonitor(inp, name='inp_mon')
train_monPatA = SpikeMonitor(trainSigsA, name='train_monPatA')
train_monPatB = SpikeMonitor(trainSigsB, name='train_monPatB')

# weigthMonitor = []
# for i in range(noSynapses):
#     weigthMonitor.append([S.w[i]])

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 15,
        }

# PLOT gBeta DISTRIBUTIONS FOR EACH SYNAPSE BEFORE LEARNING
def plotGBeta(iterationNo):
    nearEdgeAjust = 0.00001
    maxHeight = 0.0
    maxX_s = ((maxSynapticDelay/1000.0) + highISI_s)*ms*1000.0
    colourSeq = ['b','g','r','c','m','y','k']
    # xGBeta_s = np.linspace(lowISI_s + nearEdgeAjust,highISI_s - nearEdgeAjust,100)
    xGBeta_s = np.linspace(lowISI_s, highISI_s, 100)
    # x = np.linspace(0, maxX, 100)
    gB2 = [np.zeros((noSynapses,100)) for i in range(noNeurons-1)]
    for neuron in range(noNeurons-1):
        for i in range(len(q[neuron])):
            gB2[neuron][i,:] = gBeta(xGBeta_s,lowISI_s,highISI_s,q[neuron][i],r[neuron][i],False)
            m = max(gB2[neuron][i, :])
            if m > maxHeight:
                maxHeight = m
    fig, (ax1,ax2) = plt.subplots(1,2,figsize = (14,7))
    # plt.figure(figsize=(7,7))
    fig.suptitle('General Beta Distribution - i = ' + str(iterationNo), fontsize='15')
    ax1.set_xlim(0, maxX_s)
    ax2.set_xlim(0, maxX_s)
    ax1.set_ylim(0,maxHeight + maxHeight*0.05)
    ax2.set_ylim(0, maxHeight + maxHeight * 0.05)
    # plt.xlim(0, maxX_s)
    # plt.ylim(0,maxHeight + maxHeight*0.05)
    lineHeight = maxHeight/(4 * len(q[neuron]))
    for i in range(0,noSynapses//(noNeurons-1)):
        x = xGBeta_s*ms*1000.0 + S.delay[i]
        colour = colourSeq[i % len(colourSeq)]
        ax1.plot(x, gB2[0][i, :], label = "Synapse " + str(i), c = colour) # TODO: find out why label is not showing
        ax1.set_title('Neuron 1')
        ax1.set_xlabel(f'Delay + GBeta mean ({lowISI_s},{highISI_s})', fontsize='15')
        ax1.set_ylabel('Probability', fontsize='15')
        meanXDelayed_ms = meanBeta[0][i]*ms*1000.0 + S.delay[i]
        xMin = S.delay[i/(1000.0*ms)]
        xMax = meanXDelayed_ms
        yMin = i * (lineHeight + 1.0)
        yMax = yMin + lineHeight
        xRectangle = [xMin, xMax, xMax, xMin, xMin]
        yRectangle = [yMax, yMax, yMin, yMin, yMax]
        ax1.plot(xRectangle,yRectangle,ls='--', c = colour)
        # ax1.text(xMin,yMin+1,synTrans[0].ithISINo(i),c = colour, fontdict = font)
    for i in range(0,noSynapses//(noNeurons-1)):
        x = xGBeta_s*ms*1000.0 + S.delay[i + (noSynapses//(noNeurons-1))]
        colour = colourSeq[i % len(colourSeq)]
        ax2.plot(x, gB2[1][i, :], label = "Synapse " + str(i), c = colour) # TODO: find out why label is not showing
        ax2.set_title('Neuron 2')
        ax2.set_xlabel(f'Delay + GBeta mean ({lowISI_s},{highISI_s})', fontsize='15')
        ax2.set_ylabel('Probability', fontsize='15')
        # Calculate the length and position of the ISI box
        meanXDelayed_ms = meanBeta[1][i]*ms*1000.0 + S.delay[i + (noSynapses//(noNeurons-1))]
        xMin = S.delay[(i + (noSynapses//(noNeurons-1)))/(1000.0*ms)]
        xMax = meanXDelayed_ms
        yMin = i * (lineHeight + 1.0)
        yMax = yMin + lineHeight
        xRectangle = [xMin, xMax, xMax, xMin, xMin]
        yRectangle = [yMax, yMax, yMin, yMin, yMax]
        ax2.plot(xRectangle,yRectangle,ls='--', c = colour)
        # ax2.text(xMin, yMin+1, synTrans[1].ithISINo(i), c=colour, fontdict = font)
    # plt.title('General Beta Distribution - Neuron Id = ' + str(postSynNeuronId) + ' - i = ' + str(iterationNo), fontsize='15')
    # plt.xlabel(f'Values of Random Variable X ({lowISI_s},{highISI_s})', fontsize='15')
    # plt.ylabel('Probability', fontsize='15')

# UPDATE SYNAPTIC DELAYS
def updateDelays(neuronId):
    # Calculate mean delay + ISI of each synapse that transmitted per post-synaptic neuron
    meanDelay = 0
    count = 0
    spikes = group_mon.spike_trains()
    # print(f'Neuron {neuronId + 1} spike time: {spikes[neuronId + 1]}')
    lastSpikeTime = spikes[neuronId+1][len(spikes[neuronId+1])-1]
    # Check that neuronId fired within patternDuration ms
    currentTime = defaultclock.t
    timeSinceLastSpike = currentTime - lastSpikeTime
    if  timeSinceLastSpike < patternDuration:
        # print(f'Current time = {defaultclock.t/ms} - Last spike time {lastSpikeTime/ms} is within pattern duration time {patternDuration}: UPDATING DELAYS')
        # Calculate the mean delays
        for i in range(noSynapses//(noNeurons-1)):
            if synTrans[neuronId].ith(i):
                isi = meanBeta[neuronId][i]*1000*ms
                delay = S.delay[i + neuronId*(noSynapses//(noNeurons-1))]
                meanDelay += isi + delay
                count += 1
        if count > 1:
            # print(f"UPDATING DELAYS FOR NEURON {neuronId}")
            meanDelay = meanDelay/count
            # Adjust each delay to be closer to the mean
            for i in range(noSynapses//(noNeurons-1)):
                synIndex = i + neuronId * (noSynapses // (noNeurons - 1))
                if synTrans[neuronId].ith(i):
                    isi = meanBeta[neuronId][i] * 1000 * ms
                    delay = S.delay[synIndex] # TODO Check that this is referencing the correct synapse
                    adjustDelay = (meanDelay - isi - delay)*delayAdjustWeight
                    S.delay[synIndex] += adjustDelay
                    synapticDelays[synIndex] = S.delay[synIndex]  # replaced i with synIndex in synapticDelays

# UPDATE SYNAPTIC WEIGHTS
# Update the weights of all synapses that transmitted
# def updateWeights(neuronId):
#     # Check that neuronId fired within patternDuration ms
#     lastSpikeTime = spikes[neuronId + 1][len(spikes[neuronId + 1]) - 1]
#     if defaultclock.t - lastSpikeTime < patternDuration:
#         for i in range((noSynapses//(noNeurons-1))):
#             synIndex = i + neuronId * (noSynapses // (noNeurons - 1))
#             if synTrans[neuronId].ith(i):
#                 # Adjust the weight of this synapse to be closer to initialW_high
#                 S.w[synIndex] = S.w[synIndex] + (deltaW * (initialW_high - S.w[synIndex]))
#             else:
#                 # This synapse didn't transmit, reduce its weight
#                 S.w[synIndex] = S.w[synIndex] - (0.5 * deltaW * (S.w[synIndex]))

# CALCULATE MSE
def mse(atIndex):
    t = np.array([mseTrace[2][atIndex],mseTrace[3][atIndex]])
    y = np.array([mseTrace[0][atIndex],mseTrace[1][atIndex]])
    return np.square(t - y).mean()

# CALCCULATE AVERAGE MSE OVER THE LAST nMSE RUNS
def averageMSEOvernMSERuns(atIndex):
    avMSE = 0
    if len(mseTrace[0]) >= nMSE:
        runningTotal = 0
        for iMSE in range (atIndex - (nMSE - 1), atIndex + 1):
            runningTotal += mseTrace[4][iMSE]
        avMSE = runningTotal/nMSE
    return avMSE

def main():
    # MAIN LOOP
    # Use .set_spikes(...) method of SpikeGeneratorGroup as explained here:
    # https://brian.discourse.group/t/problem-changing-spike-values-in-spikegeneratorgroup-during-simulation/290
    i = 0
    plotGBetaFreq = 50
    # printTableHeader()
    trainingOn = True
    testSetSize = 100 # Training will be turned off for the last testSetSize patterns
    for thisTSST in range(len(tsstsA)):
        if i >= len(tsstsA) - testSetSize - 1:
            trainingOn = False
        # randomly choose between pattern A and B
        spikeTrain = ''
        timedTrainSig = trainSig*ms + defaultclock.t
        # if np.random.uniform(0,1) <= 0.5: # For random selections of training/test patterns
        if thisTSST % 2 == 0:
            tsst = tsstsA[thisTSST]
            spikeTrain = 'A'
            if trainingOn:
                trainSigsA.set_spikes([0],timedTrainSig)
        else:
            tsst = tsstsB[thisTSST]
            spikeTrain = 'B'
            if trainingOn:
                trainSigsB.set_spikes([0],timedTrainSig)
        timedTSST = tsst*ms + defaultclock.t
        # print(f'Initiate run {i} with noisy spike train {spikeTrain}: {tsst} adjusted to current time: {timedTSST}')
        inp.set_spikes(inpIndices,timedTSST)
        run(patternDuration) # Originally 100*ms
        # Prepare data for MSE calculation
        spikes = group_mon.spike_trains()
        for neuron in range(noNeurons-1):
            # Update counts to support MSE calculation
            currentTime = defaultclock.t
            lastSpikeTime = currentTime
            spikeCount = 0
            patternSpikeCount = 0
            while (lastSpikeTime > currentTime - patternDuration) and (spikeCount < len(spikes[neuron + 1])):
                lastSpikeTime = spikes[neuron + 1][len(spikes[neuron + 1]) - 1 - spikeCount]
                if lastSpikeTime > currentTime - patternDuration:
                    patternSpikeCount += 1
                spikeCount += 1
            if spikeTrain == 'A':
                # update spike counts for input pattern A
                if trainingOn and neuron == 0:
                    # remove the count of the training spike
                    mseTrace[neuron].append(patternSpikeCount - 1)
                else:
                    mseTrace[neuron].append(patternSpikeCount)
            elif spikeTrain == 'B':
                # update spike counts for input pattern B
                if trainingOn and neuron == 1:
                    # remove the count of the training spike
                    mseTrace[neuron].append(patternSpikeCount - 1)
                else:
                    mseTrace[neuron].append(patternSpikeCount)
            # Update delays of synapses that transmitted a spike this run
            if trainingOn:
                updateDelays(neuron)
                # updateWeights(neuron)
        if spikeTrain == 'A':
            # Update target output for this input pattern
            mseTrace[((noNeurons - 1) * 2) - 2].append(1)
            mseTrace[((noNeurons - 1) * 2) - 1].append(0)
        elif spikeTrain == 'B':
            # Update target output for this input pattern
            mseTrace[((noNeurons - 1) * 2) - 2].append(0)
            mseTrace[((noNeurons - 1) * 2) - 1].append(1)
        if i % plotGBetaFreq == 0:
            plotGBeta(i)
            show()
        # Caclulate MSE
        # print(f'MSE at iteration {i} = {mse(len(mseTrace[0])-1)}')
        mseTrace[(noNeurons-1)*2].append(mse(len(mseTrace[0])-1))
        # Caclulate running average of MSEs over the last nMSE runs
        mseTrace[(noNeurons-1)*2 + 1].append(averageMSEOvernMSERuns(len(mseTrace[0])-1))
        # Capture the weights
        # for s in range(noSynapses):
        #     weigthMonitor[s].append(S.w[s])
        # Reset the transmission record
        for neuron in range(noNeurons-1):
            synTrans[neuron].reset()
        i += 1
    plotGBeta(i)
    # Save the data to .csv file
    # /Volumes/GoogleDrive/My Drive/DOCUMENTS/RESEARCHER/Code/Python/ResonantSynapses/ResonantSynapsesV1/Results/learnMeanVarAndDelay_3Neurons_NSynapses/2022_11_29/2022_11_29__08_41_MSEData.csv
    with open('Results/learnMeanVarAndDelay_3Neurons_NSynapses/2022_11_29/2022_11_29__08_41_MSEData.csv','a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow([time.strftime("%Y-%m-%d %H:%M")])
        writer_object.writerow(["OutputSpikes and Training signal"])
        for i in range(len(mseTrace)):
            writer_object.writerow(mseTrace[i][:])
        # writer_object.writerow(['Weights'])
        # for i in range(len(weigthMonitor)):
        #     writer_object.writerow(weigthMonitor[i][:])
        writer_object.writerow(' ')
        f_object.close()
    # PLOT THE INSTANTANEOUS AND RUNNING AVERAGE MSE
    plt.figure(figsize=(14,7))
    plt.plot(mseTrace[len(mseTrace)-2], ls='-',c='m', label='Instantaneous MSE')
    plt.plot(mseTrace[len(mseTrace)-1], ls='-',c='r', label='Running average MSE')
    plt.legend(loc="upper left")
    plt.title('Instantaneous and running average MSE')
    # PLOT THE WEIGHT VALUES
    # colourSeq = ['b','g','r','c','m','y','k']
    # for neuron in range(noNeurons-1):
    #     plt.figure(figsize=(14,7))
    #     plt.ylim(0,1.2)
    #     for i in range(noSynapses//(noNeurons-1)):
    #         plt.plot(weigthMonitor[i + neuron*(noSynapses//(noNeurons-1))], ls = '-', c = colourSeq[i % len(colourSeq)], label = str(i))
    #     plt.title(f'Synaptic Weights for Neuron {neuron + 1}')
    figure(figsize=(14,7))
    plot(stateMonitor.t / ms, stateMonitor.v[0], label='Neuron 0')
    title('Neuron 0')
    figure(figsize=(14,7))
    plot(stateMonitor.t / ms, stateMonitor.v[1], label='Neuron 1')
    title('Neuron 1')
    figure(figsize=(14,7))
    plot(stateMonitor.t / ms, stateMonitor.v[2], label='Neuron 2')
    title('Neuron 2')
    figure(figsize=(14,7))
    for t in group_mon.t:
        axvline(t/ms, ls = '--', c='g', lw = 1)
    for t in inp_mon.t:
        axvline(t/ms,ls = '-', c='b', lw = 1)
    xlabel('Time (ms)')
    ylabel('v')
    legend();
    plt.figure()
    plt.plot(group_mon.t/ms, group_mon.i+1, '.k')
    plt.ylim(0,4)
    # Plot the transmission times and values for each neuron
    colourSeq = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for n in range(noNeurons-1):
        plt.figure(figsize=(14,7))
        plt.title('Neuron ' + str(n + 1))
        # plot transmissions from each synapse
        for s in range(noSynapses//(noNeurons-1)):
            y = np.array(transmissionRecord[n][s][1])
            x = transmissionRecord[n][s][0]/ms
            # plt.plot(transmissionRecord[n][s][0]/ms,y + s,'o', color = colourSeq[s % len(colourSeq)])
            for trans in range(len(y)):
                xline = [x[trans], x[trans]]
                yline = [s, y[trans] + s]
                plt.plot(xline,yline,'-',color = colourSeq[s % len(colourSeq)])
    show()

if __name__ == '__main__':
    main()