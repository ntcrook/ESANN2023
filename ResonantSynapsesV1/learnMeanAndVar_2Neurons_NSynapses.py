import numpy as np
from brian2 import *
from generalBetaDistribution import gBeta, approxArea, meanGBeta, varianceGBeta, gBetaParamEst_Eleni2
import TSSTGenerator as tsstGen
import welfordsOnlineAlgorithm as woa
import splitRickerWavelet as srw

# MODEL PARAMETERS
noNeurons = 2                   # noNeurons = number of neurons
noSynapses = 5                  # noSynapses = number of synapses
N = 100                           # N = number of noisy versions of the TSST
M = 3                           # M = number of spikes in each TSST
minSynapticDelay = 10.0
maxSynapticDelay = 30.0
lowISI_s = 0.0                    # lowISI_s = lowest ISI permitted in gBeta distribution- in seconds
highISI_s = 0.05                # highISI_s = highest ISI permitted in gBeta distribution - in seconds
lowISI_ms = lowISI_s*1000.0       # lowISI_ms = lowest ISI permitted in gBeta distribution - in milliseconds
highISI_ms = highISI_s*1000.0     # highISI_ms = highest ISI permitted in gBeta distribution - in milliseconds
minISI_ms = 5.0                  # minISI_ms = minimum ISI permitted for the spike generator
gBetaScale = 1.0/200.0              # gBetaScale = Scaling factor for gBeta
gBetaParamSum = 40.0              # gBetaParamSum = q + r, used to generate random q and r values
gBetaParamMin = 10.0             # gBetaParamMin = the lowest value of q and r to maintain bell-shaped curve
q = np.random.uniform(gBetaParamMin,gBetaParamSum - gBetaParamMin,noSynapses) # Set random q values for each synapse
r = gBetaParamSum - q           # Set random r values for each synapse
synapticDelays = np.zeros(noSynapses)
# global synTransmittedThisRun
# synTransmittedThisRun = [False for i in range(noSynapses)]      # A list of synapses that successfully transmitted a spike in the current run
# synTransmittedThisRun = np.zeros(noSynapses)
# synTrans = {i:False for i in range(noSynapses)}
class SynapticTransmissions():

    synapticTrans = [False for i in range(noSynapses)]

    def update(self,synapseIndex,value):
        self.synapticTrans[synapseIndex] = value

    def getSynapsticTrans(self):
        return self.synapticTrans

    def ith(self,i):
        return self.synapticTrans[i]

    def reset(self):
        self.synapticTrans = [False for i in range(noSynapses)]

synTrans = SynapticTransmissions()

delayAdjustWeight = 0.2         # The amount by which dynaptic delays are moved closer to the mean of the delays of all synapses that transmitted this run
noResTableCols = 11                # The number of columns in results table

meanBeta = np.zeros(noSynapses) # meanBeta = The mean of the gBeta distribution for each synapse
for synapseIndex in range(noSynapses):
    meanBeta[synapseIndex] = meanGBeta(lowISI_s, highISI_s,q[synapseIndex], r[synapseIndex])
dtArea = 0.0001                  # dt use to calculate area under curve of gBeta
# lowGBetaArea = np.zeros(noSynapses) # lowGBetaArea = the area of the gBeta distribution below the mean for each synapse
# highGBetaArea = np.zeros(noSynapses) # highGBetaArea = the area of the gBeta distribution above the mean for each synpase
# for synapseIndex in range(noSynapses):
#     lowGBetaArea[synapseIndex] = approxArea(meanBeta[synapseIndex], lowISI_s, dtArea, lowISI_s, highISI_s,
#                                             q[synapseIndex],r[synapseIndex])
#     highGBetaArea[synapseIndex] = approxArea(meanBeta[synapseIndex], highISI_s, dtArea, lowISI_s, highISI_s,
#                                             q[synapseIndex], r[synapseIndex])
existingAggregates = [(0, 0, 0) for i in range(noSynapses)]  # Stores count, mean, M2 (M2 aggregates the squared distance from the mean)
                                # for each synpase for use with the Welfords online mean and variance algorithm

# SPIKE GENERATOR FOR NEURON 0
# Create original TSST with M spike together with N noisy versions of it
tsstGen = tsstGen.TSSTGen(lowISI_ms, highISI_ms, minISI_ms)
# Set the standard deviation for the randomisation of each spike
stDevs = np.ones(M)
tssts = np.zeros(N*M).reshape(N,M)
tssts[0] = [12, 20, 45]
# Create random versions of original tsst
neuronIndices = np.zeros(M)
for i in range(1,len(tssts[:,0])):
    tssts[i] = tsstGen.genNoisyVersion(tssts[0],stDevs)

# PRINT RESULTS AS A TABLE
# Collect data as dictionary
results = {}
resultIndex = 0
tableFormat = "{:<5} {:.4f} {:^13} {: e} {: e} {: e} {: e} {: e} {: e} {:<13}"
def addResult(thisResult):
    global resultIndex
    results[resultIndex] = thisResult
    print(tableFormat.format(resultIndex, thisResult[0], thisResult[1], thisResult[2], thisResult[3], thisResult[4], thisResult[5], thisResult[6],thisResult[7], thisResult[8]))
    # printTableHeader()
    resultIndex += 1

def printTableHeader():
    print("{:<5} {:<7} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13}".format('Item', 'Time', 'SynpaseId', 'Delay', 'ISI', 'gBeta:q','gBeta:r', 'gBeta:mean','gBeta:var', 'Relevant?'))

# GET gBeta PARAMETER q AND r AT RUN TIME
@implementation('numpy', discard_units=True)
@check_units(result=1)
def getBetaParam(param, synapseNumber):
    # param = 1 signifies 'q'
    # print('Synapse: ' + str(synapseNumber) + ' param: ' + str(param))
    if param == 1:
        value = q[synapseNumber];
        # print('q = ' + str(value))
    else:
        value = r[synapseNumber]
        # print('r = ' + str(value))
    return value

# def registerTransmission(synapseIndex):
#     global synTransmittedThisRun
#     synTransmittedThisRun[synapseIndex] = 1
#     print(f'Transmission record = {synTransmittedThisRun}')

# CALCULATE ISI AT RUN TIME
@implementation('numpy', discard_units=True)
@check_units(result=1)
def calcISI(synapseIndex, t,lastUpdate):
    global synTrans,existingAggregates, q, r, meanBeta, synapticDelays
    count, mu, var = existingAggregates[synapseIndex]
    pISI = 0
    isi = lowISI_s
    relevant = False
    if lastUpdate > 0:
        isi = t - lastUpdate
        # print(f">>>>>>>>>>> ISI = {isi}")
        # find the mean of the current gBete function
        # meanBeta = meanGBeta(lowISI_s, highISI_s,getBetaParam(1,synapseIndex), getBetaParam(2,synapseIndex))
        area = approxArea(meanBeta[synapseIndex], isi, dtArea, lowISI_s, highISI_s, getBetaParam(1, synapseIndex),
                   getBetaParam(2, synapseIndex))
        # adjust for error in the calculation of the area
        # area = min(area,0.5)
        pISI = 1 - 2*area
        relevant = np.random.uniform(0,1) <= pISI
        if relevant:
            # If the ISI is relevant, then adjust the mean of the gBeta
            existingAggregates[synapseIndex] = woa.update(existingAggregates[synapseIndex], isi)
            # TODO: use gBetaParamEst_Eleni2(mu,sigma,a,b) to update the parameter of this synapses gBeta distribution
            count, mu, var = existingAggregates[synapseIndex]
            # sigma = np.sqrt(var)
            if count > 10:
                # fcount, fmu, fsigma = woa.finalize(existingAggregates[synapseIndex])
                # Record that this synapse transmitted this run
                global synTrans
                synTrans.update(synapseIndex,True) # TODO: work out the scoping issue with this variable.
                # synTransmittedThisRun[synapseIndex] = 1
                # registerTransmission(synapseIndex)
                print(f'++++++++++ Synapse {synapseIndex} transmitted this run')
                qTemp, rTemp = gBetaParamEst_Eleni2(mu, var, lowISI_s, highISI_s)
                # Check that the new values of q and r will generate a bell shaped curve
                if (qTemp >= gBetaParamMin) and (rTemp >= gBetaParamMin):
                    q[synapseIndex] = qTemp
                    r[synapseIndex] = rTemp
                    meanBeta[synapseIndex] = mu
        else:
            # If not relevant, then do not transmit the spike.
            isi = lowISI_s
    result = ['' for j in range(noResTableCols)]
    result[0] = resultIndex
    gBetaMean = meanGBeta(lowISI_s, highISI_s,q[synapseIndex],r[synapseIndex])
    gBetaVar = varianceGBeta(lowISI_s, highISI_s,q[synapseIndex],r[synapseIndex])
    # call addResult with parameter: 'Time', 'SynpaseId', 'Delay', 'ISI', 'gBeta:q', 'gBeta:r', 'gBeta:mean', 'gBeta:var', 'Relevant?'
    addResult([t, synapseIndex, synapticDelays[synapseIndex], isi, q[synapseIndex], r[synapseIndex], gBetaMean, gBetaVar,str(relevant)])
                                                                        # TODO: Check that S.delay[i] references the correct synpase
    # print("calcISI(synapseIndex = " + str(synapseIndex) + " t = " + str(t) + "; lastUpdate = " + str(lastUpdate) + ") = " + str(isi) + " ISI (count, mean, var) = " + str(existingAggregates[synapseIndex]) + " p(tau_mu,isi) = " + str(pISI) + " Relevant ISI = " + str(relevant))
    return isi

# BUILD THE MODEL
# Define and create neurons
eqs = '''
    dv/dt = -v/tau : 1
    tau : second
'''
G = NeuronGroup(noNeurons, eqs, threshold='v>1', reset='v = 0', method='exact', name = 'G')
G.tau = [10, 10]*ms
# Create resonant synapses between G[0] and G[1]
synModel = '''
    w :1
    lastupdate : second
'''
v_post_Eq = '''
    isi = calcISI(synapse_number, t, lastupdate) 
    v_post += gBetaScale * gBeta(isi, lowISI_s, highISI_s, getBetaParam(1,synapse_number), getBetaParam(2,synapse_number),False)
    lastupdate = t
'''
S = Synapses(G, G, model = synModel, on_pre=v_post_Eq, multisynaptic_index='synapse_number')
S.connect(i=0, j=1, n=noSynapses)
S.w = 'j*0.01'
# randomly set synpatic delays
for i in range(noSynapses):
    S.delay[i] = np.random.uniform(minISI_ms,highISI_ms)*ms
    synapticDelays[i] = S.delay[i]

# Create SpikeGeneratorGroup and connect to neuron 0 in G
inp = SpikeGeneratorGroup(1,neuronIndices, tssts[0]*ms, name='inp', when='before_synapses')
syn = Synapses(inp, G, on_pre='v_post += 2', name='syn')
syn.connect(i=0,j=0)
inpIndices = np.zeros(M)

# STATE MONITORING
stateMonitor = StateMonitor(G, 'v', record=True)
inp_mon = SpikeMonitor(inp, name='inp_mon')
group_mon = SpikeMonitor(G, name='group_mon')

# PLOT gBeta DISTRIBUTIONS FOR EACH SYNAPSE BEFORE LEARNING
def plotGBeta(iterationNo):
    nearEdgeAjust = 0.001
    maxHeight = 0.0
    maxX_s = ((maxSynapticDelay/1000.0) + highISI_s)*ms*1000.0
    colourSeq = ['b','g','r','c','m','y','k']
    lineHeight = 5 #0.05
    # xGBeta = np.linspace(lowISI_s + nearEdgeAjust,highISI_s - nearEdgeAjust,100)
    xGBeta_s = np.linspace(lowISI_s, highISI_s, 100)
    # x = np.linspace(0, maxX, 100)
    gB2 = np.zeros((noSynapses,100))
    for i in range(len(q)):
        gB2[i,:] = gBeta(xGBeta_s,lowISI_s,highISI_s,q[i],r[i],False)
        m = max(gB2[i, :])
        if m > maxHeight:
            maxHeight = m
    plt.figure(figsize=(7,7))
    plt.xlim(0, maxX_s)
    plt.ylim(0,maxHeight + maxHeight*0.05)
    for i in range(len(q)):
        x = xGBeta_s*ms*1000.0 + S.delay[i]
        colour = colourSeq[i % len(colourSeq)]
        plt.plot(x, gB2[i, :], label = "Synapse " + str(i), c = colour) # TODO: find out why label is not showing
        meanXDelayed_ms = meanBeta[i]*ms*1000.0 + S.delay[i]
        xMin = S.delay[i/(1000.0*ms)]
        xMax = meanXDelayed_ms
        yMin = i * (lineHeight + 1.0)
        yMax = yMin + lineHeight
        xRectangle = [xMin, xMax, xMax, xMin, xMin]
        yRectangle = [yMax, yMax, yMin, yMin, yMax]
        plt.plot(xRectangle,yRectangle,ls='--', c = colour)
    plt.title('General Beta Distribution - i = ' + str(iterationNo), fontsize='15')
    plt.xlabel(f'Values of Random Variable X ({lowISI_s},{highISI_s})', fontsize='15')
    plt.ylabel('Probability', fontsize='15')
    show()

# UPDATE SYNAPTIC DELAYS
def updateDelays():
    meanDelay = 0
    count = 0
    # Calculate mean delay + ISI of each synapse that transmitted
    for i in range(noSynapses):
        if synTrans.ith(i):
            isi = meanBeta[i]*1000*ms
            delay = S.delay[i]
            meanDelay += isi + delay
            count += 1
    if count > 1:
        print("UPDATING DELAYS")
        meanDelay = meanDelay/count
        # Adjust each delay to be closer to the mean
        for i in range(noSynapses):
            if synTrans.ith(i):
                isi = meanBeta[i] * 1000 * ms
                delay = S.delay[i]
                adjustDelay = (meanDelay - isi - delay)*delayAdjustWeight
                S.delay[i] += adjustDelay
                synapticDelays[i] = S.delay[i]
                print(f'Synapse {i}: prev. delay = {delay}, updated delay = {S.delay[i]}')

# MAIN LOOP
# Use .set_spikes(...) method of SpikeGeneratorGroup as explained here:
# https://brian.discourse.group/t/problem-changing-spike-values-in-spikegeneratorgroup-during-simulation/290
i = 0
plotGBetaFreq = 5
plotGBeta(0)
printTableHeader()
for tsst in tssts:
    timedTSST = tsst*ms + defaultclock.t
    print(f'Initiate run {i} with noisy spike train: {tsst} adjusted to current time: {timedTSST}')
    inp.set_spikes(inpIndices,timedTSST)
    # print("Clock time: " + str(defaultclock.t))
    run(100*ms)
    # Update delays of synapses that transmitted a spike this run
    updateDelays()
    # print(f'Transmitted this run {synTrans}')
    # Reset the transmission record
    # synTransmittedThisRun = [False for i in range(noSynapses)]
    # synTransmittedThisRun = np.zeros(noSynapses)
    # synTrans = {i: False for i in range(noSynapses)}
    synTrans.reset()
    i += 1
    if i % plotGBetaFreq == 0:
        plotGBeta(i)

# PLOT gBeta DISTRIBUTIONS FOR EACH SYNAPSE AFTER LEARNING
# nearEdgeAjust = 0.001
# maxHeight = 0
# x = np.linspace(lowISI_s + nearEdgeAjust,highISI_s - nearEdgeAjust,100)
# gB2 = np.zeros((noSynapses,100))
# for i in range(len(q)):
#     gB2[i,:] = gBeta(x,lowISI_s,highISI_s,q[i],r[i],False)
#     m = max(gB2[i, :])
#     if m > maxHeight:
#         maxHeight = m
# plt.figure(figsize=(7,7))
# plt.xlim(lowISI_s, highISI_s)
# plt.ylim(0,maxHeight + maxHeight*0.05)
# for i in range(len(q)):
#     plt.plot(x, gB2[i, :], label = "Synapse " + str(i)) # TODO: find out why label is not showing
#     axvline(meanBeta[i], ls='--', c='g', lw=3)
# plt.title('General Beta Distribution - after learning', fontsize='15')
# plt.xlabel(f'Values of Random Variable X ({lowISI_s},{highISI_s})', fontsize='15')
# plt.ylabel('Probability', fontsize='15')

figure()
plot(stateMonitor.t / ms, stateMonitor.v[0], label='Neuron 0')
plot(stateMonitor.t / ms, stateMonitor.v[1], label='Neuron 1')
title('2 neurons with 2 synapses from neuron 0 to neuron 1,\n delays = [' + str(S.delay[0]) + ',' + str(S.delay[1]) + '] Noisy Version of TSST')
for t in group_mon.t:
    axvline(t/ms, ls = '--', c='g', lw = 3)
    print(t)
xlabel('Time (ms)')
ylabel('v')
legend();

# fig, ax = plt.subplots()
# ax.plot(inp_mon.t/ms, [2], 'o')
# ax.plot(group_mon.t/ms, [0,1], 'o')
# ax.set(yticks=[0, 1], yticklabels=['group', 'inp'], xlabel='time (ms)')
# print(scheduling_summary())
show()