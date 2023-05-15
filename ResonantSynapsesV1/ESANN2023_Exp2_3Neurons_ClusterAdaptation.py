from brian2 import *
from generalBetaDistribution import gBeta, approxArea, meanGBeta, varianceGBeta, gBetaParamEst_Eleni2
import TSSTGenerator as tsstGen
import welfordsOnlineAlgorithm as woa
from csv import writer
import time
import statistics

# MODEL PARAMETERS
thresh = 1.0                    # Firing threshold of the neurons
isiRelevanceThresh = 10.0
patternDuration = 200*ms        # The length of time that the model is run for each input pattern - orginally 100
noNeurons = 3                   # noNeurons = number of neurons
noSynPerCluster = 2             # Number of synapses per functional cluster
noSynClusters = 2               # Number of synaptic cluster
delayAdjustWeight = 0.3         # The amount by which dynaptic delays are moved closer to the mean of the delays of all synapses that transmitted this run - original value 0.2
N = 700                         # N = number of noisy versions of the TSST
M = 3                           # M = number of spikes in each TSST
transitionW = 0.8               # Transmission weight
minSynapticDelay = 10.0
maxSynapticDelay = 50.0         # original value = 30,0
trainSig = [1]                  # Timing of the training signal this pattern.
lowISI_s = 0.0                  # lowISI_s = lowest ISI permitted in gBeta distribution- in seconds
highISI_s = 0.05                # highISI_s = highest ISI permitted in gBeta distribution - in seconds original = 0.05 DON'T CHANGE THIS
lowISI_ms = lowISI_s*1000.0     # lowISI_ms = lowest ISI permitted in gBeta distribution - in milliseconds
highISI_ms = highISI_s*1000.0   # highISI_ms = highest ISI permitted in gBeta distribution - in milliseconds
minISI_ms = 5.0                 # minISI_ms = minimum ISI permitted for the spike generator
tsstLength_ms = 50.0            # lenth of the original TSST
maxISI1_ms = 15.0
# TRANSMISSION DELAY CONSTANTS
tau = 20                        # voltage delay constant
v_belowThresh = 0.1             # smal vaoltage to keep activation below threshold
v_aboveThresh = 0.5             # small voltage to add to thresh to push post_s voltage above threshold
v_d = thresh + v_aboveThresh              # target post_s voltage to ensure above threshold (i.e. decoding transmission)
v_h = thresh - v_belowThresh            # target post_s voltage to ensure above resting but below firing threshold (i.e. a holding voltage)
v_0 = 0.01                      # resting potential
t_d = -tau*math.log((v_d - transitionW)/v_h)       # The time delay between the penultimate and the last transmission resulting in above threnshold voltage on the post_s neuron (i.e. decoding the TSST)
t_h = -tau*math.log((v_h - transitionW)/v_h)       # The time delay between transmissions that are not the first or the last transmission in this sequence
t_f = -tau*math.log((v_h - transitionW)/v_0)       # The time delay for the first transmission.
T = 10                          # Target decode time after last spike
# GBETA CONTANTS
bBetaParamMaxSum = 150
gBetaParamSum = 40.0            # gBetaParamSum = q + r, used to generate random q and r values
gBetaParamMin = 1.1            # gBetaParamMin = the lowest value of q and r to maintain bell-shaped curve
q = [[] for i in range(noSynClusters)]
r = [[] for i in range(noSynClusters)]
for synCluster in range(noSynClusters):
    q[synCluster] = np.random.uniform(gBetaParamMin,gBetaParamSum - gBetaParamMin, noSynPerCluster) # Set random q values for each synapse
    r[synCluster] = gBetaParamSum - q[synCluster]           # Set random r values for each synapse
# synapticDelays = [[np.zeros(noSynPerCluster)] for s in range(noSynClusters)]

# store transmission times and values for all synapses in each cluster
transmissionRecord = [[[[],[]] for i in range(noSynPerCluster)] for n in range(noSynClusters)]

class SynapticTransmissions():
    synapticTrans = [False for i in range(noSynPerCluster)]
    isiCounts = [-1 for i in range(noSynPerCluster)]
    isiTrans = [-1 for i in range(noSynPerCluster)]
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
        self.synapticTrans = [False for i in range(noSynPerCluster)]
        self.isiCounts = [-1 for i in range(noSynPerCluster)]
        self.isiTrans = [-1 for i in range(noSynPerCluster)]

synTrans= [SynapticTransmissions() for i in range(noSynClusters)]
meanBeta = [np.zeros(noSynPerCluster) for i in range(noSynClusters)] # meanBeta = The mean of the gBeta distribution for each synapse
for synCluster in range(noSynClusters):
    for synapseIndex in range(noSynPerCluster):
        meanBeta[synCluster][synapseIndex] = meanGBeta(lowISI_s, highISI_s,q[synCluster][synapseIndex], r[synCluster][synapseIndex])
dtArea = 0.0001                  # dt use to calculate area under curve of gBeta
existingAggregates = [[] for i in range(noSynClusters)]
existingAggregates[0] = [(0, 0, 0) for i in range(noSynPerCluster)]  # Stores count, mean, M2 (M2 aggregates the squared distance from the mean)
existingAggregates[1] = [(0, 0, 0) for i in range(noSynPerCluster)]                                # for each synpase for use with the Welfords online mean and variance algorithm
existingAggregatesManual = [[] for i in range(noSynClusters)]
existingAggregatesManual[0] = [[] for i in range(noSynPerCluster)]
existingAggregatesManual[1] = [[] for i in range(noSynPerCluster)]
# SPIKE GENERATOR FOR NEURON 0
# Create original TSST with M spike together with N noisy versions of it
tsstGen = tsstGen.TSSTGen(minISI_ms, highISI_ms)
# Set the standard deviation for the randomisation of each spike
stDevs = np.array([1.5 for i in range(M)])
tsstsA = np.zeros(N*M).reshape(N,M)
# origPatternA = np.random.randint(low=(lowISI_ms + minISI_ms),high=(highISI_ms - minISI_ms),size=1)
# origISI = np.random.uniform(low=minISI_ms,high= maxISI1_ms)
isis = [[10, 40],[20,30]]
origPatternA = [minISI_ms, minISI_ms + isis[0][0], minISI_ms + isis[0][0] + isis[0][1]]
# for spike in range(1,M):
#     [isi] = np.random.randint(low=(lowISI_ms + minISI_ms),high=(highISI_ms - minISI_ms),size=1)
#     origPatternA = np.append(origPatternA, [origPatternA[spike - 1] + isi])
tsstsA[0] = origPatternA
tsstsB = np.zeros(N*M).reshape(N,M)
# origPatternB = np.random.randint(low=(lowISI_ms + minISI_ms),high=(highISI_ms - minISI_ms),size=1)
origPatternB = [minISI_ms, minISI_ms + isis[1][0], minISI_ms + isis[1][0] + isis[1][1]]
# for spike in range(1,M):
#     [isi] = np.random.randint(low=(lowISI_ms + minISI_ms),high=(highISI_ms - minISI_ms),size=1)
#     origPatternB = np.append(origPatternB, [origPatternB[spike - 1] + isi])
tsstsB[0] = origPatternB
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
# mseTrace[3][i] = target output for neuron 2 at run iThe
# mseTrace[4][i] = mse across all neurons after run i
# mseTrace[5][i] = running average mse across the last nMSE runs after run i
mseTrace = [[] for i in range((noSynClusters)*2 + 2)]
truePositives = []
falsePositives = []
falseNegatives = []
nMSE = 10 # running average MSE over last n runs

# BUILD THE MODEL
# Define and create neurons
eqs = '''
    dv/dt = -v/tau : 1 (unless refractory)
    tau : second
    ref : second
'''
G = NeuronGroup(noNeurons, eqs, threshold='v>thresh', reset='v = 0', method='exact', name = 'G', refractory = 'ref')
G.tau = [tau, tau, tau]*ms
G.ref = [0,15,15]*ms
group_mon = SpikeMonitor(G, name='group_mon')

# GET gBeta PARAMETER q AND r AT RUN TIME
@implementation('numpy', discard_units=True)
@check_units(result=1)
def getBetaParam(clusterId, param, synapseNumber):
    # param = 1 signifies 'q'
    # print('Synapse: ' + str(synapseNumber) + ' param: ' + str(param))
    if param == 1:
        value = q[clusterId][synapseNumber];
        # print('q = ' + str(value))
    else:
        value = r[clusterId][synapseNumber]
        # print('r = ' + str(value))
    return value

# CALCULATE ISI AT RUN TIME
@implementation('numpy', discard_units=True)
@check_units(result=1)
def calcISI(clusterIdReal,synapseIndex, t, lastUpdate):
    global synTrans, q, r, meanBeta
    clusterId = int(clusterIdReal)
    transition = 0
    relevant = False
    if lastUpdate > 0:
        global synTrans
        synTrans[clusterId].updateISICount(synapseIndex)
        isi = t - lastUpdate
        gBetaVal = gBeta(isi, lowISI_s, highISI_s, getBetaParam(clusterId,1,synapseIndex), getBetaParam(clusterId,2,synapseIndex),False)
        if gBetaVal > isiRelevanceThresh:
            transition = transitionW
            # Record that this synapse transmitted this run
            synTrans[clusterId].update(synapseIndex, True)
            # Capture the time and value of the transmission from this synpase
            transmissionRecord[clusterId][synapseIndex][0].append(t)
            transmissionRecord[clusterId][synapseIndex][1].append(transitionW)  #scaledGBeta
    return transition

# Create resonant synapses between G[0] and G[1]
synModel = '''
    w :1
    lastupdate : second
    clusterId :1 
    timeToTrans : 1
''' #TODO: Remove clusterId
v_post_Eq = '''
    v_post += calcISI(clusterId,synapse_number, t, lastupdate) 
    lastupdate = t
'''
S = Synapses(G, G, model = synModel, on_pre=v_post_Eq, multisynaptic_index='synapse_number')
S.connect(i=0, j=1, n=noSynPerCluster)
S.connect(i=0, j=2, n=noSynPerCluster)
# Assign cluster id's to synapses
S.clusterId[0] = 0
S.clusterId[1] = 0
S.clusterId[2] = 1
S.clusterId[3] = 1
# Initialise the axonal delays for each synapse
for syn in range(noSynPerCluster*noSynClusters):
    S.delay[syn] = np.random.uniform(10,20)*ms

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
    gB2 = [np.zeros((noSynPerCluster,100)) for i in range(noSynClusters)]
    # Calculate the adjustment to the x-axis range to ensure that the graphs are visible
    averageMeanPerCluster = [0 for i in range(noSynClusters)]
    xStart = [0 for i in range(noSynClusters)]
    xEnd= [maxX_s for i in range(noSynClusters)]
    for clusterNo in range(noSynClusters):
        for i in range(len(q[clusterNo])):
            gB2[clusterNo][i,:] = gBeta(xGBeta_s,lowISI_s,highISI_s,q[clusterNo][i],r[clusterNo][i],False)
            m = max(gB2[clusterNo][i, :])
            averageMeanPerCluster[clusterNo] += S.delay[clusterNo*noSynPerCluster + i] + meanBeta[clusterNo][i]*ms*1000
            if m > maxHeight:
                maxHeight = m
        averageMeanPerCluster[clusterNo] = averageMeanPerCluster[clusterNo]/noSynPerCluster
        xStart[clusterNo] = averageMeanPerCluster[clusterNo] - maxX_s/2
        xEnd[clusterNo] = averageMeanPerCluster[clusterNo] + maxX_s/2
    fig, (ax1,ax2) = plt.subplots(1,2,figsize = (14,7))
    # plt.figure(figsize=(7,7))
    fig.suptitle('General Beta Distribution - i = ' + str(iterationNo), fontsize='15')
    # Adjust the x axis of the gBeta plots to that the curves are visible
    ax1.set_xlim(xStart[0], xEnd[0])
    ax2.set_xlim(xStart[1], xEnd[1])
    ax1.set_ylim(0,maxHeight + maxHeight*0.05)
    ax2.set_ylim(0, maxHeight + maxHeight * 0.05)
    # plt.xlim(0, maxX_s)
    # plt.ylim(0,maxHeight + maxHeight*0.05)
    lineHeight = maxHeight/(4 * len(q[clusterNo]))
    for i in range(0,noSynPerCluster):
        x = xGBeta_s*ms*1000.0 + S.delay[i]
        colour = colourSeq[i % len(colourSeq)]
        ax1.plot(x, gB2[0][i, :], label = "Synapse " + str(0) + "." + str(i), c = colour) # TODO: find out why label is not showing
        ax1.set_title('Cluster 0')
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
    for i in range(0,noSynPerCluster):
        x = xGBeta_s*ms*1000.0 + S.delay[i + noSynPerCluster]
        colour = colourSeq[i % len(colourSeq)]
        ax2.plot(x, gB2[1][i, :], label = "Synapse " +  str(1) + "." + str(i), c = colour) # TODO: find out why label is not showing
        ax2.set_title('Cluster 1')
        ax2.set_xlabel(f'Delay + GBeta mean ({lowISI_s},{highISI_s})', fontsize='15')
        ax2.set_ylabel('Probability', fontsize='15')
        # Calculate the length and position of the ISI box
        meanXDelayed_ms = meanBeta[1][i]*ms*1000.0 + S.delay[i + noSynPerCluster]
        xMin = S.delay[(i + noSynPerCluster)/(1000.0*ms)]
        xMax = meanXDelayed_ms
        yMin = i * (lineHeight + 1.0)
        yMax = yMin + lineHeight
        xRectangle = [xMin, xMax, xMax, xMin, xMin]
        yRectangle = [yMax, yMax, yMin, yMin, yMax]
        ax2.plot(xRectangle,yRectangle,ls='--', c = colour)

# CALCULATE MSE
def mse(atIndex):
    t = np.array([mseTrace[2][atIndex],mseTrace[3][atIndex]])
    y = np.array([mseTrace[0][atIndex],mseTrace[1][atIndex]])
    return np.square(t - y).mean()

def updateTpFpFn(atIndex):
    tpCount = 0
    fpCount = 0
    fnCount = 0
    for neuronId in range(noNeurons-1):
        if mseTrace[neuronId][atIndex] == 1:
            if mseTrace[neuronId + 2][atIndex] == 1:
                # True positive
                tpCount += 1
            else:
                # False positive
                fpCount += 1
        elif mseTrace[neuronId + 2][atIndex] == 1:
            fnCount += 1
    truePositives.append(tpCount)
    falsePositives.append(fpCount)
    falseNegatives.append(fnCount)

# CALCCULATE AVERAGE MSE OVER THE LAST nMSE RUNS
def averageMSEOvernMSERuns(atIndex):
    avMSE = 0
    if len(mseTrace[0]) >= nMSE:
        runningTotal = 0
        for iMSE in range (atIndex - (nMSE - 1), atIndex + 1):
            runningTotal += mseTrace[4][iMSE]
        avMSE = runningTotal/nMSE
    return avMSE

########################################################################################################################
# UNDER CONSTRUCTION ###################################################################################################
def updateStats(clusterId,tsst, spikeTrain):
    spikes = group_mon.spike_trains()
    # NOTE THAT IN THIS VERSION clusterId + 1 = index of the post_s neuron the cluster connects to
    # Check training signal is on this neuron for this TSST
    # Check that neuronId (clusterId + 1) has fired at least once this pattern window
    if len(spikes[clusterId + 1]) > 0:
        lastSpikeTime = spikes[clusterId + 1][len(spikes[clusterId + 1]) - 1]
        # Check that neuronId fired within patternDuration ms
        currentTime = defaultclock.t
        timeSinceLastSpike = currentTime - lastSpikeTime
        adaptThisCluster = timeSinceLastSpike < patternDuration
########################################################################################################################
# DEBUG              ###################################################################################################
        if (not adaptThisCluster) and (spikeTrain == 'A') and (clusterId == 0):
            print('adaptThisCluster ERROR 1')
        if (not adaptThisCluster) and (spikeTrain == 'B') and (clusterId == 1):
            print('adaptThisCluster ERROR 2')
########################################################################################################################
        if adaptThisCluster:
            # update the delays and stats of each synpase in turn
            isis = [tsst[1]/1000 - tsst[0]/1000, tsst[2]/1000 - tsst[1]/1000]
            for synapseIndex in range(noSynPerCluster):
                # update the stats of this synapse
                # existingAggregates[clusterId][synapseIndex] = woa.update(existingAggregates[clusterId][synapseIndex], isis[synapseIndex])
                # count, mu, var = existingAggregates[clusterId][synapseIndex]
                existingAggregatesManual[clusterId][synapseIndex].append(isis[synapseIndex])
                if len(existingAggregatesManual[clusterId][synapseIndex]) > 20:
                    mu = statistics.mean(existingAggregatesManual[clusterId][synapseIndex])
                    var = statistics.variance(existingAggregatesManual[clusterId][synapseIndex])
                    qTemp, rTemp = gBetaParamEst_Eleni2(mu, var, lowISI_s, highISI_s)
                    # Check that the new values of q and r will generate a bell shaped curve
                    if (qTemp >= gBetaParamMin) and (rTemp >= gBetaParamMin) and (qTemp + rTemp < bBetaParamMaxSum):
                        q[clusterId][synapseIndex] = qTemp
                        r[clusterId][synapseIndex] = rTemp
                        meanBeta[clusterId][synapseIndex] = mu
                    else:
                        print(f"ERROR - updateStats(): either qTemp = {qTemp}, rTemp = {rTemp} or both are outside value bounds")

def updateDelays(thisTSST, patternClass):
            # update the delay of this synpase
            # print(f'patternClass = {patternClass}')
            if patternClass == 'A':
                tsst = tsstsA[thisTSST]
                # print(f'tsst = {tsst}')
                # print(f'BEFORE UPDATE: S.delay[0]  + meanBeta[0][0] = {S.delay[0]} + {meanBeta[0][0]*ms*1000} = {S.delay[0] + meanBeta[0][0]*ms*1000}')
                # print(f'BEFORE UPDATE: S.delay[1]  + meanBeta[0][1] = {S.delay[1]} + {meanBeta[0][1]*ms*1000} = {S.delay[1] + meanBeta[0][1]*ms*1000}')
                # print(f'TARGETS: S.delay[0] + meanBeta[0][0] = (tsst[2] + (T - t_d))*ms = ({tsst[2]} + ({T} - {t_d}))*ms = {(tsst[2] + (T - t_d))*ms}')
                # print(f'TARGETS: S.delay[1] + meanBeta[0][1]= (tsst[2] + T)*ms = ({tsst[2]} + {T})*ms = {(tsst[2] + T)*ms}')
                S.delay[0] += delayAdjustWeight*((tsst[2] + (T - t_d))*ms - (S.delay[0] + meanBeta[0][0]*ms*1000))
                S.delay[1] += delayAdjustWeight*((tsst[2] + T)*ms - (S.delay[1] + meanBeta[0][1]*ms*1000))
                # print(f'AFTER UPDATE: S.delay[0]  + meanBeta[0][0] = {S.delay[0]} + {meanBeta[0][0]*ms*1000} = {S.delay[0] + meanBeta[0][0]*ms*1000}')
                # print(f'AFTER UPDATE: S.delay[1]  + meanBeta[0][1] = {S.delay[1]} + {meanBeta[0][1]*ms*1000} = {S.delay[1] + meanBeta[0][1]*ms*1000}')
            else:
                tsst = tsstsB[thisTSST]
                # print(f'tsst = {tsst}')
                # print(f'BEFORE UPDATE: S.delay[2]  + meanBeta[1][0] = {S.delay[2]} + {meanBeta[1][0]*ms*1000} = {S.delay[2] + meanBeta[1][0]*ms*1000}')
                # print(f'BEFORE UPDATE: S.delay[3]  + meanBeta[1][1] = {S.delay[3]} + {meanBeta[1][1]*ms*1000} = {S.delay[3] + meanBeta[1][1]*ms*1000}')
                # print(f'TARGETS: S.delay[2] = (tsst[2] + (T - t_d))*ms = ({tsst[2]} + ({T} - {t_d}))*ms = {(tsst[2] + (T - t_d))*ms}')
                # print(f'TARGETS: S.delay[3] = (tsst[2] + T)*ms = ({tsst[2]} + {T})*ms = {(tsst[2] + T)*ms}')
                S.delay[2] += delayAdjustWeight*((tsst[2] + (T - t_d))*ms - (S.delay[2] + meanBeta[1][0]*ms*1000))
                S.delay[3] += delayAdjustWeight*((tsst[2] + T)*ms - (S.delay[3] + meanBeta[1][1]*ms*1000))
                # print(f'AFTER UPDATE: S.delay[2]  + meanBeta[1][0] = {S.delay[2]} + {meanBeta[1][0]*ms*1000} = {S.delay[2] + meanBeta[1][0]*ms*1000}')
                # print(f'AFTER UPDATE: S.delay[3]  + meanBeta[1][1] = {S.delay[3]} + {meanBeta[1][1]*ms*1000} = {S.delay[3] + meanBeta[1][1]*ms*1000}')
            # print('---------------------------------------------------------------------------------------------------')

def main():
    i = 0
    plotGBetaFreq = 50
    trainingOn = False
    testSetSize = 100 # Training will be turned off for the last testSetSize patterns
    for thisTSST in range(len(tsstsA)):
        if (i <= testSetSize) or (i >= len(tsstsA) - testSetSize - 1):
            trainingOn = False
        else:
            trainingOn = True
        timedTrainSig = trainSig*ms + defaultclock.t
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
        inp.set_spikes(inpIndices,timedTSST)
        run(patternDuration)
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
            if trainingOn:
                updateStats(neuron,tsst, spikeTrain)
        if trainingOn:
            updateDelays(thisTSST, spikeTrain)
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
            pause(0.001)
            show(block=False)
            print(f"{i} iterations completed")
        # Caclulate MSE
        mseTrace[(noNeurons-1)*2].append(mse(len(mseTrace[0])-1))
        # Caclulate running average of MSEs over the last nMSE runs
        mseTrace[(noNeurons-1)*2 + 1].append(averageMSEOvernMSERuns(len(mseTrace[0])-1))
        # Reset the transmission record
        # Update truePositives, falsePositives and falseNegatives
        updateTpFpFn(i)
        for neuron in range(noNeurons-1):
            synTrans[neuron].reset()
        i += 1
    plotGBeta(i)
    # Calculate Precision and recall
    totalTruePos_PreTraining = sum(truePositives[0:testSetSize])
    totalFalsePos_PreTraining = sum(falsePositives[0:testSetSize])
    totalFalseNeg_PreTraining = sum(falseNegatives[0:testSetSize])
    totalTruePos_PostTraining = sum(truePositives[N-testSetSize:N-1])
    totalFalsePos_PostTraining = sum(falsePositives[N-testSetSize:N-1])
    totalFalseNeg_PostTraining = sum(falseNegatives[N-testSetSize:N-1])
    precision_PreTraining = totalTruePos_PreTraining/(totalTruePos_PreTraining + totalFalsePos_PreTraining)
    recall_PreTraining = totalTruePos_PreTraining/(totalTruePos_PreTraining + totalFalseNeg_PreTraining)
    precision_PostTraining = totalTruePos_PostTraining/(totalTruePos_PostTraining + totalFalsePos_PostTraining)
    recall_PostTraining = totalTruePos_PostTraining/(totalTruePos_PostTraining + totalFalseNeg_PostTraining)
    # Save the data to .csv file
    # /Volumes/GoogleDrive/My Drive/DOCUMENTS/RESEARCHER/Code/Python/ResonantSynapses/ResonantSynapsesV1/Results/learnMeanVarAndDelay_3Neurons_NSynapses/2022_11_29/2022_11_29__08_41_MSEData.csv
    with open('Results/ESANN2023/ESANN2023.csv','a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow([time.strftime("%Y-%m-%d %H:%M")])
        writer_object.writerow(["OutputSpikes and Training signal"])
        for i in range(len(mseTrace)):
            writer_object.writerow(mseTrace[i][:])
        writer_object.writerow(["ISI sequences"])
        for clusterId in range(noSynClusters):
            for syn in range(noSynPerCluster):
                writer_object.writerow(existingAggregatesManual[clusterId][syn])
        writer_object.writerow(' ')
        writer_object.writerow(["True Positives/False Positives/False Negatives"])
        writer_object.writerow(truePositives)
        writer_object.writerow(falsePositives)
        writer_object.writerow(falseNegatives)
        writer_object.writerow(['Precision Pre-training =', precision_PreTraining])
        writer_object.writerow(['Recall Pre-training =', recall_PreTraining])
        writer_object.writerow(['Precision Post-training =', precision_PostTraining])
        writer_object.writerow(['Recall Post-training =', recall_PostTraining])
        f_object.close()

    # PLOT THE INSTANTANEOUS AND RUNNING AVERAGE MSE
    plt.figure(figsize=(14,7))
    plt.plot(mseTrace[len(mseTrace)-2], ls='-',c='m', label='Instantaneous MSE')
    plt.plot(mseTrace[len(mseTrace)-1], ls='-',c='r', label='Running average MSE')
    plt.legend(loc="upper left")
    plt.title('Instantaneous and running average MSE')
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
    for n in range(noSynClusters):
        plt.figure(figsize=(14,7))
        plt.title('Functional Synapstic Cluster ' + str(n + 1))
        # plot transmissions from each synapse
        for s in range(noSynPerCluster):
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