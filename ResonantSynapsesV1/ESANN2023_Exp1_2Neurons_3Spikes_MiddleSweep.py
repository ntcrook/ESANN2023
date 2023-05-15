# p73 of Research Logbook 07/03/2023
import matplotlib.pyplot as plt
from brian2 import *
from generalBetaDistribution import gBeta, approxArea, meanGBeta, varianceGBeta, gBetaParamEst_Eleni2
from learnMeanVarAndDelay_3Neurons_NSynapses_DEPRECATED import plotGBeta

# MODEL PARAMETERS
thresh = 1.0                                            # Firing threshold of the neurons
noNeurons = 2                                           # noNeurons = number of neurons
noSynapses = 2                                          # noSynapses = number of synapses
M = 3                                                   # M = number of spikes in each TSST
# gBetaScale = (thresh * 1.8)/(100 * (M - 1))             # gBetaScale = Scaling factor for gBeta original = 1.0/200.0 changed to bring the total bBeta above threshold.
lowISI_s = 0.0                                          # lowISI_s = lowest ISI permitted in gBeta distribution- in seconds
highISI_s = 0.05                                        # highISI_s = highest ISI permitted in gBeta distribution - in seconds original = 0.05 DON'T CHANGE THIS
isiRelevanceThresh = 90.0                                # Threshold on gBeta to determin if isi is relevant
w = 0.75                                                # Transmission weight
neuronIndices = np.zeros(M)
maxSynapticDelay = 50.0 # original value = 30,0

# Create the two fixed patters
patternDuration = 40
t_firstSpike = 10
targetI1 = 15
targetI2 = patternDuration - targetI1
tsstsInit = [0,0,0]
tsst = [0,0,0]

# TRANSMISSION DELAY CONSTANTS
tau = 10                        # voltage delay constant
v_s = 0.2                       # small voltage to add to thresh to push post_s voltage above threshold
v_d = thresh + v_s              # target post_s voltage to ensure above threshold (i.e. decoding transmission)
v_h = thresh - v_s              # target post_s voltage to ensure above resting but below firing threshold (i.e. a holding voltage) - note that v_h must be > w
v_0 = 0.01                      # resting potential
t_d = -tau*math.log((v_d - w)/v_h)       # The time delay between the penultimate and the last transmission resulting in above threnshold voltage on the post_s neuron (i.e. decoding the TSST)
t_h = -tau*math.log((v_h - w)/v_h)       # The time delay between transmissions that are not the first or the last transmission in this sequence
t_f = -tau*math.log((v_h - w)/v_0)       # The time delay for the first transmission.
T = 10                          # Target decode time after last spike
muA = targetI1/1000                   # Convert ms to seconds
muB = targetI2/1000
var = 0.000015
qA, rA = gBetaParamEst_Eleni2(muA,var, lowISI_s, highISI_s)
qB, rB = gBetaParamEst_Eleni2(muB,var, lowISI_s, highISI_s)
q = [[qA, qB]]
r = [[rA, rB]]
meanBeta = [np.zeros(noSynapses//(noNeurons-1)) for i in range(noNeurons-1)] # meanBeta = The mean of the gBeta distribution for each synapse
varBeta = [np.zeros(noSynapses//(noNeurons-1)) for i in range(noNeurons-1)]
for synapseIndex in range(noSynapses//(noNeurons-1)):
    meanBeta[0][synapseIndex] = meanGBeta(lowISI_s, highISI_s,q[0][synapseIndex], r[0][synapseIndex])
    varBeta[0][synapseIndex] = varianceGBeta(lowISI_s, highISI_s,q[0][synapseIndex], r[0][synapseIndex])

# BUILD THE MODEL
# Define and create neurons
eqs = '''
    dv/dt = -v/tau : 1 (unless refractory)
    tau : second
    ref : second
'''
G = NeuronGroup(2, eqs, threshold='v> thresh', reset='v = 0', method='exact', name = 'G', refractory = 'ref')
G.tau = [tau,tau]*ms
G.ref = [0,15]*ms
group_mon = SpikeMonitor(G, name='group_mon')

# GET gBeta PARAMETER q AND r AT RUN TIME
@implementation('numpy', discard_units=True)
@check_units(result=1)
def getBetaParam(clusterId,param, synapseNumber):
    # param = 1 signifies 'q'
    # print('Synapse: ' + str(synapseNumber) + ' param: ' + str(param))
    if param == 1:
        value = q[clusterId][synapseNumber];
        print(f'Synapse: {synapseNumber} q = {value}')
    else:
        value = r[clusterId][synapseNumber]
        print(f'Synapse: {synapseNumber} r = {value}')
    return value

# CALCULATE ISI AT RUN TIME
@implementation('numpy', discard_units=True)
@check_units(result=1)
def calcISI(clusterId, synapseIndex, t,lastUpdate):
    print(f'clusterId = {int(clusterId)}')
    transmit = 0
    if lastUpdate > 0:
        isi = t - lastUpdate
        gBetaVal = gBeta(isi, lowISI_s, highISI_s, getBetaParam(0, 1, synapseIndex),
                                     getBetaParam(0, 2, synapseIndex), False)
########################################################################################################################
##############################################   DEBUG  ################################################################
        print(f'===================================> gBetaVal = {gBetaVal}')
########################################################################################################################
########################################################################################################################
        if gBetaVal > isiRelevanceThresh:
            transmit = w
    return transmit

# STATE MONITORING
stateMonitor = StateMonitor(G, 'v', record=True)

# Create SpikeGeneratorGroups for the training Signal and connect to neurons in G - assuming neuron index 0 is input neuron
inp = SpikeGeneratorGroup(1,neuronIndices, tsstsInit*ms, name='inp', when='before_synapses')
syn = Synapses(inp, G, on_pre='v_post += 1.5', name='syn')
syn.connect(i=0,j=0)
inpIndices = np.zeros(M)

synModel = '''
    w :1
    lastupdate : second
    clusterId : 1
'''
v_post_Eq = '''
    v_post += calcISI(clusterId, synapse_number, t, lastupdate) 
    lastupdate = t
'''
S = Synapses(G, G, model = synModel, on_pre=v_post_Eq, multisynaptic_index='synapse_number')
S.connect(i=0, j=1, n=M-1)
S.clusterId[0] = 0
S.clusterId[1] = 0
s1Delay = T
s0Delay = targetI2 + T - t_d
print(f's0Delay = {s0Delay}')
print(f's1Delay = {s1Delay}')
S.delay[0] = s0Delay * ms
S.delay[1] = s1Delay*ms

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
    fig, (ax1) = plt.subplots(1,1,figsize = (4,3))
    # plt.figure(figsize=(7,7))
    fig.suptitle('General Beta Distribution - i = ' + str(iterationNo), fontsize='9')
    ax1.set_xlim(0, maxX_s)
    ax1.set_ylim(0,maxHeight + maxHeight*0.05)
    # plt.xlim(0, maxX_s)
    # plt.ylim(0,maxHeight + maxHeight*0.05)
    lineHeight = maxHeight/(4 * len(q[neuron]))
    for i in range(0,noSynapses//(noNeurons-1)):
        x = xGBeta_s*ms*1000.0 + S.delay[i]
        colour = colourSeq[i % len(colourSeq)]
        ax1.plot(x, gB2[0][i, :], label = "Synapse " + str(i), c = colour) # TODO: find out why label is not showing
        ax1.set_title('Neuron 1')
        ax1.set_xlabel(f'Delay + GBeta mean ({lowISI_s},{highISI_s})', fontsize='9')
        ax1.set_ylabel('Probability', fontsize='15')
        meanXDelayed_ms = meanBeta[0][i]*ms*1000.0 + S.delay[i]
        xMin = S.delay[i/(1000.0*ms)]
        xMax = meanXDelayed_ms
        yMin = i * (lineHeight + 1.0)
        yMax = yMin + lineHeight
        xRectangle = [xMin, xMax, xMax, xMin, xMin]
        yRectangle = [yMax, yMax, yMin, yMin, yMax]
        ax1.plot(xRectangle,yRectangle,ls='--', c = colour)

def updateTSSTs(I1, I2):
    global tsst, q, r, meanBeta, varBeta
    I1 = I1
    I2 = I2
    tsst = [10, 10 + I1, 10 + I1 + I2]
    # muA = I1 / 1000  # Convert ms to seconds
    # muB = I2 / 1000
    # var = 0.000015
    # qA, rA = gBetaParamEst_Eleni2(muA, var, lowISI_s, highISI_s)
    # qB, rB = gBetaParamEst_Eleni2(muB, var, lowISI_s, highISI_s)
    # q = [[qA, qB]]
    # r = [[rA, rB]]
    # meanBeta = [np.zeros(noSynapses // (noNeurons - 1)) for i in
    #             range(noNeurons - 1)]  # meanBeta = The mean of the gBeta distribution for each synapse
    # varBeta = [np.zeros(noSynapses // (noNeurons - 1)) for i in range(noNeurons - 1)]
    # for synapseIndex in range(noSynapses // (noNeurons - 1)):
    #     meanBeta[0][synapseIndex] = meanGBeta(lowISI_s, highISI_s, q[0][synapseIndex], r[0][synapseIndex])
    #     varBeta[0][synapseIndex] = varianceGBeta(lowISI_s, highISI_s, q[0][synapseIndex], r[0][synapseIndex])
    #     print(f'mu[0][{synapseIndex}] = {meanBeta[0][synapseIndex]}')
    #     print(f'var[0][{synapseIndex}] = {varBeta[0][synapseIndex]}')
    # print(f'tsstsA = {tsstsA}')
    # print(f'tsstsB = {tsstsB}')

def main():
    global S
    for I1 in range(10,30):
        I2 = patternDuration - I1
        updateTSSTs(I1,I2)
        print(f'tsst = {tsst}')
        # Run the model
        inp.set_spikes([0,0,0], tsst * ms + defaultclock.t)
        run(200*ms)
    # Plot the results
    plotGBeta(0)

    fig, (ax1,ax2) = plt.subplots(2,figsize = (10,5))
    ax1.plot(stateMonitor.t / ms, stateMonitor.v[0], label='Neuron 0')
    ax2.plot(stateMonitor.t / ms, stateMonitor.v[1], label='Neuron 1')
    xThresh = [0, stateMonitor.t[len(stateMonitor.t) - 1]]
    yThresh = [1, 1]
    ax1.plot([0, 4000],[1,1], ls = '--')
    ax2.plot([0, 4000],[1,1], ls='--')
    # ax1.set_title('A: Neuron 1 activation (spike sequence generation')
    # ax2.set_title('B: Neuron 2: activation (decoding)')
    ax1.set_ylabel('Voltage')
    ax2.set_ylabel('Voltage')
    ax2.set_xlabel('Time (ms)')
    ax1.plot([1021],[1.65],marker = '*')
    ax1.text(0,1.55,'A', fontsize = '12')
    ax2.text(0,1.1, 'B',fontsize = '12')

    show()

if __name__ == '__main__':
    main()