from brian2 import *
from generalBetaDistribution import gBeta

# Using TimedArray to create time structured inputs to one of two neurons
# https://brian2.readthedocs.io/en/stable/user/input.html#timed-arrays

# Have 7 time periods of 10ms each, with neuron 0 stimulated in periods 0, 4 and 7, giving 3 spike with ISIs of 40ms and 20ms
timeStrucSpTr = np.zeros([8,2])
timeStrucSpTr[0,0] = 1.6
timeStrucSpTr[4,0] = 1.6
timeStrucSpTr[6,0] = 1.6
I = TimedArray(timeStrucSpTr,dt=10*ms)
gBetaScale = 1/200 # Scaling factor for gBeta so that
lowISI = 0
highISI = 0.05

@implementation('numpy', discard_units=True)
@check_units(result=1)
def calcISI(t,lastUpdate):
    isi = lowISI
    if lastUpdate > 0:
        isi = t - lastUpdate
    print("calcISI(t = " + str(t) + "; lastUpdate = " + str(lastUpdate) + ") = " + str(isi))
    return isi

eqs = '''
    dv/dt = (I(t,i)-v)/tau : 1
    tau : second
'''

G = NeuronGroup(2, eqs, threshold='v>1', reset='v = 0', method='exact')
G.tau = [10, 10]*ms

stateMonitor = StateMonitor(G, 'v', record=True)
spikeMonitor = SpikeMonitor(G,record=True)

#   Set q and r parameters for each synapse
q = np.zeros(2)
r = np.zeros(2)
#   Synapse 0 peaks at 0.02 (20ms)
q[0] = 16.0
r[0] = 25.0
#   Synapse 1 peaks at 0.04 (40ms)
q[1] = 32.0
r[1] = 9.0

@implementation('numpy', discard_units=True)
@check_units(result=1)
def getBetaParam(param, synapseNumber):
    # param = 1 signifies 'q'
    print('Synapse: ' + str(synapseNumber) + ' param: ' + str(param))
    if param == 1:
        value = q[synapseNumber];
        # print('q = ' + str(value))
    else:
        value = r[synapseNumber]
        # print('r = ' + str(value))
    return value

# print(S.N_incoming[1])

synModel = '''
    w :1
    lastupdate : second
'''
v_post_Eq = '''
    isi = calcISI(t, lastupdate) 
    v_post += gBetaScale * gBeta(isi, lowISI, highISI, getBetaParam(1,synapse_number), getBetaParam(2,synapse_number))
    lastupdate = t
'''
S = Synapses(G, G, model = synModel, on_pre=v_post_Eq, multisynaptic_index='synapse_number')
S.connect(i=0, j=1, n=2)
S.w = 'j*0.01'
# Set delay on synapse 1 to 30ms
S.delay[0] = 10*ms
# Set delay on synapse 2 to 10pm
S.delay[1] = 30*ms

run(100*ms)

plot(stateMonitor.t / ms, stateMonitor.v[0], label='Neuron 0')
plot(stateMonitor.t / ms, stateMonitor.v[1], label='Neuron 1')
title('2 neurons with 2 synapses from neuron 0 to neuron 1,\n delays = [' + str(S.delay[0]) + ',' + str(S.delay[1]) + ']')
for t in spikeMonitor.t:
    axvline(t/ms, ls = '--', c='g', lw = 3)
    print(t)
xlabel('Time (ms)')
ylabel('v')
legend();
show()

