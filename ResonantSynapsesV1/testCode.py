#############################################################################################
#   TEST generalBetaDistribution.py

# from generalBetaDistribution import gBeta, meanGBeta
# import matplotlib.pyplot as plt
# import numpy as np

# #   Set the parameters for gBeta
# low = 0
# high = 0.05
# nearEdgeAjust = 0.001
# # betaQParam = [5, 1, 5, 3, 2, 1]
# # betaRParam = [1.1, 1, 3, 5, 5, 5]
# # col = ['r','b','y','c','g','m']
# x = np.linspace(low + nearEdgeAjust,high - nearEdgeAjust,100)
# # gB = np.zeros((len(betaQParam), 100))
# maxHeight = 0
# for i in range(len(betaQParam)):
#     gB[i,:] = gBeta(x, low, high, betaQParam[i], betaRParam[i])
#     m = max(gB[i,:])
#     if m > maxHeight:
#         maxHeight = m
#
# print('Mean values')
# for i in range(len(betaQParam)):
#     print(mean(low, high, betaQParam[i], betaRParam[i]))
#
# plt.figure(figsize=(7,7))
# plt.xlim(low, high)
# plt.ylim(0,maxHeight + maxHeight*0.05)
# for i in range(len(betaQParam)):
#     plt.plot(x, gB[i,:], col[i])
#     plt.axvline(x=mean(low, high, betaQParam[i], betaRParam[i]), color=col[i])
# plt.title('General Beta Distribution', fontsize='15')
# plt.xlabel(f'Values of Random Variable X ({low},{high})', fontsize='15')
# plt.ylabel('Probability', fontsize='15')
# plt.show()

#############################################################################################
#   Test range of values for q and r
#
# totQandR = 40
# r = 1
# q = totQandR
# noSamples = 10
# increment = totQandR/noSamples
# stepSize = 0.2
# gB2 = np.zeros((noSamples,100))
# # maxHeight = 0
# # for i in range(noSamples):
# #     r = r + increment
# #     q = q - increment
# #     print('q = ' + str(q) + ' r = ' + str(r))
# #     gB2[i,:] = gBeta(x,low,high,q,r)
# #     m = max(gB2[i, :])
# #     if m > maxHeight:
# #         maxHeight = m
#
# gBetaParamSum = 40              # gBetaParamSum = q + r, used to generate random q and r values
# gBetaParamMin = 1.2             # gBetaParamMin = the lowest value of q and r to maintain bell-shaped curve
# #   SET RANDOM q AND r VALUES FOR EACH SYNAPSE
# q = np.random.uniform(gBetaParamMin,gBetaParamSum - gBetaParamMin,2)
# r = gBetaParamSum - q
#
# q = [6.788163534648131]
# r = [33.211836465351865]
#
#
# maxHeight = 0
# for i in range(len(q)):
#     print('q = ' + str(q[i]) + ' r = ' + str(r[i]))
#     gB2[i,:] = gBeta(x,low,high,q[i],r[i])
#     m = max(gB2[i, :])
#     if m > maxHeight:
#         maxHeight = m
#
# #   Plot for a range of q and r values
# plt.figure(figsize=(7,7))
# plt.xlim(low, high)
# plt.ylim(0,maxHeight + maxHeight*0.05)
# for i in range(noSamples):
#     plt.plot(x, gB2[i, :])
# plt.title('General Beta Distribution - varying q and r  from ' + str(1 + stepSize) + ' to ' + str(noSamples*0.2), fontsize='15')
# plt.xlabel(f'Values of Random Variable X ({low},{high})', fontsize='15')
# plt.ylabel('Probability', fontsize='15')
#
# #   Plot for specific q and r value
# plt.figure(figsize=(7,7))
# plt.xlim(low, high)
# q = 32.0
# r = 9.0
# plt.ylim(0,maxHeight + maxHeight*0.05)
# plt.plot(x, gBeta(x,low,high,q,r))
# plt.title('General Beta Distribution - q = ' + str(q) + ' r = ' + str(r), fontsize='15')
# plt.xlabel(f'Values of Random Variable X ({low},{high})', fontsize='15')
# plt.ylabel('Probability', fontsize='15')
#
# plt.show()


