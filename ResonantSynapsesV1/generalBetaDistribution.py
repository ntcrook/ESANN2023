#############################################################################################
#   General Beta Distribution
#
from brian2 import *
import numpy as np
import scipy.stats
# from scipy.stats import beta, gamma
from scipy.special import gamma

# TODO: Look at the following for approx area: https://machinelearningmastery.com/probability-density-estimation/
def approxArea(x1, x2, dt, a, b, q, r):
    if x2 > x1:
        t = x1
        x1 = x2
        x2 = t
    area = 0
    if dt > 0:
        x = x2
        while x < x1:
            thisSlice = gBeta(x, a, b, q, r, False) * dt
            area += thisSlice
            x += dt
    return area

def meanGBeta(a, b, q, r):
    # TODO: evaluate the mean using 50% of the area?
    origMean = a + ((q/(q + r)) * (b - a))
    # Need to correct for error in the mean calculation
    correctedMean = origMean
    # CORRECTION REMOVED - DIVISION BY ZERO OCCURRED HERE
    # dt = 0.0001
    # deltaMean = 0.0001
    # areaToCorrectedMean = approxArea(a,correctedMean,dt,a,b,q,r)
    # areaToOrigMean = approxArea(a, origMean,dt,a,b,q,r)
    # if areaToCorrectedMean < 0.5:
    #     while areaToCorrectedMean < 0.5:
    #         correctedMean += deltaMean
    #         areaToCorrectedMean = approxArea(a,correctedMean,dt,a,b,q,r)
    # else:
    #     while areaToCorrectedMean > 0.5:
    #         correctedMean -= deltaMean
    #         areaToCorrectedMean = approxArea(a, correctedMean, dt, a, b, q, r)
    return correctedMean

def varianceGBeta(a, b, q, r):
    return (q*r*((b - a)**2))/(((q + r)**2)*(q + r + 1))

@implementation('numpy', discard_units=True)
@check_units(X=1, a=1, b = 1, q=1, r = 1, displayResults = 1, result=1)
#   Taken from: https://vitalflux.com/beta-distribution-explained-with-python-examples/
def gBeta(x, a, b, q, r, displayResults):
    # NOTE THAT THIS VERSION DOES NOT INSIST ON a <= x <= b BECAUSE IT OPERATES ON VECTORS
    if displayResults:
        print("gBeta params: X = " + str(x) + " a = " + str(a) + " b = " + str(b) + " q = " + str(q) + " r = " + str(r))
    errorMessage = "  ====================================="
    result = 0
    denominator = (b - a) ** (q + r - 1)
    if any(x < a) or any(x > b):
        errorMessage = " x = " + str(x) + " OUT OF RANGE"
    elif denominator <= sys.float_info.min:
        errorMessage += "(b - a) ** (q + r - 1) = " + str(denominator) + " ==> DIVISION BY ZERO"
    else:
        inverseBeta = gamma(q + r) / (gamma(q) * (gamma(r)))
        result = inverseBeta * ((((x - a) ** (q - 1)) * ((b - x) ** (r - 1))) / denominator)
    if displayResults:
        print("gBeta() = " + str(result) + errorMessage)
        print("Mean of gBeta: " + str(meanGBeta(a, b, q, r)))
        print("Variance of gBeta: " + str(varianceGBeta(a, b, q, r)))
        print("===========================================================")
    return result

def getGBetaDist(a,b,q,r, noSamples):
    x = np.linspace(a, b, noSamples)
    return gBeta(x,a,b,q,r)

# Parameter Estimator for q and r
# Eleni's second solution
def gBetaParamEst_Eleni2(mu,var,a,b):
    # print(f"gBetaParamEst_Eleni2: mu = {mu} var = {var} a = {a} b = {b}")
    c = a
    d = b
    # var = sigma**2
    l = (d-mu)/(mu-c)
    a = (l*(d-c)**2-var*((l+1)**2))/(var*(1+3*l+3*l**2+l**3))
    b = a*l
    return a, b


###################################################################################################################
# TEST CODE
###################################################################################################################
#  Test gBeta with different max ISI values





# Testing Alex and Eleni's gBetaParamEst
# if __name__ == "__main__":
#     low = 0
#     high = 0.05
#     #   mean of around 0.02 (20ms)
#     q = 16.0
#     r = 25.0
#     mean = meanGBeta(low,high,q,r)
#     variance = varianceGBeta(low,high,q,r)
#     sigma = np.sqrt(variance)
#
#     # alphaAlex, betaAlex = gBetaParamEst_Alex(mean,sigma,low,high)
#     # alphaEleni, betaEleni = gBetaParamEst_Eleni(mean,sigma,low,high)
#     # alphaNigel, betaNigel = gBetaParamEst_Nigel(mean,sigma,low,high)
#     alphaEleni2, betaEleni2 = gBetaParamEst_Eleni2(mean,variance,low,high)
#     print(f'Original alpha = {q}, beta = {r}, mean = {mean}, sigma = {sigma}')
#     # print(f'Solution from Alexs formula: alpha = {alphaAlex}, beta = {betaAlex}')
#     # print(f'Solution from Elenis formula: alpha = {alphaEleni}, beta = {betaEleni}')
#     # print(f'Solution from Nigels formula: alpha = {alphaNigel}, beta = {betaNigel}')
#     print(f'Solution from Elenis second formula: alpha = {alphaEleni2}, beta = {betaEleni2}')
###################################################################################################################
# Test approxArea(x1, x2, dt, a, b, q, r)
# low = 0
# high = 0.05
# # high = 5
# #   mean of around 0.02 (20ms)
# q = 33.666316037092614
# r = 6.3336839629073864
#
# dt = 0.0001
# tau_mu = meanGBeta(low, high, q, r)
# print("tau_mu = " + str(tau_mu))
# isi = low
# # while isi < high:
# #     print("ISI = " + str(isi) + " area = " + str(approxArea(tau_mu,isi, dt, low,high,q,r)))
# #     isi += dt
# area = approxArea(tau_mu,tau_mu+(2*dt), dt, low,high,q,r)
# print(f"Area to close to mean = {area}")
# pISI = 1 - 2*area
# print(f"Prob of ISI close to mean = {pISI}")
# # PLOT gBeta DISTRIBUTION
# nearEdgeAjust = 0.001
# x = np.linspace(low + nearEdgeAjust,high - nearEdgeAjust,100)
# gB = gBeta(x,low,high,q,r,False)
# maxHeight =  max(gB)
# plt.figure(figsize=(7,7))
# plt.xlim(low, high)
# plt.ylim(0,maxHeight + maxHeight*0.05)
# plt.plot(x, gB) # TODO: find out why label is not showing
# plt.title('General Beta Distribution', fontsize='15')
# plt.xlabel(f'Values of Random Variable X ({low},{high})', fontsize='15')
# plt.ylabel('Probability', fontsize='15')
# # axvline(isi, ls = '--', c='g', lw = 3)
# axvline(tau_mu, ls = '--', c='b', lw = 3)
# show()
###################################################################################################################
# Test corrected mean(...) function
# low = 0
# high = 0.05
# # high = 5
# #   mean of around 0.02 (20ms)
# q = np.random.uniform(1.2,37.5,1)
# r = 40 - q
#
# dt = 0.0001
# origTau_mu, correctedTau_mu, areaToOrigMean, areaToCorrectedMean = meanGBeta(low, high, q, r)
# print(f"oriTtau_mu = {origTau_mu}, correctedTau_mu = {correctedTau_mu}, areaToOrigMean = {areaToOrigMean}, areaToCorrectedMean = {areaToCorrectedMean}")
#
# # PLOT gBeta DISTRIBUTION
if __name__ == "__main__":
    low = 0
    high = 0.05
    nearEdgeAjust = 0.001
    x = np.linspace(low + nearEdgeAjust,high - nearEdgeAjust,100)
    q = 2
    r = 2
    gB = gBeta(x,low,high,q,r,False)
    mu = meanGBeta(low, high, q, r)
    print(mu)
    maxHeight =  max(gB)
    plt.figure(figsize=(7,7))
    plt.xlim(low, high)
    plt.ylim(0,maxHeight + maxHeight*0.05)
    plt.plot(x, gB) # TODO: find out why label is not showing
    plt.title(f'General Beta Distribution q={q}, r={r}', fontsize='15')
    plt.xlabel(f'Values of Random Variable X ({low},{high})', fontsize='15')
    plt.ylabel('Probability', fontsize='15')
    # axvline(isi, ls = '--', c='g', lw = 3)
    axvline(mu, ls = '--', c='r', lw = 3)
    show()

# print(gBeta(0.01, 0.01, 0.05, 5, 2.4))
# result = 0.058722752193097756
