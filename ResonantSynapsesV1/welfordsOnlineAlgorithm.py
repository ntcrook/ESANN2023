# Taken from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# It is often useful to be able to compute the variance in a single pass, inspecting each value x_i
# only once; for example, when the data is being collected without enough storage to keep all the values, or when costs of
# memory access dominate those of computation. For such an online algorithm, a recurrence relation is required between
# quantities from which the required statistics can be calculated in a numerically stable fashion.

import numpy as np

# For a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)

################################################ TEST CODE #############################################################
# print('Test 1: generate samples from a known normal distribution and compare with calcualted mean and variance')
#
# mu, sigma = 5, 2
# samples = np.random.normal(mu,sigma,10000)
# existingAggregate = (0, 0, 0)
# for x in samples:
#     existingAggregate = update(existingAggregate,x)
# (mean, variance, sampleVariance) = finalize(existingAggregate)
# print(f'mean = {mean}, variance = {variance}, sample variance = {sampleVariance}')
# print('Test 1 PASSED')
#
# print('/nTest 2: show running mean and variance as data items are added (after data item 2')
#
# mu, sigma = 5, 2
# samples = np.random.normal(mu,sigma,100)
# existingAggregate = (0, 0, 0)
# for i in range(len(samples)):
#     existingAggregate = update(existingAggregate,samples[i])
#     if i > 1:
#         (mean, variance, sampleVariance) = finalize(existingAggregate)
#         print(f'Sample[{i}] = {samples[i]}: mean = {mean}, variance = {variance}, sample variance = {sampleVariance}')
# print('Test 2 PASSED')