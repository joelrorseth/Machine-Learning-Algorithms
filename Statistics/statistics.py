#
# statistics.py
# A quick implementation of some basic statistics calculations
#

from collections import Counter
import math

# << Vector calculations >>
# Dot product (sum of componentwise products)
def dot(v, w):
    return sum(vi * wi for vi, wi in zip(v, w))

# Sum of squares
def sum_of_squares(v):
    return dot(v, v)



# << Central Tendencies >>
# The average of all elements
def mean(array):
    return sum(array) / len(array)


# The middle value
def median(array):

    length = len(array)
    sorted_array = sorted(array)
    mid = length // 2

    # Sort then find midpoint, or calculate if of even length
    if length % 2 == 1:
        return sorted_array[mid]
    else:
        return (sorted_array[mid-1] + sorted_array[mid]) / 2


# The value which less than a given percentile of data lies
def quantile(array, p):
    return sorted(array)[ int(p * len(array)) ]


# The most common value in a dataset
def mode(array):

    c = Counter(array)
    count = max(c.values())

    # Return all elements which appeared the most (if tied)
    return [e for  e, count in c.iteritems() if c == count]


# Dispersion of an array
def range(array):
    return max(array) - min(array)


# Map the deviations of each element from the mean
def deviations_mean(array):
    array_mean = mean(array)
    return [x - array_mean for x in array]


# Variance
def variance(array):
    deviations = deviations_mean(array)
    return sum_of_squares(deviations) / (len(array) - 1)


# Standard Deviation
def standard_deviation(array):
    return math.sqrt(variance(array))


# IQ Range (a better range measure to handle outlier bias)
def interquartile_range(array):
    return quantile(array, 0.75) - quantile(array, 0.25)
