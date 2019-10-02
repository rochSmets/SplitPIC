
from split import Split
import numpy as np
import scipy.signal as sig


# _____________________________________________________________________________
#
# this first block contains the input data, needed to calculate
# the Q2 value for a given set of weights & shifts & patterns
# only this part of the code needs to be modified
# _____________________________________________________________________________

# explicit : dimension of the spline, its order and the refinement factor
# for the children
dimension  = 2
refinement = 2
order      = 1

# mesh size of the parent : isotrop if all values are equal
mesh = (1.0, 2.0, 0.5)

# this is the list of pattern : should be by growing orders
pattern = ['p0', 'p1', 'p2']

# these 2 lists contain the weights and shifts values to test
weights = [0.2500, 0.1250, 0.0625]
shifts  = [0.0000, 1.0000, 1.0000]

# _____________________________________________________________________________
#
# this second block should not be modified.
# its aim is just to check the Q2 values for a given set of shifts and weights
# _____________________________________________________________________________

# set the global values
mysplit = Split(dimension, refinement, order, pattern, mesh)

# make the list for objective function
list4Objective = mysplit.makeEntryList(weights, shifts)

# calculate the q2 value
q2 = mysplit.objective(list4Objective, [0])

# display the parameters initialized above
mysplit.displayParameters()

# the delta values are define on the fine grid : the one of the children
print("list of weights            : {0}".format(weights))
print("list of shifts             : {0}\n".format(shifts))

# value of q2
print("value of Q2                : {0:.6f}\n".format(q2))
