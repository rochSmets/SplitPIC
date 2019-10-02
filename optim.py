
from split import Split
import numpy as np
import nlopt


# _____________________________________________________________________________
#
# this first block contains the input data, needed to calculate
# the optimum values of weights & shifts for a given set of patterns
# only this part of the code needs to be modified
# _____________________________________________________________________________

# explicit : dimension of the spline, its order and the refinement factor
# for the children
dimension  = 2
refinement = 2
order      = 3

# mesh size : should be 1.0 if isotrop for simplification
mesh = (1.0, 1.0, 1.0)

# this is the list of pattern : should be by growing orders
pattern = ['p0', 'p1', 'p2']

# these 2 lists contain lower & uper bound values (in a list) for optimization
weights = [[0.18, 0.28], [0.100, 0.150], [0.030, 0.090]]
shifts  = [[0.00, 0.00], [0.80, 1.60], [0.80, 1.60]]
# weights = [[0.25, 0.25]]
# shifts  = [[0.40, 0.80]]

precision = 1e-4

# _____________________________________________________________________________
#
# this second block should not be modified :
# it is the one calculating the optimization using the nlopt library
# note that not all the type of optimization are acceptable :
# nlopt.LN_NELDERMEAD is the most efficient, but converge to the same limit
# as the one for nlopt.LN_COBYLA (faster).
# note that it is not an optimization problem using constraints :
# the constraint we need to fullfill is that the sum of all weights equal 1.
# this constraint is integrated in the way the list of weights is built
# in the Split class
# _____________________________________________________________________________

# set the global values
mysplit = Split(dimension, refinement, order, pattern, mesh)

# make the list of lower bound values
wlb_ = [weights[i][0] for i in range(weights.__len__())]
slb_ = [shifts[i][0] for i in range(shifts.__len__())]
lb = mysplit.makeEntryList(wlb_, slb_)

# make the list of upper bound values
wub_ = [weights[i][1] for i in range(weights.__len__())]
sub_ = [shifts[i][1] for i in range(shifts.__len__())]
ub = mysplit.makeEntryList(wub_, sub_)

# initial guess is the average from lower & upper bounds
iniGuess = 0.5*(np.array(lb) + np.array(ub))

# number of free parameter to optimize
n = iniGuess.shape[0]

# display the parameters initialized above
mysplit.displayParameters()

# define the type of optimization
#opt = nlopt.opt(nlopt.LN_COBYLA, n)
opt = nlopt.opt(nlopt.LN_NELDERMEAD, n)

# define the lower & upper bounds for the free parameters
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)

# define the objective function
opt.set_min_objective(mysplit.objective)

# define the precision needed
opt.set_xtol_rel(precision)

# optimization from the initial guess
x = opt.optimize(iniGuess)

# the delta values are define on the fine grid : the one of the children
w, s = mysplit.getWeightsAndShifts(x)
print("list of weights            : {0}".format(np.round(w, 6)))
print("list of shifts             : {0}\n".format(np.round(s, 6)))

# min value of q2 after optimization
print("optimum value of Q2        : {0:.6f}\n".format(opt.last_optimum_value()))
