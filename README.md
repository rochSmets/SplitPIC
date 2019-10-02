# splitPIC

To run this code, you need : `import python3, numpy, scipy.signal, nlopt`

- The `split.py` file contains the definition of the `split` object and the method `objective`, defining the cost function. This file should not be modified.

- The `optim.py` file is the one to find the optimum values for the weights and shifts, depending on the dimension, refinement factor, order of the b-spline, mesh size and list of patterns. All these values should be initialized in the first block, the second one should not be modified. simply run
> python3 optim.py

- The `check.py` file is the one to check the Q2 value (the cost function) for a given set of children, providing a list of appropriate size for the weights and shifts. simply run
> python3 check.py

For any questions : roch.smets@polytechnique.edu
