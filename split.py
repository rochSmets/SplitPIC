
import numpy as np
import scipy.signal as sig


class Split(object):

    def __init__(self, dim, ref, order, patt, mesh):

        """ define the needed var : dimension,
                                    refine (factor)
                                    order (of interpollation)
                                    list of pattern indice(s)
                                    (list of) # of particles per pattern
                                    (list of) shift patterns
            and the method needed for optimization : objective
        """

        self.dimension = dim
        self.refine = ref
        self.order = order
        self.mesh = mesh

        self.patternIndices = self.setPatternIndices(patt)
        self.numOfPattern = self.patternIndices.__len__()
        self.numPerPattern = self.setNumPerPattern(self.dimension, self.patternIndices)
        self.shiftPattern = self.setShiftPattern(self.dimension, self.patternIndices)

        # needed to set the points for the integral calculation
        # smin & smax have to be large enough to include the whole support
        # for all possible spline order
        # ns has also to be large enough in order to calculate a good integral
        self.smin = np.array([-2.0, -2.0, -2.0])*np.array(mesh)
        self.smax = np.array([+2.0, +2.0, +2.0])*np.array(mesh)
        self.ns = 500
        self.ds = (np.array(self.smax)-np.array(self.smin))/self.ns
        self.s = list(np.linspace(self.smin[i], self.smax[i], self.ns) for i in range(3))


    def setPatternIndices(self, pattern):

        """ set the list of pattern order : the order has to be the last term (digit)
                                            in the string name of each pattern (eg 'p1')
        """

        patternIndices = []

        for i in range(pattern.__len__()):
            patternIndices.append(int(pattern[i][pattern[i].__len__()-1]))

        return patternIndices


    def setNumPerPattern(self, dimension, patternIndices):

        """ set the list of the # of particles for each pattern
        """

        index = [-1, 0, 1]

        numPerPattern = []

        for p in range(self.numOfPattern):
            num = 0
            for i in index:
                if (dimension == 1):
                    if ((i**2) == patternIndices[p]):
                        num+=1
                else:
                    for j in index:
                        if (dimension == 2):
                            if ((i**2+j**2) == patternIndices[p]):
                                num+=1
                        else:
                            for k in index:
                                if ((i**2+j**2+k**2) == patternIndices[p]):
                                    num+=1

            numPerPattern.append(num)

        return numPerPattern


    def setShiftPattern(self, dimension, patternIndices):

        """ set the full list of shifts : the lenght of this list equals
            the total number of particles associated to all patterns.
            each element is a int or a list of int of size equals the dimension
        """
        index = [-1, 0, 1]

        shiftPattern = []

        for p in range(self.numOfPattern):
            for i in index:
                if (dimension == 1):
                    if ((i**2) == patternIndices[p]):
                        shiftPattern.append([i])
                else:
                    for j in index:
                        if (dimension == 2):
                            if ((i**2+j**2) == patternIndices[p]):
                                shiftPattern.append([i, j])
                        else:
                            for k in index:
                                if ((i**2+j**2+k**2) == patternIndices[p]):
                                    shiftPattern.append([i, j, k])

        return shiftPattern


    def objective(self, x, grad):

        """ this one aims at selecting the objective function
            for the appropriate dimension
        """

        if self.dimension == 1:
            return self.objective1D(x)
        elif self.dimension == 2:
            return self.objective2D(x)
        elif self.dimension ==3:
            return self.objective3D(x)
        else:
            raise ValueError("bad dimension value")


    def objective1D(self, x):

        """ the objective function for the 1 dimensional case
        """

        p0x = sig.bspline(self.s[0]/self.mesh[0], self.order)/self.mesh[0]

        parent = p0x

        babies = self.setFamily1D(x)
        allbabies = np.sum(babies, axis = 0)

        return np.sum(np.square(parent-allbabies))*self.ds[0]*self.mesh[0]


    def objective2D(self, x):

        """ the objective function for the 2 dimensional case
        """

        p0x = sig.bspline(self.s[0]/self.mesh[0], self.order)/self.mesh[0]
        p0y = sig.bspline(self.s[1]/self.mesh[1], self.order)/self.mesh[1]

        parent = np.tensordot(p0x, p0y, axes = 0)

        babies = self.setFamily2D(x)
        allbabies = np.sum(babies, axis = 0)

        return np.sum(np.square(parent-allbabies))*self.ds[0]*self.ds[1]\
            *(self.mesh[0]*self.mesh[1])


    def objective3D(self, x):

        """ the objective function for the 3 dimensional case
        """

        p0x = sig.bspline(self.s[0]/self.mesh[0], self.order)/self.mesh[0]
        p0y = sig.bspline(self.s[1]/self.mesh[1], self.order)/self.mesh[1]
        p0z = sig.bspline(self.s[2]/self.mesh[2], self.order)/self.mesh[2]

        parent = np.tensordot(p0x, np.tensordot(p0y, p0z, axes = 0), axes = 0)

        babies = self.setFamily3D(x)
        allbabies = np.sum(babies, axis = 0)

        return np.sum(np.square(parent-allbabies))*self.ds[0]*self.ds[1]*self.ds[2]\
            *(self.mesh[0]*self.mesh[1]*self.mesh[2])


    def setFamily1D(self, x):

        """ create the list of the set of babies for all the patterns asked,
            in the 1 dimensional case. each baby is a numpy array of size ns
            and babies is a ndarray of numOfBabies values (one for each baby)
        """

        weight, shift = self.getWeightsAndShifts(x)
        numOfBabies_ = self.shiftPattern.__len__()

        babies = np.ndarray(shape=(numOfBabies_, self.ns), dtype = float)

        index = 0
        for p in range(self.numOfPattern):

            for i in range(self.numPerPattern[p]):
                b0 = sig.bspline(self.s[0]*self.refine/self.mesh[0]
                    +shift[p]*self.shiftPattern[index][0], self.order)

                babies[index] = weight[p]*(self.refine/self.mesh[0])*b0

                index +=1

        return babies


    def setFamily2D(self, x):

        """ create the list of the set of babies for all the patterns asked,
            in the 2 dimensional case. each baby is a numpy array of size ns**2
            and babies is a ndarray of numOfBabies values (one for each baby)
        """

        weight, shift = self.getWeightsAndShifts(x)
        numOfBabies_ = self.shiftPattern.__len__()

        babies = np.ndarray(shape=(numOfBabies_, self.ns, self.ns), dtype = float)

        index = 0
        for p in range(self.numOfPattern):

            for i in range(self.numPerPattern[p]):
                b0 = sig.bspline(self.s[0]*self.refine/self.mesh[0]
                    +shift[p]*self.shiftPattern[index][0], self.order)
                b1 = sig.bspline(self.s[1]*self.refine/self.mesh[1]
                    +shift[p]*self.shiftPattern[index][1], self.order)

                babies[index] = weight[p]*(self.refine/self.mesh[0])\
                                         *(self.refine/self.mesh[1])\
                                *np.tensordot(b0, b1, axes = 0)

                index +=1

        return babies


    def setFamily3D(self, x):

        """ create the list of the set of babies for all the patterns asked,
            in the 3 dimensional case. each baby is a numpy array of size ns**3
            and babies is a ndarray of numOfBabies values (one for each baby)
        """

        weight, shift = self.getWeightsAndShifts(x)
        numOfBabies_ = self.shiftPattern.__len__()

        babies = np.ndarray(shape=(numOfBabies_, self.ns, self.ns, self.ns), dtype = float)

        index = 0
        for p in range(self.numOfPattern):

            for i in range(self.numPerPattern[p]):
                b0 = sig.bspline(self.s[0]*self.refine/self.mesh[0]
                    +shift[p]*self.shiftPattern[index][0], self.order)
                b1 = sig.bspline(self.s[1]*self.refine/self.mesh[1]
                    +shift[p]*self.shiftPattern[index][1], self.order)
                b2 = sig.bspline(self.s[2]*self.refine/self.mesh[2]
                    +shift[p]*self.shiftPattern[index][2], self.order)

                babies[index] = weight[p]*(self.refine/self.mesh[0])\
                                         *(self.refine/self.mesh[1])\
                                         *(self.refines/self.mesh[2])\
                                *np.tensordot(b0, np.tensordot(b1, b2, axes = 0), axes = 0)

                index +=1

        return babies


    def makeEntryList(self, weights, shifts):

        """ build a single list including the weights and shifts from the 2 lists
            containing separately these values. this single list is then used
            as entry parameter for the objective and constraint functions
        """

        x = []
        # append the weight except the last one (because of the constraint)
        for p in range(self.numOfPattern-1):
            x.append(weights[p])

        # then append the shifts (all except the one of pattern 0)
        for p in range(self.numOfPattern):
            if self.patternIndices[p] != 0:
                x.append(shifts[p])

        return x


    def getWeightsAndShifts(self, x):

        """ this function aims at extracting a list of weight and a list
            of shifts from the single list (x) used as entry
            for the objective function
        """

        weights = []
        shifts = []

        norm = 1.0

        # the structure of the list "x" contain the weights for growing
        # pattern values, and then the shifts.
        # if pattern 0 is to be considered, the shift is not appended.
        # the last weight of the list is not appended neither, because
        # of the norm constraint (summ of all weights = 1)
        for p in range(self.numOfPattern-1):
            weights.append(x[p])
            norm -= weights[p]*self.numPerPattern[p]
        weights.append(norm/self.numPerPattern[self.numOfPattern-1])

        # the # of shifts values in x equals numOfPattern if pattern 0
        # is not included. if pattern 0 is included, then, there are
        # numOfPattern-1 shifts in x
        if 0 in self.patternIndices:
            numOfShifts_ = self.numOfPattern-1
        else:
            numOfShifts_ = self.numOfPattern

        for p in range(self.numOfPattern):
            if self.patternIndices[p] != 0:
                shifts.append(x[numOfShifts_-1+p])
            else:
                shifts.append(0.0)

        return weights, shifts


    def displayParameters(self):

        """ just to display the parameters of the split...
        """

        print("")
        print("dimension of the spline    : {0}".format(self.dimension))
        print("order of the spline        : {0}".format(self.order))
        print("refine factor              : {0}\n".format(self.refine))
        print("mesh size in each dir      : {0}\n".format(self.mesh))
        print("list of indices of pattern : {0}".format(self.patternIndices))
        print("# of particles per pattern : {0}".format(self.numPerPattern))
        print("total # of particles       : {0}\n".format(np.sum(self.numPerPattern)))

