"""
Multiple-sample EGO optimizer
Author: Reino Ruusu <reino.ruusu@semantum.fi>

Based on ego.py by
Author: Remi Lafage <remi.lafage@onera.fr> and Nathalie Bartoli

This package is distributed under New BSD license.
"""
# TODO : documentation

from __future__ import division
import six
import numpy as np

from types import FunctionType

from scipy.stats import norm
from scipy.optimize import minimize, shgo

# import matplotlib.pyplot as pyplot

from smt.utils.options_dictionary import OptionsDictionary
from smt.applications.application import SurrogateBasedApplication
from smt.utils.misc import compute_rms_error

from smt.surrogate_models import KPLS, KRG, KPLSK
from smt.sampling_methods import LHS


class EGO_MS(SurrogateBasedApplication):
    def _initialize(self):
        super(EGO_MS, self)._initialize()
        declare = self.options.declare

        declare("plot", False, types=bool, desc="Plot intermediate results")
        declare("fun", None, types=FunctionType, desc="Function to minimize")
        declare(
            "criterion",
            "EI",
            types=str,
            values=["EI", "SBO", "UCB"],
            desc="criterion for next evaluation point determination: Expected Improvement, \
            Surrogate-Based Optimization or Upper Confidence Bound",
        )
        declare("n_iter", None, types=int, desc="Number of optimizer steps")
        declare("n_sample", None, types=int, desc="Number of samples evaluated at a single iteration")
        declare(
            "n_max_optim",
            20,
            types=int,
            desc="Maximum number of internal optimizations",
        )
        declare("n_start", 20, types=int, desc="Number of optimization start points")
        declare(
            "n_doe",
            None,
            types=int,
            desc="Number of points of the initial LHS doe, only used if xdoe is not given",
        )
        declare(
            "krgoptions",
            {},
            types=dict,
            desc="Kriging model options",
        )
        declare("xdoe", None, types=np.ndarray, desc="Initial doe inputs")
        declare("ydoe", None, types=np.ndarray, desc="Initial doe outputs")
        declare("xlimits", None, types=np.ndarray, desc="Bounds of function fun inputs")
        declare("verbose", False, types=bool, desc="Print computation information")

    def optimize(self, fun):
        """
        Optimizes fun

        Parameters
        ----------

        fun: function to optimize: ndarray[n, nx] or ndarray[n] -> ndarray[n, 1]

        Returns
        -------

        [nx, 1]: x optimum
        [1, 1]: y optimum
        int: index of optimum in data arrays
        [ndoe + n_iter * n_sample, nx]: coord-x data
        [ndoe + n_iter * n_sample, 1]: coord-y data
        [ndoe, nx]: coord-x initial doe
        [ndoe, 1]: coord-y initial doe
        """
        xlimits = self.options["xlimits"]
        sampling = LHS(xlimits=xlimits, criterion="ese")

        xdoe = self.options["xdoe"]
        if xdoe is None:
            self.log("Build initial DOE with LHS")
            n_doe = self.options["n_doe"]
            x_doe = sampling(n_doe)
        else:
            self.log("Initial DOE given")
            x_doe = np.atleast_2d(xdoe)

        ydoe = self.options["ydoe"]
        if ydoe is None:
            y_doe = fun(x_doe)
        else: # to save time if y_doe is already given to EGO
            y_doe = ydoe

        # to save the initial doe
        x_data = x_doe
        y_data = y_doe

        _, nx = np.shape(x_data)

        #self.gpr = KRG(print_global=False)
        self.gpr = KRG(print_global=False, **self.options["krgoptions"])

        bounds = xlimits

        criterion = self.options["criterion"]
        n_iter = self.options["n_iter"]
        n_sample = self.options["n_sample"]
        n_start = self.options["n_start"]
        n_max_optim = self.options["n_max_optim"]

        plot = self.options["plot"]
        if plot:
            x = [np.linspace(bounds[i,0], bounds[i,1], 201) for i in range(0,nx)]
            grid_x, grid_y = np.meshgrid(*x)

        for k in range(n_iter):
            x_sample = np.empty(shape=(0, nx))
            x_data_tmp = np.copy(x_data)
            y_data_tmp = np.copy(y_data)

            for j in range(n_sample):
                self.gpr.set_training_values(x_data_tmp, y_data_tmp)
                self.gpr.train()

                if criterion == "EI":
                    self.obj_k = lambda x: -self.EI(np.atleast_2d(x), y_data_tmp)
                elif criterion == "SBO":
                    self.obj_k = lambda x: self.SBO(np.atleast_2d(x))
                elif criterion == "UCB":
                    self.obj_k = lambda x: self.UCB(np.atleast_2d(x))

                # if plot:
                    # x_tmp = np.vstack((np.ravel(grid_x), np.ravel(grid_y))).T
                    # grid_z = np.reshape(self.gpr.predict_values(x_tmp), grid_x.shape)
                    # #grid_z = np.reshape(self.obj_k(x_tmp), grid_x.shape)
                    # grid_z2 = np.reshape(fun(x_tmp), grid_x.shape)
                    # pyplot.clf()
                    # lvls = np.linspace(np.sqrt(np.min(grid_z2)), np.sqrt(np.max(grid_z2)), 10)**2
                    # pyplot.contour(grid_x, grid_y, grid_z, lvls)
                    # pyplot.contour(grid_x, grid_y, grid_z2, lvls)
                    # pyplot.scatter(x_data_tmp[:,0], x_data_tmp[:,1])
                    # pyplot.savefig('test.png')

                success = False
                n_optim = 1  # in order to have some success optimizations with SLSQP
                while not success and n_optim <= n_max_optim:
                    opt_all = []
                    x_start = sampling(max(2,n_start))[0:n_start,:]
                    for ii in range(n_start):
                        opt_all.append(
                            shgo(
                                self.obj_k,
                                bounds=bounds,
                                options={'f_tol': 1e-6, 'disp': False},
                                sampling_method='sobol'
                            )
                        )

                    opt_all = np.asarray(opt_all)

                    opt_success = opt_all[[opt_i["success"] for opt_i in opt_all]]
                    obj_success = np.array([opt_i["fun"] for opt_i in opt_success])
                    success = obj_success.size != 0
                    if not success:
                        self.log("New start point for the internal optimization")
                        n_optim += 1

                if n_optim >= n_max_optim:
                    self.log("Internal optimization failed at EGO iter = {}".format(k))
                    break
                elif success:
                    self.log("Internal optimization succeeded at EGO iter = {}".format(k))

                ind_min = np.argmin(obj_success)
                opt = opt_success[ind_min]
                x_et_k = np.atleast_2d(opt["x"])
                y_et_k_low = self.blind(np.atleast_2d(x_et_k))

                y_data_tmp = np.atleast_2d(np.append(y_data_tmp, y_et_k_low)).T
                x_data_tmp = np.atleast_2d(np.append(x_data_tmp, x_et_k, axis=0))

                x_sample = np.atleast_2d(np.append(x_sample, x_et_k, axis=0))

            y_sample = fun(x_sample)

            y_data = np.concatenate((y_data, y_sample))
            x_data = np.concatenate((x_data, x_sample), axis=0)

        ind_best = np.argmin(y_data)
        x_opt = x_data[ind_best]
        y_opt = y_data[ind_best]

        return x_opt, y_opt, ind_best, x_data, y_data, x_doe, y_doe

    def initialize(self, x_data, y_data):
        self.xlimits = self.options["xlimits"]
        self.sampling = LHS(xlimits=self.xlimits, criterion="ese")

        self.x_data = np.atleast_2d(x_data)
        self.y_data = np.atleast_2d(y_data)
        if self.y_data.shape[0] == 1:
            self.y_data = self.y_data.T

        nx = self.x_data.shape[0]
        self.y_valid = np.zeros(nx, 'bool')

        self.gpr = KRG(print_global=False, **self.options["krgoptions"])

    def get_candidate(self):
        """
        Get a new x values for model evaluation

        Performs an optimization to find an optimal candidate for model
        evaluation. The surrogate model data is augmented with an estimate
        of the model output value and the input values are returned for actual
        model evaluation.

        The caller should evaluate the actual objective function at the
        returned input values and then call update_model() to update the
        surrogate model with the actual model output.

        Any number of get_candidate() calls can be made in between calls to
        update_model(). Until the actual model output is provided, the estimate
        that was made by the call to get_candidate() is kept as a sample value
        for the surrogate model.
        """
        x_data_tmp = np.copy(self.x_data)
        y_data_tmp = np.copy(self.y_data)

        self.gpr.set_training_values(x_data_tmp, y_data_tmp)
        self.gpr.train()

        criterion = self.options["criterion"]
        n_start = self.options["n_start"]
        n_max_optim = self.options["n_max_optim"]

        if criterion == "EI":
            self.obj_k = lambda x: -self.EI(np.atleast_2d(x), y_data_tmp)
        elif criterion == "SBO":
            self.obj_k = lambda x: self.SBO(np.atleast_2d(x))
        elif criterion == "UCB":
            self.obj_k = lambda x: self.UCB(np.atleast_2d(x))

        success = False
        n_optim = 1  # in order to have some success optimizations with SLSQP
        while not success and n_optim <= n_max_optim:
            opt_all = []
            x_start = self.sampling(max(2,n_start))[0:n_start,:]
            for ii in range(n_start):
                opt_all.append(
                    shgo(
                        self.obj_k,
                        bounds=self.xlimits,
                        options={'f_tol': 1e-6, 'disp': False},
                        sampling_method='sobol'
                    )
                )

            opt_all = np.asarray(opt_all)

            opt_success = opt_all[[opt_i["success"] for opt_i in opt_all]]
            obj_success = np.array([opt_i["fun"] for opt_i in opt_success])
            success = obj_success.size != 0
            if not success:
                self.log("New start point for the internal optimization")
                n_optim += 1

        if n_optim >= n_max_optim:
            self.log("Internal optimization failed at EGO")
            return None
        elif success:
            self.log("Internal optimization succeeded at EGO")

        ind_min = np.argmin(obj_success)
        opt = opt_success[ind_min]
        x_et_k = np.atleast_2d(opt["x"])
        y_et_k_low = self.blind(np.atleast_2d(x_et_k))

        y_data_tmp = np.append(y_data_tmp, y_et_k_low, axis=0)
        x_data_tmp = np.append(x_data_tmp, x_et_k, axis=0)

        self.x_data = x_data_tmp
        self.y_data = y_data_tmp
        self.y_valid = np.append(self.y_valid, False)

        return x_et_k

    def update_model(self, x_et_k, y_et_k):
        """
        Update the surrogate model with a new sample of the actual objective
        function.

        The collection of training samples for the surrogate model is updated
        so that if an existing sample matches the x values, its y values are
        updated to the given values. Otherwise a new sample point is added to
        the training set.

        This function should be called with a single sample at a time.
        """
        x_et_k = np.atleast_2d(x_et_k)
        y_et_k = np.atleast_2d(y_et_k)

        m = (self.x_data == x_et_k).all(axis = 1)
        if not m.any():
            # Add a new independent sample point
            self.x_data = np.append(self.x_data, x_et_k, axis=0)
            self.y_data = np.append(self.y_data, y_et_k, axis=0)
            self.y_valid = np.append(self.y_valid, False)
        else:
            self.y_data[m,:] = y_et_k
            self.y_valid[m] = True

    def get_result(self):
        """
        Get optimization result

        Returns a tuple of (
            optimal x,
            optimal y,
            index of best sample,
            x samples,
            y samples
        )
        """
        x_data = np.copy(self.x_data[self.y_valid,:])
        y_data = np.copy(self.y_data[self.y_valid,:])

        ind_best = np.argmin(y_data)
        x_opt = x_data[ind_best]
        y_opt = y_data[ind_best]

        return x_opt, y_opt, ind_best, x_data, y_data

    def log(self, msg):
        if self.options["verbose"]:
            print(msg)

    def EI(self, points, y_data):
        """ Expected improvement """
        f_min = np.min(y_data)
        pred = self.gpr.predict_values(points)
        sig = np.sqrt(self.gpr.predict_variances(points))
        args0 = (f_min - pred) / sig
        args1 = (f_min - pred) * norm.cdf(args0)
        args2 = sig * norm.pdf(args0)
        if sig.size == 1 and sig == 0.0:  # can be use only if one point is computed
            return 0.0

        ei = args1 + args2
        return ei

    def SBO(self, point):
        """ Surrogate based optimization: min the surrogate model by suing the mean mu """
        res = self.gpr.predict_values(point)
        return res

    def UCB(self, point):
        """ Upper confidence bound optimization: minimize by using mu - 3*sigma """
        pred = self.gpr.predict_values(point)
        var = self.gpr.predict_variances(point)
        res = pred - 3.0 * np.sqrt(var)
        return res

    def blind(self, point):
        """ Blind prediction: Assume value at upper confidence bound mu + 3*sigma """
        pred = self.gpr.predict_values(point)
        var = self.gpr.predict_variances(point)
        res = pred + 1.0 * np.sqrt(var)
        return res
