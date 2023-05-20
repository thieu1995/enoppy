#!/usr/bin/env python
# Created by "Thieu" at 08:04, 21/09/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# Paper: A Test-suite of Non-Convex Constrained Optimization Problems from the Real-World and Some Baseline Results

import numpy as np
from scipy.optimize import fminbound
from enoppy.engineer import Engineer


class HeatExchangerNetworkDesignCase1Problem(Engineer):
    """
    Industrial Chemical Processes
    [x1, x2, x3, x4,..., x9]
    Heat Exchanger Network Design (case 1)
    """

    name = "Heat Exchanger Network Design Case 1 (Industrial Chemical Processes)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 9
        self._n_objs = 1
        self._n_eq_cons = 8
        self._n_cons = 8
        self._bounds = np.array([(0., 10.), (0., 200), (0., 100.), (0., 200.), (1000, 2000000),
                        (0., 600.), (100, 600.), (100., 600.), (100., 900)])
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 35 * x[0] ** 0.6 + 35 * x[1] ** 0.6
        return np.array([f1])

    def get_eq_cons(self, x):
        hx = np.zeros(self.n_eq_cons)
        hx[0] = 200 * x[0] * x[3] - x[2]
        hx[1] = 200 * x[1] * x[5] - x[4]
        hx[2] = x[2] - 10000 * (x[6] - 100)
        hx[3] = x[4] - 10000 * (300 - x[6])
        hx[4] = x[2] - 10000 * (600 - x[7])
        hx[5] = x[4] - 10000 * (900 - x[8])
        hx[6] = x[3] * np.log(x[7] - 100 + 1e-8) - x[3] * np.log((600 - x[6]) + 1e-8) - x[7] + x[6] + 500
        hx[7] = x[5] * np.log(x[8] - x[6] + 1e-8) - x[5] * np.log(600) - x[8] + x[6] + 600
        return hx

    def get_cons(self, x):
        hx_list = self.get_eq_cons(x)
        gx = np.array([np.abs(hval) - self.epsilon for hval in hx_list])
        return gx

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class HeatExchangerNetworkDesignCase2Problem(Engineer):
    """
    Industrial Chemical Processes
    [x1, x2, x3, x4,..., x10, x11]
    Heat Exchanger Network Design (case 2)
    """

    name = "Heat Exchanger Network Design Case 2 (Industrial Chemical Processes)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 11
        self._n_objs = 1
        self._n_eq_cons = 9
        self._n_cons = 9
        self._bounds = np.array([(10**4, 81.9*10**4), (10**4, 113.1*10**4), (10**4, 205*10**4), (0., 5.074*10**(-2)), (0., 5.074*10**(-2)),
                        (0., 5.074*10**(-2)), (100, 200.), (100., 300.), (100., 300), (100., 300), (100, 400.)])
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = (x[0] / (120 * x[3])) ** 0.6 + (x[1] / (80 * x[4])) ** 0.6 + (x[2] / (40 * x[5])) * 0.6
        return np.array([f1])

    def get_eq_cons(self, x):
        hx = np.zeros(self.n_eq_cons)
        hx[0] = x[0] - 1e4 * (x[6] - 100)
        hx[1] = x[1] - 1e4 * (x[7] - x[6])
        hx[2] = x[2] - 1e4 * (500 - x[7])
        hx[3] = x[0] - 1e4 * (300 - x[8])
        hx[4] = x[1] - 1e4 * (400 - x[9])
        hx[5] = x[2] - 1e4 * (600 - x[10])
        hx[6] = x[3] * np.log(np.abs(x[8] - 100) + 1e-8) - x[3] * np.log(300 - x[6] + 1e-8) - x[8] - x[6] + 400
        hx[7] = x[4] * np.log(np.abs(x[9] - x[6]) + 1e-8) - x[4] * np.log(np.abs(400 - x[7]) + 1e-8) - x[9] + x[6] - x[7] + 400
        hx[8] = x[5] * np.log(np.abs(x[10] - x[7]) + 1e-8) - x[5] * np.log(100) - x[10] + x[7] + 100
        return hx

    def get_cons(self, x):
        hx_list = self.get_eq_cons(x)
        gx = np.array([np.abs(hval) - self.epsilon for hval in hx_list])
        return gx

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class HaverlyPoolingProblem(Engineer):
    """
    Industrial Chemical Processes
    [x1, x2, x3, x4,..., x9]
    Haverly's Pooling Problem
    """

    name = "Haverly's Pooling Problem (Industrial Chemical Processes)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 9
        self._n_objs = 1
        self._n_eq_cons = 4
        self._n_ineq_cons = 2
        self._n_cons = 6
        self._bounds = np.array([(0., 100.), (0., 200.), (0., 100.), (0., 100.), (0., 100.), (0., 100.), (0., 200.), (0., 100.), (0., 200.)])
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        """
        Maximum to minimum by using negative sign
        """
        f1 = -(9 * x[0] + 15 * x[1] - 6 * x[2] - 16 * x[3] - 10 * (x[4] + x[5]))
        return np.array([f1])

    def get_eq_cons(self, x):
        hx = np.zeros(self.n_eq_cons)
        hx[0] = x[6] + x[7] - x[2] - x[3]
        hx[1] = x[0] - x[6] - x[4]
        hx[2] = x[1] - x[7] - x[5]
        hx[3] = x[8] * x[6] + x[8] * x[7] - 3 * x[2] - x[3]
        return hx

    def get_ineq_cons(self, x):
        gx = np.zeros(self.n_ineq_cons)
        gx[0] = x[8] * x[6] + 2 * x[4] - 2.5 * x[0]
        gx[1] = x[8] * x[7] + 2 * x[5] - 1.5 * x[1]
        return gx

    def get_cons(self, x):
        hx_list = self.get_eq_cons(x)
        hx_values = np.array([np.abs(hval) - self.epsilon for hval in hx_list])
        gx_values = self.get_ineq_cons(x)
        return np.concatenate((hx_values, gx_values))

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class BlendingPoolingSeparationProblem(Engineer):
    """
    Industrial Chemical Processes
    [x1, x2, x3, x4,..., x37, x38]
    Blending-Pooling-Separation problem
    """

    name = "Blending-Pooling-Separation problem (Industrial Chemical Processes)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 38
        self._n_objs = 1
        self._n_eq_cons = 32
        self._n_cons = 32
        self._bounds = np.array([
            (0., 90), (0., 150), (0., 90), (0., 150), (0., 90), (0., 90), (0., 150), (0., 90), (0., 90),
            (0., 90), (0., 150), (0., 150), (0., 90), (0., 90), (0., 150), (0., 90), (0., 150), (0., 90),
            (0., 150), (0., 90), (0., 1), (0., 1.2), (0., 1), (0., 1), (0., 1), (0., 0.5), (0., 1), (0., 1),
            (0., 0.5), (0., 0.5), (0., 0.5), (0., 1.2), (0., 0.5), (0., 1.2), (0., 1.2), (0., 0.5), (0., 1.2), (0., 1.2)
        ])
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 0.9979 + 0.00432 * x[4] + 0.01517 * x[12]
        return np.array([f1])

    def get_eq_cons(self, x):
        hx = np.zeros(self.n_eq_cons)
        hx[0] = x[0] + x[1] + x[2] + x[3] - 300
        hx[1] = x[5] - x[6] - x[7]
        hx[2] = x[8] - x[9] - x[10] - x[11]
        hx[3] = x[13] - x[14] - x[15] - x[16]
        hx[4] = x[17] - x[18] - x[19]
        hx[5] = x[4] * x[20] - x[5] * x[21] - x[8] * x[22]
        hx[6] = x[4] * x[23] - x[5] * x[24] - x[8] * x[25]
        hx[7] = x[4] * x[26] - x[5] * x[27] - x[8] * x[28]
        hx[8] = x[12] * x[29] - x[13] * x[30] - x[17] * x[31]
        hx[9] = x[12] * x[32] - x[13] * x[33] - x[17] * x[34]
        hx[10] = x[12] * x[35] - x[13] * x[36] - x[17] * x[36]
        hx[11] = 1 / 3 * x[0] + x[14] * x[30] - x[4] * x[20]
        hx[12] = 1 / 3 * x[0] + x[14] * x[33] - x[4] * x[23]
        hx[13] = 1 / 3 * x[0] + x[14] * x[36] - x[4] * x[26]
        hx[14] = 1 / 3 * x[1] + x[9] * x[22] - x[12] * x[29]
        hx[15] = 1 / 3 * x[1] + x[9] * x[25] - x[12] * x[32]
        hx[16] = 1 / 3 * x[1] + x[9] * x[28] - x[12] * x[35]
        hx[17] = 1 / 3 * x[2] + x[6] * x[21] + x[10] * x[22] + x[15] * x[30] + x[18] * x[31] - 30
        hx[18] = 1 / 3 * x[2] + x[6] * x[24] + x[10] * x[25] + x[15] * x[33] + x[18] * x[34] - 50
        hx[19] = 1 / 3 * x[2] + x[6] * x[27] + x[10] * x[28] + x[15] * x[36] + x[18] * x[37] - 30
        hx[20] = x[20] + x[23] + x[26] - 1
        hx[21] = x[21] + x[24] + x[27] - 1
        hx[22] = x[22] + x[25] + x[28] - 1
        hx[23] = x[29] + x[32] + x[35] - 1
        hx[24] = x[30] + x[33] + x[36] - 1
        hx[25] = x[31] + x[34] + x[37] - 1
        hx[26] = x[24]
        hx[27] = x[27]
        hx[28] = x[22]
        hx[29] = x[36]
        hx[30] = x[31]
        hx[31] = x[34]
        return hx

    def get_cons(self, x):
        hx_list = self.get_eq_cons(x)
        hx_values = np.array([np.abs(hval) - self.epsilon for hval in hx_list])
        return hx_values

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class PropaneIsobutaneNButaneNonsharpSeparationProblem(Engineer):
    """
    Industrial Chemical Processes
    [x1, x2, x3, x4,..., x47, x48]
    Propane, Isobutane, n-Butane Nonsharp Separation
    """

    name = "Propane, Isobutane, n-Butane Nonsharp Separation (Industrial Chemical Processes)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 48
        self._n_objs = 1
        self._n_eq_cons = 38
        self._n_cons = 38
        bounds = [(0., 150), ]*20 + [(0., 1), ]*3 + [(0.85, 1), (0., 30), (0.85, 1), (0., 30), (0.85, 1), (0., 30), (0., 1),
                                        (0.85, 1), (0., 30), (0., 1), (0., 1), (0., 30), (0., 1), (0., 30)] + [(0., 1), ]*11
        self._bounds = np.array(bounds)
        self.c = np.array([[0.23947, 0.75835], [-0.0139904, -0.0661588], [0.0093514, 0.0338147],
                      [0.0077308, 0.0373349], [-0.0005719, 0.0016371], [0.0042656, 0.0288996]])
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = self.c[0, 0] + (self.c[1, 0] + self.c[2, 0] * x[23] + self.c[3, 0] * x[27] + self.c[4, 0] * x[32] + self.c[5, 0] * x[33]) * x[4] + \
             self.c[0, 1] + (self.c[1, 1] + self.c[2, 1] * x[25] + self.c[3, 1] * x[30] + self.c[4, 1] * x[37] + self.c[5, 1] * x[38]) * x[12]
        return np.array([f1])

    def get_eq_cons(self, x):
        hx = np.zeros(self.n_eq_cons)
        hx[0] = x[0] + x[1] + x[2] + x[3] - 300
        hx[1] = x[5] - x[6] - x[7]
        hx[2] = x[8] - x[9] - x[10] - x[11]
        hx[3] = x[13] - x[14] - x[15] - x[16]
        hx[4] = x[17] - x[18] - x[19]
        hx[5] = x[5] * x[20] - x[23] * x[24]
        hx[6] = x[13] * x[21] - x[25] * x[26]
        hx[7] = x[8] * x[22] - x[27] * x[28]
        hx[8] = x[17] * x[29] - x[30] * x[31]
        hx[9] = x[24] - x[4] * x[32]
        hx[10] = x[28] - x[4] * x[33]
        hx[11] = x[34] - x[4] * x[35]
        hx[12] = x[36] - x[12] * x[37]
        hx[13] = x[26] - x[12] * x[38]
        hx[14] = x[31] - x[12] * x[39]
        hx[15] = x[24] - x[5] * x[20] - x[8] * x[40]
        hx[16] = x[28] - x[5] * x[41] - x[8] * x[22]
        hx[17] = x[34] - x[5] * x[42] - x[8] * x[43]
        hx[18] = x[36] - x[13] * x[44] - x[17] * x[45]
        hx[19] = x[26] - x[13] * x[21] - x[17] * x[46]
        hx[20] = x[31] - x[13] * x[47] - x[17] * x[29]
        hx[21] = 1 / 3 * x[0] + x[14] * x[44] - x[24]
        hx[22] = 1 / 3 * x[0] + x[14] * x[21] - x[28]
        hx[23] = 1 / 3 * x[0] + x[14] * x[47] - x[34]
        hx[24] = 1 / 3 * x[1] + x[9] * x[40] - x[36]
        hx[25] = 1 / 3 * x[1] + x[9] * x[22] - x[26]
        hx[26] = 1 / 3 * x[1] + x[9] * x[43] - x[31]
        hx[27] = 1 / 3 * x[2] + x[6] * x[20] + x[10] * x[40] + x[15] * x[44] + x[18] * x[45] - 30
        hx[28] = 1 / 3 * x[2] + x[6] * x[41] + x[10] * x[22] + x[15] * x[21] + x[18] * x[46] - 50
        hx[29] = 1 / 3 * x[2] + x[6] * x[42] + x[10] * x[43] + x[15] * x[47] + x[18] * x[29] - 30
        hx[30] = x[32] + x[33] + x[35] - 1
        hx[31] = x[20] + x[41] + x[42] - 1
        hx[32] = x[40] + x[22] + x[43] - 1
        hx[33] = x[37] + x[38] + x[39] - 1
        hx[34] = x[44] + x[21] + x[47] - 1
        hx[35] = x[45] + x[46] + x[29] - 1
        hx[36] = x[42]
        hx[37] = x[45]
        return hx

    def get_cons(self, x):
        hx_list = self.get_eq_cons(x)
        hx_values = np.array([np.abs(hval) - self.epsilon for hval in hx_list])
        return hx_values

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class OptimalOperationAlkylationUnitProblem(Engineer):
    """
    Industrial Chemical Processes
    [x1, x2, x3, x4,..., x7]
    Optimal Operation of Alkylation Unit
    """

    name = "Optimal Operation of Alkylation Unit (Industrial Chemical Processes)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 7
        self._n_objs = 1
        self._n_ineq_cons = 14
        self._n_cons = 14
        self._bounds = np.array([(1000., 2000), (0., 100), (2000, 4000.), (0., 100), (0., 100), (0., 20), (0, 200.)])
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = -1.715 * x[0] - 0.035 * x[0] * x[5] - 4.0565 * x[2] - 10.0 * x[1] + 0.063 * x[2] * x[4]
        return np.array([f1])

    def get_ineq_cons(self, x):
        gx = np.zeros(self.n_ineq_cons)
        gx[0] = 0.0059553571 * x[5] ** 2 * x[0] + 0.88392857 * x[2] - 0.1175625 * x[5] * x[0] - x[0]
        gx[1] = 1.1088 * x[0] + 0.1303533 * x[0] * x[5] - 0.0066033 * x[0] * x[5] ** 2 - x[2]
        gx[2] = 6.66173269 * x[5] ** 2 + 172.39878 * x[4] - 56.596669 * x[3] - 191.20592 * x[5] - 10000
        gx[3] = 1.08702 * x[5] + 0.32175 * x[3] - 0.03762 * x[5] ** 2 - x[4] + 56.85075
        gx[4] = 0.006198 * x[6] * x[3] * x[2] + 2462.3121 * x[1] - 25.125634 * x[1] * x[3] - x[2] * x[3]
        gx[5] = 161.18996 * x[2] * x[3] + 5000.0 * x[1] * x[3] - 489510.0 * x[1] - x[2] * x[3] * x[6]
        gx[6] = 0.33 * x[6] - x[4] + 44.333333
        gx[7] = 0.022556 * x[4] - 0.007595 * x[6] - 1.0
        gx[8] = 0.00061 * x[2] - 0.0005 * x[0] - 1.0
        gx[9] = 0.819672 * x[0] - x[2] + 0.819672
        gx[10] = 24500.0 * x[1] - 250.0 * x[1] * x[3] - x[2] * x[3]
        gx[11] = 1020.4082 * x[3] * x[1] + 1.2244898 * x[2] * x[3] - 100000 * x[1]
        gx[12] = 6.25 * x[0] * x[5] + 6.25 * x[0] - 7.625 * x[2] - 100000
        gx[13] = 1.22 * x[2] - x[5] * x[0] - x[0] + 1.0
        return gx

    def get_cons(self, x):
        gx_values = self.get_ineq_cons(x)
        return gx_values

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class ReactorNetworkDesignProblem(Engineer):
    """
    Industrial Chemical Processes
    [x1, x2, x3, x4,..., x6]
    Reactor Network Design Problem
    """
    name = "Reactor Network Design (Industrial Chemical Processes)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 6
        self._n_objs = 1
        self._n_eq_cons = 4
        self._n_ineq_cons = 1
        self._n_cons = 5
        self._bounds = np.array([(0., 1), (0., 1), (0., 1), (0., 1), (0.00001, 16), (0.00001, 16)])
        self.k1 = 0.09755988
        self.k2 = 0.99 * self.k1
        self.k3 = 0.0391908
        self.k4 = 0.9 * self.k3
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = -x[3]
        return np.array([f1])

    def get_eq_cons(self, x):
        hx = np.zeros(self.n_eq_cons)
        hx[0] = x[0] + self.k1 * x[1] * x[4] - 1
        hx[1] = x[1] - x[0] + self.k2 * x[1] * x[5]
        hx[2] = x[2] + x[0] + self.k3 * x[2] * x[4] - 1
        hx[3] = x[3] - x[2] + x[1] - x[0] + self.k4 * x[3] * x[5]
        return hx

    def get_ineq_cons(self, x):
        gx = x[4] ** 0.5 + x[5] ** 0.5 - 4
        return np.array([gx, ])

    def get_cons(self, x):
        hx_list = self.get_eq_cons(x)
        hx_values = np.array([np.abs(hval) - self.epsilon for hval in hx_list])
        gx_values = self.get_ineq_cons(x)
        return np.concatenate((hx_values, gx_values))

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class ProcessSynthesis01Problem(Engineer):
    """
    Process design and synthesis problems
    [x1, x2]
    Process synthesis problem 01
    """
    name = "Process synthesis 01 problem (Process design and synthesis problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 2
        self._n_objs = 1
        self._n_ineq_cons = 2
        self._n_cons = 2
        self._bounds = np.array([(0., 1.6), (0., 1.99)])
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 2 * x[0] + x[1]
        return np.array([f1])

    def amend_position(self, x, lb=None, ub=None):
        x[1] = int(x[1])
        return x

    def get_ineq_cons(self, x):
        g1 = 1.25 - x[0] ** 2 - x[1]
        g2 = x[0] + x[1] - 1.6
        return np.array([g1, g2])

    def get_cons(self, x):
        gx_values = self.get_ineq_cons(x)
        return gx_values

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        if type(x[1]) != int:
            x = self.amend_position(x, self.lb, self.ub)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class ProcessSynthesisAndDesignProblem(Engineer):
    """
    Process design and synthesis problems
    [x1, x2, x3]
    Process synthesis and design problem
    """
    name = "Process synthesis and design problem (Process design and synthesis problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 3
        self._n_objs = 1
        self._n_ineq_cons = 1
        self._n_eq_cons = 1
        self._n_cons = 2
        self._bounds = np.array([(0.5, 1.4), (0.5, 1.4), (0., 1.99)])
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 2 * x[0] + x[1] - x[2]
        return np.array([f1])

    def amend_position(self, x, lb=None, ub=None):
        x[2] = int(x[2])
        return x

    def get_eq_cons(self, x):
        h1 = x[0] - 2 * np.exp(-x[1])
        return np.array([h1, ])

    def get_ineq_cons(self, x):
        g1 = -x[0] + x[1] + x[2]
        return np.array([g1, ])

    def get_cons(self, x):
        hx_list = self.get_eq_cons(x)
        hx_values = np.array([np.abs(hval) - self.epsilon for hval in hx_list])
        gx_values = self.get_ineq_cons(x)
        return np.concatenate((hx_values, gx_values))

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        if type(x[2]) != int:
            x = self.amend_position(x, self.lb, self.ub)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class ProcessFlowSheetingProblem(Engineer):
    """
    Process design and synthesis problems
    [x1, x2, x3]
    Process flow sheeting problem
    """
    name = "Process flow sheeting problem (Process design and synthesis problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10 ** (-4)
        self._n_dims = 3
        self._n_objs = 1
        self._n_ineq_cons = 3
        self._n_cons = 3
        self._bounds = np.array([(-2.22554, -1), (0.2, 1.0), (0., 1.99)])
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 5 * (x[0] - 0.5) ** 2 + 0.8 -0.7 * x[2]
        return np.array([f1])

    def amend_position(self, x, lb=None, ub=None):
        x[2] = int(x[2])
        return x

    def get_ineq_cons(self, x):
        g1 = -np.exp(x[0] - 0.2) - x[1]
        g2 = x[1] + 1.1 * x[2] + 1
        g3 = x[0] - x[2] - 0.2
        return np.array([g1, g2, g3])

    def get_cons(self, x):
        gx_values = self.get_ineq_cons(x)
        return gx_values

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        if type(x[2]) != int:
            x = self.amend_position(x, self.lb, self.ub)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class TwoReactorProblem(Engineer):
    """
    Process design and synthesis problems
    [x1, x2, ..., x8]
    Two-reactor problem
    """
    name = "Two-reactor problem (Process design and synthesis problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10 ** (-4)
        self._n_dims = 8
        self._n_objs = 1
        self._n_eq_cons = 5
        self._n_ineq_cons = 4
        self._n_cons = 9
        bounds = [(0., 100.), ] * 6 + [(0., 1.99), ]*2
        self._bounds = np.array(bounds)
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 7.5 * x[6] + 5.5 * x[7] + 7 * x[4] + 6 * x[5] + 5 * (x[0] + x[1])
        return np.array([f1])

    def amend_position(self, x, lb=None, ub=None):
        x[-1] = int(x[-1])
        x[-2] = int(x[-2])
        return x

    def get_eq_cons(self, x):
        h1 = x[6] + x[7] - 1
        h2 = x[2] - 0.9*(1 - np.exp(0.5*x[4]))*x[0]
        h3 = x[3] - 0.8*(1 - np.exp(0.4 * x[5])) * x[1]
        h4 = x[2] + x[3] - 10
        h5 = x[2]*x[6] + x[3]*x[7] - 10
        return np.array([h1, h2, h3, h4, h5])

    def get_ineq_cons(self, x):
        g1 = x[4] - 10*x[6]
        g2 = x[5] - 10*x[7]
        g3 = x[0] - 20*x[6]
        g4 = x[1] - 20*x[7]
        return np.array([g1, g2, g3, g4])

    def get_cons(self, x):
        hx_list = self.get_eq_cons(x)
        hx_values = np.array([np.abs(hval) - self.epsilon for hval in hx_list])
        gx_values = self.get_ineq_cons(x)
        return np.concatenate((hx_values, gx_values))

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        if type(x[-1]) != int or type(x[-2]) != int:
            x = self.amend_position(x, self.lb, self.ub)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class ProcessSynthesis02Problem(Engineer):
    """
    Process design and synthesis problems
    [x1, x2,..., x9]
    Process synthesis problem 02
    """
    name = "Process synthesis 02 problem (Process design and synthesis problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 7
        self._n_objs = 1
        self._n_ineq_cons = 9
        self._n_cons = 9
        bounds = [(0., 100.), ]*3 + [(0., 1.99), ]*4
        self._bounds = np.array(bounds)
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = (1-x[3])**2 + (1-x[4])**2 + (1-x[5])**2 - np.log(1 + x[6]) + (1 - x[0])**2 + (2-x[1])**2 + (3-x[2])**2
        return np.array([f1])

    def amend_position(self, x, lb=None, ub=None):
        x[3] = int(x[3])
        x[4] = int(x[4])
        x[5] = int(x[5])
        x[6] = int(x[6])
        return x

    def get_ineq_cons(self, x):
        g1 = np.sum(x[:5]) - 5
        g2 = x[0]**2 + x[1]**2 + x[2]**2 + x[5]**3 - 5.5
        g3 = x[0] + x[3] - 1.2
        g4 = x[1] + x[4] - 1.8
        g5 = x[2] + x[5] - 2.5
        g6 = x[0] + x[6] - 1.2
        g7 = x[4]**2 + x[1]**2 - 1.64
        g8 = x[5]**2 + x[2]**2 - 4.25
        g9 = x[4]**2 + x[2]**2 - 4.64
        return np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9])

    def get_cons(self, x):
        gx_values = self.get_ineq_cons(x)
        return gx_values

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        if type(x[3]) != int or type(x[4]) != int or type(x[5]) != int or type(x[6]) != int:
            x = self.amend_position(x, self.lb, self.ub)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class ProcessDesignProblem(Engineer):
    """
    Process design and synthesis problems
    [x1, x2,..., x5]
    Process design Problem
    """
    name = "Process design Problem (Process design and synthesis problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 5
        self._n_objs = 1
        self._n_ineq_cons = 3
        self._n_cons = 3
        self._bounds = np.array([(27., 45), (27., 45), (27., 45), (78, 102.99), (33, 45.99)])
        self.a = [85.334407, 0.0056858, 0.0006262, 0.0022053, 80.51249, 0.0071317, 0.0029955, 0.0021813, 9.300961, 0.0047026, 0.0012547, 0.0019085]
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = -5.357854 * x[0] ** 2 + 0.835689 * x[3]*x[2] - 37.29329 * x[3] + 40792.141
        return np.array([f1])

    def amend_position(self, x, lb=None, ub=None):
        x[3] = int(x[3])
        x[4] = int(x[4])
        return x

    def get_ineq_cons(self, x):
        g1 = -92 + self.a[2]*x[3]*x[1] + self.a[0] + self.a[1]*x[3]*x[2] - self.a[3]*x[3]*x[2]
        g2 = -110 + self.a[6]*x[3]*x[1] + self.a[4] + self.a[5]*x[4]*x[2] + self.a[7]*x[0]**2
        g3 = self.a[8] + self.a[10]*x[3]*x[0] + self.a[9]*x[3]*x[2] - 25 + self.a[11]*x[0]*x[1]
        return np.array([g1, g2, g3])

    def get_cons(self, x):
        gx_values = self.get_ineq_cons(x)
        return gx_values

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        if type(x[3]) != int or type(x[4]) != int:
            x = self.amend_position(x, self.lb, self.ub)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class MultiProductBatchPlantProblem(Engineer):
    """
    Process design and synthesis problems
    [x1, x2,..., x10]
    Multi-product batch plant
    """
    name = "Multi-product batch plant (Process design and synthesis problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 10
        self._n_objs = 1
        self._n_ineq_cons = 13
        self._n_cons = 13
        self._bounds = np.array([
            (1, 3.99), (1, 3.99), (1, 3.99), (250, 2500), (250, 2500), (250, 2500), (20/3, 20), (16/3, 16), (40, 700), (10, 450)
        ])
        self.S = np.array([[2, 3, 4], [4, 6, 3]])
        self.t = np.array([[8, 20, 8], [16, 4, 4]])
        self.H = 6000
        self.alp = 250
        self.beta = 0.6
        self.Q1 = 40000
        self.Q2 = 20000
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        N1, N2, N3, V1, V2, V3, TL1, TL2, B1, B2 = x
        f1 = self.alp * (N1 * V1 ** self.beta + N2 * V2 ** self.beta + N3 * V3 ** self.beta)
        return np.array([f1])

    def amend_position(self, x, lb=None, ub=None):
        x[0] = int(x[0])
        x[1] = int(x[1])
        x[2] = int(x[2])
        return x

    def get_ineq_cons(self, x):
        N1, N2, N3, V1, V2, V3, TL1, TL2, B1, B2 = x
        g1 = self.S[0, 0] * B1 - V1
        g2 = self.S[0, 1] * B1 - V2
        g3 = self.S[0, 2] * B1 - V3
        g4 = self.S[1, 0] * B2 - V1
        g5 = self.S[1, 1] * B2 - V2
        g6 = self.S[1, 2] * B2 - V3
        g7 = self.Q1*TL1/B1 + self.Q2*TL2/B2 - self.H
        g8 = self.t[0, 0] - N1 * TL1
        g9 = self.t[0, 1] - N2 * TL1
        g10 = self.t[0, 2] - N3 * TL1
        g11 = self.t[1, 0] - N1 * TL2
        g12 = self.t[1, 1] - N2 * TL2
        g13 = self.t[1, 2] - N3 * TL2
        return np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13])

    def get_cons(self, x):
        gx_values = self.get_ineq_cons(x)
        return gx_values

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        if type(x[0]) != int or type(x[1]) != int or type(x[2]) != int:
            x = self.amend_position(x, self.lb, self.ub)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class WeightMinimizationSpeedReducerProblem(Engineer):
    """
    Mechanical design problems
    [x1, x2,..., x7]
    Weight minimization of a speed reducer
    """
    name = "Weight minimization of a speed reducer (Mechanical design problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 7
        self._n_objs = 1
        self._n_ineq_cons = 11
        self._n_cons = 11
        self._bounds = np.array([
            (2.6, 3.6), (0.7, 0.8), (17, 28), (7.3, 8.3), (7.3, 8.3), (2.9, 3.9), (5, 5.5)
        ])
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 0.7854 * x[0] * x[1] ** 2 * (3.3333 * x[2] ** 2 + 14.9334 * x[2] - 43.0934) - 1.508 * x[0] * (x[5] ** 2 + x[6] ** 2) + \
         7.477 * (x[5] ** 3 + x[6] ** 3) + 0.7854 * (x[3] * x[5] ** 2 + x[4] * x[6] ** 2)
        return np.array([f1])

    def get_ineq_cons(self, x):
        g1 = -x[0] * x[1] ** 2 * x[2] + 27
        g2 = -x[0] * x[1] ** 2 * x[2] ** 2 + 397.5
        g3 = -x[1] * x[5] ** 4 * x[2] * x[3] ** (-3) + 1.93
        g4 = -x[1] * x[6] ** 4 * x[2] / x[4] ** 3 + 1.93
        g5 = 10 * x[5] ** (-3) * np.sqrt(16.91 * 10 ** 6 + (745 * x[3] / (x[1] * x[2])) ** 2) - 1100
        g6 = 10 * x[6] ** (-3) * np.sqrt(157.5 * 10 ** 6 + (745 * x[4] / (x[1] * x[2])) ** 2) - 850
        g7 = x[1] * x[2] - 40
        g8 = -x[0] / x[1] + 5
        g9 = x[0] / x[1] - 12
        g10 = 1.5 * x[5] - x[3] + 1.9
        g11 = 1.1 * x[6] - x[4] + 1.9
        return np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])

    def get_cons(self, x):
        gx_values = self.get_ineq_cons(x)
        return gx_values

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class OptimalDesignIndustrialRefrigerationSystemProblem(Engineer):
    """
    Mechanical design problems
    [x1, x2,..., x14]
    Optimal design of industrial refrigeration system
    """
    name = "Optimal design of industrial refrigeration system (Mechanical design problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 14
        self._n_objs = 1
        self._n_ineq_cons = 15
        self._n_cons = 15
        self._bounds = np.array([(0.001, 5.), ] * 14)
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 63098.88 * x[1] * x[3] * x[11] + 5441.5 * x[1] ** 2 * x[11] + 115055.5 * x[1] ** 1.664 * x[5] + 6172.27 * x[1] ** 2 * x[5] + \
         63098.88 * x[0] * x[2] * x[10] + 5441.5 * x[0] ** 2 * x[10] + 115055.5 * x[0] ** 1.664 * x[4] + 6172.27 * x[0] ** 2 * x[4] + \
         140.53 * x[0] * x[10] + 281.29 * x[2] * x[10] + 70.26 * x[0] ** 2 + 281.29 * x[0] * x[2] + 281.29 * x[2] ** 2 + \
         14437 * x[7] ** 1.8812 * x[11] ** 0.3424 * x[9] * x[13] ** (-1) * x[0] ** 2 * \
         x[6] * x[8] ** (-1) + 20470.2 * x[6] ** (2.893) * x[10] ** 0.316 * x[0] ** 2
        return np.array([f1])

    def get_ineq_cons(self, x):
        gx = np.zeros(self.n_ineq_cons)
        gx[0] = 1.524 * x[6] ** (-1) - 1
        gx[1] = 1.524 * x[7] ** (-1) - 1
        gx[2] = 0.07789 * x[0] - 2 * x[6] ** (-1) * x[8] - 1
        gx[3] = 7.05305 * x[8] ** (-1) * x[0] ** 2 * x[9] * x[7] ** (-1) * x[1] ** (-1) * x[13] ** (-1) - 1
        gx[4] = 0.0833 / x[12] * x[13] - 1
        gx[5] = 0.04771 * x[9] * x[7] ** 1.8812 * x[11] ** 0.3424 - 1
        gx[6] = 0.0488 * x[8] * x[6] ** 1.893 * x[10] ** 0.316 - 1
        gx[7] = 0.0099 * x[0] / x[2] - 1
        gx[8] = 0.0193 * x[1] / x[3] - 1
        gx[9] = 0.0298 * x[0] / x[4] - 1
        gx[10] = 47.136 * x[1] ** 0.333 / x[9] * x[11] - 1.333 * x[7] * x[12] ** 2.1195 + 62.08 * x[12] ** 2.1195 * x[7] ** 0.2 / (x[11] * x[9]) - 1
        gx[11] = 0.056 * x[1] / x[5] - 1
        gx[12] = 2 / x[8] - 1
        gx[13] = 2 / x[9] - 1
        gx[14] = x[11] / x[10] - 1
        return gx

    def get_cons(self, x):
        gx_values = self.get_ineq_cons(x)
        return gx_values

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class TensionCompressionSpringDesignProblem(Engineer):
    """
    Mechanical design problems
    [x1, x2, x3]
    Tension/compression spring design
    """
    name = "Tension/compression spring design (Mechanical design problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 3
        self._n_objs = 1
        self._n_ineq_cons = 4
        self._n_cons = 4
        self._bounds = np.array([(0.05, 2.), (0.25, 1.3), (2.0, 15.0)])
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = x[0] ** 2 * x[1] * (x[2] + 2)
        return np.array([f1])

    def get_ineq_cons(self, x):
        gx = np.zeros(self.n_ineq_cons)
        gx[0] = 1 - (x[1] ** 3 * x[2]) / (71785 * x[0] ** 4)
        gx[1] = (4 * x[1] ** 2 - x[0] * x[1]) / (12566 * (x[1] * x[0] ** 3 - x[0] ** 4)) + 1 / (5108 * x[0] ** 2) - 1
        gx[2] = 1 - 140.45 * x[0] / (x[1] ** 2 * x[2])
        gx[3] = (x[0] + x[1]) / 1.5 - 1
        return gx

    def get_cons(self, x):
        gx_values = self.get_ineq_cons(x)
        return gx_values

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class PressureVesselDesignProblem(Engineer):
    """
    Mechanical design problems
    [x1, x2, x3, x4]
    Pressure vessel design
    """
    name = "Pressure vessel design (Mechanical design problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 4
        self._n_objs = 1
        self._n_ineq_cons = 4
        self._n_cons = 4
        self._bounds = np.array([(1, 99.99), (1, 99.99), (10., 200), (10., 200)])
        self.check_penalty_func(f_penalty)

    def amend_position(self, x, lb=None, ub=None):
        x[0] = int(x[0])
        x[1] = int(x[1])
        return x

    def get_objs(self, x):
        z1 = 0.0625 * x[0]
        z2 = 0.0625 * x[1]
        f1 = 0.6224 * z1 * x[2] * x[3] + 1.7781 * z2 * x[2] ** 2 + 3.1661 * z1 ** 2 * x[3] + 19.84 * z1 ** 2 * x[2]
        return np.array([f1])

    def get_ineq_cons(self, x):
        z1 = 0.0625 * x[0]
        z2 = 0.0625 * x[1]
        gx = np.zeros(self.n_ineq_cons)
        gx[0] = 0.00954 * x[2] - z2
        gx[1] = 0.0193 * x[2] - z1
        gx[2] = x[3] - 240
        gx[3] = 1296000 - np.pi * x[2] ** 2 * x[3] - 4 / 3 * np.pi * x[2] ** 3
        return gx

    def get_cons(self, x):
        gx_values = self.get_ineq_cons(x)
        return gx_values

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        if type(x[0]) != int or type(x[1]) != int:
            x = self.amend_position(x, self.lb, self.ub)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class WeldedBeamDesignProblem(Engineer):
    """
    Mechanical design problems
    [x1, x2, x3, x4]
    Welded beam design
    """
    name = "Welded beam design (Mechanical design problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 4
        self._n_objs = 1
        self._n_ineq_cons = 5
        self._n_cons = 5
        self._bounds = np.array([(0.125, 2), (0.1, 10), (0.1, 10), (0.1, 2)])
        self.P = 6000
        self.L = 14
        self.delta_max = 0.25
        self.E = 30 * 1e6
        self.G = 12 * 1e6
        self.T_max = 13600
        self.sigma_max = 30000
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 1.10471 * x[0] ** 2 * x[1] + 0.04811 * x[2] * x[3] * (14 + x[1])
        return np.array([f1])

    def get_ineq_cons(self, x):
        Pc = 4.013 * self.E * np.sqrt(x[2] ** 2 * x[3] ** 6 / 30) / self.L ** 2 * (1 - x[2] / (2 * self.L) * np.sqrt(self.E / (4 * self.G)))
        sigma = 6 * self.P * self.L / (x[3] * x[2] ** 2)
        delta = 6 * self.P * self.L ** 3 / (self.E * x[2] ** 2 * x[3])
        J = 2 * (np.sqrt(2) * x[0] * x[1] * (x[1] ** 2 / 4 + (x[0] + x[2]) ** 2 / 4))
        R = np.sqrt(x[1] ** 2 / 4 + (x[0] + x[2]) ** 2 / 4)
        M = self.P * (self.L + x[1] / 2)
        ttt = M * R / J
        tt = self.P / (np.sqrt(2) * x[0] * x[1])
        t = np.sqrt(tt ** 2 + 2 * tt * ttt * x[1] / (2 * R) + ttt ** 2)
        gx = np.zeros(self.n_ineq_cons)
        gx[0] = t - self.T_max
        gx[1] = sigma - self.sigma_max
        gx[2] = x[0] - x[3]
        gx[3] = delta - self.delta_max
        gx[4] = self.P - Pc
        return gx

    def get_cons(self, x):
        gx_values = self.get_ineq_cons(x)
        return gx_values

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class ThreeBarTrussDesignProblem(Engineer):
    """
    Mechanical design problems
    [x1, x2]
    Three-bar truss design problem
    """
    name = "Three-bar truss design problem (Mechanical design problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 2
        self._n_objs = 1
        self._n_ineq_cons = 3
        self._n_cons = 3
        self._bounds = np.array([(0., 1.), (0., 1.)])
        self.ll = 100
        self.PP = 2
        self.xichma = 2
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = (2 * np.sqrt(2) * x[0] + x[1]) * 100
        return np.array([f1])

    def get_ineq_cons(self, x):
        gx = np.zeros(self.n_ineq_cons)
        gx[0] = x[1] / (np.sqrt(2) * x[0] ** 2 + 2 * x[0] * x[1]) * self.PP - self.xichma
        gx[0] = (np.sqrt(2) * x[0] + x[1]) / (np.sqrt(2) * x[0] ** 2 + 2 * x[0] * x[1]) * self.PP - self.xichma
        gx[2] = 1 / (np.sqrt(2) * x[1] + x[0]) * self.PP - self.xichma
        return gx

    def get_cons(self, x):
        gx_values = self.get_ineq_cons(x)
        return gx_values

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class MultipleDiskClutchBrakeDesignProblem(Engineer):
    """
    Mechanical design problems
    [x1, x2, x3, x4, x5]
    Multiple disk clutch brake design problem
    """
    name = "Multiple disk clutch brake design problem (Mechanical design problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 5
        self._n_objs = 1
        self._n_ineq_cons = 8
        self._n_cons = 8
        self._bounds = np.array([(60, 80.99), (90, 110.99), (1, 3.99), (0, 1000.99), (2, 9.99)])

        self.Mf = 3
        self.Ms = 40
        self.Iz = 55
        self.n = 250
        self.Tmax = 15
        self.s = 1.5
        self.delta = 0.5
        self.Vsrmax = 10
        self.rho = 0.0000078
        self.pmax = 1
        self.mu = 0.6
        self.Lmax = 30
        self.delR = 20
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = np.pi * (x[1] ** 2 - x[0] ** 2) * x[2] * (x[4] + 1) * self.rho
        return np.array([f1])

    def get_ineq_cons(self, x):
        Rsr = 2 / 3 * (x[1] ** 3 - x[0] ** 3) / (x[1] ** 2 * x[0] ** 2)
        Vsr = np.pi * Rsr * self.n / 30
        A = np.pi * (x[1] ** 2 - x[0] ** 2)
        Prz = x[3] / A
        w = np.pi * self.n / 30
        Mh = 2 / 3 * self.mu * x[3] * x[4] * (x[1] ** 3 - x[0] ** 3) / (x[1] ** 2 - x[0] ** 2)
        T = self.Iz * w / (Mh + self.Mf)

        gx = np.zeros(self.n_ineq_cons)
        gx[0] = Prz - self.pmax
        gx[1] = Prz * Vsr - self.pmax * self.Vsrmax
        gx[2] = x[0] + self.delR -x[1]
        gx[3] = (x[4] + 1) * (x[2] + self.delta) - self.Lmax
        gx[4] = self.s * self.Ms - Mh
        gx[5] = -T
        gx[6] = Vsr - self.Vsrmax
        gx[7] = T - self.Tmax
        return gx

    def get_cons(self, x):
        gx_values = self.get_ineq_cons(x)
        return gx_values

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class PlanetaryGearTrainDesignOptimizationProblem(Engineer):
    """
    Mechanical design problems
    [x1, x2,...,x9]
    Planetary gear train design optimization problem
    """
    name = "Planetary gear train design optimization problem (Mechanical design problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 9
        self._n_objs = 1
        self._n_ineq_cons = 10
        self._n_eq_cons = 1
        self._n_cons = 11
        self._bounds = np.array([(17, 96.99), (14, 54.99), (14, 51.99), (17, 46.99), (14, 51.99), (48, 124.99),
                                 (3, 5.99), (0, 5.99), (0, 5.99)])
        self.mind = [1.75, 2, 2.25, 2.5, 2.75, 3.0]
        self.Dmax = 220
        self.dlt22 = 0.5
        self.dlt33 = 0.5
        self.dlt55 = 0.5
        self.dlt35 = 0.5
        self.dlt34 = 0.5
        self.dlt56 = 0.5
        self.check_penalty_func(f_penalty)

    def amend_position(self, x, lb=None, ub=None):
        x = np.array(x, dtype=int)
        return x

    def get_objs(self, x):
        N1, N2, N3, N4, N5, N6, p = x[:6]
        i1 = N6 / N4
        i01 = 3.11
        i2 = N6 * (N1 * N3 + N2 * N4) / (N1 * N3 * (N6 - N4))
        i02 = 1.84
        iR = -(N2 * N6 / (N1 * N3))
        i0R = -3.11
        f1 = max([i1 - i01, i2 - i02, iR - i0R])
        return np.array([f1])

    def get_eq_cons(self, x):
        N1, N2, N3, N4, N5, N6, p = x[:6]
        hx = np.remainder(N6 - N4, p)
        return np.array([hx, ])

    def get_ineq_cons(self, x):
        N1, N2, N3, N4, N5, N6, p = x[:6]
        m1 = self.mind[x[7]]
        m2 = self.mind[x[8]]
        gx = np.zeros(self.n_ineq_cons)
        gx[0] = m2 * (N6 + 2.5) - self.Dmax
        gx[1] = m1 * (N1 + N2) + m1 * (N2 + 2) - self.Dmax
        gx[2] = m2 * (N4 + N5) + m2 * (N5 + 2) - self.Dmax
        gx[3] = np.abs(m1 * (N1 + N2) - m2 * (N6 - N3)) - m1 - m2
        gx[4] = -((N1 + N2) * np.sin(np.pi / p) - N2 - 2 - self.dlt22)
        gx[5] = -((N6 - N3) * np.sin(np.pi / p) - N3 - 2 - self.dlt33)
        gx[6] = -((N4 + N5) * np.sin(np.pi / p) - N5 - 2 - self.dlt55)
        beta = np.arccos(((N6 - N3) ** 2 + (N4 + N5) ** 2 - (N3 + N5) ** 2) / (2 * (N6 - N3) * (N4 + N5)))
        if beta == beta.real:
            gx[7] = (N3 + N5 + 2 + self.dlt35) ** 2 - ((N6 - N3) ** 2 + (N4 + N5) ** 2 - 2 * (N6 - N3) * (N4 + N5) * np.cos(2 * np.pi / p - beta))
        else:
            gx[7] = 1e6
        gx[8] = -(N6 - 2 * N3 - N4 - 4 - 2 * self.dlt34)
        gx[9] = -(N6 - N4 - 2 * N5 - 4 - 2 * self.dlt56)
        return gx

    def get_cons(self, x):
        hx_list = self.get_eq_cons(x)
        hx_values = np.array([np.abs(hval) - self.epsilon for hval in hx_list])
        gx_values = self.get_ineq_cons(x)
        return np.concatenate((hx_values, gx_values))

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        if type(x[0]) != int:
            x = self.amend_position(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class StepConePulleyProblem(Engineer):
    """
    Mechanical design problems
    [x1, x2, x3, x4, x5]
    Step-cone pulley problem
    """
    name = "Step-cone pulley problem (Mechanical design problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 5
        self._n_objs = 1
        self._n_ineq_cons = 8
        self._n_eq_cons = 3
        self._n_cons = 11
        self._bounds = np.array([(0, 60.), (0, 60.), (0, 90.), (0, 90.), (0, 90.)])
        self.N = 350
        self.N1 = 750
        self.N2 = 450
        self.N3 = 250
        self.N4 = 150
        self.rho = 7200
        self.a = 3
        self.mu = 0.35
        self.s = 1.75 * 1e6
        self.t = 8 * 1e-3
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        d1 = x[0] * 1e-3
        d2 = x[1] * 1e-3
        d3 = x[2] * 1e-3
        d4 = x[3] * 1e-3
        w = x[4] * 1e-3
        f1 = self.rho * w * np.pi / 4 * (d1 ** 2 * (1 + (self.N1 / self.N) ** 2) + d2 ** 2 * (1 + (self.N2 / self.N) ** 2) +
                                    d3 ** 2 * (1 + (self.N3 / self.N) ** 2) + d4 ** 2 * (1 + (self.N4 / self.N) ** 2))
        return np.array([f1, ])

    def get_eq_cons(self, x):
        d1 = x[0] * 1e-3
        d2 = x[1] * 1e-3
        d3 = x[2] * 1e-3
        d4 = x[3] * 1e-3
        C1 = np.pi * d1 / 2 * (1 + self.N1 / self.N) + (self.N1 / self.N - 1) ** 2 * d1 ** 2 / (4 * self.a) + 2 * self.a
        C2 = np.pi * d2 / 2 * (1 + self.N2 / self.N) + (self.N2 / self.N - 1) ** 2 * d2 ** 2 / (4 * self.a) + 2 * self.a
        C3 = np.pi * d3 / 2 * (1 + self.N3 / self.N) + (self.N3 / self.N - 1) ** 2 * d3 ** 2 / (4 * self.a) + 2 * self.a
        C4 = np.pi * d4 / 2 * (1 + self.N4 / self.N) + (self.N4 / self.N - 1) ** 2 * d4 ** 2 / (4 * self.a) + 2 * self.a
        hx = np.zeros(self.n_eq_cons)
        hx[0] = C1 - C2
        hx[1] = C1 - C3
        hx[2] = C1 - C4
        return np.array([hx, ])

    def get_ineq_cons(self, x):
        d1 = x[0] * 1e-3
        d2 = x[1] * 1e-3
        d3 = x[2] * 1e-3
        d4 = x[3] * 1e-3
        w = x[4] * 1e-3
        R1 = np.exp(self.mu * (np.pi - 2 * np.arcsin((self.N1 / self.N - 1) * d1 / (2 * self.a))))
        R2 = np.exp(self.mu * (np.pi - 2 * np.arcsin((self.N2 / self.N - 1) * d2 / (2 * self.a))))
        R3 = np.exp(self.mu * (np.pi - 2 * np.arcsin((self.N3 / self.N - 1) * d3 / (2 * self.a))))
        R4 = np.exp(self.mu * (np.pi - 2 * np.arcsin((self.N4 / self.N - 1) * d4 / (2 * self.a))))
        P1 = self.s * self.t * w * (1 - np.exp(-self.mu * (np.pi - 2 * np.arcsin((self.N1 / self.N - 1) * d1 / (2 * self.a))))) * np.pi * d1 * self.N1 / 60
        P2 = self.s * self.t * w * (1 - np.exp(-self.mu * (np.pi - 2 * np.arcsin((self.N2 / self.N - 1) * d2 / (2 * self.a))))) * np.pi * d2 * self.N2 / 60
        P3 = self.s * self.t * w * (1 - np.exp(-self.mu * (np.pi - 2 * np.arcsin((self.N3 / self.N - 1) * d3 / (2 * self.a))))) * np.pi * d3 * self.N3 / 60
        P4 = self.s * self.t * w * (1 - np.exp(-self.mu * (np.pi - 2 * np.arcsin((self.N4 / self.N - 1) * d4 / (2 * self.a))))) * np.pi * d4 * self.N4 / 60
        gx = np.zeros(self.n_ineq_cons)
        gx[0] = -R1 + 2
        gx[1] = -R2 + 2
        gx[2] = -R3 + 2
        gx[3] = -R4 + 2
        gx[4] = -P1 + (0.75 * 745.6998)
        gx[5] = -P2 + (0.75 * 745.6998)
        gx[6] = -P3 + (0.75 * 745.6998)
        gx[7] = -P4 + (0.75 * 745.6998)
        return gx

    def get_cons(self, x):
        hx_list = self.get_eq_cons(x)
        hx_values = np.array([np.abs(hval) - self.epsilon for hval in hx_list])
        gx_values = self.get_ineq_cons(x)
        return np.concatenate((hx_values, gx_values))

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class RobotGripperProblem(Engineer):
    """
    Mechanical design problems
    [x1, x2, x3, x4, x5]
    Robot gripper problem
    """
    name = "Robot gripper problem (Mechanical design problems)"

    def __init__(self, f_penalty=None):
        super().__init__()
        self.epsilon = 10**(-4)
        self._n_dims = 7
        self._n_objs = 1
        self._n_ineq_cons = 8
        self._n_eq_cons = 3
        self._n_cons = 11
        self._bounds = np.array([(0, 60.), (0, 60.), (0, 90.), (0, 90.), (0, 90.)])
        self.N = 350
        self.N1 = 750
        self.N2 = 450
        self.N3 = 250
        self.N4 = 150
        self.rho = 7200
        self.a = 3
        self.mu = 0.35
        self.s = 1.75 * 1e6
        self.t = 8 * 1e-3
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        d1 = x[0] * 1e-3
        d2 = x[1] * 1e-3
        d3 = x[2] * 1e-3
        d4 = x[3] * 1e-3
        w = x[4] * 1e-3
        f1 = self.rho * w * np.pi / 4 * (d1 ** 2 * (1 + (self.N1 / self.N) ** 2) + d2 ** 2 * (1 + (self.N2 / self.N) ** 2) +
                                    d3 ** 2 * (1 + (self.N3 / self.N) ** 2) + d4 ** 2 * (1 + (self.N4 / self.N) ** 2))
        return np.array([f1, ])

    def get_eq_cons(self, x):
        d1 = x[0] * 1e-3
        d2 = x[1] * 1e-3
        d3 = x[2] * 1e-3
        d4 = x[3] * 1e-3
        C1 = np.pi * d1 / 2 * (1 + self.N1 / self.N) + (self.N1 / self.N - 1) ** 2 * d1 ** 2 / (4 * self.a) + 2 * self.a
        C2 = np.pi * d2 / 2 * (1 + self.N2 / self.N) + (self.N2 / self.N - 1) ** 2 * d2 ** 2 / (4 * self.a) + 2 * self.a
        C3 = np.pi * d3 / 2 * (1 + self.N3 / self.N) + (self.N3 / self.N - 1) ** 2 * d3 ** 2 / (4 * self.a) + 2 * self.a
        C4 = np.pi * d4 / 2 * (1 + self.N4 / self.N) + (self.N4 / self.N - 1) ** 2 * d4 ** 2 / (4 * self.a) + 2 * self.a
        hx = np.zeros(self.n_eq_cons)
        hx[0] = C1 - C2
        hx[1] = C1 - C3
        hx[2] = C1 - C4
        return np.array([hx, ])

    def get_ineq_cons(self, x):
        d1 = x[0] * 1e-3
        d2 = x[1] * 1e-3
        d3 = x[2] * 1e-3
        d4 = x[3] * 1e-3
        w = x[4] * 1e-3
        R1 = np.exp(self.mu * (np.pi - 2 * np.arcsin((self.N1 / self.N - 1) * d1 / (2 * self.a))))
        R2 = np.exp(self.mu * (np.pi - 2 * np.arcsin((self.N2 / self.N - 1) * d2 / (2 * self.a))))
        R3 = np.exp(self.mu * (np.pi - 2 * np.arcsin((self.N3 / self.N - 1) * d3 / (2 * self.a))))
        R4 = np.exp(self.mu * (np.pi - 2 * np.arcsin((self.N4 / self.N - 1) * d4 / (2 * self.a))))
        P1 = self.s * self.t * w * (1 - np.exp(-self.mu * (np.pi - 2 * np.arcsin((self.N1 / self.N - 1) * d1 / (2 * self.a))))) * np.pi * d1 * self.N1 / 60
        P2 = self.s * self.t * w * (1 - np.exp(-self.mu * (np.pi - 2 * np.arcsin((self.N2 / self.N - 1) * d2 / (2 * self.a))))) * np.pi * d2 * self.N2 / 60
        P3 = self.s * self.t * w * (1 - np.exp(-self.mu * (np.pi - 2 * np.arcsin((self.N3 / self.N - 1) * d3 / (2 * self.a))))) * np.pi * d3 * self.N3 / 60
        P4 = self.s * self.t * w * (1 - np.exp(-self.mu * (np.pi - 2 * np.arcsin((self.N4 / self.N - 1) * d4 / (2 * self.a))))) * np.pi * d4 * self.N4 / 60
        gx = np.zeros(self.n_ineq_cons)
        gx[0] = -R1 + 2
        gx[1] = -R2 + 2
        gx[2] = -R3 + 2
        gx[3] = -R4 + 2
        gx[4] = -P1 + (0.75 * 745.6998)
        gx[5] = -P2 + (0.75 * 745.6998)
        gx[6] = -P3 + (0.75 * 745.6998)
        gx[7] = -P4 + (0.75 * 745.6998)
        return gx

    def get_cons(self, x):
        hx_list = self.get_eq_cons(x)
        hx_values = np.array([np.abs(hval) - self.epsilon for hval in hx_list])
        gx_values = self.get_ineq_cons(x)
        return np.concatenate((hx_values, gx_values))

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)



def OBJ11(x, n):
    a = x[0]
    b = x[1]
    c = x[2]
    e = x[3]
    f = x[4]
    l = x[5]
    Zmax = 99.9999
    P = 100
    if n == 1:
        def fhd(z):
            return P * b * np.sin(np.arccos((a ** 2 + (l - z) ** 2 + e ** 2 - b ** 2) / (2 * a * np.sqrt((l - z) ** 2 + e ** 2))) + \
                                  np.arccos((b ** 2 + (l - z) ** 2 + e ** 2 - a ** 2) / (2 * b * np.sqrt((l - z) ** 2 + e ** 2)))) / \
                   (2 * c * np.cos(np.arccos((a ** 2 + (l - z) ** 2 + e ** 2 - b ** 2) / (2 * a * np.sqrt((l - z) ** 2 + e ** 2)))
                                   + np.arctan(e / (l - z))))

        fhd_func = fhd
    else:
        def fhd(z):
            return -(P * b * np.sin(np.arccos((a ** 2 + (l - z) ** 2 + e ** 2 - b ** 2) / (2 * a * np.sqrt((l - z) ** 2 + e ** 2))) +
                                    np.arccos((b ** 2 + (l - z) ** 2 + e ** 2 - a ** 2) / (2 * b * np.sqrt((l - z) ** 2 + e ** 2)))) /
                     (2 * c * np.cos(np.arccos((a ** 2 + (l - z) ** 2 + e ** 2 - b ** 2) / (2 * a * np.sqrt((l - z) ** 2 + e ** 2))) +
                                     np.arctan(e / (l - z)))))
        fhd_func = fhd
    return fminbound(fhd_func, 0, Zmax)


# def robot_gripper_problem(x):  ## Not done
#     ### Robot gripper problem
#     out = constant.benchmark_function(24)
#     D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
#
#     a = x[0]
#     b = x[1]
#     c = x[2]
#     e = x[3]
#     ff = x[4]
#     l = x[5]
#     delta = x[6]
#
#     Ymin = 50
#     Ymax = 100
#     YG = 150
#     Zmax = 99.9999
#     P = 100
#     alpha_0 = np.arccos((a ** 2 + l ** 2 + e ** 2 - b ** 2) / (2 * a * np.sqrt(l ** 2 + e ** 2))) + np.arctan(e / l)
#     beta_0 = np.arccos((b ** 2 + l ** 2 + e ** 2 - a ** 2) / (2 * b * np.sqrt(l ** 2 + e ** 2))) - np.arctan(e / l)
#     alpha_m = np.arccos((a ** 2 + (l - Zmax) ** 2 + e ** 2 - b ** 2) / (2 * a * np.sqrt((l - Zmax) ** 2 + e ** 2))) + np.arctan(e / (l - Zmax))
#     beta_m = np.arccos((b ** 2 + (l - Zmax) ** 2 + e ** 2 - a ** 2) / (2 * b * np.sqrt((l - Zmax) ** 2 + e ** 2))) - np.arctan(e / (l - Zmax))
#     ## objective function
#     fx = np.zeros(D)
#     for i in range(0, D):
#         fx[i] = -1 * OBJ11(x, 2) + OBJ11(x, 1)
#     ## constraints
#     Yxmin = 2 * (e + ff + c * np.sin(beta_m + delta))
#     Yxmax = 2 * (e + ff + c * np.sin(beta_0 + delta))
#     gx = np.zeros(g)
#     gx[0] = Yxmin - Ymin
#     gx[1] = -Yxmin
#     gx[2] = Ymax - Yxmax
#     gx[3] = Yxmax - YG
#     gx[4] = l ** 2 + e ** 2 - (a + b) ** 2
#     gx[5] = b ** 2 - (a - e) ** 2 - (l - Zmax) ** 2
#     gx[6] = Zmax - l
#     hx = 0
#     tt = np.int(np.imag(fx[0]) != 0)
#     fx[tt] = 1e4
#     tt = np.int(np.imag(gx[0]) != 0)
#     gx[tt] = 1e4
#     return fx, gx, hx


# def hydro_static_thrust_bearing_design_problem(x):
#     ## Hydro-static thrust bearing design problem
#     out = constant.benchmark_function(25)
#     D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
#
#     R = x[0]
#     Ro = x[1]
#     mu = x[2]
#     Q = x[3]
#     gamma = 0.0307
#     C = 0.5
#     n = -3.55
#     C1 = 10.04
#     Ws = 101000
#     Pmax = 1000
#     delTmax = 50
#     hmin = 0.001
#     gg = 386.4
#     N = 750
#     P = (np.log10(np.log10(8.122 * 1e6 * mu + 0.8)) - C1) / n
#     delT = 2 * (10 ** P - 560)
#     Ef = 9336 * Q * gamma * C * delT
#     h = (2 * np.pi * N / 60) ** 2 * 2 * np.pi * mu / Ef * (R ** 4 / 4 - Ro ** 4 / 4) - 1e-5
#     Po = (6 * mu * Q / (np.pi * h ** 3)) * np.log(R / Ro)
#     W = np.pi * Po / 2 * (R ** 2 - Ro ** 2) / (np.log(R / Ro) - 1e-5)
#     ##  objective function
#     fx = (Q * Po / 0.7 + Ef) / 12
#     ##  constraints
#     gx = np.zeros(g)
#     hx = 0
#     gx[0] = Ws - W
#     gx[1] = Po - Pmax
#     gx[2] = delT - delTmax
#     gx[3] = hmin - h
#     gx[4] = Ro - R
#     gx[5] = gamma / (gg * Po) * (Q / (2 * np.pi * R * h)) - 0.001
#     gx[6] = W / (np.pi * (R ** 2 - Ro ** 2) + 1e-5) - 5000
#     return fx, gx, hx


p1 = HENDC1P = HeatExchangerNetworkDesignCase1Problem
p2 = HENDC2P = HeatExchangerNetworkDesignCase2Problem
p3 = HPP = HaverlyPoolingProblem
p4 = BPSP = BlendingPoolingSeparationProblem
p5 = PINBNSP = PropaneIsobutaneNButaneNonsharpSeparationProblem
p6 = OOAUP = OptimalOperationAlkylationUnitProblem
p7 = RNDP = ReactorNetworkDesignProblem
p8 = PS01P = ProcessSynthesis01Problem
p9 = PSADP = ProcessSynthesisAndDesignProblem
p10 = PFSP = ProcessFlowSheetingProblem
p11 = TRP = TwoReactorProblem
p12 = PS02P = ProcessSynthesis02Problem
p13 = PDP = ProcessDesignProblem
p14 = MPBP = MultiProductBatchPlantProblem
p15 = WMSRP = WeightMinimizationSpeedReducerProblem
p16 = ODIRSP = OptimalDesignIndustrialRefrigerationSystemProblem
p17 = CCSDP = TensionCompressionSpringDesignProblem
p18 = PVDP = PressureVesselDesignProblem
p19 = WBDP = WeldedBeamDesignProblem
p20 = TBTDP = ThreeBarTrussDesignProblem
p21 = MDCBDP = MultipleDiskClutchBrakeDesignProblem
p22 = PGTDOP = PlanetaryGearTrainDesignOptimizationProblem
p23 = SCPP = StepConePulleyProblem

# p24 = RGP = robot_gripper_problem
# p25 = HSTBDP = hydro_static_thrust_bearing_design_problem
