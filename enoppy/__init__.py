#!/usr/bin/env python
# Created by "Thieu" at 11:23, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
#
# Examples:
# >>> from enoppy.cec_based.cec2014 import F12014
# >>>
# >>> f1 = F12014(ndim=30, f_bias=100)
# >>>
# >>> lower_bound = f1.lb                       # Numpy array
# >>> lower_bound_as_list = f1.lb.to_list()     # Python list
# >>> upper_bound = f1.ub
# >>> fitness = f1.evaluate
# >>>
# >>> solution = np.random.uniform(0, 1, 30)
# >>> print(f1.evaluate(solution))
# >>> print(fitness.evaluate(solution))
# >>>
# >>> print(f1.get_paras())         # Print the parameters of function if has
# >>>
# >>> Plot 2d or plot 3d contours
# >>> Warning ! Only working on 2d functions objects !
# >>> Warning !! change n_space to reduce the computing time
# >>>
# >>> import enoppy
# >>> f2 = enoppy.cec_based.F22005(ndim=2)
# >>> enoppy.plot_2d(f22005, n_space=1000, ax=None)
# >>> enoppy.plot_3d(f22005, n_space=1000, ax=None)

__version__ = "0.1.0"

import inspect
import re
from .utils import *
