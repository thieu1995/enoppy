#!/usr/bin/env python
# Created by "Thieu" at 11:23, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
#
# Examples:
# >>> from enoppy.paper_based import moeosma_2023
#
# >>> p1 = moeosma_2023.SpeedReducerProblem()
#
# >>> print(p1.get_paras())
# >>> print(p1.bounds)
# >>> print(p1.n_dims)
# >>> print(p1.n_objs)
# >>> print(p1.n_cons)
# >>> print(p1.lb)
# >>> print(p1.ub)
#
# >>> x0 = p1.create_solution()
# >>> print(x0)
# >>> x0 = p1.amend_position(x0)
#
# >>> print(p1.get_objs(x0)) # Get all objectives
# >>> print(p1.get_cons(x0)) # Get all constraints
# >>> print(p1.evaluate(x0)) # Evaluate the x0 (calculate fitness value with objectives and constraints combination)

__version__ = "0.1.1"

import inspect
import re
from .utils import *
