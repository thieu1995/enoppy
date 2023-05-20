#!/usr/bin/env python
# Created by "Thieu" at 09:42, 20/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from scipy import optimize
from enoppy.paper_based import ihaoavoa_2022


def penalty_func(list_objs, list_cons):
    list_cons[list_cons < 0] = 0
    return np.sum(list_objs) + 1e5*np.sum(list_cons**2)


prob = ihaoavoa_2022.TensionCompressionSpringProblem(penalty_func)

# Define the initial guess for the variables
x0 = np.random.uniform(prob.lb, prob.ub)

# Solve the modified optimization problem
result = optimize.minimize(prob.evaluate, x0=x0, bounds=prob.bounds)

# Print the optimal solution
print("Optimal solution:")
print("x =", result.x)
print("Objective value:", prob.evaluate(result.x))
print("List cons:", prob.get_cons(result.x))
print("list obj:", prob.get_objs(result.x))
