#!/usr/bin/env python
# Created by "Thieu" at 00:36, 30/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import SMA, FloatVar
from enoppy.paper_based import moeosma_2023

prob = moeosma_2023.SpeedReducerProblem()

problem = {
    "bounds": FloatVar(lb=prob.lb, ub=prob.ub),
    "obj_func": prob.evaluate,
    "minmax": "min",
    "obj_weights": [0.5, 0.5]
}

## Run the algorithm
model = SMA.OriginalSMA(epoch=100, pop_size=50, pr=0.03)
g_best = model.solve(problem)
print(f"Best solution: {g_best.solution}, Best fitness: {g_best.target.fitness}")
