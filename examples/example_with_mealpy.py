#!/usr/bin/env python
# Created by "Thieu" at 00:36, 30/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.bio_based import SMA
from enoppy.paper_based import moeosma_2023

prob = moeosma_2023.SpeedReducerProblem()

problem_dict1 = {
    "fit_func": prob.evaluate,
    "lb": prob.lb,
    "ub": prob.ub,
    "minmax": "min",
    "log_to": None,
}

## Run the algorithm
model = SMA.BaseSMA(problem_dict1, epoch=100, pop_size=50, pr=0.03)
best_position, best_fitness = model.solve()
print(f"Best solution: {best_position}, Best fitness: {best_fitness}")
