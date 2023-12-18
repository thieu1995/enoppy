#!/usr/bin/env python
# Created by "Thieu" at 20:59, 29/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from enoppy.paper_based import moeosma_2023

p1 = moeosma_2023.SpeedReducerProblem()

print(p1.get_paras())
print(p1.bounds)
print(p1.n_dims)
print(p1.n_objs)
print(p1.n_cons)
print(p1.lb)
print(p1.ub)

x0 = p1.create_solution()
print(x0)
x0 = p1.amend_position(x0)

print(p1.get_objs(x0))
print(p1.get_cons(x0))
print(p1.evaluate(x0))
