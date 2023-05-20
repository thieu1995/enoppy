===========
Quick Start
===========

------------
Installation
------------

* Install the `current PyPI release <https://pypi.python.org/pypi/enoppy />`_::

   $ pip install enoppy==0.1.0


* Install directly from source code::

   $ git clone https://github.com/thieu1995/enoppy.git
   $ cd enoppy
   $ python setup.py install


---------------
Lib's structure
---------------

Current's structure::

   docs
   examples
   enoppy
      paper_based
         pdo_2022.py
         rwco_2020.py
      problem_based
         chemical.py
         mechanism.py
      utils
         operator.py
         validator.py
         visualize.py
      __init__.py
      engineer.py
   README.md
   setup.py


-----
Usage
-----

After installation, you can import ENOPPY as any other Python module::

   $ python
   >>> import enoppy
   >>> enoppy.__version__


Let's go through some examples.

--------
Examples
--------

How to get the problem and use it::

   from enoppy.paper_based.moeosma_2023 import SpeedReducerProblem
   # SRP = SpeedReducerProblem
   # SP = SpringProblem
   # HTBP = HydrostaticThrustBearingProblem
   # VPP = VibratingPlatformProblem
   # CSP = CarSideImpactProblem
   # WRMP = WaterResourceManagementProblem
   # BCP = BulkCarriersProblem
   # MPBPP = MultiProductBatchPlantProblem

   srp_prob = SpeedReducerProblem()
   print("Lower bound for this problem: ", srp_prob.lb)
   print("Upper bound for this problem: ", srp_prob.ub)
   x0 = srp_prob.create_solution()
   print("Get the objective values of x0: ", srp_prob.get_objs(x0))
   print("Get the constraint values of x0: ", srp_prob.get_cons(x0))
   print("Evaluate with default penalty function: ", srp_prob.evaluate(x0))


Design my own penalty function::

   import numpy as np
   from enoppy.paper_based.moeosma_2023 import HTBP
   # HTBP = HydrostaticThrustBearingProblem

   def penalty_func(list_objectives, list_constraints):
      list_constraints[list_constraints < 0] = 0
      return np.sum(list_objectives) + 1e5 * np.sum(list_constraints**2)

   htbp_prob = HTBP(f_penalty=penalty_func)
   print("Lower bound for this problem: ", htbp_prob.lb)
   print("Upper bound for this problem: ", htbp_prob.ub)
   x0 = htbp_prob.create_solution()
   print("Get the objective values of x0: ", htbp_prob.get_objs(x0))
   print("Get the constraint values of x0: ", htbp_prob.get_cons(x0))
   print("Evaluate with default penalty function: ", htbp_prob.evaluate(x0))


For more usage examples please look at [examples](/examples) folder.


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
