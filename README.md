
<p align="center"><img src=".github/img/logo.png" alt="ENOPPY" title="ENOPPY"/></p>

---



[![GitHub release](https://img.shields.io/badge/release-0.1.0-yellow.svg)](https://github.com/thieu1995/enoppy/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/enoppy) 
[![PyPI version](https://badge.fury.io/py/enoppy.svg)](https://badge.fury.io/py/enoppy)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/enoppy.svg)
![PyPI - Status](https://img.shields.io/pypi/status/enoppy.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/enoppy.svg)
[![Downloads](https://pepy.tech/badge/enoppy)](https://pepy.tech/project/enoppy)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/enoppy/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/enoppy/actions/workflows/publish-package.yaml)
![GitHub Release Date](https://img.shields.io/github/release-date/thieu1995/enoppy.svg)
[![Documentation Status](https://readthedocs.org/projects/enoppy/badge/?version=latest)](https://enoppy.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
[![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/thieu1995/enoppy.svg)](http://isitmaintained.com/project/thieu1995/enoppy "Average time to resolve an issue")
[![Percentage of issues still open](http://isitmaintained.com/badge/open/thieu1995/enoppy.svg)](http://isitmaintained.com/project/thieu1995/enoppy "Percentage of issues still open")
![GitHub contributors](https://img.shields.io/github/contributors/thieu1995/enoppy.svg)
[![GitTutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3711948.svg)](https://doi.org/10.5281/zenodo.3711948)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


ENOPPY (ENgineering Optimization Problems in PYthon) is the largest python library for real-world engineering 
optimization problems. Contains all engineering problems from CEC competition functions from 2005, 2008, 2010, 2013, 
2014, 2015, 2017, 2019, 2020, 2021, 2022. 

* **Free software:** GNU General Public License (GPL) V3 license
* **Total problems**: > 50 problems
* **Documentation:** https://enoppy.readthedocs.io/en/latest/
* **Python versions:** 3.6.x, 3.7.x, 3.8.x, 3.9.x, 3.10.x
* **Dependencies:** numpy, scipy, pandas, matplotlib




# Installation

### Install with pip

Install the [current PyPI release](https://pypi.python.org/pypi/enoppy):
```sh 
$ pip install enoppy==0.1.0
```

Or install the development version from GitHub:

```bash
pip install git+https://github.com/thieu1995/enoppy
```

### Install from source

In case you want to install directly from the source code, use:
```sh 
$ git clone https://github.com/thieu1995/enoppy.git
$ cd enoppy
$ python setup.py install
```


## Lib's structure

```code 

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
```


# Usage

After installation, you can import ENOPPY as any other Python module:

```sh
$ python
>>> import enoppy
>>> enoppy.__version__
```

Let's go through some examples.


### Examples

How to get the function and use it

#### 1st way

```python
from enoppy.cec_based.cec2014 import F12014

func = F12014(ndim=30)
func.evaluate(func.create_solution())

## or

from enoppy.cec_based import F102014

func = F102014(ndim=50)
func.evaluate(func.create_solution())
```


#### 2nd way

```python

import enoppy

funcs = enoppy.get_functions_by_classname("F12014")
func = funcs[0](ndim=10)
func.evaluate(func.create_solution())

## or

all_funcs_2014 = enoppy.get_functions_based_classname("2014")
print(all_funcs_2014)

```

For more usage examples please look at [examples](/examples) folder.



### Get helps (questions, problems)

* Official source code repo: https://github.com/thieu1995/enoppy
* Official document: https://enoppy.readthedocs.io/
* Download releases: https://pypi.org/project/enoppy/
* Issue tracker: https://github.com/thieu1995/enoppy/issues
* Notable changes log: https://github.com/thieu1995/enoppy/blob/master/ChangeLog.md
* Examples with different meapy version: https://github.com/thieu1995/enoppy/blob/master/examples.md

* This project also related to our another projects which are "meta-heuristics" and "neural-network", check it here
    * https://github.com/thieu1995/mealpy
    * https://github.com/thieu1995/metaheuristics
    * https://github.com/thieu1995/permetrics
    * https://github.com/aiir-team


**Want to have an instant assistant? Join our telegram community at [link](https://t.me/+fRVCJGuGJg1mNDg1)**
We share lots of information, questions, and answers there. You will get more support and knowledge there.


### Cite Us

If you are using enoppy in your project, we would appreciate citations:

```code 
@software{thieu_nguyen_2020_3711682,
  author       = {Nguyen Van Thieu},
  title        = {ENOPPY: A Python Library for Engineering Optimization Problems},
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3620960},
  url          = {https://doi.org/10.5281/zenodo.3620960.}
}
```


### References 

```code
1. http://benchmarkfcns.xyz/fcns
2. https://en.wikipedia.org/wiki/Test_functions_for_optimization
3. https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/
4. http://www.sfu.ca/~ssurjano/optimization.html
5. A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)
6. Problem Definitions and Evaluation Criteria for the CEC 2014 Special Session and Competition on Single Objective Real-Parameter Numerical Optimization 
```
