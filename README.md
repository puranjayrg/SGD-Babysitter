# SGDBabysitter

[![Build Status](https://travis-ci.com/JayCata/SGDBabysitter.jl.svg?branch=master)](https://travis-ci.com/JayCata/SGDBabysitter.jl)
[![CodeCov](https://codecov.io/gh/JayCata/SGDBabysitter.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JayCata/SGDBabysitter.jl)

In neural network optimization problems, the **Stochastic Gradient Descent (SGD)** optimization algorithm is widely popular for its performance, low computational cost, as well as its desirable property of implicit regularization. It is often the optimizer of choice - if not in its base version, then as the basis for other popular optimizers that build upon it such as Adam, among others.  

However, SGD can often face convergence issues and be infamously **difficult to converge to minima**, requiring **constant manual fine-tuning**, or 'babysitting' of optimizer hyperparameters to ensure convergence to useful minima.    

In this project, we **develop an automatic 'babysitter' for SGD**  - an algorithm that dynamically adjusts the learning rate and batch size of SGD in **real-time during training**. To fine-tune the hyperparameters of the optimizer, the algorithm considers the change in validation error over time and the angle between gradients. More detail on the logic behind the algorithm can be found in the [full report](doc/Project%20Write%20Up.pdf). 

We chose to use Julia for this project to practice another language popular for deep learning tasks that we are less familiar with. However, the code can easily be ported to Python.

---
## Contents

+ The detailed report for the project going over the multiple test cases, as well as the details of our optimization algorithm for the babysitter can be found in the `doc` folder or by clicking [here.](doc/Project%20Write%20Up.pdf)
+ The algorithm for SGDBabysitter can be found in the `test` folder in `sgdfunc.jl` or by clicking [here.](test/sgdfunc.jl)
+ The skeletal code for the neural network can be found in the `test` folder or by clicking [here.](test/NeuralNet.jl)
+ Our code for testing on the simulated data sets can be found in the `test` folder or by clicking [here.](test/DataImp.jl)
+ Our code for testing on the real data sets can be found in the `test` folder or by clicking [here](test/testscript.jl), [here](test/otherdata_test.jl), and [here](test/conctest.jl) for the MNIST, Parkinson's, and concrete data sets respectively.
