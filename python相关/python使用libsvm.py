#! /usr/lib/python3
# -*- coding: utf-8 -*-
#http://stackoverflow.com/questions/4214868/an-example-using-python-bindings-for-svm-library-libsvm

   在libsvm-3.16的python文件夹下主要包括了两个文件svm.py和svmutil.py。
    svmutil.py接口主要包括了high-level的函数，这些函数的使用和LIBSVM的MATLAB接口大体类似
    svmutil中主要包含了以下几个函数：
    svm_train()        : train an SVM model
    svm_predict()      : predict testing data
    svm_read_problem() : read the data from a LIBSVM-format file.
    svm_load_model()   : load a LIBSVM model.
    svm_save_model()   : save model to a file.
    evaluations()      : evaluate prediction results.

This example demonstrates a one-class SVM classifier; it's about as simple as possible while still showing the complete LIBSVM workflow.

Step 1: Import NumPy & LIBSVM

  import numpy as NP
    from svm import *
Step 2: Generate synthetic data: for this example, 500 points within a given boundary (note: quite a few real data sets are are provided on the LIBSVM website)

Data = NP.random.randint(-5, 5, 1000).reshape(500, 2)
Step 3: Now, choose some non-linear decision boundary for a one-class classifier:

rx = [ (x**2 + y**2) < 9 and 1 or 0 for (x, y) in Data ]
Step 4: Next, arbitrarily partition the data w/r/t this decision boundary:

Class I: those that lie on or within an arbitrary circle

Class II: all points outside the decision boundary (circle)

The SVM Model Building begins here; all steps before this one were just to prepare some synthetic data.

Step 5: Construct the problem description by calling svm_problem, passing in the decision boundary function and the data, then bind this result to a variable.

px = svm_problem(rx, Data)
Step 6: Select a kernel function for the non-linear mapping

For this exmaple, i chose RBF (radial basis function) as my kernel function

pm = svm_parameter(kernel_type=RBF)
Step 7: Train the classifier, by calling svm_model, passing in the problem description (px) & kernel (pm)

v = svm_model(px, pm)
Step 8: Finally, test the trained classifier by calling predict on the trained model object ('v')

v.predict([3, 1])
# returns the class label (either '1' or '0')
For the example above, I used version 3.0 of LIBSVM (the current stable release at the time this answer was posted).

Finally, w/r/t the part of your question regarding the choice of kernel function, Support Vector Machines are not specific to a particular kernel function--e.g., i could have chosen a different kernel (gaussian, polynomial, etc.).

LIBSVM includes all of the most commonly used kernel functions--which is a big help because you can see all plausible alternatives and to select one for use in your model, is just a matter of calling svm_parameter and passing in a value for *kernel_type* (a three-letter abbreviation for the chosen kernel).

Finally, the kernel function you choose for training must match the kernel function used against the testing data.

