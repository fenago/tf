How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras
================================================================================


Hyperparameter optimization is a big part of deep learning.

The reason is that neural networks are notoriously difficult to
configure and there are a lot of parameters that need to be set. On top
of that, individual models can be very slow to train.

In this lab you will discover how you can use the grid search
capability from the scikit-learn python machine learning library to tune
the hyperparameters of Keras deep learning models.

After completing this lab, you will know:

-   How to wrap Keras models for use in scikit-learn and how to use grid
    search.
-   How to grid search common neural network parameters such as learning
    rate, dropout rate, epochs and number of neurons.
-   How to define your own hyperparameter tuning experiments on your own
    projects.

Let's get started.


Overview
--------

In this lab, I want to show you both how you can use the scikit-learn
grid search capability and give you a suite of examples that you can
copy-and-paste into your own project as a starting point.

Below is a list of the topics we are going to cover:

1.  How to use Keras models in scikit-learn.
2.  How to use grid search in scikit-learn.
3.  How to tune batch size and training epochs.
4.  How to tune optimization algorithms.
5.  How to tune learning rate and momentum.
6.  How to tune network weight initialization.
7.  How to tune activation functions.
8.  How to tune dropout regularization.
9.  How to tune the number of neurons in the hidden layer.



How to Use Keras Models in scikit-learn
---------------------------------------

Keras models can be used in scikit-learn by wrapping them with the
**KerasClassifier** or **KerasRegressor** class.

To use these wrappers you must define a function that creates and
returns your Keras sequential model, then pass this function to the
**build\_fn** argument when constructing the **KerasClassifier** class.

For example:

```
def create_model():
	...
	return model

model = KerasClassifier(build_fn=create_model)
```


The constructor for the **KerasClassifier** class can take default
arguments that are passed on to the calls to **model.fit()**, such as
the number of epochs and the [batch size].

For example:

```
def create_model():
	...
	return model
 
model = KerasClassifier(build_fn=create_model, epochs=10)
```


The constructor for the **KerasClassifier** class can also take new
arguments that can be passed to your custom **create\_model()**
function. These new arguments must also be defined in the signature of
your **create\_model()** function with default parameters.

For example:

```
def create_model(dropout_rate=0.0):
	...
	return model

model = KerasClassifier(build_fn=create_model, dropout_rate=0.2)
```

How to Use Grid Search in scikit-learn
--------------------------------------

Grid search is a model hyperparameter optimization technique.

In scikit-learn this technique is provided in the **GridSearchCV**
class.

When constructing this class you must provide a dictionary of
hyperparameters to evaluate in the **param\_grid** argument. This is a
map of the model parameter name and an array of values to try.

By default, accuracy is the score that is optimized, but other scores
can be specified in the **score** argument of the **GridSearchCV**
constructor.

By default, the grid search will only use one thread. By setting the
**n\_jobs** argument in the **GridSearchCV** constructor to -1, the
process will use all cores on your machine. Depending on your Keras
backend, this may interfere with the main neural network training
process.

The **GridSearchCV** process will then construct and evaluate one model
for each combination of parameters. Cross validation is used to evaluate
each individual model and the default of 3-fold cross validation is
used, although this can be overridden by specifying the **cv** argument
to the **GridSearchCV** constructor.

Below is an example of defining a simple grid search:


```
param_grid = dict(epochs=[10,20,30])
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
```

Once completed, you can access the outcome of the grid search in the
result object returned from **grid.fit()**. The **best\_score\_** member
provides access to the best score observed during the optimization
procedure and the **best\_params\_** describes the combination of
parameters that achieved the best results.


Problem Description
-------------------

Now that we know how to use Keras models with scikit-learn and how to
use grid search in scikit-learn, let's look at a bunch of examples.

All examples will be demonstrated on a small standard machine learning
dataset called the [Pima Indians onset of diabetes classification
dataset]
This is a small dataset with all numerical attributes that is easy to
work with.

1.  [Download the dataset](https://github.com/fenago/tf/blob/main/deep-learning/data/pima-indians-diabetes.csv)


As we proceed through the examples in this lab, we will aggregate the
best parameters. This is not the best way to grid search because
parameters can interact, but it is good for demonstration purposes.

### Note on Parallelizing Grid Search

All examples are configured to use parallelism (**n\_jobs=-1**).

If you get an error like the one below:


```
INFO (theano.gof.compilelock): Waiting for existing lock by process '55614' (I am process '55613')
INFO (theano.gof.compilelock): To manually release the lock, delete ...
```

Kill the process and change the code to not perform the grid search in
parallel, set **n\_jobs=1**.


How to Tune Batch Size and Number of Epochs
-------------------------------------------

In this first simple example, we look at tuning the batch size and
number of epochs used when fitting the network.

The batch size in [iterative gradient
descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Iterative_method)
is the number of patterns shown to the network before the weights are
updated. It is also an optimization in the training of the network,
defining how many patterns to read at a time and keep in memory.

The number of epochs is the number of times that the entire training
dataset is shown to the network during training. Some networks are
sensitive to the batch size, such as LSTM recurrent neural networks and
Convolutional Neural Networks.

Here we will evaluate a suite of different mini batch sizes from 10 to
100 in steps of 20.

The full code listing is provided below.

```
# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

**Note**: Your [results may vary] given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

Running this example produces the following output.

```
Best: 0.686198 using {'epochs': 100, 'batch_size': 20}
0.348958 (0.024774) with: {'epochs': 10, 'batch_size': 10}
0.348958 (0.024774) with: {'epochs': 50, 'batch_size': 10}
0.466146 (0.149269) with: {'epochs': 100, 'batch_size': 10}
0.647135 (0.021236) with: {'epochs': 10, 'batch_size': 20}
0.660156 (0.014616) with: {'epochs': 50, 'batch_size': 20}
0.686198 (0.024774) with: {'epochs': 100, 'batch_size': 20}
0.489583 (0.075566) with: {'epochs': 10, 'batch_size': 40}
0.652344 (0.019918) with: {'epochs': 50, 'batch_size': 40}
0.654948 (0.027866) with: {'epochs': 100, 'batch_size': 40}
0.518229 (0.032264) with: {'epochs': 10, 'batch_size': 60}
0.605469 (0.052213) with: {'epochs': 50, 'batch_size': 60}
0.665365 (0.004872) with: {'epochs': 100, 'batch_size': 60}
0.537760 (0.143537) with: {'epochs': 10, 'batch_size': 80}
0.591146 (0.094954) with: {'epochs': 50, 'batch_size': 80}
0.658854 (0.054904) with: {'epochs': 100, 'batch_size': 80}
0.402344 (0.107735) with: {'epochs': 10, 'batch_size': 100}
0.652344 (0.033299) with: {'epochs': 50, 'batch_size': 100}
0.542969 (0.157934) with: {'epochs': 100, 'batch_size': 100}
```

We can see that the batch size of 20 and 100 epochs achieved the best
result of about 68% accuracy.


How to Tune the Training Optimization Algorithm
-----------------------------------------------

Keras offers a suite of different state-of-the-art optimization
algorithms.

In this example, we tune the optimization algorithm used to train the
network, each with default parameters.

This is an odd example, because often you will choose one approach a
priori and instead focus on tuning its parameters on your problem (e.g.
see the next example).

Here we will evaluate the [suite of optimization algorithms supported by
the Keras API](http://keras.io/optimizers/).

The full code listing is provided below.

```
# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model(optimizer='adam'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

**Note**: Your [results may vary] given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

Running this example produces the following output.

```
Best: 0.704427 using {'optimizer': 'Adam'}
0.348958 (0.024774) with: {'optimizer': 'SGD'}
0.348958 (0.024774) with: {'optimizer': 'RMSprop'}
0.471354 (0.156586) with: {'optimizer': 'Adagrad'}
0.669271 (0.029635) with: {'optimizer': 'Adadelta'}
0.704427 (0.031466) with: {'optimizer': 'Adam'}
0.682292 (0.016367) with: {'optimizer': 'Adamax'}
0.703125 (0.003189) with: {'optimizer': 'Nadam'}
```

The results suggest that the ADAM optimization algorithm is the best
with a score of about 70% accuracy.


How to Tune Learning Rate and Momentum
--------------------------------------

It is common to pre-select an optimization algorithm to train your
network and tune its parameters.

By far the most common optimization algorithm is plain old [Stochastic
Gradient Descent](http://keras.io/optimizers/#sgd) (SGD) because it is
so well understood. In this example, we will look at optimizing the SGD
learning rate and momentum parameters.

Learning rate controls how much to update the weight at the end of each
batch and the momentum controls how much to let the previous update
influence the current weight update.

We will try a suite of small standard learning rates and a momentum
values from 0.2 to 0.8 in steps of 0.2, as well as 0.9 (because it can
be a popular value in practice).

Generally, it is a good idea to also include the number of epochs in an
optimization like this as there is a dependency between the amount of
learning per batch (learning rate), the number of updates per epoch
(batch size) and the number of epochs.

The full code listing is provided below.

```
# Use scikit-learn to grid search the learning rate and momentum
import numpy
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD
# Function to create model, required for KerasClassifier
def create_model(learn_rate=0.01, momentum=0):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	optimizer = SGD(lr=learn_rate, momentum=momentum)
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

**Note**: Your [results may
vary]
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

Running this example produces the following output.

```
Best: 0.680990 using {'learn_rate': 0.01, 'momentum': 0.0}
0.348958 (0.024774) with: {'learn_rate': 0.001, 'momentum': 0.0}
0.348958 (0.024774) with: {'learn_rate': 0.001, 'momentum': 0.2}
0.467448 (0.151098) with: {'learn_rate': 0.001, 'momentum': 0.4}
0.662760 (0.012075) with: {'learn_rate': 0.001, 'momentum': 0.6}
0.669271 (0.030647) with: {'learn_rate': 0.001, 'momentum': 0.8}
0.666667 (0.035564) with: {'learn_rate': 0.001, 'momentum': 0.9}
0.680990 (0.024360) with: {'learn_rate': 0.01, 'momentum': 0.0}
0.677083 (0.026557) with: {'learn_rate': 0.01, 'momentum': 0.2}
0.427083 (0.134575) with: {'learn_rate': 0.01, 'momentum': 0.4}
0.427083 (0.134575) with: {'learn_rate': 0.01, 'momentum': 0.6}
0.544271 (0.146518) with: {'learn_rate': 0.01, 'momentum': 0.8}
0.651042 (0.024774) with: {'learn_rate': 0.01, 'momentum': 0.9}
0.651042 (0.024774) with: {'learn_rate': 0.1, 'momentum': 0.0}
0.651042 (0.024774) with: {'learn_rate': 0.1, 'momentum': 0.2}
0.572917 (0.134575) with: {'learn_rate': 0.1, 'momentum': 0.4}
0.572917 (0.134575) with: {'learn_rate': 0.1, 'momentum': 0.6}
0.651042 (0.024774) with: {'learn_rate': 0.1, 'momentum': 0.8}
0.651042 (0.024774) with: {'learn_rate': 0.1, 'momentum': 0.9}
0.533854 (0.149269) with: {'learn_rate': 0.2, 'momentum': 0.0}
0.427083 (0.134575) with: {'learn_rate': 0.2, 'momentum': 0.2}
0.427083 (0.134575) with: {'learn_rate': 0.2, 'momentum': 0.4}
0.651042 (0.024774) with: {'learn_rate': 0.2, 'momentum': 0.6}
0.651042 (0.024774) with: {'learn_rate': 0.2, 'momentum': 0.8}
0.651042 (0.024774) with: {'learn_rate': 0.2, 'momentum': 0.9}
0.455729 (0.146518) with: {'learn_rate': 0.3, 'momentum': 0.0}
0.455729 (0.146518) with: {'learn_rate': 0.3, 'momentum': 0.2}
0.455729 (0.146518) with: {'learn_rate': 0.3, 'momentum': 0.4}
0.348958 (0.024774) with: {'learn_rate': 0.3, 'momentum': 0.6}
0.348958 (0.024774) with: {'learn_rate': 0.3, 'momentum': 0.8}
0.348958 (0.024774) with: {'learn_rate': 0.3, 'momentum': 0.9}
```

We can see that relatively SGD is not very good on this problem,
nevertheless best results were achieved using a learning rate of 0.01
and a momentum of 0.0 with an accuracy of about 68%.


How to Tune Network Weight Initialization
-----------------------------------------

Neural network weight initialization used to be simple: use small random
values.

Now there is a suite of different techniques to choose from. [Keras
provides a laundry list](http://keras.io/initializations/).

In this example, we will look at tuning the selection of network weight
initialization by evaluating all of the available techniques.

We will use the same weight initialization method on each layer.
Ideally, it may be better to use different weight initialization schemes
according to the activation function used on each layer. In the example
below we use rectifier for the hidden layer. We use sigmoid for the
output layer because the predictions are binary.

The full code listing is provided below.

```
# Use scikit-learn to grid search the weight initialization
import numpy
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model(init_mode='uniform'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, kernel_initializer=init_mode, activation='relu'))
	model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

**Note**: Your [results may
vary]
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

Running this example produces the following output.

```
Best: 0.720052 using {'init_mode': 'uniform'}
0.720052 (0.024360) with: {'init_mode': 'uniform'}
0.348958 (0.024774) with: {'init_mode': 'lecun_uniform'}
0.712240 (0.012075) with: {'init_mode': 'normal'}
0.651042 (0.024774) with: {'init_mode': 'zero'}
0.700521 (0.010253) with: {'init_mode': 'glorot_normal'}
0.674479 (0.011201) with: {'init_mode': 'glorot_uniform'}
0.661458 (0.028940) with: {'init_mode': 'he_normal'}
0.678385 (0.004872) with: {'init_mode': 'he_uniform'}
```

We can see that the best results were achieved with a uniform weight
initialization scheme achieving a performance of about 72%.

How to Tune the Neuron Activation Function
------------------------------------------

The activation function controls the non-linearity of individual neurons
and when to fire.

Generally, the rectifier activation function is the most popular, but it
used to be the sigmoid and the tanh functions and these functions may
still be more suitable for different problems.

In this example, we will evaluate the suite of [different activation
functions available in Keras](http://keras.io/activations/). We will
only use these functions in the hidden layer, as we require a sigmoid
activation function in the output for the binary classification problem.

Generally, it is a good idea to prepare data to the range of the
different transfer functions, which we will not do in this case.

The full code listing is provided below.

```
# Use scikit-learn to grid search the activation function
import numpy
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model(activation='relu'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation=activation))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

**Note**: Your [results may
vary]
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

Running this example produces the following output.

```
Best: 0.722656 using {'activation': 'linear'}
0.649740 (0.009744) with: {'activation': 'softmax'}
0.720052 (0.032106) with: {'activation': 'softplus'}
0.688802 (0.019225) with: {'activation': 'softsign'}
0.720052 (0.018136) with: {'activation': 'relu'}
0.691406 (0.019401) with: {'activation': 'tanh'}
0.680990 (0.009207) with: {'activation': 'sigmoid'}
0.691406 (0.014616) with: {'activation': 'hard_sigmoid'}
0.722656 (0.003189) with: {'activation': 'linear'}
```


Surprisingly (to me at least), the 'linear' activation function achieved
the best results with an accuracy of about 72%.


How to Tune Dropout Regularization
----------------------------------

In this example, we will look at tuning the [dropout rate for
regularization]
in an effort to limit overfitting and improve the model's ability to
generalize.

To get good results, dropout is best combined with a weight constraint
such as the max norm constraint.

For more on using dropout in deep learning models with Keras see the
lab:

-   [Dropout Regularization in Deep Learning Models With
    Keras]

This involves fitting both the dropout percentage and the weight
constraint. We will try dropout percentages between 0.0 and 0.9 (1.0
does not make sense) and maxnorm weight constraint values between 0 and
5.

The full code listing is provided below.

```
# Use scikit-learn to grid search the dropout rate
import numpy
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.constraints import maxnorm
# Function to create model, required for KerasClassifier
def create_model(dropout_rate=0.0, weight_constraint=0):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='linear', kernel_constraint=maxnorm(weight_constraint)))
	model.add(Dropout(dropout_rate))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```


**Note**: Your [results may
vary]
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

Running this example produces the following output.

```
Best: 0.723958 using {'dropout_rate': 0.2, 'weight_constraint': 4}
0.696615 (0.031948) with: {'dropout_rate': 0.0, 'weight_constraint': 1}
0.696615 (0.031948) with: {'dropout_rate': 0.0, 'weight_constraint': 2}
0.691406 (0.026107) with: {'dropout_rate': 0.0, 'weight_constraint': 3}
0.708333 (0.009744) with: {'dropout_rate': 0.0, 'weight_constraint': 4}
0.708333 (0.009744) with: {'dropout_rate': 0.0, 'weight_constraint': 5}
0.710937 (0.008438) with: {'dropout_rate': 0.1, 'weight_constraint': 1}
0.709635 (0.007366) with: {'dropout_rate': 0.1, 'weight_constraint': 2}
0.709635 (0.007366) with: {'dropout_rate': 0.1, 'weight_constraint': 3}
0.695312 (0.012758) with: {'dropout_rate': 0.1, 'weight_constraint': 4}
0.695312 (0.012758) with: {'dropout_rate': 0.1, 'weight_constraint': 5}
0.701823 (0.017566) with: {'dropout_rate': 0.2, 'weight_constraint': 1}
0.710938 (0.009568) with: {'dropout_rate': 0.2, 'weight_constraint': 2}
0.710938 (0.009568) with: {'dropout_rate': 0.2, 'weight_constraint': 3}
0.723958 (0.027126) with: {'dropout_rate': 0.2, 'weight_constraint': 4}
0.718750 (0.030425) with: {'dropout_rate': 0.2, 'weight_constraint': 5}
0.721354 (0.032734) with: {'dropout_rate': 0.3, 'weight_constraint': 1}
0.707031 (0.036782) with: {'dropout_rate': 0.3, 'weight_constraint': 2}
0.707031 (0.036782) with: {'dropout_rate': 0.3, 'weight_constraint': 3}
0.694010 (0.019225) with: {'dropout_rate': 0.3, 'weight_constraint': 4}
0.709635 (0.006639) with: {'dropout_rate': 0.3, 'weight_constraint': 5}
0.704427 (0.008027) with: {'dropout_rate': 0.4, 'weight_constraint': 1}
0.717448 (0.031304) with: {'dropout_rate': 0.4, 'weight_constraint': 2}
0.718750 (0.030425) with: {'dropout_rate': 0.4, 'weight_constraint': 3}
0.718750 (0.030425) with: {'dropout_rate': 0.4, 'weight_constraint': 4}
0.722656 (0.029232) with: {'dropout_rate': 0.4, 'weight_constraint': 5}
0.720052 (0.028940) with: {'dropout_rate': 0.5, 'weight_constraint': 1}
0.703125 (0.009568) with: {'dropout_rate': 0.5, 'weight_constraint': 2}
0.716146 (0.029635) with: {'dropout_rate': 0.5, 'weight_constraint': 3}
0.709635 (0.008027) with: {'dropout_rate': 0.5, 'weight_constraint': 4}
0.703125 (0.011500) with: {'dropout_rate': 0.5, 'weight_constraint': 5}
0.707031 (0.017758) with: {'dropout_rate': 0.6, 'weight_constraint': 1}
0.701823 (0.018688) with: {'dropout_rate': 0.6, 'weight_constraint': 2}
0.701823 (0.018688) with: {'dropout_rate': 0.6, 'weight_constraint': 3}
0.690104 (0.027498) with: {'dropout_rate': 0.6, 'weight_constraint': 4}
0.695313 (0.022326) with: {'dropout_rate': 0.6, 'weight_constraint': 5}
0.697917 (0.014382) with: {'dropout_rate': 0.7, 'weight_constraint': 1}
0.697917 (0.014382) with: {'dropout_rate': 0.7, 'weight_constraint': 2}
0.687500 (0.008438) with: {'dropout_rate': 0.7, 'weight_constraint': 3}
0.704427 (0.011201) with: {'dropout_rate': 0.7, 'weight_constraint': 4}
0.696615 (0.016367) with: {'dropout_rate': 0.7, 'weight_constraint': 5}
0.680990 (0.025780) with: {'dropout_rate': 0.8, 'weight_constraint': 1}
0.699219 (0.019401) with: {'dropout_rate': 0.8, 'weight_constraint': 2}
0.701823 (0.015733) with: {'dropout_rate': 0.8, 'weight_constraint': 3}
0.684896 (0.023510) with: {'dropout_rate': 0.8, 'weight_constraint': 4}
0.696615 (0.017566) with: {'dropout_rate': 0.8, 'weight_constraint': 5}
0.653646 (0.034104) with: {'dropout_rate': 0.9, 'weight_constraint': 1}
0.677083 (0.012075) with: {'dropout_rate': 0.9, 'weight_constraint': 2}
0.679688 (0.013902) with: {'dropout_rate': 0.9, 'weight_constraint': 3}
0.669271 (0.017566) with: {'dropout_rate': 0.9, 'weight_constraint': 4}
0.669271 (0.012075) with: {'dropout_rate': 0.9, 'weight_constraint': 5}
```

We can see that the dropout rate of 20% and the maxnorm weight
constraint of 4 resulted in the best accuracy of about 72%.


How to Tune the Number of Neurons in the Hidden Layer
-----------------------------------------------------

The number of neurons in a layer is an important parameter to tune.
Generally the number of neurons in a layer controls the representational
capacity of the network, at least at that point in the topology.

Also, generally, a large enough single layer network can approximate any
other neural network, [at least in
theory](https://en.wikipedia.org/wiki/Universal_approximation_theorem).

In this example, we will look at tuning the number of neurons in a
single hidden layer. We will try values from 1 to 30 in steps of 5.

A larger network requires more training and at least the batch size and
number of epochs should ideally be optimized with the number of neurons.

The full code listing is provided below.

```
# Use scikit-learn to grid search the number of neurons
import numpy
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.constraints import maxnorm
# Function to create model, required for KerasClassifier
def create_model(neurons=1):
	# create model
	model = Sequential()
	model.add(Dense(neurons, input_dim=8, kernel_initializer='uniform', activation='linear', kernel_constraint=maxnorm(4)))
	model.add(Dropout(0.2))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```


**Note**: Your [results may
vary]
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

Running this example produces the following output.

```
Best: 0.714844 using {'neurons': 5}
0.700521 (0.011201) with: {'neurons': 1}
0.714844 (0.011049) with: {'neurons': 5}
0.712240 (0.017566) with: {'neurons': 10}
0.705729 (0.003683) with: {'neurons': 15}
0.696615 (0.020752) with: {'neurons': 20}
0.713542 (0.025976) with: {'neurons': 25}
0.705729 (0.008027) with: {'neurons': 30}
```
We can see that the best results were achieved with a network with 5 neurons in the hidden layer with an accuracy of about 71%.


Tips for Hyperparameter Optimization
------------------------------------

This section lists some handy tips to consider when tuning
hyperparameters of your neural network.

-   **k-fold Cross Validation**. You can see that the results from the
    examples in this lab show some variance. A default cross-validation
    of 3 was used, but perhaps k=5 or k=10 would be more stable.
    Carefully choose your cross validation configuration to ensure your
    results are stable.
-   **Review the Whole Grid**. Do not just focus on the best result,
    review the whole grid of results and look for trends to support
    configuration decisions.
-   **Parallelize**. Use all your cores if you can, neural networks are
    slow to train and we often want to try a lot of different
    parameters. Consider spinning up a lot of [AWS
    instances].
-   **Use a Sample of Your Dataset**. Because networks are slow to
    train, try training them on a smaller sample of your training
    dataset, just to get an idea of general directions of parameters
    rather than optimal configurations.
-   **Start with Coarse Grids**. Start with coarse-grained grids and
    zoom into finer grained grids once you can narrow the scope.
-   **Do not Transfer Results**. Results are generally problem specific.
    Try to avoid favorite configurations on each new problem that you
    see. It is unlikely that optimal results you discover on one problem
    will transfer to your next project. Instead look for broader trends
    like number of layers or relationships between parameters.
-   **Reproducibility is a Problem**. Although we set the seed for the
    random number generator in NumPy, the results are not 100%
    reproducible. There is more to reproducibility when grid searching
    wrapped Keras models than is presented in this lab.

Summary
-------

In this lab, you discovered how you can tune the hyperparameters of
your deep learning networks in Python using Keras and scikit-learn.

Specifically, you learned:

-   How to wrap Keras models for use in scikit-learn and how to use grid
    search.
-   How to grid search a suite of different standard neural network
    parameters for Keras models.
-   How to design your own hyperparameter optimization experiments.