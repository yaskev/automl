# AutoML tool for binary classification


AutoML is a library which allows you to get a simple model for binary classification. The main purpose of this library is to provide a baseline model which
performance is to be exceeded by a more sophisticated one.


## How to use the project

You should import the ```automl``` package and use the class ```ModelSelector```. To construct ```ModelSelector``` you need to specify the following arguments:
* A design matrix. The data does _not_ have to be normalized beforehand.
* An array of labels with the size equal to the number of samples in the design matrix.
* Metric to choose the best model based on its values. Must be of type ```MetricType``` and can be one of the following: ```ACCURACY```, ```BALANCED_ACCURACY```, ```PRECISION```, ```RECALL```, ```ROC_AUC```.
* Optional: labels encoder. ```OrdinalEncoder``` is used by default
* Optional: scaler. ```StandardScaler``` is used by default

By default the ```ModelSelector``` object is constructed with all possible ```ModelType```s enabled. To enable or disable some of them use the corresponding
functions: ```turn_off_models``` and ```turn_on_models```. Their only argument is of type ```ModelType``` and can be one of the following:

* BAGGED_TREES for bagging classifier
* BOOSTED_TREES for gradient boosting classifier
* LOG_REGRESSION for logistic regression
* RANDOM_FOREST for random forest classifier
* SVM for support vector machine


After you have created an instance of ```ModelSelector``` and configured the desired ```ModelType```s, you can call the function ```get_baseline_model``` which accepts the following arguments:
* fast_mode: bool. If set to ```True```, models' hyperparameters won't be optimized
* verbose: bool. If set to ```True```, intermediate steps results will be printed

The return values are the best possible fitted model(```Model``` type) and the value of the metric(```float```).


The resulting model has the following methods:
* fit(design_matrix: ```pd.DataFrame```, labels: ```np.ndarray```)
* predict(test_data: ```pd.DataFrame```)
* score(test_data: ```pd.DataFrame```, test_labels: ```np.ndarray```, metric: ```MetricType```)
* get_model_parameters()

## Examples
You can find an example in the file ```example.py```
