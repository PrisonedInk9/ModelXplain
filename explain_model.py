import logging
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def _check_estimator(estimator, *attributes):
    for attr in attributes:
        if not hasattr(estimator, attr):
            raise ValueError(f"Provided estimator does not have '{attr}' method")

    try:
        estimated_model = estimator.copy()
    except AttributeError:
        from copy import deepcopy
        estimated_model = deepcopy(estimator)
    except:
        raise ValueError("Cannot make copy of an estimator")
    return estimated_model


def _check_importances_args(estimated_model, X, y, **kwargs):
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        feature_names = list(range(X.shape[1]))
        try:
            X = pd.DataFrame(data=X, columns=feature_names)
        except:
            raise ValueError(f"Cannot create DataFrame from `X`")

    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy()
    else:
        y = np.asarray(y)
    
    estimator = _check_estimator(estimated_model, "fit", "predict")

    normalize_result = kwargs.get("normalize_result", True)
    normalize_num = kwargs.get("normalize_num", 1.0)
    error_type = kwargs.get("kwargs", "divide")
    metric = kwargs.get("metric", mean_squared_error)

    if normalize_num <= 0:
        raise ValueError(f"Incorrect normalization value, expected to be > 0, got {normalize_num}")

    if error_type not in ['divide', 'subtract']:
        raise ValueError(f"Incorrect error_type value, expected to be in ['divide', 'subtract'], got {error_type}")

    if not callable(metric):
        raise ValueError("Provided metric function is not callable")

    return estimator, X, y, feature_names, normalize_result, normalize_num, error_type, metric


def get_loco_feature_importances(estimated_model, X, y, data_split=False, fit_args=None, **kwargs):
    """
    This function calculates LOCO feature importances.
        1) Input trained model, table of features and column of target values
        2) Make a prediction and evaluate start loss array with any metrics acceptable
        3) Reshape your model with datasets with every feature alternately excluded and write error for every
            excluded feature into separate array
        4) For every feature divide the error vector for the excluded feature by the error vector of the full dataset or 
            subtract the error vector of the full dataset from the error vector for the excluded feature 
        5) Normalize the result vector if needed

    Parameters
    ----------
    estimated_model:    Fitted sklearn, XGBoost, CatBoost or any other model class type with `fit` and `predict` methods
        Input model which we want to calculate LOCO for
    X:                  Array like data
        A table of features values
    y:                  Array like data
        A column of target values
    data_split:         bool
        option for splitting data when calculatig
        Default is False
    fit_args:           dict
        Optional argumets for fitting model
        Default is None

    Keyword Arguments
    -----------------
    normalize_result:   bool
        should we normalize results?
        Default is True
    normalize_num:      int or float
        value for normalizing features upon
        Default is 1.0
    error_type:         str
        Option for choosing method for calculating error. One of the 'divide', 'subtract'
        Default is 'divide'
    metric:             metric function
        option for choosing error calculation method.
        Default is mean_squared_error

    Returns
    -------
    LOCO Values:        np.array
        LOCO Feature importances
    """
    estimator, X, y, feature_names, normalize_result, normalize_num, error_type, metric  \
        = _check_importances_args(estimated_model, X, y, **kwargs)
    
    fit_args = {} if fit_args is None else fit_args

    prediction = estimator.predict(X)
    start_loss = metric(y, prediction)

    result = []
    for feature in feature_names:
        logging.debug(f"Processing feature = '{feature}'")
        if data_split:
            df_temp = X.copy()
            df_train, df_test, y_train, y_test = train_test_split(df_temp, y, test_size=0.23, random_state=0)

            df_train = df_train.drop(feature,  axis=1)
            df_test = df_test.drop(feature,  axis=1)

            estimator.fit(df_train, y_train, **fit_args)
            prediction = estimator.predict(df_test)

            result.append(metric(y_test, prediction))
        else:
            df_temp = X.copy()
            df_temp = df_temp.drop(feature, axis=1)

            estimator.fit(df_temp, y, **fit_args)
            prediction = estimator.predict(df_temp)

            result.append(metric(y, prediction))

    result = np.asarray(result)
    
    if error_type == 'divide':
        if start_loss == 0:
            start_loss = 1
            logging.warning("ATTENTION. As start_loss is zero, it was changed to 1 to better represent result.")
        result = (result / start_loss) - 1
    else:
        result = result - start_loss

    if normalize_result:
        result = normalize_num * result / result.max()

    return result

def get_pfi_feature_importances(estimated_model, X, y, shuffle_num=3, **kwargs):
    """
    this function calculates PFI feature importances

        1) Input trained model, table of features and column of target values
        2) Make a prediction and evaluate start loss array with any metrics acceptable
        3) Calculate the importance of features by dividing the vector of errors of the dataset with shuffled 
           values by the vector of errors of the original dataset or subtracting the vector of error values of 
           the original dataset from the vector of errors of the dataset with shuffled values 
        4) Normalize the result vector if needed

    Parameters
    ----------
    estimated_model:    Fitted sklearn, XGBoost, CatBoost or any other model class type with `fit` and `predict` methods
        Input model which we want to calculate PFI for
    X:                  Array like data
        A table of features values
    y:                  Array like data
        A column of target values
    shuffle_num:         int
        number of shuffles of selected feature

    Keyword Arguments
    -----------------
    normalize_result:   bool
        should we normalize results?
        Default is True
    normalize_num:      int or float
        value for normalizing features upon
        Default is 1.0
    error_type:         str
        Option for choosing method for calculating error. One of the 'divide', 'subtract'
        Default is 'divide'
    metric:             metric function
        option for choosing error calculation method.
        Default is mean_squared_error

    Returns
    -------
    PFI Values:        np.array
        PFI Feature importances
    """
    estimator, X, y, feature_names, normalize_result, normalize_num, error_type, metric \
        = _check_importances_args(estimated_model, X, y, **kwargs)

    if shuffle_num < 1:
        raise ValueError(f"Incorrect argument: shuffle_num. \nExpected: positive integer not less 1, got {shuffle_num}")
    
    prediction = estimator.predict(X)
    start_loss = metric(y, prediction)

    n_objects = X.shape[0]
    result = []
    for feature in feature_names:
        logging.debug(f"Processing feature = '{feature}'")
        feature_sum = 0
        for _ in range(shuffle_num):
            df_shuffled = X.copy()
            idx = np.random.choice(np.arange(n_objects), size=n_objects, replace=False)
            df_shuffled[feature] = X[feature].values[idx]

            prediction = estimated_model.predict(df_shuffled)
            feature_sum += metric(y, prediction)

        result.append(feature_sum / shuffle_num)
        
    result = np.asarray(result)

    if error_type == 'divide':
        result = result / start_loss
    else:
        result = result - start_loss
    
    if normalize_result:
        result = normalize_num * result / result.max()

    return result