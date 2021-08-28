# Ultimate model explanation library
# 
# Have fun!

# importing all necessary libraries beforehand

import shap
import numpy as np
import pandas as pd
import logging
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt


# =============== AUXILIARY FUNCTIONS ===============


def _check_dataset_model(estimator, X, *attributes):
    """
        Auxiliary function to check dataset and if model is suitable for working with
    """

    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        if isinstance(X, np.ndarray):
            feature_names = list(range(X.shape[1]))
        else:
            feature_names = list(range(len(X[0])))
        try:
            X = pd.DataFrame(data=X, columns=feature_names)
        except:
            raise ValueError(f"Cannot create DataFrame from `X`")

    for attr in attributes:
        if not hasattr(estimator, attr):
            raise ValueError(f"Provided estimator does not have '{attr}' method")

    try:
        estimated_model = estimator.copy()
    except AttributeError:
        from copy import deepcopy
        estimated_model = deepcopy(estimator)
    except:
        raise ValueError("Cannot make copy of the provided estimator")

    return estimated_model, X, feature_names


def _check_y(y):
    """
        Auxiliary function to check target variable
    """

    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy()
    else:
        try:
            y = np.asarray(y)
        except:
            raise ValueError(f"Cannot create Numpy Array from `y`")
    return y


def _check_importances_args(estimated_model, X, y, **kwargs):
    """
        Auxiliary function for checking arguments for get_loco_feature_importances and get_pfi_feature_importances

        Parameters
        ----------
        estimated_model:    Fitted sklearn, XGBoost, CatBoost or any other model class type with
                            `fit` and `predict` methods
            Input model which we want to calculate LOCO for
        X:                  Array like data
            Features dataset
        y:                  Array like data
            Target dataset
        kwargs:
            optional arguments
        Keyword Arguments
        -----------------
        normalize_result:

        normalize_num:

        error_type:         string

        metric:

        Returns
        -------
        Checked input parameters
    """

    estimator, X, feature_names = _check_dataset_model(estimated_model, X, "fit", "predict")
    y = _check_y(y)

    normalize_result = kwargs.get("normalize_result", True)
    normalize_num = kwargs.get("normalize_num", 1.0)
    error_type = kwargs.get("error_type", "divide")
    metric = kwargs.get("metric", mean_squared_error)

    if normalize_num <= 0:
        raise ValueError(f"Incorrect normalization value, expected to be > 0, got {normalize_num}")

    if error_type not in ['divide', 'subtract']:
        raise ValueError(f"Incorrect error_type value, expected to be in ['divide', 'subtract'], got {error_type}")

    if not callable(metric):
        raise ValueError("Provided metric function is not callable")

    return estimator, X, y, feature_names, normalize_result, normalize_num, error_type, metric


def _check_integer_values(**kwarg):     # Проверка на integer
    out = []
    for name, val in kwarg.items():
        try:
            out.append(int(val))
        except:
            raise TypeError('Incorrect type of ' + name + ': must be integer')

    if len(out) > 1:
        return tuple(out)
    else:
        return out[0]


def _check_float_values(**kwarg):     # Проверка на float
    out = []
    for name, val in kwarg.items():
        try:
            out.append(val)
        except:
            raise TypeError('Incorrect type of ' + name + ': must be float')

    if len(out) > 1:
        return tuple(out)
    else:
        return out[0]


# ================= MAIN FUNCTIONS =================


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

    estimator, X, y, feature_names, normalize_result, normalize_num, error_type, metric = \
        _check_importances_args(estimated_model, X, y, **kwargs)

    fit_args = {} if fit_args is None else fit_args

    prediction = estimator.predict(X)
    start_loss = metric(y, prediction)

    result = []
    for feature in feature_names:
        logging.debug(f"Processing feature = '{feature}'")
        if data_split:
            df_temp = X.copy()
            df_train, df_test, y_train, y_test = train_test_split(df_temp, y, test_size=0.23, random_state=0)

            df_train = df_train.drop(feature, axis=1)
            df_test = df_test.drop(feature, axis=1)

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
        This function calculates PFI feature importances
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

    estimator, X, y, feature_names, normalize_result, normalize_num, error_type, metric = _check_importances_args(estimated_model, X, y, **kwargs)

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


def pdp_values_2D(estimated_model, X, target_name, n_splits):
    """
        This function calculates PDP and ICE values for target_name feature

        Parameters
        ----------
        estimated_model:    Fitted sklearn, XGBoost, CatBoost or any other model class type with `fit` and `predict` methods
            Input model which we want to calculate PDP for
        X:                  Array like data
            A table of features values
        target_name:                  str or int
            Name of the feature to calculate PDP for
        n_splits:         int
            Number of splits for target_name feature range

        Returns
        -------
        PDP Values:        tuple of 3 np.array
            The output contains of 3 np.arrays: feature_list, pdp_list, ice_list
    """

    estimator, X, _ = _check_dataset_model(estimated_model, X, 'predict')
    if target_name not in X.columns:
        raise ValueError('The provided target_name was not found in X')
    n_splits = _check_integer_values(n_splits=n_splits)
    if n_splits <= 0:
        raise ValueError('The provided n_splits has to be positive non-zero value. Got:' + str(n_splits))

    pdp_list = []
    ice_list = []
    X_copy = X.copy()
    x_max = X[target_name].max()
    x_min = X[target_name].min()

    step = abs(x_max - x_min) / n_splits
    feature_vals = np.arange(x_min, x_max + step, step)[:n_splits+1]

    for feature_val in feature_vals:
        X_copy[target_name] = feature_val

        predict = estimator.predict(X_copy)
        predict = np.array(predict)

        pdp_list.append(predict.mean())
        ice_list.append(predict)

    pdp_list = np.array(pdp_list)
    ice_list = np.array(ice_list)

    return feature_vals, pdp_list, ice_list


def pdp_plot_2D(estimated_model, X, target_feature, n_splits):
    """
        This function plots PDP and ICE values for target_name feature

        Parameters
        ----------
        estimated_model:    Fitted sklearn, XGBoost, CatBoost or any other model class type with `fit` and `predict` methods
            Input model which we want to calculate PDP for
        X:                  Array like data
            A table of features values
        target_feature:     str or int
            Name of the feature to calculate PDP for
        n_splits:           int
            Number of splits for target_name feature range

        Returns
        -------
        Nothing, just plot
    """

    estimator, X, _ = _check_dataset_model(estimated_model, X, 'predict')
    if target_feature not in X.columns:
        raise ValueError('The provided target_name was not found in X')
    n_splits = _check_integer_values(n_splits=n_splits)
    if n_splits <= 0:
        raise ValueError('The provided n_splits has to be positive non-zero value. Got:' + str(n_splits))

    feature_list, pdp_list, ice_list = pdp_values_2D(estimator, X, target_feature, n_splits)

    pdp_color = 'red'
    ice_color = 'blue'

    fig, ax = plt.subplots(figsize=(12, 6))
    ice = ax.plot(feature_list, ice_list, color=ice_color, alpha=0.1)
    pdp = ax.plot(feature_list, pdp_list, color=pdp_color, lw=2)
    ax.set_title('2D PDP for {} feature'.format(str(target_feature)))
    ax.set_xlabel('Value changes of {} feature'.format(str(target_feature)))
    ax.set_ylabel('Target variable')
    ax.minorticks_on()
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', linestyle=':', color='grey')

    ax.legend((pdp[0], ice[0]), ['PDP line', 'ICE values'])
    plt.show()
    return


def pdp_interval_values(estimated_model, X, target_feature, grid_points_val, target_val_lower, target_val_upper):
    """
        Function for finding intervals of analysed feature, where target variable is in the provided interval

        Parameters
        ----------
        estimated_model:    Fitted sklearn, XGBoost, CatBoost or any other model class type with `fit`
                            and `predict` methods
            Input model which we want to calculate PDP values for
        X:                  Array like data
            A table of features' values
        target_feature:     string
            Target feature for finding intervals
        grid_points_val:    integer
            Number of points for calculating PDP
        target_val_lower:   float
            Lower-boundary value of interval
        target_val_upper:   float
            Upper-boundary value of interval

        Returns
        -------
        PDP intervals values    list([tuple], value, value))
    """

    estimator, X, _ = _check_dataset_model(estimated_model, X, 'predict')
    if target_feature not in X.columns:
        raise ValueError('The provided target_name was not found in X')
    grid_points_val = _check_integer_values(grid_points_val=grid_points_val)
    if grid_points_val <= 1:
        raise ValueError('The provided grid_points_val has to be >= 2. Got:' + str(grid_points_val))
    target_val_lower, target_val_upper = _check_float_values(target_val_lower=target_val_lower,
                                                             target_val_upper=target_val_upper)

    pdp_goals = pdp_values_2D(estimator, X, target_feature, grid_points_val - 1)

    intervals = np.zeros([grid_points_val, 3])
    for i in range(0, grid_points_val):
        intervals[i, 0] = pdp_goals[1][i]
        intervals[i, 1] = pdp_goals[0][i]
        if pdp_goals[1][i] > target_val_lower and pdp_goals[1][i] < target_val_upper:
            intervals[i, 2] = 1
        else:
            intervals[i, 2] = -1

    interval_list = list()
    average_val_list = list()
    median_val_list = list()

    flag = True
    current_interval_list = np.array([])
    start = 0
    end = 0

    for i in range(0, grid_points_val):
        if intervals[i, 2] == 1 and flag and i != grid_points_val - 1:
            start = intervals[i, 1]
            end = intervals[i, 1]  # Начало интервала
            flag = False
            current_interval_list = np.empty(0)
            current_interval_list = np.append(current_interval_list, [start])

        elif intervals[i, 2] == 1 and not flag and i != grid_points_val - 1:
            end = intervals[i, 1]
            current_interval_list = np.append(current_interval_list, [end])  # Продолжение записи интервала

        elif intervals[i, 2] == 1 and flag and i == grid_points_val - 1:
            start = intervals[i, 1]  # Единичный интервал в конце списка
            average_val_list.append(start)
            median_val_list.append(start)
            current_interval_list.clear()
            interval_list.append((start, start))

        elif intervals[i, 2] == 1 and not flag and i == grid_points_val - 1:
            # запись последнего интервала, если последний элемент подходит по условиям
            end = intervals[i, 1]
            current_interval_list = np.append(current_interval_list, [end])
            average_val_list.append(np.average(current_interval_list))
            median_val_list.append(np.median(current_interval_list))
            interval_list.append((start, end))
            current_interval_list = np.empty(0)
            flag = True

        elif intervals[i, 2] == -1 and not flag and i != grid_points_val - 1:
            # Конец интервала; его запись в список
            average_val_list.append(np.average(current_interval_list))
            median_val_list.append(np.median(current_interval_list))
            interval_list.append((start, end))
            current_interval_list = np.empty(0)
            flag = True

    finals_list = [interval_list, average_val_list, median_val_list]
    return finals_list


def pdp_values_3D(estimated_model, X, target_name_1, target_name_2, n_splits):
    """
        This function calculates PDP values for interaction of 2 features

        Parameters
        ----------
        estimated_model:    Fitted sklearn, XGBoost, CatBoost or any other model class type with `fit`
                            and `predict` methods (WARNING: SHAP does not support Decision-tree-based models!)
            Input model which we want to calculate ICE values for
        X:                  Array like data
            A table of features' values
        target_name_1      string or value
            1st feature name
        target_name_2      string or value
            2nd feature name
        n_splits:           int
            Number of splits for features' range

        Returns
        --------
        Outputs tuple of 3 objects:
        PDP interaction 2D-matrix for 3 features,
        range for target_name_1 feature,
        range for target_name_2 feature
    """

    estimator, X, _ = _check_dataset_model(estimated_model, X, 'predict')
    n_splits = _check_integer_values(n_splits=n_splits)
    if n_splits <= 0:
        raise ValueError('The provided n_splits has to be positive non-zero value. Got:' + str(n_splits))
    if target_name_1 not in X.columns:
        raise ValueError('The provided target_name_1 was not found in X')
    if target_name_2 not in X.columns:
        raise ValueError('The provided target_name_2 was not found in X')

    x_max_1 = X[target_name_1].max()
    x_min_1 = X[target_name_1].min()
    step_1 = abs(x_max_1 - x_min_1) / n_splits
    feature_vals_1 = np.arange(x_min_1, x_max_1 + step_1, step_1)[:n_splits + 1]

    x_max_2 = X[target_name_2].max()
    x_min_2 = X[target_name_2].min()
    step_2 = abs(x_max_2 - x_min_2) / n_splits
    feature_vals_2 = np.arange(x_min_2, x_max_2 + step_2, step_2)[:n_splits + 1]

    X_copy = X.copy()
    PDP_result = np.zeros((n_splits + 1, n_splits + 1))

    for i, f_val_1 in enumerate(feature_vals_1):
        X_copy[target_name_1] = f_val_1

        for j, f_val_2 in enumerate(feature_vals_2):
            X_copy[target_name_2] = f_val_2

            predict = estimator.predict(X_copy)
            predict = np.array(predict)
            PDP_result[i][j] = predict.mean()

    return PDP_result, feature_vals_1, feature_vals_2


def pdp_plot_3D(estimated_model, X, target_name_1, target_name_2, n_splits):
    """
        This function is plotting PDP for 2 features and outputs heatmap plot
        Parameters
        ----------
        estimated_model:    Fitted sklearn, XGBoost, CatBoost or any other model class type with `fit`
                            and `predict` methods (WARNING: SHAP does not support Decision-tree-based models!)
            Input model which we want to calculate ICE values for
        X:                  Array like data
            A table of features' values
        n_splits:           int
            Number of splits for features' range
        target_name_1      string or value
            1st feature name
        target_name_2      string or value
            2nd feature name

        Returns
        --------
        Outputs PDP heatmap interaction plot for 2 features
    """

    estimator, X, _ = _check_dataset_model(estimated_model, X, 'predict')
    n_splits = _check_integer_values(n_splits=n_splits)
    if n_splits <= 0:
        raise ValueError('The provided n_splits has to be positive non-zero value. Got:' + str(n_splits))
    if target_name_1 not in X.columns:
        raise ValueError('The provided target_name_1 was not found in X')
    if target_name_2 not in X.columns:
        raise ValueError('The provided target_name_2 was not found in X')

    PDP_result, feature_vals_1, feature_vals_2 = pdp_values_3D(estimator, X, target_name_1, target_name_2, n_splits)

    feature_vals_1 = np.round(feature_vals_1, 3)
    feature_vals_2 = np.round(feature_vals_2, 3)

    plt.subplots(figsize=(12, 10))
    result_2_plot = pd.DataFrame(data=PDP_result)
    sns.heatmap(result_2_plot, xticklabels=feature_vals_1, yticklabels=feature_vals_2, annot=True)
    plt.title('3D PDP for {} and {} feature'.format(str(target_name_1), str(target_name_2)))
    plt.xlabel('Value changes of {} feature'.format(str(target_name_1)))
    plt.ylabel('Value changes of {} feature'.format(str(target_name_2)))
    plt.show()
    return


# NEEDS MORE TESTING
def shap_plot(estimated_model, X):
    """
        Just a simple overlay of the shap library method  waterfall for calculating SHAP plots

        Parameters
        ----------
        estimated_model:    Fitted sklearn, XGBoost, CatBoost or any other model class type with `fit`
                            and `predict` methods (WARNING: SHAP does not support Decision-tree-based models!)
             Input model which we want to calculate ICE values for
        X:                  Array like data
             A table of features' values

        Returns
        --------
        Outputs SHAP plot
     """

    estimator, X, _ = _check_dataset_model(estimated_model, X, 'fit', 'predict')

    explainer = shap.Explainer(estimator)
    shap_values = explainer(X)

    inp = shap_values[0]
    inp.base_values = inp.base_values[0]

    shap.plots.waterfall(inp)


def lime_plot(estimated_model, X, **kwargs):
    """
        Just a simple overlay of the lime library method explain_instance for calculating LIME plots

        Parameters
        ----------
        estimated_model:    Fitted sklearn, XGBoost, CatBoost or any other model class type with `fit`
                            and `predict` methods (WARNING: SHAP does not support Decision-tree-based models!)
            Input model which we want to calculate ICE values for
        X:                  Array like data
            A table of features' values
        Keyword Arguments
        -----------------
         max_feature_amount  integer
            Maximum amount of features for plotting LIME for
            100 by default
         selection_num       integer
            number of elements for plotting LIME
            0 by default
         work_mode           string
            work mode, 'regression' by default.
            (ATTENTION - 'classification' MODE IS NOT SUPPORTED YET)
        Returns
        --------
        Outputs LIME plot
    """

    max_feature_amount = kwargs.get("max_feature_amount", 100)
    selection_num = kwargs.get("selection_num", 0)
    work_mode = kwargs.get("work_mode", 'regression')

    estimator, X, _ = _check_dataset_model(estimated_model, X, 'fit', 'predict')
    max_feature_amount, selection_num = _check_integer_values(max_feature_amount = max_feature_amount,
                                                              selection_num = selection_num)
    if max_feature_amount <= 0:
        raise ValueError("Incorrect feature amount. You should have at least one feature. Got:"+str(max_feature_amount))
    if selection_num < 0:
        raise ValueError("Incorrect selection amount. You should have at least one element. Got:"+str(selection_num))

    explainer = LimeTabularExplainer(training_data=X.to_numpy(),
        feature_names=list(X.columns),
        mode=work_mode, random_state=0)

    exp = explainer.explain_instance(X.to_numpy()[selection_num], estimator.predict, num_features=max_feature_amount)
    exp.as_pyplot_figure()
    plt.show()
    return


def pdp_custom_4D(estimated_model, X, n_splits, target_name_1, target_name_2, target_name_3):
    """
        This function calculates 3d-matrix of PDP values for 3-features interaction

        Parameters
        ----------
        estimated_model:    Fitted sklearn, XGBoost, CatBoost or any other model class type with `fit`
                            and `predict` methods (WARNING: SHAP does not support Decision-tree-based models!)
            Input model which we want to calculate ICE values for
        X:                  Array like data
            A table of features' values
        n_splits:           int
            Number of splits for features' range
        feature_name_1      string
            1st feature name
        feature_name_2      string
            2nd feature name
        target_name_3      string
            2nd feature name

        Returns
        --------
        Outputs PDP interaction 3D-matrix for 3 features
    """

    estimator, X, _ = _check_dataset_model(estimated_model, X, 'predict')
    n_splits = _check_integer_values(n_splits=n_splits)
    if n_splits <= 0:
        raise ValueError('The provided n_splits has to be positive non-zero value. Got:' + str(n_splits))

    if target_name_1 not in X.columns:
        raise ValueError('The provided target_name_1 was not found in X')
    if target_name_2 not in X.columns:
        raise ValueError('The provided target_name_2 was not found in X')
    if target_name_3 not in X.columns:
        raise ValueError('The provided target_name_3 was not found in X')

    feature_list_1 = list()
    feature_list_2 = list()
    feature_list_3 = list()

    x_max_1 = X[target_name_1].max()
    x_min_1 = X[target_name_1].min()

    x_max_2 = X[target_name_2].max()
    x_min_2 = X[target_name_2].min()

    x_max_3 = X[target_name_3].max()
    x_min_3 = X[target_name_3].min()

    X_copy = X.copy()

    step_1 = abs(x_max_1 - x_min_1) / n_splits
    step_2 = abs(x_max_2 - x_min_2) / n_splits
    step_3 = abs(x_max_3 - x_min_3) / n_splits

    x_axis = int(n_splits + 1)
    y_axis = int(n_splits + 1)
    z_axis = int(n_splits + 1)

    s = (x_axis, y_axis, z_axis)
    PDP_result = np.zeros(s)

    counter_1 = x_min_1
    counter_2 = x_min_2
    counter_3 = x_min_3

    for i in range(n_splits + 1):

        feature_list_1.append(counter_1 + (i * step_1))
        X_copy[target_name_1] = counter_1 + i * step_1

        for j in range(n_splits + 1):

            feature_list_2.append(counter_2 + j * step_2)
            X_copy[target_name_2] = counter_2 + j * step_2

            for k in range(n_splits + 1):
                feature_list_3.append(counter_3 + k * step_3)
                X_copy[target_name_3] = counter_3 + k * step_3

                temp = estimator.predict(X_copy)
                PDP_result[i][j][k] = temp.mean()

    return PDP_result


# ============= ADDITIONAL FUNCTIONS FOR DATA ANALYZE =============


def target_count(df, target, feature_intervals, fun='mean'):
    """
        Target metric count for objects belongs to pair of features and they intervals
        This function was provided by Sergey

        Parameters
        ----------
        df:                     pandas DataFrame
            A table of objects' features and a target variable
        target:                 string or int
            Name of the column with target variable
        feature_intervals:      {key1: value1, key2: value2}
            2 objects must be
            key - column name from df
            value - array-like object with left and right values of interested interval
        fun:                    string
            name of aggregate action for target values:
            ('mean', 'median', 'classification')

        Returns
        --------
        aggregated by "fun" target values of objects corresponded to conditions of intervals
    """

    # choose objects inside of feature intervals
    cond = None
    for key in feature_intervals.keys():
        if not (cond is None):
            cond = cond & (df[key] >= feature_intervals[key][0]) & (df[key] <= feature_intervals[key][1])
        else:
            cond = (df[key] >= feature_intervals[key][0]) & (df[key] <= feature_intervals[key][1])
    df_filter = df[cond]
    # choose objects corresponds in interval
    in_interval_set = df_filter[target]

    if fun == 'mean':
        res = np.nanmean(in_interval_set)
    elif fun == 'median':
        res = np.nanmedian(in_interval_set)
    elif fun == 'classification':
        if np.size(in_interval_set, 0) != 0:
            res = np.sum(in_interval_set)/np.size(in_interval_set, 0)
        else:
            res = 1
    else:
        res = np.nanmean(in_interval_set)
    return round(res, 3)


def feature_target_dependency_3d_chart(df, target, col_lim, fun='mean'):
    """
        Plots graphs of paired influence of parameters on the target variable
        This function was provided by Sergey

        Parameters
        ----------
        df:                 pandas DataFrame
            A table of objects' features and a target variable
        target:             string or int
            Name of the column with target variable
        col_lim:            {key1: value1, key2: value2}
            2 objects must be
            key - column name from df
            value - array-like,
                value[0] - min edge,
                value[1] - max edge,
                value[2] - step for mesh grid
        fun:                string
            name of aggregate action for target values:
            ('mean', 'median', 'classification')

        Returns
        --------
        3D-plot showing dependency between target and 2 features
    """

    grid = []
    keys = col_lim.keys()
    for key in keys:
        X = np.arange(col_lim[key][0], col_lim[key][1], col_lim[key][2])
        grid.append(X)

    x_len = len(grid[0])
    y_len = len(grid[1])

    Z = np.zeros((y_len, x_len))
    for i in range(y_len)[:-1]:
        for k in range(x_len)[:-1]:
            grid_lim = {keys[1]:[grid[1][i], grid[1][i+1]], keys[0]:[grid[0][k], grid[0][k+1]]}
            Z[i,k] = target_count(df, target, grid_lim, fun)
    X, Y = np.meshgrid(grid[0], grid[1])

    plt.figure(figsize=(14, 10))
    surf2d = plt.contourf(X, Y, Z, 8, cmap='BuPu', grid=True, alpha=0.7)
    plt.colorbar(surf2d)
    plt.show()
    return


'''

 _____                       _         _                         __     __      __      _ 
(_   _)                     ( )       ( )_            _  _     /'__`\ /' _`\  /'__`\  /' )
  | |   __   _ __   _ _    _| |   _ _ | ,_)   _ _    | || |   (_)  ) )| ( ) |(_)  ) )(_, |
  | | /'__`\( '__)/'_` ) /'_` | /'_` )| |   /'_` )   | || |      /' / | | | |   /' /   | |
  | |(  ___/| |  ( (_| |( (_| |( (_| || |_ ( (_| |   | || |    /' /( )| (_) | /' /( )  | |
  (_)`\____)(_)  `\__,_)`\__,_)`\__,_)`\__)`\__,_)   | || |   (_____/'`\___/'(_____/'  (_)
  
  
'''




