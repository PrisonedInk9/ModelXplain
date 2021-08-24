# Ultimate model explanation library
# 
# Have fun!

# importing all nessessary libraries beforehand

import shap
import numpy as np
import pandas as pd
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
from pdpbox import pdp


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


def pdp_plot_2D(estimated_model, X, target_feature, grid_points_val=30):
    """
        Just a simple overlay of the pdpbox library function pdp_isolate for PDP/ICE plotting

        Parameters
        ----------
        estimated_model:    Fitted sklearn, XGBoost, CatBoost or any other model class type
                            with `fit` and `predict` methods
            Input model which we want to plot PDP for
        X:                  Array like data
            A table of features' values
        target_feature:         string
            target feature for plotting PDP for
        Keyword Arguments
        -----------------
        grid_points_val         integer
            Number of points for plotting PDP.
            30 by default.
        Returns
        -------
        Plots PDP. No output.
        """

    feature_names = None # ЗАГЛУШКА

    #ВСТАВИТЬ ПРОВЕРКИ ПОСЛЕ ИМПЛИМЕНТАЦИИ КАСТОМНОГО PDP

    pdp_goals = pdp.pdp_isolate(model=estimated_model, dataset=X, model_features=feature_names,
                            feature=target_feature, num_grid_points=grid_points_val, grid_type='equal')

    pdp.pdp_plot(pdp_goals, target_feature)
    plt.show()


def pdp_values(estimated_model, X, target_feature, target_val_upper, target_val_lower, grid_points_val=30):

    """
        Just a simple overlay of the pdpbox library function pdp_isolate for calculating PDP values

        Parameters
        ----------
        estimated_model:    Fitted sklearn, XGBoost, CatBoost or any other model class type with `fit`
                            and `predict` methods
            Input model which we want to calculate PDP values for
        X:                  Array like data
            A table of features' values
        target_feature:         string
            target feature for calculating PDP values for
        target_val_upper        float
            upper-boundary value of interval
        target_val_lower        float
            lower-boundary value of interval
        Keyword Arguments
        -----------------
        grid_points_val         integer
            Number of points for plotting PDP.
            30 by default.
        Returns
        -------
        PDP values           list([tuple], value, value))
    """

    feature_names = None  # ЗАГЛУШКА

    # ВСТАВИТЬ ПРОВЕРКИ ПОСЛЕ ИМПЛИМЕНТАЦИИ КАСТОМНОГО PDP

    pdp_goals = pdp.pdp_isolate(model=estimated_model, dataset=X, model_features=feature_names, 
                            feature=target_feature, num_grid_points=grid_points_val, grid_type='equal')

    intervals = np.zeros([grid_points_val, 3])
    
    for i in range (0, grid_points_val): 
        intervals[i, 0] = pdp_goals.pdp[i]
        intervals[i, 1] = pdp_goals.display_columns[i]
        if pdp_goals.pdp[i] > target_val_lower and pdp_goals.pdp[i] < target_val_upper:
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
        if intervals[i, 2] == 1 and flag and i != grid_points_val-1:
            start = intervals[i, 1]
            end = intervals[i, 1]  # Начало интервала
            flag = False
            current_interval_list = np.empty(0)
            current_interval_list = np.append(current_interval_list, [start])

        elif intervals[i, 2] == 1 and not flag and i != grid_points_val-1:
            end = intervals[i, 1]
            current_interval_list = np.append(current_interval_list, [end])  # Продолжение записи интервала
             
        elif intervals[i, 2] == 1 and flag and i == grid_points_val-1:
            start = intervals[i, 1]  # Единичный интервал в конце списка
            average_val_list.append(start)
            median_val_list.append(start)
            current_interval_list.clear()
            interval_list.append((start, start))

        elif intervals[i, 2] == 1 and not flag and i == grid_points_val-1:
            # запись последнего интервала, если последний элемент подходит по условиям
            end = intervals[i, 1]
            current_interval_list = np.append(current_interval_list, [end])
            average_val_list.append(np.average(current_interval_list))
            median_val_list.append(np.median(current_interval_list))
            interval_list.append((start, end))
            current_interval_list = np.empty(0)
            flag = True
            
        elif intervals[i, 2] == -1 and not flag and i != grid_points_val-1:
            # Конец интервала; его запись в список
            average_val_list.append(np.average(current_interval_list))
            median_val_list.append(np.median(current_interval_list))
            interval_list.append((start, end))
            current_interval_list = np.empty(0)
            flag = True

    finals_list = [interval_list, average_val_list, median_val_list]
    
    return finals_list


def ice_values(estimated_model, X, target_feature, grid_val_start = None, grid_val_end = None):
    """
        Just a simple overlay of the pdpbox library function pdp_isolate for calculating ICE values

        Parameters
        ----------
        estimated_model:    Fitted sklearn, XGBoost, CatBoost or any other model class type with `fit`
                            and `predict` methods
            Input model which we want to calculate ICE values for
        X:                  Array like data
            A table of features values
        target_feature:         string
            target feature for calculating ICE values for
        Keyword Arguments
        -----------------
        grid_val_start        float
            start of the interval
        grid_val_end           float
            end of the interval
        Returns
        -------
        ICE values       pandas.DataFrame
    """

    feature_names = None  # ЗАГЛУШКА
    g_range = None        # ЗАГЛУШКА

    # ВСТАВИТЬ ПРОВЕРКИ ПОСЛЕ ИМПЛИМЕНТАЦИИ КАСТОМНОГО PDP


    pdp_goals = pdp.pdp_isolate(model=estimated_model, dataset=X, model_features=feature_names,
                            feature=target_feature, num_grid_points=30, grid_type='equal',
                            grid_range=g_range)

    return pdp_goals.ice_lines


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

    explainer = shap.Explainer(estimated_model)
    shap_values = explainer(X)
    print(type(shap_values))
    shap.plots.waterfall(shap_values[0])


def lime_plot(estimated_model, X, max_feature_amount=10, selection_num=25, work_mode='regression'):
    """
        Just a simple overlay of the shap library method  waterfall for calculating SHAP plots

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
            10 by default
        selection_num       integer
            number of elements for plotting LIME
        work_mode           string
            work mode, 'regression' by default.
            (ATTENTION - 'classification' MODE IS NOT SUPPORTED YET)

        Returns
        --------
        Outputs LIME plot
    """
    
    #ДОБАВИТЬ ПРОВЕРКИ

    explainer = LimeTabularExplainer(training_data=X.to_numpy(),
        feature_names=list(X.columns),
        mode=work_mode, random_state=0)

    exp = explainer.explain_instance(X.to_numpy()[selection_num], estimated_model.predict, num_features=max_feature_amount)
    exp.as_pyplot_figure()
    plt.tight_layout()


def pdp_plot_3D(estimated_model, X, feature_name_1, feature_name_2):
    """
        Just a simple overlay of the shap library method  waterfall for calculating SHAP plots

        Parameters
        ----------
        estimated_model:    Fitted sklearn, XGBoost, CatBoost or any other model class type with `fit`
                            and `predict` methods (WARNING: SHAP does not support Decision-tree-based models!)
            Input model which we want to calculate ICE values for
        X:                  Array like data
            A table of features' values
        feature_name_1      string
            1st feature name
        feature_name_2      string
            2nd feature name

        Returns
        --------
        Outputs PDP interaction plot for 2 features as a heatmap
    """

    feature_names = None  # ЗАГЛУШКА

    # ДОБАВИТЬ ПРОВЕРКИ ПОСЛЕ ИМПЛИМЕНТАЦИИ КАСТОМНОГО PDP

    pdp_goal = pdp.pdp_interact(model=estimated_model, dataset=X, model_features=feature_names, 
                                  features=[feature_name_1, feature_name_2])
    
    fig, axes = pdp.pdp_interact_plot(pdp_interact_out=pdp_goal,
                                  feature_names=[feature_name_1, feature_name_2],
                                  plot_type='contour',
                                  x_quantile=True,
                                  plot_pdp=True)

    fig.show()


## Target metric count for objects belongs to pair of features and they intervals. IMPORTANT: MIGHT BE NOT RELIABLE

def targetMetric(df, target, fun='mean', colLim={0: [0, 0], 1:[0, 0]}):
    '''
    InPuts:
      _df - dataSet in pandas DataFrame
      _fun - name of agregate action for target values ('mean', 'median', 'classification')
      colLim - dict of pairs:
          key: columnname from _df
         value: list of left and right values of interested interval
    OutPuts:
     function return aggregate of target values of objects corresponded to conditions of intervals per each interesting columns
  '''
   ## chose objects inside of feture intervals
    _cond = None
    for _key in colLim:
        if not (_cond is None):
            _cond = (_cond) & (df[_key] >= colLim[_key][0]) & (df[_key] <= colLim[_key][1])
        else:
            _cond = (df[_key] >= colLim[_key][0]) & (df[_key] <= colLim[_key][1])
    _df_filter = df[_cond]
    # choose objects coresponds in interval
    inIntervalSet = _df_filter[target]

    if fun == 'mean':
        res = np.nanmean(inIntervalSet)
    elif fun == 'median':
        res = np.nanmedian(inIntervalSet)
    elif fun == 'classification':
        if np.size(inIntervalSet, 0) != 0 :
            res = np.sum(inIntervalSet)/np.size(inIntervalSet, 0)
        else:
            res = 1
    else:
        res = np.nanmean(inIntervalSet)
    return round(res,3)


# Подготовка данных для графиков парного влияния параметров на брак
#2D heatmap                                                                     IMPORTANT: MIGHT BE NOT RELIABLE
# сформируем данные для отображения карты в 2 и 3 мерном пространстве

# Создаем массивы NumPy с координатами точек по осям X и У.
# Используем метод meshgrid, при котором по векторам координат
# создается матрица координат. Задаем нужную функцию Z(x, y).

def gridConstruct(_dt, _colLim):
#    colLim - dictionary:
#       key - column name in _dt,
#       value - ndarray,
#       value[0] means min edge,
#       value[1] means max edge,
#       value[2] means step for mesh gfrid
    _grid = []
    _keys = []  ##auxiliary list to remember column names
    for _key in _colLim:
        _keys.append(_key)
        X = np.arange(_colLim[_key][0],  _colLim[_key][1], _colLim[_key][2])
        _grid.append(X)
    xlen = len(_grid[0])
    ylen = len(_grid[1])

    Z = np.zeros((ylen,xlen))
    _gridLim = _colLim.copy()
    for i in range(ylen)[:-1]:
        for k in range(xlen)[:-1]:
            _gridLim = {_keys[1]:[_grid[1][i], _grid[1][i+1]], _keys[0]:[_grid[0][k], _grid[0][k+1]]}
            Z[i,k] = targetMetric(_dt, 'classification', _gridLim)
    X, Y = np.meshgrid(_grid[0], _grid[1])
    return X,Y,Z


#2D plot IMPORTANT: MIGHT BE NOT RELIABLE

def d3ChartShow(X,Y,Z, _cols):
    # 3D-график
    plt.figure(figsize=(14,10))
    surf2d  = plt.contourf(X, Y, Z, 8, cmap='BuPu', grid=True, alpha=0.7)
    plt.colorbar(surf2d)
    plt.xlabel(_cols[0])
    plt.ylabel(_cols[1])
    plt.title(_cols[2])
    plt.show()

'''

 _____                       _         _                         __     __      __      _ 
(_   _)                     ( )       ( )_            _  _     /'__`\ /' _`\  /'__`\  /' )
  | |   __   _ __   _ _    _| |   _ _ | ,_)   _ _    | || |   (_)  ) )| ( ) |(_)  ) )(_, |
  | | /'__`\( '__)/'_` ) /'_` | /'_` )| |   /'_` )   | || |      /' / | | | |   /' /   | |
  | |(  ___/| |  ( (_| |( (_| |( (_| || |_ ( (_| |   | || |    /' /( )| (_) | /' /( )  | |
  (_)`\____)(_)  `\__,_)`\__,_)`\__,_)`\__)`\__,_)   | || |   (_____/'`\___/'(_____/'  (_)
  
  
'''




