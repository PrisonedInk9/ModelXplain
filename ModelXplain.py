#importing all nessessary libraries beforehand

import time
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
import logging as log


from sklearn.model_selection import train_test_split
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
from pdpbox import info_plots
from PyALE import ale
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
from lime.lime_tabular import LimeTabularExplainer


def get_loco_feature_importances(estimated_model, X, y, feature_names=None, normalize_result=True, normalize_num=1.0,
                                 error_type='divide', metrics='mean_squared_error', verbose=False, data_split=False):
    # === FUNCTION SUMMARY ====================================================================================================

    # Input: Trained model estimated_model, feature matrix X, target value y, error measure

    #  1) Input trained model, table of features and coloumn of target values
    #  2) Make a prediction and evaluate start loss array with any metrics acceptable
    #  3) Reshape your model with datasets with every feature alternately excluded and write error for every excluded feature
    #    into separate array
    #  4) For every feature divide the error vector for the excluded feature by the error vector of the full dataset or
    #    subtract the error vector of the full dataset from the error vector for the excluded feature
    #  5) Normalize the result vector if needed

    # TL;DR - this function calculates LOCO feature importances

    # ===LIST OF ARGUMENTS: ===================================================================================================

    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit and .predict methods)  ///
    #  PRE-TRAINED input model which we want to calculate PFI for

    #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values

    #  y  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values

    #  error_type  ///  (string - 'divide' or 'subtract')  ///  option for choosing method for calculating error

    #  metrics  ///  (string - any metrics from  metrics_dict.keys())  ///  option for choosing error calculation method

    #  normalize_num  /// (number)  ///  value for normilizing features upon

    #  verbose  ///  (bool) (OPTIONAL- REMOVAL REQUIRED)  ///  option for outputting detailed

    # === OUTPUT ==============================================================================================================

    # Output  /// (numpy.array) ///  function returns LOCO feature importances

    if error_type != 'divide' and error_type != 'subtract':
        print("Incorrect error type.")
        return

    if normalize_num == 0:
        logging.warning(
            "Incorrect normalization value (zero). Best normalization values for result representation are 1 and 100.")
        return

    metrics_dict = {'mean_squared_error': mean_squared_error}
    if metrics not in metrics_dict.keys():
        logging.warning("Incorrect metrics.")
        return
        # checking inputs
    df = None
    if isinstance(X, (np.ndarray)):
        if feature_names is None:
            feature_names = np.array(range(X.shape[1]))
        df = pd.DataFrame(data=X.to_numpy(), columns=feature_names)
    elif isinstance(X, (pd.core.frame.DataFrame)):
        df = X.copy()
        if feature_names is None:
            feature_names = np.array(X.columns)
    else:
        logging.warning("Incorrect type of X.")
        return

    n_features = df.shape[1]
    result = np.zeros(n_features)

    try:
        estimated_model.fit(df, y)
    except:
        logging.warning("Estimated model must have fit method.")

    try:
        prediction = estimated_model.predict(df)  # training model to calculate initial error
    except:
        logging.warning("Estimated model must have predict method.")

    start_loss = metrics_dict[metrics](y, prediction)

    for i in range(n_features):
        current_feature = feature_names[i]

        df_temp = df.copy()
        df_temp = df_temp.drop(current_feature, axis=1)  # calculating error per every feature's eclusion

        estimated_model.fit(df_temp, y)

        prediction = estimated_model.predict(df_temp)
        feature_loss = metrics_dict[metrics](y, prediction)

        result[i] += feature_loss

    if error_type == 'divide':
        if start_loss == 0:
            start_loss = 1
            if verbose:
                logging.warning("ATTENTION. As start_loss is zero, it was changed to 1 to better represent result.")
        result = (result / start_loss) - 1
    else:
        result = result - start_loss

    if verbose:
        logging.warning('Result:', result)
    if normalize_result:
        result = normalize_num * (result / result.max())
        if verbose:
            logging.warning('Normalized result:', result)

    return result


def get_pfi_feature_importances(estimated_model, X, y, feature_names=None, normalize_result=True, normalize_num=1.0,
                                error_type='divide', metrics='mean_squared_error', shuffle_num=3, verbose=False):
    # === FUNCTION SUMMARY ====================================================================================================

    #  1) Input trained model, table of features and coloumn of target values
    #  2) Make a prediction and evaluate start loss array with any metrics acceptable
    #  3) Calculate the importance of features by dividing the vector of errors of the dataset with shuffled
    #     values by the vector of errors of the original dataset or subtracting the vector of error values of
    #     the original dataset from the vector of errors of the dataset with shuffled values
    #  4) Normalize the result vector if needed

    # TL;DR - this function calculates PFI feature importances

    # ===LIST OF ARGUMENTS: ===================================================================================================

    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit and .predict methods)  ///
    #  input model which we want to calculate PFI for

    #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values

    #  y  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values

    #  error_type  ///  (string - 'divide' or 'subtract')  ///  option for choosing method for calculating error

    #  metrics  ///  (string - any metrics from  metrics_dict.keys())  ///  option for choosing error calculation method

    # shuffle_num  ///  number  ///  number of shuffles of selected feature

    #  normalize_num  /// (number)  ///  value for normilizing features upon

    #  verbose  ///  (bool) (OPTIONAL-REMOVAL REQUIRED)  ///  option for outputting detailed

    # === OUTPUT ==============================================================================================================

    # Output  /// (numpy.array) ///  function returns PFI feature importances

    if error_type != 'divide' and error_type != 'subtract':
        logging.warning("Incorrect error type.")
        return

    if shuffle_num == 0:
        logging.warning("Incorrect shuffle num. Shuffle amount should be at least 1.")
        return

    if normalize_num == 0:
        logging.warning(
            "Incorrect normalization value (zero). Best normalization values for result representation are 1 and 100.")
        return

    metrics_dict = {'mean_squared_error': mean_squared_error}
    if metrics not in metrics_dict.keys():
        logging.warning("Incorrect metrics.")
        return

    df = None
    if isinstance(X, (np.ndarray)):  # check inputs
        if feature_names is None:
            feature_names = np.array(range(X.shape[1]))
        df = pd.DataFrame(data=X.to_numpy(), columns=feature_names)
    elif isinstance(X, (pd.core.frame.DataFrame)):
        df = X.copy()
        if feature_names is None:
            feature_names = np.array(X.columns)
    else:
        logging.warning("Incorrect type of X.")
        return

    # -- main part --

    n_features = df.shape[1]
    result = np.zeros(n_features)

    try:
        estimated_model.fit(df, y)
    except:
        logging.warning("Estimated model must have fit method.")

    try:
        prediction = estimated_model.predict(df)
    except:
        logging.warning("Estimated model must have predict method.")

    start_loss = metrics_dict[metrics](y, prediction)

    for i in range(n_features):
        current_feature = feature_names[i]

        # shuffling values to kill any correllation

        for j in range(shuffle_num):
            df_shuffled = df.copy()
            df_shuffled[current_feature] = df[current_feature].sample(frac=1).reset_index(drop=True)

            prediction = estimated_model.predict(df_shuffled)
            feature_loss = metrics_dict[metrics](y, prediction)

            result[i] += feature_loss

    result = result / shuffle_num

    if error_type == 'divide':
        result = result / start_loss

    else:
        result = result - start_loss

    if normalize_result:
        result = normalize_num * (result / result.max())

    return result


def get_genie_feature_importances(estimated_model, X_train, y_train, normalize_result=True,
                                  prefit=True, normalize_num=1.0, verbose=False):
    # === FUNCTION SUMMARY ====================================================================================================

    # Just a simple overlay of the standard SKlearn Genie criteria-based feature importance method

    # ===LIST OF ARGUMENTS: ===================================================================================================

    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit and .feature_importances_
    # methods)  /// input model which we want to calculate feature importances for for

    # X_train ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for training model

    # y_train  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for training model

    # normalize_num  /// (number)  ///  value for normilizing features upon

    # prefit  ///  (bool)  ///  indicator of whether you provide a pre-trained model or not

    #  verbose  ///  (bool) (OPTIONAL-REMOVAL REQUIRED)  ///  option for outputting detailed

    # === OUTPUT ==============================================================================================================

    # Output  /// (numpy.array) ///  function returns feature importances

    if not isinstance(X, (pd.DataFrame)):
        logging.warning("MISSING ARGUMENT: X.")
        return

    if not isinstance(y, (pd.DataFrame)):
        logging.warning("MISSING ARGUMENT: y.")
        return

    if normalize_num == 0:
        logging.warning(
            "Incorrect normalization value (zero). Best normalization values for result representation are 1 and 100.")
        return

    if prefit == False:
        try:
            estimated_model.fit(X_train, y_train)
        except:
            logging.warning("Estimated model must have fit method.")

    try:
        feature_importance = estimated_model.feature_importances_
    except:
        logging.warning("Estimated model must have feature_importances_ method.")

    if normalize_result == True:
        feature_importance = normalize_num * (feature_importance / feature_importance.max())

    return feature_importance


def ice_plot_2D(estimated_model, X_train, y_train, X_test, y_test, feature_names, target_feature, prefit=True,
                verbose=False):
    # === FUNCTION SUMMARY ====================================================================================================

    # Just a simple overlay of the pdpbox library function pdp_isolate for PDP/ICE plotting

    # ===LIST OF ARGUMENTS: ===================================================================================================

    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit and .feature_importances_
    #  methods)  /// input model which we want to plot partial dependence for

    #  X_train ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for training model

    #  y_train  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for training model

    #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for plotting PDP

    #  y  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for plotting PDP

    #  feature_names  ///  (list)  ///   a list of feature names

    #  target_name  ///  (string)  ///  name of the target feature to work with

    #  prefit  ///  (bool)  ///  indicator of whether you provide a pre-trained model or not  (OPTIONAL)

    #  verbose  ///  (bool) (OPTIONAL-REMOVAL REQUIRED)  ///  option for outputting detailed (OPTIONAL)

    # === OUTPUT ==============================================================================================================

    # Output  ///  plots a PDP plot

    if not isinstance(X, (pd.DataFrame)):
        logging.warning("MISSING ARGUMENT: X.")
        return

    if not isinstance(y, (pd.DataFrame)):
        logging.warning("MISSING ARGUMENT: y.")
        return

    if not isinstance(feature_names, (list)):
        logging.warning("MISSING ARGUMENT: feature_names.")
        return

    if not isinstance(target_feature, (string)):
        logging.warning("MISSING ARGUMENT: target_feature.")
        return

    from matplotlib import pyplot as plt
    from pdpbox import pdp, get_dataset, info_plots

    if prefit == False:
        try:
            estimated_model.fit(X_train, y_train)
        except:
            logging.warning("Estimated model must have fit method.")

    pdp_goals = pdp.pdp_isolate(model=estimated_model, dataset=X_test, model_features=feature_names,
                                feature=target_feature, num_grid_points=30, grid_type='equal', grid_range=(1, 5))

    pdp.pdp_plot(pdp_goals, target_feature)
    plt.show()

    def pdp_values(estimated_model, X_train, y_train, X_test, y_test, feature_names, target_feature, target_val_upper,
                   target_val_lower, grid_points_val=30, prefit=True, verbose=False):

        # === FUNCTION SUMMARY ====================================================================================================

        # Just a simple overlay of the pdpbox library function pdp_isolate for calculating PDP values of test feature, at which the
        # target variable falls into the specified interval

        # ===LIST OF ARGUMENTS: ==================================================================================================

        #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit and .feature_importances_
        #  methods)  /// input model which we want to plot partial dependence for

        #  X_train ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for training model

        #  y_train  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for training model

        #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for calculatig intervals

        #  y  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for calculatig intervals

        #  feature_names  ///  (list)  ///   a list of feature names

        #  target_feature  ///  (string)  ///  name of the target feature to work with

        #  target_val_upper  ///  (value)  ///  upper-bound value of target value

        #  target_val_lower  ///  (value)  ///  lower-bound value of target value

        #  grid_points_val  ///  (value)  ///  number of PDP points

        #  prefit  ///  (bool)  ///  indicator of whether you provide a pre-trained model or not  (OPTIONAL)

        #  verbose  ///  (bool) (OPTIONAL-REMOVAL REQUIRED)  ///  option for outputting detailed (OPTIONAL)

        # === OUTPUT ==============================================================================================================

        # Output  ///  (numpy.array)  ///  2-coloumn array of intervals of PDP values on which target value falls into
        # defined interval  (NEEDS MORE WORK)

        if not isinstance(X, (pd.DataFrame)):
            logging.warning("MISSING ARGUMENT: X.")
            return

        if not isinstance(y, (pd.DataFrame)):
            logging.warning("MISSING ARGUMENT: y.")
            return

        if not isinstance(feature_names, (Lists)):
            logging.warning("MISSING ARGUMENT: feature_names.")
            return

        if not isinstance(target_feature, (string)):
            logging.warning

        if not isinstance(target_val_upper, (Numbers)):
            logging.warning("MISSING ARGUMENT: target_val_upper.")
            return

        if not isinstance(target_val_lower, (Numbers)):
            logging.warning("MISSING ARGUMENT: target_val_lower.")
            return

        if grid_points_val == 0:
            logging.warning("Incorrect number of grid pints (zero). Input value that is at least equal to 1")

        if prefit == False:
            try:
                estimated_model.fit(X_train, y_train)
            except:
                logging.warning("Estimated model must have fit method.")

        pdp_goals = pdp.pdp_isolate(model=estimated_model, dataset=X_test, model_features=feature_names,
                                    feature=target_feature, num_grid_points=grid_points_val, grid_type='equal',
                                    grid_range=(1, 5))

        intervals = np.zeros([grid_points_val, 3])
        intervals_values = np.zeros([2, 2])

        for i in range(0, grid_points_val):
            intervals[i, 0] = pdp_goals.pdp[i]
            intervals[i, 1] = pdp_goals.display_columns[i]
            if pdp_goals.pdp[i] > target_val_lower and pdp_goals.pdp[i] < target_val_upper:
                intervals[i, 2] = 1
            else:
                intervals[i, 2] = -1

        flag = True

        for i in range(0, grid_points_val):
            if intervals[i, 2] == 1 and flag:
                start = intervals[i, 1]
                end = intervals[i, 1]
                flag = False

            elif intervals[i, 2] == 1 and i == grid_points_val - 1:
                end = intervals[i, 1]
                intervals_values = np.append(intervals_values, [start, end])
                print('Interval: [', start, ' , ', end, ']1')

            elif intervals[i, 2] == -1 and i == grid_points_val - 1:
                intervals_values = np.append(intervals_values, [start, end])
                print('Interval: [', start, ' , ', end, ']2')


            elif intervals[i, 2] == 1 and not flag:
                end = intervals[i, 1]

            elif intervals[i, 2] == -1 and not flag:
                flag = True
                intervals_values = np.append(intervals_values, [start, end])
                print('Interval: [', start, ' , ', end, ']3')

        return intervals_values


def pdp_values(estimated_model, X_train, y_train, X_test, y_test, feature_names, target_feature, target_val_upper,
               target_val_lower, grid_points_val=30, prefit=True, verbose=False):
    # === FUNCTION SUMMARY ====================================================================================================

    # Just a simple overlay of the pdpbox library function pdp_isolate for calculating PDP values of test feature, at which the
    # target variable falls into the specified interval

    # ===LIST OF ARGUMENTS: ==================================================================================================

    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit and .feature_importances_
    #  methods)  /// input model which we want to plot partial dependence for

    #  X_train ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for training model

    #  y_train  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for training model

    #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for calculatig intervals

    #  y  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for calculatig intervals

    #  feature_names  ///  (list)  ///   a list of feature names

    #  target_feature  ///  (string)  ///  name of the target feature to work with

    #  target_val_upper  ///  (value)  ///  upper-bound value of target value

    #  target_val_lower  ///  (value)  ///  lower-bound value of target value

    #  grid_points_val  ///  (value)  ///  number of PDP points

    #  prefit  ///  (bool)  ///  indicator of whether you provide a pre-trained model or not  (OPTIONAL)

    #  verbose  ///  (bool) (OPTIONAL-REMOVAL REQUIRED)  ///  option for outputting detailed (OPTIONAL)

    # === OUTPUT ==============================================================================================================

    # Output  ///  (numpy.array)  ///  2-coloumn array of intervals of PDP values on which target value falls into
    # defined interval  (NEEDS MORE WORK)

    if not isinstance(X, (pd.DataFrame)):
        logging.warning("MISSING ARGUMENT: X.")
        return

    if not isinstance(y, (pd.DataFrame)):
        logging.warning("MISSING ARGUMENT: y.")
        return

    if not isinstance(feature_names, (Lists)):
        logging.warning("MISSING ARGUMENT: feature_names.")
        return

    if not isinstance(target_feature, (string)):
        logging.warning

    if not isinstance(target_val_upper, (Numbers)):
        logging.warning("MISSING ARGUMENT: target_val_upper.")
        return

    if not isinstance(target_val_lower, (Numbers)):
        logging.warning("MISSING ARGUMENT: target_val_lower.")
        return

    if grid_points_val == 0:
        logging.warning("Incorrect number of grid pints (zero). Input value that is at least equal to 1")

    if prefit == False:
        try:
            estimated_model.fit(X_train, y_train)
        except:
            logging.warning("Estimated model must have fit method.")

    pdp_goals = pdp.pdp_isolate(model=estimated_model, dataset=X_test, model_features=feature_names,
                                feature=target_feature, num_grid_points=grid_points_val, grid_type='equal',
                                grid_range=(1, 5))

    intervals = np.zeros([grid_points_val, 3])
    intervals_values = np.zeros([2, 2])

    for i in range(0, grid_points_val):
        intervals[i, 0] = pdp_goals.pdp[i]
        intervals[i, 1] = pdp_goals.display_columns[i]
        if pdp_goals.pdp[i] > target_val_lower and pdp_goals.pdp[i] < target_val_upper:
            intervals[i, 2] = 1
        else:
            intervals[i, 2] = -1

    flag = True

    for i in range(0, grid_points_val):
        if intervals[i, 2] == 1 and flag:
            start = intervals[i, 1]
            end = intervals[i, 1]
            flag = False

        elif intervals[i, 2] == 1 and i == grid_points_val - 1:
            end = intervals[i, 1]
            intervals_values = np.append(intervals_values, [start, end])
            print('Interval: [', start, ' , ', end, ']1')

        elif intervals[i, 2] == -1 and i == grid_points_val - 1:
            intervals_values = np.append(intervals_values, [start, end])
            print('Interval: [', start, ' , ', end, ']2')


        elif intervals[i, 2] == 1 and not flag:
            end = intervals[i, 1]

        elif intervals[i, 2] == -1 and not flag:
            flag = True
            intervals_values = np.append(intervals_values, [start, end])
            print('Interval: [', start, ' , ', end, ']3')

    return intervals_values


def ice_values(estimated_model, X_train, y_train, X_test, y_test, feature_names, target_feature,
               grid_val_start, grid_val_end, normalize_result=True, prefit=True, verbose=False):
    # === FUNCTION SUMMARY ====================================================================================================

    # Just a simple overlay of the pdpbox library function pdp_isolate for calculating ICE values

    # ===LIST OF ARGUMENTS: ===================================================================================================

    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit and .feature_importances_
    # methods)  /// input model which we want to calculate ICE values for

    #  X_train ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for training model

    #  y_train  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for training model

    #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for plotting PDP

    #  y  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for plotting PDP

    #  feature_names  ///  (list)  ///   a list of feature names

    #  target_name  ///  (string)  ///  name of the target feature to work with

    #  grid_val_start  ///  (number)  ///  start ov the interval

    #  grid_val_end  ///  (number)  ///  end of the interval

    #  prefit  ///  (bool)  ///  indicator of whether you provide a pre-trained model or not  (OPTIONAL)

    #  verbose  ///  (bool) (OPTIONAL-REMOVAL REQUIRED)  ///  option for outputting detailed (OPTIONAL)

    # === OUTPUT ==============================================================================================================

    # pdp_goals.ice_lines  ///  (pandas.DataFrame)  /// outputs ICE values

    if not isinstance(X, (pd.DataFrame)):
        logging.warning("MISSING ARGUMENT: X.")
        return

    if not isinstance(y, (pd.DataFrame)):
        logging.warning("MISSING ARGUMENT: y.")
        return

    if not isinstance(grid_val_start, (Numbers)):
        logging.warning("MISSING ARGUMENT: grid_val_start.")
        return

    if not isinstance(grid_val_end, (Numbers)):
        logging.warning("MISSING ARGUMENT: grid_val_end.")

    if prefit == False:
        try:
            estimated_model.fit(X_train, y_train)
        except:
            logging.warning("Estimated model must have fit method.")

    pdp_goals = pdp.pdp_isolate(model=estimated_model, dataset=X_test, model_features=feature_names,
                                feature=target_feature, num_grid_points=30, grid_type='equal',
                                grid_range=(grid_val_start, grid_val_end))

    return pdp_goals.ice_lines


def shap_plot(estimated_model, X, X_train='foo', y_train='foo', prefit=True, verbose=False):
    # === FUNCTION SUMMARY ====================================================================================================

    # Just a simple overlay of the shap library method  waterfall for calculating SHAP plots

    # ===LIST OF ARGUMENTS: ===================================================================================================

    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit method)
    # /// input model which we want to plot SHAP for

    #  X_train ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for training model

    #  y_train  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for training model

    #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for plotting PDP

    #  prefit  ///  (bool)  ///  indicator of whether you provide a pre-trained model or not  (OPTIONAL)

    #  verbose  ///  (bool) (OPTIONAL-REMOVAL REQUIRED)  ///  option for outputting detailed (OPTIONAL)

    # === OUTPUT ==============================================================================================================

    # Output  /// outputs SHAP plot

    if not isinstance(X, (pd.DataFrame)):
        logging.warning("MISSING ARGUMENT: X.")
        return

    if prefit == False:
        try:
            estimated_model.fit(X_train, y_train)
        except:
            logging.warning("Estimated model must have fit method.")

    explainer = shap.Explainer(estimated_model)
    shap_values = explainer(X)
    print(type(shap_values))
    shap.plots.waterfall(shap_values[0])


def lime_plot(estimated_model, X, X_train, y_train, max_feature_amount=10, selection_num=25, prefit=True,
              work_mode='regression', verbose=False):
    # === FUNCTION SUMMARY ====================================================================================================

    # Just a simple overlay of the shap library method  waterfall for calculating SHAP plots

    # ===LIST OF ARGUMENTS: ===================================================================================================

    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit and .predict method)
    # /// input model which we want to plot LIME for

    #  X_train ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for training model

    #  y_train  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for training model

    #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for plotting PDP

    #  prefit  ///  (bool)  ///  indicator of whether you provide a pre-trained model or not  (OPTIONAL)

    #  selection_num  ///  (value)  ///  number of elements for plotting LIME (???)

    #  work_mode  ///  (string)  ///  work mode, 'regression' bu default. (ATTENTION - 'classification' MODE IS NOT SUPPORTED YET)

    #  verbose  ///  (bool) (OPTIONAL-REMOVAL REQUIRED)  ///  option for outputting detailed (OPTIONAL)

    # === OUTPUT ==============================================================================================================

    # Output  /// outputs SHAP plot

    if not isinstance(X, (pd.DataFrame)):
        logging.warning("MISSING ARGUMENT: .")
        return

    if not isinstance(selection_num, (Numbers)):
        logging.warning("MISSING ARGUMENT: selection_num.")
        return

    if not isinstance(max_feature_amount, (Numbers)):
        logging.warning("MISSING ARGUMENT: max_feature_amount.")
        return

    if prefit == False:
        try:
            estimated_model.fit(X_train, y_train)
        except:
            logging.warning("Estimated model must have fit method.")

    explainer = LimeTabularExplainer(training_data=X.to_numpy(),
                                     feature_names=list(X.columns),
                                     mode=work_mode, random_state=0)

    exp = explainer.explain_instance(X.to_numpy()[selection_num], estimated_model.predict,
                                     num_features=max_feature_amount)

    exp.as_pyplot_figure()

    from matplotlib import pyplot as plt
    plt.tight_layout()


def pdp_3d_plot(estimated_model, X_train, y_train, X, feature_names, feature_name_1, feature_name_2,
                prefit=True, verbose=False):
    # === FUNCTION SUMMARY ====================================================================================================

    # Just a simple overlay of the shap library method  waterfall for calculating 3D PDP plot for 2 features' interaction

    # ===LIST OF ARGUMENTS: ===================================================================================================

    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit and .predict method)
    # /// input model which we want to plot LIME for

    #  X_train ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for training model

    #  y_train  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for training model

    #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for plotting PDP

    #  feature_name_1  ///  (string)  ///  1st feature name to plot pdp interaction for

    #  feature_name_2  ///  (string)  ///  2nd feature name to plot pdp interaction for

    #  prefit  ///  (bool)  ///  indicator of whether you provide a pre-trained model or not  (OPTIONAL)

    #  verbose  ///  (bool) (OPTIONAL-REMOVAL REQUIRED)  ///  option for outputting detailed (OPTIONAL)

    # === OUTPUT ==============================================================================================================

    # Output  /// outputs 3D "heat" PDP plot

    if not isinstance(X, (pd.DataFrame)):
        logging.warning("MISSING ARGUMENT: .")
        return

    if not isinstance(feature_name_1, (string)):
        logging.warning("MISSING ARGUMENT: feature_name_1.")
        return

    if not isinstance(feature_name_2, (string)):
        logging.warning("MISSING ARGUMENT: feature_name_2.")
        return

    pdp_goal = pdp.pdp_interact(model=estimated_model, dataset=X, model_features=feature_names,
                                features=[feature_name_1, feature_name_2])

    fig, axes = pdp.pdp_interact_plot(pdp_interact_out=pdp_goal,
                                      feature_names=[feature_name_1, feature_name_2],
                                      plot_type='contour',
                                      x_quantile=True,
                                      plot_pdp=True)

    fig.show()


 _____                       _         _                         __     __      __      _
(_   _)                     ( )       ( )_            _  _     /'__`\ /' _`\  /'__`\  /' )
  | |   __   _ __   _ _    _| |   _ _ | ,_)   _ _    ( )( )   (_)  ) )| ( ) |(_)  ) )(_, |
  | | /'__`\( '__)/'_` ) /'_` | /'_` )| |   /'_` )   | || |      /' / | | | |   /' /   | |
  | |(  ___/| |  ( (_| |( (_| |( (_| || |_ ( (_| |   | || |    /' /( )| (_) | /' /( )  | |
  (_)`\____)(_)  `\__,_)`\__,_)`\__,_)`\__)`\__,_)   | || |   (_____/'`\___/'(_____/'  (_)
