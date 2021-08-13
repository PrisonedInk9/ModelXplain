# Ultimate model explanation library
# 
# Have fun!

#importing all nessessary libraries beforehand

import time 
import shap
import numpy as np
import pandas as pd
import ipywidgets as widgets
import logging

import statistics as stat
from sklearn.model_selection import train_test_split
from pdpbox import info_plots
from PyALE import ale
from sklearn.metrics import mean_squared_error
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots


def get_loco_feature_importances(estimated_model, X, y, feature_names=None, normalize_result=True, normalize_num=1.0, 
                                 error_type='divide', metrics='mean_squared_error', verbose=False, data_split=False,
                                 prefit = True, X_train = None, y_train = None):
    
    
   
    # === FUNCTION SUMMARY ============================================================================================

    # Input: Trained model estimated_model, feature matrix X, target value y, error measure 

    #  1) Input trained model, table of features and coloumn of target values
    #  2) Make a prediction and evaluate start loss array with any metrics acceptable
    #  3) Reshape your model with datasets with every feature alternately excluded and write error for every excluded feature 
    #    into separate array
    #  4) For every feature divide the error vector for the excluded feature by the error vector of the full dataset or 
    #    subtract the error vector of the full dataset from the error vector for the excluded feature 
    #  5) Normalize the result vector if needed
    
    # TL;DR - this function calculates LOCO feature importances

    # ===LIST OF ARGUMENTS: ===========================================================================================
    
    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit and .predict methods)  /// 
    #  input model which we want to calculate PFI for
    
    #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values
    
    #  y  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values
    
    #  error_type  ///  (string - 'divide' or 'subtract')  ///  option for choosing method for calculating error
    
    #  metrics  ///  (string - any metrics from  metrics_dict.keys())  ///  option for choosing error calculation method
    
    #  normalize_num  /// (number)  ///  value for normilizing features upon calculating error per every feature's eclusion (OPTIONAL)
    
    #  data_split  ///  (bool)   ///  option for splitting data when calculatig 
    
    #  prefit  ///  (bool)  ///  indicator of whether you provide a pre-trained model or not  (OPTIONAL)
    
    #  X_train ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for training model (OPTIONAL)
    
    #  y_train  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for training model (OPTIONAL)
      
    #  verbose  ///  (bool) (OPTIONAL- REMOVAL REQUIRED)  ///  option for outputting detailed 
    
    
    # === OUTPUT ======================================================================================================
    
    # Output  /// (numpy.array) ///  function returns LOCO feature importances
    
    if estimated_model is None:
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: estimated_model.")
        return
    
    if not (isinstance(X_train, (pd.DataFrame)) or isinstance(X_train, (np.ndarray))):
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE : X_train.")
        return
    
    if not (isinstance(y_train, (pd.DataFrame)) or isinstance(y_train, (np.ndarray))):
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: y_train.")
        return
    
    if not prefit:
        try:
            estimated_model.fit(X_train, y_train)
        except:
            logging.warning("Estimated model must have fit method.")
            
    if error_type != 'divide' and error_type != 'subtract':
        print("Incorrect error type.")
        return
    
    if normalize_num <= 0:
        logging.warning("Incorrect normalization value (zero). Best normalization values for result representation are 1 and 100.")
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
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: X.")
        return
    
    n_features = df.shape[1]
    result = np.zeros(n_features)
    
    try:
        estimated_model.fit(df, y)
    except:
        logging.warning("Estimated model must have fit method.")
        return

    try:   
        prediction = estimated_model.predict(df) # training model to calculate initial error
    except:
        logging.warning("Estimated model must have predict method.")
        return

    start_loss = metrics_dict[metrics](y, prediction)

    for i in range(n_features):
        current_feature = feature_names[i]
        
        if data_split:
        
            df_temp = df.copy()
            df_train, df_test, y_train, y_test = train_test_split(df_temp, y, test_size=0.23, random_state=0)
        
            df_train = df_train.drop(current_feature,  axis=1)
            df_test = df_test.drop(current_feature,  axis=1)
        
            estimated_model.fit(df_train, y_train)
        
            prediction = estimated_model.predict(df_test)
            feature_loss = metrics_dict[metrics](y_test, prediction)
        
        else:
            df_temp = df.copy()
            df_temp = df_temp.drop(current_feature, axis=1)   # calculating error per every feature's eclusion
        
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
                                error_type='divide', metrics='mean_squared_error', shuffle_num=3, verbose=False,
                                prefit = True, X_train = None, y_train = None):    
    
    # === FUNCTION SUMMARY ============================================================================================

    #  1) Input trained model, table of features and coloumn of target values
    #  2) Make a prediction and evaluate start loss array with any metrics acceptable
    #  3) Calculate the importance of features by dividing the vector of errors of the dataset with shuffled 
    #     values by the vector of errors of the original dataset or subtracting the vector of error values of 
    #     the original dataset from the vector of errors of the dataset with shuffled values 
    #  4) Normalize the result vector if needed
    
    # TL;DR - this function calculates PFI feature importances

    # ===LIST OF ARGUMENTS: ===========================================================================================
    
    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit and .predict methods)  /// 
    #  input model which we want to calculate PFI for
    
    #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values
    
    #  y  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values
    
    #  error_type  ///  (string - 'divide' or 'subtract')  ///  option for choosing method for calculating error
    
    #  metrics  ///  (string - any metrics from  metrics_dict.keys())  ///  option for choosing error calculation method
    
    #  shuffle_num  ///  number  ///  number of shuffles of selected feature
    
    #  normalize_num  /// (number)  ///  value for normilizing features upon
    
    #  verbose  ///  (bool) (OPTIONAL-REMOVAL REQUIRED)  ///  option for outputting detailed 
       
    # === OUTPUT ======================================================================================================
    
    # Output  /// (numpy.array) ///  function returns PFI feature importances  
    
    if estimated_model is None:
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: estimated_mode.")
        return
    
    if not (isinstance(X_train, (pd.DataFrame)) or isinstance(X_train, (np.ndarray))):
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE : X_train.")
        return
    
    if not (isinstance(y_train, (pd.DataFrame)) or isinstance(y_train, (np.ndarray))):
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: y_train.")
        return
    
    if not prefit:
        try:
            estimated_model.fit(X_train, y_train)
        except:
            logging.warning("Estimated model must have fit method.")
            return
    
    if error_type != 'divide' and error_type != 'subtract':
        logging.warning("Incorrect error type.")
        return
    
    if shuffle_num <= 0:
        logging.warning("Incorrect shuffle num. Shuffle amount should be at least 1.")
        return
    
    if normalize_num <= 0:
        logging.warning("Incorrect normalization value (zero). Best normalization values for result representation are 1 and 100.")
        return
    
    metrics_dict = {'mean_squared_error': mean_squared_error}
    if metrics not in metrics_dict.keys():
        logging.warning("Incorrect metrics.")
        return
    
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
    
    # -- main part --
    
    n_features = df.shape[1]
    result = np.zeros(n_features)
    
    try:
        estimated_model.fit(df, y)
    except:
        logging.warning("Estimated model must have fit method.")
        return
        
    try:
        prediction = estimated_model.predict(df)
    except:
        logging.warning("Estimated model must have predict method.")
        return
     
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


def ice_plot_2D(estimated_model, X, y, feature_names, target_feature, prefit=True, 
                X_train=None, y_train=None, verbose=False):
    
    
    # === FUNCTION SUMMARY ============================================================================================

    # Just a simple overlay of the pdpbox library function pdp_isolate for PDP/ICE plotting

    # ===LIST OF ARGUMENTS: ===========================================================================================
    
    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit and .feature_importances_ 
    #  methods)  /// input model which we want to plot partial dependence for
    
    #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for plotting PDP
    
    #  y  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for plotting PDP 
    
    #  feature_names  ///  (list)  ///   a list of feature names 
    
    #  target_name  ///  (string)  ///  name of the target feature to work with
    
    #  prefit  ///  (bool)  ///  indicator of whether you provide a pre-trained model or not  (OPTIONAL)
    
    #  X_train ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for training model (OPTIONAL)
    
    #  y_train  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for training model (OPTIONAL)
    
    #  verbose  ///  (bool) (OPTIONAL-REMOVAL REQUIRED)  ///  option for outputting detailed (OPTIONAL)
       
    # === OUTPUT ======================================================================================================
    
    # Output  ///  plots a PDP plot
    
    if estimated_model is None:
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: estimated_mode.")
        return
    
    if not (isinstance(X, (pd.DataFrame)) or isinstance(X, (np.ndarray))): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: X.")
        return
        
    if not (isinstance(y, (pd.DataFrame)) or isinstance(y, (np.ndarray))): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: y.")
        return
    
    if not isinstance(feature_names, (list)): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: feature_names.")
        return
    
    if not isinstance(target_feature, (str)): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: target_feature.")
        return
    
    if not (isinstance(X_train, (pd.DataFrame)) or isinstance(X_train, (np.ndarray))):
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE : X_train.")
        return
    
    if not (isinstance(y_train, (pd.DataFrame)) or isinstance(y_train, (np.ndarray))):
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: y_train.")
        return
    
    if not prefit:
        try:
            estimated_model.fit(X_train, y_train)
        except:
            logging.warning("Estimated model must have fit method.")

    pdp_goals = pdp.pdp_isolate(model=estimated_model, dataset=X, model_features=feature_names,
                            feature=target_feature, num_grid_points=30, grid_type='equal', grid_range=(1, 5))

    pdp.pdp_plot(pdp_goals, target_feature)
    plt.show()


def pdp_values(estimated_model, X, y, feature_names, target_feature, target_val_upper, target_val_lower, grid_points_val = 30,
               normalize_result=True, prefit=True, X_train = None, y_train = None, normalize_num=1.0, verbose=False):
    
    
    # === FUNCTION SUMMARY ============================================================================================

    # Just a simple overlay of the pdpbox library function pdp_isolate for calculating PDP values

    # ===LIST OF ARGUMENTS: ===========================================================================================
    
    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit and .feature_importances_ 
    #  methods)  /// input model which we want to plot partial dependence for
    
    #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for plotting PDP
    
    #  y  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for plotting PDP 
    
    #  feature_names  ///  (list)  ///   a list of feature names 
    
    #  target_name  ///  (string)  ///  name of the target feature to work with
    
    #  normalize_num  /// (number)  ///  value for normilizing features upon  (OPTIONAL)
    
    #  prefit  ///  (bool)  ///  indicator of whether you provide a pre-trained model or not  (OPTIONAL)
    
    #  X_train ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for training model (OPTIONAL)
    
    #  y_train  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for training model (OPTIONAL) 
    
    #  verbose  ///  (bool) (OPTIONAL-REMOVAL REQUIRED)  ///  option for outputting detailed (OPTIONAL)
    
    # === OUTPUT ======================================================================================================
    
    # Output  ///  (list([tuple], value, value))  ///  outputs the intervals of feature values where the target variable falls 
    # within the specified interval, the average and median within these intervals  
    
    if estimated_model is None:
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: estimated_mode.")
        return
    
    if not (isinstance(X, (pd.DataFrame)) or isinstance(X, (np.ndarray))): 
            logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: X.")
            return
        
    if not (isinstance(y, (pd.DataFrame)) or isinstance(y, (np.ndarray))): 
            logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: y.")
            return
    
    if not isinstance(feature_names, (list)): 
            logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: feature_names.")
            return
    
    if not isinstance(target_feature, (str)): 
            logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: target_feature.")
            return    
    
    if not (isinstance(X_train, (pd.DataFrame)) or isinstance(X_train, (np.ndarray))):
            logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE : X_train.")
            return
    
    if not (isinstance(y_train, (pd.DataFrame)) or isinstance(y_train, (np.ndarray))):
            logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: y_train.")
            return  
        
    if not(isinstance(target_val_upper, int) or isinstance(target_val_upper, np.int_) or isinstance(target_val_upper, np.intc)):
            logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE : target_val_upper.")
            return
    
    if not(isinstance(target_val_lower, int) or isinstance(target_val_lower, np.int_) or isinstance(target_val_lower, np.intc)):
            logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: target_val_lower.")
            return
    
    if not prefit:
        try:
            estimated_model.fit(X_train, y_train)
        except:
            logging.warning("Estimated model must have fit method.")

    pdp_goals = pdp.pdp_isolate(model=estimated_model, dataset=X, model_features=feature_names, 
                            feature=target_feature, num_grid_points=grid_points_val, grid_type='equal', grid_range=(1, 5))

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

    print(interval_list)
    print(average_val_list)
    print(median_val_list)
    
    finals_list = [interval_list, average_val_list, median_val_list]
    
    return finals_list


def ice_values(estimated_model, X, y, feature_names, target_feature, grid_val_start, grid_val_end, 
               normalize_result=True, prefit=True, X_train = None, y_train = None, verbose=False):

    # === FUNCTION SUMMARY =========================================================================================

    # Just a simple overlay of the pdpbox library function pdp_isolate for calculating ICE values

    # ===LIST OF ARGUMENTS: ========================================================================================
    
    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit and .feature_importances_ 
    # methods)  /// input model which we want to calculate ICE values for
    
    #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for plotting PDP
    
    #  y  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for plotting PDP 
    
    #  feature_names  ///  (list)  ///   a list of feature names 
    
    #  target_name  ///  (string)  ///  name of the target feature to work with
    
    #  grid_val_start  ///  (number)  ///  start ov the interval
    
    #  grid_val_end  ///  (number)  ///  end of the interval
    
    #  prefit  ///  (bool)  ///  indicator of whether you provide a pre-trained model or not  (OPTIONAL)
    
    #  X_train ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for training model (OPTIONAL)
    
    #  y_train  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for training model (OPTIONAL)
    
    #  verbose  ///  (bool) (OPTIONAL-REMOVAL REQUIRED)  ///  option for outputting detailed (OPTIONAL)
       
    # === OUTPUT ======================================================================================================
    
    # pdp_goals.ice_lines  ///  (pandas.DataFrame)  /// outputs ICE values
     
    if estimated_model is None:
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: estimated_mode.")
        return
    
    if not (isinstance(X, (pd.DataFrame)) or isinstance(X, (np.ndarray))): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: X.")
        return
        
    if not (isinstance(y, (pd.DataFrame)) or isinstance(y, (np.ndarray))): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: y.")
        return
    
    if not(isinstance(grid_val_start, int) or isinstance(grid_val_start, np.int_) or isinstance(grid_val_start, np.intc)): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: grid_val_start.")
        return
    
    if not(isinstance(grid_val_end, int) or isinstance(grid_val_end, np.int_) or isinstance(grid_val_end, np.intc)): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: grid_val_end.")

    if not (isinstance(X_train, (pd.DataFrame)) or isinstance(X_train, (np.ndarray))): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: X_train.")
        return
        
    if not (isinstance(y_train, (pd.DataFrame)) or isinstance(y_train, (np.ndarray))): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: y_train.")
        return
    
    if prefit == False:
        try:
            estimated_model.fit(X_train, y_train)
        except:
            logging.warning("Estimated model must have fit method.")

    pdp_goals = pdp.pdp_isolate(model=estimated_model, dataset=X, model_features=feature_names,
                            feature=target_feature, num_grid_points=30, grid_type='equal', 
                            grid_range=(grid_val_start, grid_val_end))

    return pdp_goals.ice_lines


def shap_plot(estimated_model, X, prefit=True, X_train=None, y_train=None, verbose=False):

   
    # === FUNCTION SUMMARY ====================================================================================================

    # Just a simple overlay of the shap library method  waterfall for calculating SHAP plots

    # ===LIST OF ARGUMENTS: ===================================================================================================
    
    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit method)  
    #  /// input model which we want to plot SHAP for
     
    #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for plotting PDP
    
    #  X_train ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for training model (OPTIONAL)
    
    #  y_train  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for training model (OPTIONAL)
    
    #  prefit  ///  (bool)  ///  indicator of whether you provide a pre-trained model or not  (OPTIONAL)
    
    #  verbose  ///  (bool) (OPTIONAL-REMOVAL REQUIRED)  ///  option for outputting detailed (OPTIONAL)
       
    # === OUTPUT ==============================================================================================================
    
    # Output  /// outputs SHAP plot
    
    if estimated_model is None:
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: estimated_mode.")
        return
    
    if not (isinstance(X, (pd.DataFrame)) or isinstance(X, (np.ndarray))): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: X.")
        return
    
    if not (isinstance(X_train, (pd.DataFrame)) or isinstance(X_train, (np.ndarray))): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: X_train.")
        return
        
    if not (isinstance(y_train, (pd.DataFrame)) or isinstance(y_train, (np.ndarray))): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: y_train.")
        return
    
    if not prefit:
        try:
            estimated_model.fit(X_train, y_train)
        except:
            logging.warning("Estimated model must have fit method.")

    explainer = shap.Explainer(estimated_model)
    shap_values = explainer(X)
    print(type(shap_values))
    shap.plots.waterfall(shap_values[0])


def lime_plot(estimated_model, X, max_feature_amount = 10, selection_num = 25, prefit = True, 
              X_train = None, y_train = None, work_mode = 'regression', verbose = False):

    # === FUNCTION SUMMARY ============================================================================================

    # Just a simple overlay of the shap library method  waterfall for calculating SHAP plots

    # ===LIST OF ARGUMENTS: ===========================================================================================
    
    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit and .predict method)  
    #/// input model which we want to plot LIME for
    
    #  X_train ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for training model 
    
    #  y_train  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for training model 
    
    #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for plotting PDP
    
    #  prefit  ///  (bool)  ///  indicator of whether you provide a pre-trained model or not  (OPTIONAL)
    
    #  selection_num  ///  (value)  ///  number of elements for plotting LIME (OPTIONAL)
    
    #  work_mode  ///  (string)  ///  work mode, 'regression' bu default. (ATTENTION - 'classification' MODE IS NOT SUPPORTED YET)
    
    #  verbose  ///  (bool) (OPTIONAL-REMOVAL REQUIRED)  ///  option for outputting detailed (OPTIONAL)
       
    # === OUTPUT ==============================================================================================================
    
    # Output  /// outputs SHAP plot
    
    if estimated_model is None:
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: estimated_mode.")
        return
    
    if not (isinstance(X, (pd.DataFrame)) or isinstance(X, (np.ndarray))): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: X.")
        return
    
    if not (isinstance(X_train, (pd.DataFrame)) or isinstance(X_train, (np.ndarray))): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: X_train.")
        return
        
    if not (isinstance(y_train, (pd.DataFrame)) or isinstance(y_train, (np.ndarray))): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: y_train.")
        return
    
    if not(isinstance(max_feature_amount, int) or isinstance(max_feature_amount, np.int_) or isinstance(max_feature_amount, np.intc)) or max_feature_amount <= 0: 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: max_feature_amount.")
        return
        
    if not(isinstance(selection_num, int) or isinstance(selection_num, np.int_) or isinstance(selection_num, np.intc)) or selection_num <= 0: 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: selection_num.")
        return
    
    if not prefit:
        try:
            estimated_model.fit(X_train, y_train)
        except:
            logging.warning("Estimated model must have fit method.")

    explainer = LimeTabularExplainer(training_data=X.to_numpy(),
        feature_names=list(X.columns),
        mode=work_mode, random_state=0)

    exp = explainer.explain_instance(X.to_numpy()[selection_num], estimated_model.predict, num_features=max_feature_amount)

    exp.as_pyplot_figure()

    from matplotlib import pyplot as plt
    plt.tight_layout()


def pdp_3d_plot(estimated_model, X, feature_names, feature_name_1, feature_name_2, 
                prefit = True, X_train = None, y_train = None, verbose = False):

    # === FUNCTION SUMMARY ============================================================================================

    # Just a simple overlay of the shap library method  waterfall for calculating 3D PDP plot for 2 features' interaction

    # ===LIST OF ARGUMENTS: ===========================================================================================
    
    #  estimated_model  ///  (sklearn, XGBoost, CatBoost or any other model class type with .fit and .predict method)  
    #/// input model which we want to plot LIME for
    
    #  X  ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for plotting PDP
    
    #  feature_name_1  ///  (string)  ///  1st feature name to plot pdp interaction for
    
    #  feature_name_2  ///  (string)  ///  2nd feature name to plot pdp interaction for
    
    #  prefit  ///  (bool)  ///  indicator of whether you provide a pre-trained model or not  (OPTIONAL)
    
    #  X_train ///  (numpy.array or pandas.DataFrame)  ///  a table of features values for training model (OPTIONAL)
    
    #  y_train  ///  (numpy.array or pandas.DataFrame)  ///  a coloumn of target values for training model (OPTIONAL)
    
    #  verbose  ///  (bool) (OPTIONAL-REMOVAL REQUIRED)  ///  option for outputting detailed (OPTIONAL)
       
    # === OUTPUT ==============================================================================================================
    
    # Output  /// outputs 3D "heat" PDP plot
     
    if estimated_model is None:
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: estimated_mode.")
        return
    
    if not (isinstance(X, (pd.DataFrame)) or isinstance(X, (np.ndarray))): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: X.")
        return
    
    if not (isinstance(X_train, (pd.DataFrame)) or isinstance(X_train, (np.ndarray))): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: X_train.")
        return
        
    if not (isinstance(y_train, (pd.DataFrame)) or isinstance(y_train, (np.ndarray))): 
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: y_train.")
        return
             
    if not isinstance(feature_name_1, (str)):
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: feature_name_1.")
        return
    
    if not isinstance(feature_name_2, (str)):
        logging.warning("MISSING ARGUMENT OR INCORRECT ARGUMENT TYPE: feature_name_2.")
        return

    pdp_goal = pdp.pdp_interact(model=estimated_model, dataset=X, model_features=feature_names, 
                                  features=[feature_name_1, feature_name_2])
    
    fig, axes = pdp.pdp_interact_plot(pdp_interact_out=pdp_goal,
                                  feature_names=[feature_name_1, feature_name_2],
                                  plot_type='contour',
                                  x_quantile=True,
                                  plot_pdp=True)

    fig.show()


'''

_____                       _         _                         __     __      __      _ 
(_   _)                     ( )       ( )_            _  _     /'__`\ /' _`\  /'__`\  /' )
  | |   __   _ __   _ _    _| |   _ _ | ,_)   _ _    ( )( )   (_)  ) )| ( ) |(_)  ) )(_, |
  | | /'__`\( '__)/'_` ) /'_` | /'_` )| |   /'_` )   | || |      /' / | | | |   /' /   | |
  | |(  ___/| |  ( (_| |( (_| |( (_| || |_ ( (_| |   | || |    /' /( )| (_) | /' /( )  | |
  (_)`\____)(_)  `\__,_)`\__,_)`\__,_)`\__)`\__,_)   | || |   (_____/'`\___/'(_____/'  (_)
  
  
'''




