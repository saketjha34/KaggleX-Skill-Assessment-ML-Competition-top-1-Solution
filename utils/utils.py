import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler , MaxAbsScaler , StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import os
import joblib
from sklearn.base import RegressorMixin
import pandas
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import Parallel, delayed
import optuna


def get_horse_power(x : pandas.Series,
                    df: pandas.DataFrame)->pandas.DataFrame:
    """
    Extracts the horsepower values from a pandas Series and adds them to a DataFrame.

    Parameters:
    x (pandas.Series): A pandas Series containing strings, some of which include horsepower values 
                       (formatted as 'XXXHP').
    df (pandas.DataFrame): A pandas DataFrame to which the horsepower values will be added.

    Returns:
    pandas.DataFrame: The original DataFrame with an additional column 'horse power' containing the 
                      extracted horsepower values as floats. If a string in the Series does not 
                      contain a horsepower value, NaN is inserted.
    Example:
    >>> import pandas as pd
    >>> data = {'description': ['200HP engine', 'No HP value', '150HP motor']}
    >>> df = pd.DataFrame(data)
    >>> get_horse_power(df['description'], df)
       description       horse power
    0  200HP engine       200.0
    1  No HP value        NaN
    2  150HP motor        150.0
    """
    x.to_list()
    hp = []
    for i in x:
        if  i.split()[0].endswith('HP'):
            hp.append(float(i.split()[0].strip('HP')))
        else:
            hp.append(np.nan)
    df['horse power'] = hp
    return df

def get_capacity(x : pandas.Series,
                 df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Extracts the capacity values from a pandas Series and adds them to a DataFrame.

    Parameters:
    x (pandas.Series): A pandas Series containing strings, some of which include capacity values 
                       (formatted as 'XXXL').
    df (pandas.DataFrame): A pandas DataFrame to which the capacity values will be added.

    Returns:
    pandas.DataFrame: The original DataFrame with an additional column 'capacity' containing the 
                      extracted capacity values as floats. If a string in the Series does not 
                      contain a capacity value, NaN is inserted.

    Example:
    >>> import pandas as pd
    >>> data = {'description': ['200HP 2.0L engine', 'No capacity value', '150HP 3.5L motor']}
    >>> df = pd.DataFrame(data)
    >>> get_capacity(df['description'], df)
               description  capacity
    0  200HP 2.0L engine       2.0
    1  No capacity value       NaN
    2  150HP 3.5L motor        3.5
    """
    x.to_list()
    capacity = []
    x_new  = [i.split() for i in x]
    for i in x_new:
        if len(i) >=2:
            if i[1].endswith('L'):
                capacity.append(float(i[1].strip('L')))
            else:
                capacity.append(np.nan)
        else:
            capacity.append(np.nan)
    df['capacity'] = capacity
    return df


models: dict[RegressorMixin,dict[str,float]] = {
    
    LinearRegression : {
        'n_jobs' : -1
    },

    SGDRegressor : {
        'eta0' : 0.001 ,
        'max_iter': 2000 ,
        'penalty': 'l1' ,
        'learning_rate': 'adaptive'
    },

    DecisionTreeRegressor : {
        'max_depth': 32 , 
        'max_features': 'sqrt' , 
        'max_leaf_nodes': 100 , 
        'min_samples_leaf': 4
    },

    RandomForestRegressor : {
                'n_estimators': 400 ,
                'max_depth': 32 , 
                'min_samples_split': 10 ,
                'min_samples_leaf':4 , 
                'n_jobs':-1
    },

    GradientBoostingRegressor:{
        'learning_rate': 0.001,
        'n_estimators' : 400,
    },

    XGBRegressor : {
        'n_estimators': 400,
        'max_depth': 10,
        'learning_rate': 0.01,
        'subsample': 0.5,

    },

    LGBMRegressor : {
        'n_estimators': 500,
        'num_leaves': 32 , 
        'learning_rate': 0.01,
        'subsample': 0.7,
    },

    MLPRegressor : {
        'activation': 'identity',
        'solver': 'adam',
        'batch_size': 32,
        'max_iter': 400,
        'random_state': 42,
        'beta_1': 0.5,

    },
    
    CatBoostRegressor : {
        'n_estimators': 500,
        "od_type": 'Iter',  
        "bootstrap_type": 'Bernoulli',
    },

}

def evaluate_model(model : type[RegressorMixin],
                   params : dict,
                   X_train : np.ndarray,
                   train_targets : np.ndarray,
                   X_val: np.ndarray, 
                   val_targets :np.ndarray) -> tuple[float,float,float,float,float,float]:
    
    """
    Evaluates a regression model on training and validation datasets.

    Parameters:
    model (type[RegressorMixin]): The regression model class.
    params (dict): The parameters to initialize the regression model.
    X_train (np.ndarray): The training feature data.
    train_targets (np.ndarray): The training target data.
    X_val (np.ndarray): The validation feature data.
    val_targets (np.ndarray): The validation target data.

    Returns:
    tuple: A tuple containing:
        - train_mae (float): Mean Absolute Error on the training data.
        - val_mae (float): Mean Absolute Error on the validation data.
        - train_rmse (float): Root Mean Squared Error on the training data.
        - val_rmse (float): Root Mean Squared Error on the validation data.
        - train_r2 (float): R-squared score on the training data.
        - val_r2 (float): R-squared score on the validation data.

    Example:
    >>> from sklearn.linear_model import Ridge
    >>> import numpy as np
    >>> X_train = np.array([[1, 2], [3, 4], [5, 6]])
    >>> train_targets = np.array([1, 2, 3])
    >>> X_val = np.array([[7, 8], [9, 10]])
    >>> val_targets = np.array([4, 5])
    >>> params = {'alpha': 1.0}
    >>> evaluate_model(Ridge, params, X_train, train_targets, X_val, val_targets)
    (0.0, 1.0, 0.0, 1.0, 1.0, 0.0)
    """
    regressor = model(**params).fit(X_train, train_targets)
    train_preds = regressor.predict(X_train)
    val_preds = regressor.predict(X_val)

    train_mae = mean_absolute_error(train_targets, train_preds)
    val_mae = mean_absolute_error(val_targets, val_preds)

    train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
    val_rmse = mean_squared_error(val_targets, val_preds, squared=False)

    train_r2 = r2_score(train_targets, train_preds)
    val_r2 = r2_score(val_targets, val_preds)

    return (train_mae, val_mae, train_rmse, val_rmse, train_r2, val_r2)

def try_models(model_dict: dict[RegressorMixin,dict[str,float]], 
               X_train: np.array, 
               train_targets: np.array, 
               X_val: np.array, 
               val_targets: np.array) -> pd.DataFrame:
    """
    Evaluates multiple regression models and returns a DataFrame with their performance metrics.

    Parameters:
    model_dict (dict): A dictionary where keys are regression model classes and values are 
                       dictionaries of parameters for initializing the models.
    X_train (np.ndarray): The training feature data.
    train_targets (np.ndarray): The training target data.
    X_val (np.ndarray): The validation feature data.
    val_targets (np.ndarray): The validation target data.

    Returns:
    pd.DataFrame: A DataFrame containing the model names, parameters, and performance metrics 
                  (MAE, RMSE, R-squared) for both training and validation datasets.

    Example:
    >>> from sklearn.linear_model import Ridge, Lasso
    >>> import numpy as np
    >>> import pandas as pd
    >>> X_train = np.array([[1, 2], [3, 4], [5, 6]])
    >>> train_targets = np.array([1, 2, 3])
    >>> X_val = np.array([[7, 8], [9, 10]])
    >>> val_targets = np.array([4, 5])
    >>> model_dict = {
    >>>     Ridge: {'alpha': 1.0},
    >>>     Lasso: {'alpha': 0.1}
    >>> }
    >>> try_models(model_dict, X_train, train_targets, X_val, val_targets)
          models               params  train_mae  val_mae  train_rmse  val_rmse  train_r2  val_r2
    0     Ridge   {'alpha': 1.0}        0.0       1.0       0.0        1.0       1.0       0.0
    1     Lasso   {'alpha': 0.1}        0.1       0.9       0.1        0.9       0.99      0.01
    """
    
    results = Parallel(n_jobs=-1)(delayed(evaluate_model)(model, params, X_train, train_targets, X_val, val_targets) for model, params in model_dict.items())

    metrics = ['train_mae', 'val_mae', 'train_rmse', 'val_rmse', 'train_r2', 'val_r2']
    results_dict = {metric: [result[i] for result in results] for i, metric in enumerate(metrics)}

    df = pd.DataFrame({
        'models': list(model_dict.keys()),
        'params': list(model_dict.values()),
        **results_dict
    })

    return df


def evalmodel(model:type[RegressorMixin],
              X_train:np.array , 
              train_targets:np.array ,
              X_val:np.array,
              val_targets:np.array,
              **params) -> dict[str , float]:
    """
    Evaluates a regression model on training and validation datasets and returns a dictionary of performance metrics.

    Parameters:
    model (type[RegressorMixin]): The regression model class.
    X_train (np.ndarray): The training feature data.
    train_targets (np.ndarray): The training target data.
    X_val (np.ndarray): The validation feature data.
    val_targets (np.ndarray): The validation target data.
    **params: Additional parameters to initialize the regression model.

    Returns:
    dict: A dictionary containing:
        - 'Train RMSE': Root Mean Squared Error on the training data.
        - 'Val RMSE': Root Mean Squared Error on the validation data.
        - 'Train R2 Score': R-squared score on the training data.
        - 'Val R2 Score': R-squared score on the validation data.

    Example:
    >>> from sklearn.linear_model import Ridge
    >>> import numpy as np
    >>> X_train = np.array([[1, 2], [3, 4], [5, 6]])
    >>> train_targets = np.array([1, 2, 3])
    >>> X_val = np.array([[7, 8], [9, 10]])
    >>> val_targets = np.array([4, 5])
    >>> params = {'alpha': 1.0}
    >>> evalmodel(Ridge, X_train, train_targets, X_val, val_targets, **params)
    {
        'Train RMSE': '0.0',
        'Val RMSE': '1.0',
        'Train R2 Score': '1.0',
        'Val R2 Score': '0.0'
    }
    """

    model = model(**params).fit(X_train,train_targets)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    return {
        'Train RMSE:': f'{mean_squared_error(train_preds,train_targets , squared=False)}',
        'Val RMSE:' :  f'{mean_squared_error(val_preds,val_targets , squared=False)}',
        'Train R2 Score:' :  f'{r2_score(val_preds,val_targets)}',
        'Val R2 Score:' :  f'{r2_score(val_preds,val_targets)}',
    }


def plot_model_importance(model , 
                          X_train: pd.DataFrame , 
                          train_targets: pd.Series ,
                          get_importance_df : bool = False,
                          **params) -> pd.DataFrame:
    """
    Trains a model and plots the feature importances.

    Parameters:
    - model : The classifier to be used for training.
    - X_train (pd.DataFrame): The training data features.
    - train_targets (pd.Series): The training data targets.
    - **params: Additional parameters to be passed to the model.

    Returns:
    - pd.DataFrame: A dataframe containing feature names and their importance scores.
    """
    regressor = model(**params)
    regressor.fit(X_train, train_targets)

    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': regressor.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 30))
    sns.set(style="whitegrid")
    barplot = sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')

    barplot.set_title('Feature Importance of Model', fontsize=16, weight='bold')
    barplot.set_xlabel('Importance', fontsize=14)
    barplot.set_ylabel('Feature', fontsize=14)
    barplot.tick_params(axis='x', rotation=90, labelsize=12)
    barplot.tick_params(axis='y', labelsize=12)

    for index, value in enumerate(importance_df['importance']):
        barplot.text(value, index, f'{value:.2f}', color='black', ha="left", va="center", fontsize=12)
    
    plt.tight_layout()
    plt.show()
    if get_importance_df == True:
        return importance_df
    
def submit_prediction(model:type[RegressorMixin],
                      model_name : str,
                      submission_df : pd.DataFrame, 
                      X_train: np.array, 
                      train_targets : np.array,
                      X_test:np.array, 
                      **params) -> dict[str , float]:
    """
    Trains a regression model, makes predictions on the test dataset, and saves the predictions to a CSV file.

    Parameters:
    model (type[RegressorMixin]): The regression model class.
    model_name (str): The name of the model, used for naming the submission file.
    submission_df (pd.DataFrame): The DataFrame used for submission, to which the predictions will be added.
    X_train (np.ndarray): The training feature data.
    train_targets (np.ndarray): The training target data.
    X_test (np.ndarray): The test feature data.
    **params: Additional parameters to initialize the regression model.

    Returns:
    dict: A dictionary containing the parameters used to initialize the model and the trained model instance.

    Example:
    >>> from sklearn.linear_model import Ridge
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pathlib import Path
    >>> X_train = np.array([[1, 2], [3, 4], [5, 6]])
    >>> train_targets = np.array([1, 2, 3])
    >>> X_test = np.array([[7, 8], [9, 10]])
    >>> submission_df = pd.DataFrame({'id': [1, 2]})
    >>> params = {'alpha': 1.0}
    >>> submit_prediction(Ridge, 'ridge_model', submission_df, X_train, train_targets, X_test, **params)
    submission_df saved at : submission\\submisson_df_ridge_model.csv
    {
        'alpha': 1.0,
        'model': Ridge(alpha=1.0)
    }
    """
    model = model(**params).fit(X_train,train_targets)
    test_preds = model.predict(X_test)
    submission_df['price'] = test_preds
    submission_path = Path('submission')
    submission_df.to_csv(f'..\\{submission_path}\\submisson_df_{model_name}.csv' , index=False)
    params['model'] = model
    print(f"submission_df saved at : {submission_path}\\submisson_df_{model_name}.csv")
    return params


def save_model(model : type[RegressorMixin], 
               model_params : dict[str,float], 
               filepath:str) -> None:
    """
    Save a model and its parameters to a specified file path using joblib.

    Parameters:
    model (type[ClassifierMixin]) : The model to be saved.
    model_params (dict[str,float]) : The parameters used to create the model.
    filepath (str): The file path where the model and its parameters will be saved.
    
    Returns:
    None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    model_data = {
        'model': model,
        'params': model_params
    }

    joblib.dump(model_data, filepath)
    print(f"Model and parameters saved to {filepath}")

def load_model(filepath:str) -> tuple[type[RegressorMixin],dict[str,float]]:
    """
    Load a model and its parameters from a specified file path using joblib.

    Parameters:
    filepath (str): The file path from where the model and its parameters will be loaded.
    
    Returns:
    model (type[ClassifierMixin]) : The loaded model.
    model_params (dict[str,float]): The parameters used to create the model.
    """
    model_data = joblib.load(filepath)
    
    model = model_data['model']
    model_params = model_data['params']
    
    print(f"Model and parameters loaded from {filepath}")
    return model, model_params
