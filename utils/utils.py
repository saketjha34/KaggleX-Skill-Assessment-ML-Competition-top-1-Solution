import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler , MaxAbsScaler , StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
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


