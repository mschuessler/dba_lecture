import pandas as pd
import numpy as np


def fix_column_names(df, character_map={' ':'_','ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss'}):
    """
    Cleans and standardizes the column names of a DataFrame.

    This function performs the following transformations on the column names:
    - Strips leading and trailing whitespace.
    - Converts all characters to lowercase.
    - Replaces spaces with underscores.
    - Removes parentheses.

    Parameters:
    df (pandas.DataFrame): The DataFrame whose column names need to be fixed.

    Returns:
    pandas.Index: The cleaned and standardized column names.
    """
    if isinstance(df, pd.DataFrame):
        columns = df.columns
    elif isinstance(df, pd.Index):
        columns = df
    else:
        raise TypeError("The 'df' parameter must be a pandas DataFrame, Series, or Index.")
    columns = columns.str.strip().str.lower()
    for char, replacement in character_map.items():
        columns = columns.str.replace(char, replacement)
    columns = columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
    
    return columns


def recode_to_categorical(column, recode, ordered=True):
    """
    Recode a pandas Series to a categorical type with specified categories.

    Parameters:
    column (pd.Series): The pandas Series to be recoded.
    recode (dict): An ordered dictionary where keys are the original values in the column 
                   and values are the new categorical values. Should contain all possible values in the column.

    Returns:
    pd.Categorical: A pandas Categorical object with the recoded values and specified categories.
    """
    if not isinstance(column, pd.Series):
        raise TypeError("The 'column' parameter must be a pandas Series.")
    if not isinstance(recode, dict):
        raise TypeError("The 'recode' parameter must be a dictionary.")

    return pd.Categorical(
        column.replace(recode),
        categories=list(recode.values()),
        ordered=ordered)


def better_describe(df, columns=None):
    """
    Generate a descriptive statistics summary for specified columns in a DataFrame, including skewness, kurtosis, and normality check.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (list): List of column names to describe.

    Returns:
    pandas.DataFrame: A DataFrame containing the descriptive statistics, skewness, kurtosis, and normality check for the specified columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(columns, (list, pd.Index)):
        raise TypeError(
            "The 'columns' parameter must be a list or a pandas Index.")
    if isinstance(columns, pd.Index):
        columns = columns.tolist()
    desc = df[columns].describe().T
    desc["skew"] = df[columns].skew()
    desc["kurtosis"] = df[columns].kurtosis()
    desc["normal"] = (desc["kurtosis"] < 1) & (
        desc["kurtosis"] > -1) & (desc["skew"] < 0.5) & (desc["skew"] > -0.5)
    desc = desc.T
    return desc


def informative_columns(df):
    """
    Analyzes the columns of a DataFrame to determine their informational content.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be analyzed.

    Returns:
    tuple: A tuple containing three lists:
        - no_information (list): Columns with no information (empty).
        - always_the_same (list): Columns that are always set to the same value.
        - contains_information (list): Columns with more than one unique value.
    """
    no_information = list(df.columns[df.nunique() == 0])
    always_the_same = list(df.columns[df.nunique() == 1])
    contains_information = list(df.columns[df.nunique() > 1])
    return no_information, always_the_same, contains_information


def drop_no_information_columns(df, keep=None, drop=None):
    """
    Drop columns with no information (empty) from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be analyzed.

    Returns:
    pandas.DataFrame: The DataFrame with columns containing no information dropped.
    """
    no_information, always_the_same, _ = informative_columns(df)
    delete = no_information + always_the_same

    if drop and isinstance(drop, list):
        delete = delete + drop
    if keep and isinstance(keep, list):
        delete = delete - set(keep)

    print(f"Dropping columns: {delete}")

    return df.drop(columns=delete)

