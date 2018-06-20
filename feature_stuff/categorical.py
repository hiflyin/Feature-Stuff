
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import gc


def encode_labels(df, cols = None):
    '''
    Inputs:
        df: a pandas dataframe containing the column for which to calculate target encoding (categ_col)
        cols: all columns' names for which to do label encoding . If is None (default) then all object columns are taken.
    Output: df with cols replaced the coresponding label encodings while maintaining all existing None values at their positions.
    '''

    le = LabelEncoder()
    for col in cols:
        # pick some random value from the col - will make it null back at the end anyway
        null_replacement = df[col].values[0]
        # save col null positions and set ones for the rest
        nan_col = np.array([1 if not pd.isnull(x) else x for x in df[col]])
        # replace nulls in the original array, and fit on it
        a = np.array([x if not pd.isnull(x) else null_replacement for x in df[col]])
        le.fit(a)
        # transform the data and add the nulls back
        df[col] = le.transform(a) * nan_col

    return(df)


def add_dummies(df, cols = None, drop = True):
    '''
    Inputs:
        df: a pandas Dataframe containing the columns to add dummies for.
        cols: a list or array of the names of the columns to dummy. If is None (default) then all object columns are taken.
        drop: if the categorical columns are to be dropped after adding the dummies. Default = True.

    Output: the dataframe with the added dummies. NaNs will be ignored rather than considered a distinct category.

    TO DO: TypeErrors?
    '''

    if cols is None:
        cols = [col for col in df.columns if df[col].dtype == 'object']

    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col).astype(np.int8)
        df = pd.concat([df, dummies], axis=1)
        if drop:
            df.drop([col], inplace=True, axis=1)

    del dummies
    gc.collect()
    return(df)


def add_dummies_selected_cat(col, df, categs, drop = True):
    '''
    Inputs:
        col: the name of column to be considered.
        df: a pandas Dataframe containing the columns to add dummies for.
        categs: the names of the categories in col to add dummies for.
        drop: if the categorical columns are to be dropped after adding the dummies. Default = True.

    Output: the dataframe with the added dummies. NaNs will be ignored rather than considered a distinct category.
    '''

    aux = df[col]
    df.loc[~df[col].isin(categs), col] = None
    dummies = pd.get_dummies(df[col], prefix=col).astype(np.int8)
    df = pd.concat([df, dummies], axis=1)

    if drop:
        df.drop([col], inplace=True, axis=1)
    else:
        df[col] = aux
    del dummies
    gc.collect()
    return(df)


