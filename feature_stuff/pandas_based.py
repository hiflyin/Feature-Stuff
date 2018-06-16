
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import gc
import graph_search_algorithms
import model_features_insights_extractions

'''
Summary:  generic function for adding interaction features to a data frame either by passing them as a list or
    by passing an a boosted trees model to extract the interactions from.

Inputs:
    df: a pandas dataframe
    model: boosted trees model (currently xgboost supported only). Can be None in which case the interactions have to be provided
    interactions: list in which each element is a list of features/columns in df, default: None

Output: df containing the group values added to it

TO DO: check if interactions has to by np array rather than list
'''
def addInteractions(df, model = None, interactions = None):

    if interactions is None:
        interactions = graph_search_algorithms.get_paths_from_trees(model_features_insights_extractions.get_xgboost_trees(model))

    path_id = 0  # as inter_id
    for path in interactions:

        if len(path) > 1:
            inter_col_name = 'inter_' + str(path_id)

            df[inter_col_name] = df[path[0]]
            for i in range(1, len(path)):

                if path[i] in df.columns.values:
                    df[inter_col_name] = df[inter_col_name] * df[path[i]]

            path_id += 1

    return df

'''
Summary:  generic and memory efficient enrichment of features dataframe with group values

Inputs:
    df: a pandas dataframe
    group_cols: columns to group by
    counted: column to compute the aggregate/ group values  on
    agg_name: name of the new function
    agg_function: aggregate function name
    agg_type: default is float32

Output: df containing the group values added to it
'''
def add_group_values( df, group_cols, counted, agg_name, agg_function,  agg_type='float32'):

    gp = df[group_cols+[counted]].groupby(group_cols)[counted].apply(agg_function).reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return(df)

'''
Inputs:
    counts: a pandas series as counts of number of samples falling in each category
    prior_weight: a prior weight to put on each category

Output: a pandas series of weights such that each category is weighted as a mean between the prior weight and the
    number of samples  we have in the respective category. The more samples we have in this category the larger the
    weight will be.
'''
def meanPriorSmoothing(counts, prior_weight=1):
    return counts / (counts + prior_weight)

'''
Inputs:
    counts: a pandas series as counts of number of samples falling in each category
    prior_weight: a prior weight to put on each category

Output: a pandas series of weights such that each category is weighted as a sigmoid of the number of samples  we have
    in the respective category minus the prior weight. The more samples we have in this category the larger the
    weight will be.
'''
def exponentialPriorSmoothing(counts, prior_weight=1):
    return 1 / (1 + np.exp(-(counts - prior_weight)))

'''
Inputs:
    df: a pandas dataframe containing the column for which to calculate target encoding (categ_col)
    ref_df: a pandas dataframe containing the column for which to calculate target encoding and the target variable (y_col)
    categ_col: the name of the categorical column for which to calculate target encoding
    y_col: the name of the target column, or target variable to predict
    smoothing_func: the name of the function to be used for calculating the weights of the corresponding target variable
        value inside ref_df. Default: exponentialPriorSmoothing.
    aggr_func: the statistic used to aggregate the target variable values inside each category of the categ_col
    smoothing_prior_weight: a prior weight to put on each category. Default 1.

Output: df containing a new column called <categ_col + "_bayes_" + aggr_func> containing the encodings of categ_col
'''
def targetEncoding(df, ref_df, categ_col, y_col, smoothing_func=exponentialPriorSmoothing, aggr_func="mean", smoothing_prior_weight=1):

    ref_df["y_col"] = y_col

    df_grouped = ref_df.groupby([categ_col])["y_col"].agg([aggr_func, "count"]).reset_index()

    smoothing = smoothing_prior_weight
    if smoothing_func != None:
        smoothing = smoothing_func(df_grouped["count"], smoothing_prior_weight)

    df_grouped[categ_col + "_bayes_" + aggr_func] = df_grouped[aggr_func] * smoothing + sum(y_col)/len(y_col) * (1 - smoothing)
    df_grouped.drop(["count", aggr_func], 1, inplace=True)
    df = pd.merge(df, df_grouped, how='left', on=[categ_col])

    ref_df.drop("y_col", 1, inplace=True)
    del df_grouped
    gc.collect()

    return(df)

'''
Inputs:
    df: a pandas dataframe containing the column for which to calculate target encoding (categ_col) and the target variable (y_col)
    categ_cols: a list or array with the the names of the categorical columns for which to calculate target encoding
    y_col: the name of the target column, or target variable to predict
    cv_folds: a list with fold pairs for cross-val target encoding
    smoothing_func: the name of the function to be used for calculating the weights of the corresponding target variable
        value inside ref_df. Default: exponentialPriorSmoothing.
    aggr_func: the statistic used to aggregate the target variable values inside each category of the categ_col
    smoothing_prior_weight: a prior weight to put on each category. Default 1.

Output: df containing a new column called <categ_col + "_bayes_" + aggr_func> containing the encodings of categ_col
'''
def cv_targetEncoding(df, categ_cols, y_col, cv_folds, smoothing_func=exponentialPriorSmoothing, aggr_func="mean", smoothing_prior_weight=1):


    df_parts = []
    fold_id = 0
    for fold0, fold1 in cv_folds:

        te_df = df.loc[fold1, :]
        ref_df = df.loc[fold0, :]
        for col in categ_cols:
            te_df = targetEncoding(te_df, ref_df, col, y_col, smoothing_func=smoothing_func, aggr_func=aggr_func, smoothing_prior_weight=smoothing_prior_weight)
        df_parts.append(te_df)
        fold_id += 1

    df = pd.concat(df_parts)

    del df_parts
    gc.collect()

    return(df)

'''
Inputs:
    df: a pandas dataframe containing the column for which to calculate target encoding (categ_col)
    cols: all columns' names for which to do label encoding . If is None (default) then all object columns are taken.
Output: df with cols replaced the coresponding label encodings while maintaining all existing None values at their positions.
'''
def encode_labels(df, cols = None):

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

'''
Inputs:
    df: a pandas Dataframe containing the columns to add dummies for.
    cols: a list or array of the names of the columns to dummy. If is None (default) then all object columns are taken.
    drop: if the categorical columns are to be dropped after adding the dummies. Default = True.

Output: the dataframe with the added dummies. NaNs will be ignored rather than considered a distinct category.

TO DO: TypeErrors?
'''
def add_dummies(df, cols = None, drop = True):

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

'''
Inputs:
    col: the name of column to be considered.
    df: a pandas Dataframe containing the columns to add dummies for.
    categs: the names of the categories in col to add dummies for.
    drop: if the categorical columns are to be dropped after adding the dummies. Default = True.

Output: the dataframe with the added dummies. NaNs will be ignored rather than considered a distinct category.
'''
def add_dummies_selected_cat(col, df, categs, drop = True):

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


