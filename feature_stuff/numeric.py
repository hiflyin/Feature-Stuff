
import pandas as pd
import numpy as np
import gc
import graph_search_algorithms
import model_features_insights_extractions
from scipy import spatial

def add_interactions(df, model = None, interactions = None):
    '''
    Summary:  generic function for adding interaction features to a data frame either by passing them as a list or
        by passing a boosted trees model to extract the interactions from.

    Inputs:
        df: a pandas dataframe
        model: boosted trees model (currently xgboost supported only). Can be None in which case the interactions have to be provided
        interactions: list in which each element is a list of features/columns in df, default: None

    Output: df containing the group values added to it

    TO DO: check if interactions has to by np array rather than list
    '''

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


def add_group_values(df, group_cols, counted, agg_name, agg_function,  agg_type='float32'):
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

    gp = df[group_cols+[counted]].groupby(group_cols)[counted].apply(agg_function).reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return(df)


def meanPriorSmoothing(counts, prior_weight=1):
    '''
    Inputs:
        counts: a pandas series as counts of number of samples falling in each category
        prior_weight: a prior weight to put on each category

    Output: a pandas series of weights such that each category is weighted as a mean between the prior weight and the
        number of samples  we have in the respective category. The more samples we have in this category the larger the
        weight will be.
    '''
    return counts / (counts + prior_weight)


def exponentialPriorSmoothing(counts, prior_weight=1):
    '''

    Inputs:
        counts: a pandas series as counts of number of samples falling in each category
        prior_weight: a prior weight to put on each category

    Output: a pandas series of weights such that each category is weighted as a sigmoid of the number of samples  we have
        in the respective category minus the prior weight. The more samples we have in this category the larger the
        weight will be.
    '''
    return 1 / (1 + np.exp(-(counts - prior_weight)))


def target_encoding(df, ref_df, categ_col, y_col, smoothing_func=exponentialPriorSmoothing, aggr_func="mean", smoothing_prior_weight=1):
    '''
    Summary:  target encoding of a feature column using exponential prior smoothing or mean prior smoothing
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
    y_col_name = "y_" + max([len(x) for x in df.columns.values])*"x"
    all_group_col_name = "g_" + max([len(x) for x in df.columns.values])*"x"

    ref_df[y_col_name] = y_col
    ref_df[all_group_col_name] = 0


    df_grouped = ref_df.groupby([categ_col])[y_col_name].agg([aggr_func, "count"]).reset_index()

    smoothing = smoothing_prior_weight
    if smoothing_func != None:
        smoothing = smoothing_func(df_grouped["count"], smoothing_prior_weight)
    df_grouped[categ_col + "_bayes_" + aggr_func] = df_grouped[aggr_func] * smoothing + \
                                                    ref_df.groupby(all_group_col_name)[y_col_name].agg([aggr_func]).values[0][0] * (1 - smoothing)

    df_grouped.drop(["count", aggr_func], 1, inplace=True)
    df = pd.merge(df, df_grouped, how='left', on=[categ_col])

    ref_df.drop([y_col_name,all_group_col_name], 1, inplace=True)
    del df_grouped
    gc.collect()

    return(df)


def cv_targetEncoding(df, categ_cols, y_col, cv_folds, smoothing_func=exponentialPriorSmoothing, aggr_func="mean", smoothing_prior_weight=1, verbosity =0):
    '''
    Inputs:
        df: a pandas dataframe containing the column for which to calculate target encoding (categ_col) and the target variable (y_col)
        categ_cols: a list or array with the the names of the categorical columns for which to calculate target encoding
        y_col: a numpy array of the target variable to predict
        cv_folds: a list with fold pairs as tuples of numpy arrays for cross-val target encoding
        smoothing_func: the name of the function to be used for calculating the weights of the corresponding target variable
            value inside ref_df. Default: exponentialPriorSmoothing.
        aggr_func: the statistic used to aggregate the target variable values inside each category of the categ_col
        smoothing_prior_weight: a prior weight to put on each category. Default 1.
        verbosity: 0-none, 1-high_level, 2-detailed

    Output: df containing a new column called <categ_col + "_bayes_" + aggr_func> containing the encodings of categ_col
    '''

    df_parts = []
    fold_id = 0

    for fold0, fold1 in cv_folds:

        if verbosity >=1 :
            print("working on fold: {}".format(fold_id+1))
        te_df = df.loc[fold1, :]
        ref_df = df.loc[fold0, :]
        for col in categ_cols:
            if verbosity == 2:
                print("working on column: {}".format(col))
            te_df = target_encoding(te_df, ref_df, col, y_col[fold0], smoothing_func=smoothing_func, aggr_func=aggr_func,
                                    smoothing_prior_weight=smoothing_prior_weight)
            te_df.index = fold1
        df_parts.append(te_df)
        fold_id += 1
    df = pd.concat(df_parts).loc[df.index.values,:]
    del df_parts
    gc.collect()

    return(df)


def standardize_cols(df, cols):
    '''
    Summary:  simple standardizing - substract mean and divide by std

    Inputs:
        df: a pandas dataframe
        cols: columns to standardize
        counted: column to compute the aggregate/ group values  on
        agg_name: name of the new function
        agg_function: aggregate function name
        agg_type: default is float32

    Output: df containing the standardized columns
    '''

    for col in cols:

        std = df[col].std()
        if std != 0:
            df[col] = (df[col] - df[col].mean()) / std

    return df


def add_knn_values(df, dist_cols, k, col_prefix, val_col, pred_points_indexes):

    '''
    Summary:  given a dataframe creates a new feature with knn of the values of a given feature. For example,
        given a dataframe with house prices to predict and coordinates as predictors, we can add a new feature
        by computing the prices of the houses sold in the neighbourhood.

    Inputs:
        df: a pandas dataframe
        dist_cols: columns based on which to compute knn distances
        k: number of neighbours for knn
        col_prefix: prefix name of the new column
        val_col: column  based on which to average the k neighbours
        pred_points_indexes: indexes of rows to use for the computation of KNN

    Output: df containing the KNN columns
    '''

    # init k cols
    k_cols = [col_prefix + str(x + 1) for x in range(k)]
    for col in k_cols:
        df[col] = df[val_col].values  # df['e'] = e.values

    standard_train = df[dist_cols].copy()
    standard_train = standardize_cols(standard_train, dist_cols)

    # build a kd tree
    tree = spatial.KDTree(standard_train.loc[pred_points_indexes,:].values.tolist())

    track = 0
    for j in standard_train.index.values:

        track += 1

        new_point = standard_train.loc[j, dist_cols]

        # we take k +1 assuming one is the point itself
        distance, index = tree.query(new_point, k=(k+1), p=2)  # eps, distance_upper_bound)

        pos = standard_train.index[index].values.tolist()

        # as long as the point we update is not the same as the neighbour ie itself
        try:
            pos.remove(j)
        except:
            del pos[-1]
            pass
        df.loc[j, k_cols] = df.loc[pos, val_col].values

        if track % 50000 == 0:

            print track

    return df


