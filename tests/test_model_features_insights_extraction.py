

from model_features_insights_extractions import *
import pandas as pd
import xgboost as xgb
import graph_search_algorithms


def createDF(arrs):

    d = {}
    for i in range(len(arrs)):
        d["x" + str(i)] = arrs[i]
    df = pd.DataFrame(data=d, index=range(len(arrs[0])))

    return df

def test_get_xgboost_trees():

    data = createDF([[0,1,0,1], range(4), [1,0,1,0]])
    target = data.x0 * data.x1 + data.x2*data.x1

    model = xgb.train({'max_depth': 4, "seed": 123}, xgb.DMatrix(data, label=target), num_boost_round=2)
    trees = get_xgboost_trees(model)

    # should have at least 2 feature splits
    assert trees[0].shape[0] == 2
    assert trees[1].shape[0] == 2

    # should contain at least one of the binary variables
    assert "x0" in trees[0].var_name.values or "x2" in trees[0].var_name.values
    interactions = graph_search_algorithms.get_paths_from_trees(trees)
    assert interactions == sorted([['x1'], ['x0', 'x1']]) or interactions == sorted([['x1'], ['x0', 'x1']])
