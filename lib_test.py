import feature_processing as fp
import pandas as pd
import xgboost as xgb
import numpy as np

def createDF(arrs):

    d = {}
    for i in range(len(arrs)):
        d["x" + str(i)] = arrs[i]
    df = pd.DataFrame(data=d, index=range(len(arrs[0])))

    return df

def show(message, l=50):
    print(" ")
    print(np.repeat("*", l).tostring())
    print("   " + message)
    print(np.repeat("*", l).tostring())


show("Examples on how to use the feature processing library !", 200)

show(" >> Example on extracting interactions form tree based models\n and adding them as new features to your dataset.", 100)

data = createDF([[0,1,0,1], range(4), [1,0,1,0]])
target = data.x0 * data.x1 + data.x2*data.x1
model = xgb.train({'max_depth': 4, "seed": 123}, xgb.DMatrix(data, label=target), num_boost_round=2)

show("original data")
print(data)
data = fp.addInteractions(data, model)
show("data with interactions")
print(data)

interactions = fp.model_features_insights_extractions.get_xgboost_interactions(model)

show("extracted interactions")
print(interactions)

show(" >> Example on target encoding", 100)

train_data = createDF([[0, 1, 0, 1], range(4)])
test_data = createDF([[1, 0, 0, 1], range(4)])
target = range(4)

train_data = fp.targetEncoding(train_data, train_data, "x0", target, smoothing_func=fp.exponentialPriorSmoothing,
                           aggr_func="mean", smoothing_prior_weight=1)
test_data = fp.targetEncoding(test_data, train_data, "x0", target, smoothing_func=fp.exponentialPriorSmoothing,
                           aggr_func="mean", smoothing_prior_weight=1)

show("train data with target encoding")
print(train_data)
show("test data with target encoding")
print(test_data)

show(" >> Example  generic and memory efficient enrichment of features dataframe with group values", 100)
data = createDF([[0, 1, 0, 1], range(4), [1, 0, 1, 0]])
data = fp.add_group_values(data, ["x0"], "x1", "x1_sum_per_x0", sum, agg_type='float32')
show("data with group values")
print(data)



