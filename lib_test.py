import feature_processing as fp
import pandas as pd
import xgboost as xgb

def createDF(arrs):

    d = {}
    for i in range(len(arrs)):
        d["x" + str(i)] = arrs[i]
    df = pd.DataFrame(data=d, index=range(len(arrs[0])))

    return df


data = createDF([[0,1,0,1], range(4), [1,0,1,0]])
target = data.x0 * data.x1 + data.x2*data.x1
model = xgb.train({'max_depth': 4, "seed": 123}, xgb.DMatrix(data, label=target), num_boost_round=2)
print "original data"
print data
data = fp.addInteractions(data, model)
print "data with interactions"
print data

interactions = fp.model_features_insights_extractions.get_xgboost_interactions(model)

print "extracted interactions"
print interactions


