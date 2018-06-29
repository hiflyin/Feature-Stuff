import matplotlib.pyplot as plt

curr_months = data["v23_month"]
avg_probs_preds = []
index = 0
for month in range(12):
    data["v23_month"] = month
    mean_pred = xgb_model.predict(xgb.DMatrix(data[xgb_model.feature_names])).mean()
    avg_probs_preds.append(mean_pred)

data["v23_month"] = curr_months

plt.figure(figsize=(14,6))
plt.scatter(range(1,13), avg_probs_preds, s=15)
plt.ylabel('average probability of buying', color = "blue")
plt.xlabel('pickup month', color = "blue")
plt.title("Average Probability of Buying a Car for a Given Pickup Month - Based on Curent Best Model", color = "red")
plt.show()

'''
TO DO:

get grid param and split - ig grid > unique values use values directly
compute average of each grid segment
for each such average compute values and  return
add function to plot

consider parwise vars:
consider parallel for loop




'''