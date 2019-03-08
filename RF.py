from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as mc
import Preprocess as load
import time as time
import numpy as np

# Load data
train_data, train_label, test_data, test_label = load.read_data()

start = time.clock()

# Create model
rf = RandomForestRegressor(max_depth=10,
                           random_state=0,
                           n_estimators=200
                           )

# Traning model
pre_rf = rf.fit(train_data, train_label).predict(test_data)

# Calculate score
score_rf = rf.score(test_data, test_label)
pre_rf = np.reshape(pre_rf, [-1, 1])
[m,n] = np.shape(test_label)

rsquare =1- (((pre_rf- test_label) ** 2).sum()) / (((test_label - test_label.mean()) ** 2).sum())
prepro = rf.get_params()

se = mc.mean_squared_error(test_label, pre_rf)

print("R-square:" + str(score_rf))
print("Square-error:" + str(se))

elapsed = (time.clock() - start)
print("Time used: " + str(elapsed) + " sec")

