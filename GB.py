from sklearn.ensemble import GradientBoostingRegressor
import sklearn.metrics as mc
import Preprocess as load
import time as time
import numpy as np

# Load data
train_data, train_label, test_data, test_label = load.read_data()

start = time.clock()

# Create model
gb = GradientBoostingRegressor(loss='huber',
                               learning_rate=0.1,
                               max_depth=8,
                               alpha=0.95,
                               random_state=0,
                               n_estimators=300,
                               subsample=0.7
                               )

# Training model
pre_gb = gb.fit(train_data, train_label).predict(test_data)

# Calculate score
score_gb = gb.score(test_data, test_label)
pre_gb = np.reshape(pre_gb, [-1, 1])

se = mc.mean_squared_error(test_label, pre_gb)
r2 = mc.r2_score(test_label, pre_gb)

print(gb.feature_importances_)
print("R-square:" + str(r2))
print("Square-error:" + str(se))

elapsed = (time.clock() - start)
print("Time used: " + str(elapsed) + " sec")
