import Preprocess as input_data
import sklearn.metrics as mc
import xgboost as xgb
import time as time
import numpy as np

# Load data
train_feat, train_label, test_feat, test_label = input_data.read_data()

start = time.clock()

# Create model
model = xgb.XGBRegressor(gamma=0.9,
                         learning_rate=0.05,
                         max_depth=11,
                         n_estimators=150,
                         subsample=0.8
                         )

# Training model
model.fit(train_feat, train_label)
prediction = model.predict(test_feat)
prediction = np.reshape(prediction, [-1, 1])

se = mc.mean_squared_error(test_label, prediction)
r2 = mc.r2_score(test_label, prediction)

print(model.feature_importances_)
print('se:' + str(se))
print('r2:' + str(r2))

