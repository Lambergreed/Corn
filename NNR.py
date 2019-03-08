from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPRegressor
import Preprocess as input_data
import sklearn.metrics as mc
import time as time

# Load data
train_feat, train_label, test_feat, test_label = input_data.read_data()

start = time.clock()

# Create model
model = MLPRegressor(hidden_layer_sizes=(100, 50),
                     activation='relu',
                     solver='sgd',
                     momentum=0.95,
                     max_iter=200,
                     random_state=1,
                     alpha=1e-5,
                     warm_start=True
                     )

# Training model
# model.fit(train_feat, train_label)
# prediction = model.predict(test_feat)
# prediction = np.reshape(prediction, [-1, 1])

predicted = cross_val_predict(model, train_feat, train_label, cv=10)

# se = mc.mean_squared_error(test_label, prediction)
# r2 = mc.r2_score(test_label, prediction)

print(mc.r2_score(train_label, predicted))
print(mc.mean_squared_error(train_label, predicted))
print(mc.explained_variance_score(train_label, predicted))

# print('se:' + str(se))
# print('r2:' + str(r2))

elapsed = (time.clock() - start)
print("Time used: " + str(elapsed) + " sec")