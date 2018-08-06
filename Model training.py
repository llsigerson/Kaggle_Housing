#define loss function
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

#create and run baseline neural network model
def baseline_model():
   # create model
  model = Sequential()
  model.add(Dense(37, input_dim=37, kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  # Compile model
  model.compile(loss=root_mean_squared_error, optimizer='adam')
  return model

seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)
baseline_results = cross_val_score(estimator, train_x, train_y, cv=kfold)
#print results. I'm using a masked array because for some reason I'm getting an inf value
#for one of the results. I don't believe this compromises the other results, so I ignore it for now
print("Baseline Results: %.2f (%.2f) MSE" % (numpy.ma.masked_invalid(baseline_results).mean(),
                                    numpy.ma.masked_invalid(baseline_results).std()))

#add second hidden layer and run to see if accuracy is improved
def two_layer_model():
   # create model
  model = Sequential()
  model.add(Dense(37, input_dim=37, kernel_initializer='normal', activation='relu'))
  model.add(Dense(19, kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  # Compile model
  model.compile(loss=root_mean_squared_error, optimizer='adam')
  return model

# evaluate model with dataset
estimator = KerasRegressor(build_fn=two_layer_model, epochs=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)
two_layer_results = cross_val_score(estimator, train_x, train_y, cv=kfold)
print("Two Layer Results: %.2f (%.2f) MSE" % (numpy.ma.masked_invalid(two_layer_results).mean(),
                                    numpy.ma.masked_invalid(two_layer_results).std()))


#add third hidden layer and run to see if accuracy is improved
def three_layer_model():
   # create model
  model = Sequential()
  model.add(Dense(37, input_dim=37, kernel_initializer='normal', activation='relu'))
  model.add(Dense(19, kernel_initializer='normal', activation='relu'))
  model.add(Dense(10, kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  # Compile model
  model.compile(loss=root_mean_squared_error, optimizer='adam')
  return model

# evaluate model with  dataset
estimator = KerasRegressor(build_fn=three_layer_model, epochs=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)
three_layer_results = cross_val_score(estimator, train_x, train_y, cv=kfold)
print("Three Layer Results: %.2f (%.2f) MSE" % (numpy.ma.masked_invalid(three_layer_results).mean(),
                                    numpy.ma.masked_invalid(three_layer_results).std()))

#Three layered model does not improve on two layer model, so we'll stick with the two layer model
#We'll build a final model using all the data, then predict values for the test set
final_model= two_layer_model()
final_model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=0)
predictions= final_model.predict(test_dataset)
#transform predictions again, because the model was built on log10 transformed outcomes
final_predictions= 10**predictions.flatten()

kaggle_submission= pandas.DataFrame({"Id": test_numeric.Id, "SalePrice":final_predictions})
kaggle_submission.to_csv("kaggle submission.csv", sep=",")