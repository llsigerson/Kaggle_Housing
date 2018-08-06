# load dataset
train_dataframe = pandas.read_csv("train.csv")
test_dataframe = pandas.read_csv("test.csv")


#explore data
train_dataframe.columns
train_dataframe.info()
#look at individual categorical predictors to see if they can easily be converted to numeric
train_dataframe.Fence.value_counts()
#it doesn't look like they can, so I'll use only predictors that are already numeric
#I considered dummy coding categorical predictors, but I worried that this would make the
#results too hard to interpret and take too long to run the models.
#  I could consider doing so in the future to maximize model accuracy

numerics = ['int64', 'float64']
train_numeric= train_dataframe.select_dtypes(include=numerics).astype("float")
test_numeric= test_dataframe.select_dtypes(include=numerics).astype("float")
#fill na values with means for each column
train_numeric=train_numeric.fillna(train_numeric.mean())
test_numeric= test_numeric.fillna(test_numeric.mean())
#convert to array
train_dataset = train_numeric.values
test_dataset= test_numeric.values
# split into input (X) and output (Y) variables, take log of output
train_x = train_dataset[:,0:37]
train_y = numpy.log10(train_dataset[:,-1])