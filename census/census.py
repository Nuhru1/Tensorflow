# import pandas library to read the dataset which is csv file
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# reasd the csv file
census = pd.read_csv("<path to your census dataset>")

# show the first five elements of the dataset
census.head()

census['income_bracket'].unique()
#census['marital_status'].unique()

#Since tensorflow can't understand strings as labels, let's convert those string into (0 for <50k and 1 for <=50k)

def label_fix(label):
    if label == ' <=50K':
        return 0
    else:
        return 1
    
census['income_bracket'] = census['income_bracket'].apply(label_fix)

# the neural network need to be fed with feartures and labels
# Our features will be all columns except the last one which is the label column 
x_data = census.drop('income_bracket', axis = 1)
y_label = census['income_bracket']

# split the data (features and label) into train and test set, 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(x_data, y_label, test_size = 0.2, random_state = 101)

census.columns

# create feature columns for categorical values

gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size = 1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size = 1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size = 1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size = 1000)

#create feature columns for continuous values
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

# create a vecrtor for all these features
feat_cols = [gender, occupation, marital_status, relationship, education, workclass, native_country,
            age, education_num, capital_gain, capital_loss, hours_per_week]

# create the input function
input_func = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train, batch_size = 100, num_epochs = 5 , shuffle = True)

# create the linearClissifier 
model = tf.estimator.LinearClassifier(feature_columns = feat_cols)

#train our model
model.train(input_fn = input_func, steps = 6000)

# create a prediction function 
pred_fn = tf.estimator.inputs.pandas_input_fn(x = X_test, batch_size = len(X_test), shuffle= False)

# use mdel.predict() and pass your input function, this will produce a generator of predictions
predictions = list(model.predict(input_fn = pred_fn))


# create a list with only the class idx key values from the prediction key dictionaries
# these are the predictions you will use to compare against the real y_test values

final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])

    
final_preds[:10]

# use sklearn tot see our the acuracy 
from sklearn.metrics import classification_report
print(classification_report(y_test, final_preds))


