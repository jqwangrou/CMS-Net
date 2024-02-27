#
# # -*- coding: utf-8 -*-

from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


tf.reset_default_graph()  # Reset default graph

tf.keras.backend.clear_session()
def load_data(data_path):
    dataset = pd.read_csv(data_path, header=None, dtype=str)
    exp1 = dataset.iloc[1:, 2:].astype(float)
    scaler = StandardScaler()
    exp1_normalized = scaler.fit_transform(exp1.T).T
    exp1_normalized_df = pd.DataFrame(exp1_normalized, columns=dataset.columns[2:])

    train_data = np.array(exp1_normalized_df.iloc[:, :].values, dtype=np.float32)
    train_label = np.array(dataset.iloc[1:, 1].values, dtype=np.int32)

    x_train, x_test, y_train, y_test = train_test_split(train_data, train_label,test_size=0.3,random_state=42)

    return x_train, x_test, y_train, y_test
data_dir = "./data/dataset/"
data_path = data_dir + "sva_result_516_train.csv"
x_train, x_test, y_train, y_test = load_data(data_path)
x_train_dl1 = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
x_test_dl1 = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = load_model('./data/model_classes.h5')  # Replace with your model weight file path
model.summary()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    def feature_importance_analysis(model, x):
        # Gets indexes for the input and output layers
        input_index = model.input
        output_index = model.output

        # Calculate the gradient of the feature against the output
        grads = tf.keras.backend.gradients(output_index, input_index)[0]
        # grads = tf.keras.backend.gradients(input_index, output_index)[0]
        # Definition calculation diagram
        sess = tf.keras.backend.get_session()

        # Calculated gradient
        feature_importance_values = sess.run(grads, feed_dict={input_index: x})

        # Take the absolute value as the feature importance
        feature_importance_values = np.abs(feature_importance_values)

        return feature_importance_values
    # Feature analysis
    feature_importance = feature_importance_analysis(model, x_test_dl1)
    # print(feature_importance)

    # Output the importance of each feature
    for i, importance in enumerate(feature_importance):
        print(f"sample {i+1} Importance: {importance}")

    features = []
    importances = []

    for i, importance in enumerate(feature_importance):
        features.append(f"sample {i+1}")
        importances.append(importance)


    df = pd.DataFrame({'sample': features, '重要性': importances})

    print(df)
    # df.to_csv('./data/feature_importance.csv', index=False,sep=',')