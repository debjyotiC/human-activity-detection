import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Specify Data Path and assign it to a variable
file_path = "fall_20_feature_train_data.xlsx" #provide path of your data. Source could be either on your local machine or internet
data = pd.read_excel(file_path)

X = data.iloc[:,:-1].values
Y = data.iloc[::,-1].values

Y = tf.keras.utils.to_categorical(Y, num_classes=2)

X = np.array(X)
X = X.reshape(Y.shape[0],1,20,1)  #X,Y

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=42,stratify=Y)


# Define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(20, kernel_size=(3,3),strides=1, activation='relu', input_shape=(1,20,1),data_format="channels_first",padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3),data_format="channels_first"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(20, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.summary()

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=20, epochs=20, verbose=True, validation_data=(X_test, Y_test))

model.save("har.model") #final model
