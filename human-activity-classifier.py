import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Features/features_all_classes.csv')

X = df.drop(columns=['Label']).values
y = df['Label'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

classes = len(np.unique(y_encoded))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(X_train.shape, y_train.shape)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Input(shape=(X_train.shape[1],)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(classes, activation='softmax')
# ])

X_train_cnn = X_train.reshape(-1, 19, 1, 1)
X_test_cnn = X_test.reshape(-1, 19, 1, 1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3,1), strides=1, activation='relu', input_shape=(19, 1, 1), padding='same'),
    tf.keras.layers.Conv2D(32, kernel_size=(3,1), strides=1, activation='relu', padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(classes, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_cnn, y_train, epochs=100, batch_size=10, validation_data=(X_test_cnn, y_test))

test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

y_pred = model.predict(X_test_cnn)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("images/cm.png", dpi=300)
plt.show()

