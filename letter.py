import pandas as pd

data = pd.read_csv("letter-recognition.csv")
data.sample(5)




X = data.drop(columns=['letter'])
Y = data['letter']




from sklearn.preprocessing import LabelEncoder

lr = LabelEncoder()
y = lr.fit_transform(Y)




from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)




from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout

model = Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(26, activation='softmax'))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")




history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=64, validation_split=0.1)




y_prob = model.predict(X_test_scaled)
y_pred = y_prob.argmax(axis=1)




import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

predictions = model.predict(X_test_scaled)
predicted_labels = np.argmax(predictions, axis=1)

p = lr.inverse_transform(predicted_labels)
t = lr.inverse_transform(y_test)

conf_matrix = confusion_matrix(t, p)

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=data['letter'].unique(), yticklabels=data['letter'].unique())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()