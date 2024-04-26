import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




data = pd.read_csv("IMDB Dataset.csv")
data.head(5)




from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding
from tensorflow.keras.preprocessing.text import Tokenizer




tokenizer = Tokenizer()
tokenizer.fit_on_texts(data["review"])
X = tokenizer.texts_to_sequences(data["review"])
X = pad_sequences(X)
X[0]




from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()
y = lr.fit_transform(data["sentiment"])
y




from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)




model = Sequential()
model.add(Embedding(input_dim = len(tokenizer.word_index)+1, output_dim=128,input_length=X.shape[1]))
model.add(LSTM(units=64, dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer="adam",loss = "binary_crossentropy",metrics=["accuracy"])




history = model.fit(X_train,y_train,epochs=5,batch_size=128,validation_split=0.2)




loss,accuracy,precision,recall = model.evaluate(X_test,y_test)




# Evaluate the model
loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.legend()
plt.show()