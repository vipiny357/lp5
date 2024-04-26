import pandas as pd

boston = pd.read_csv("BostonHousing.csv")
boston.sample(5)




X = boston.drop(columns=['medv'])
y = boston['medv']




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)




from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




from keras.layers import Dense
from keras.models import Sequential

model = Sequential()

model.add(Dense(128, activation='relu', input_dim=13))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(optimizer = 'adam',loss = 'mean_squared_error')




model.fit(X_train, y_train, epochs = 100)




y_pred = model.predict(X_test)




from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
r2 = r2_score(y_test, y_pred)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:   ", r2)
print("RMSE: ", rmse)




import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,8))
sns.regplot(x=y_pred, y=y_test, data=boston)
plt.axis('tight')
plt.ylabel('The True Prices', fontsize=20)

plt.xlabel('Predicted Prices', fontsize=20)
plt.title("Predicted Boston Housing Prices vs. Actual", fontsize=20)

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

plt.show()