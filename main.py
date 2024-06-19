import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

training_data=pd.read_csv("Train_set.csv")
training_set=training_data.iloc[:, 1:2].values
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)

# Creating data structure with 60 timesteps and 1 output
x_train=[]
y_train=[]
for i in range(60,1259):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i,0])

x_train, y_train= np.array(x_train), np.array(y_train)
# x_train.shape= (1199, 60)
x_train=x_train.reshape(1199, 60,1)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=60, activation='relu',
                               return_sequences=True, input_shape=(60,1)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(units=60, activation='relu',
                               return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(units=80, activation='relu',
                               return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(units=120, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
# for classification problems, we use cross entropy loss functions but for
# regression problems, we use mean_squared_error loss functions

model.fit(x_train, y_train, batch_size=32, epochs=100)

test_data=pd.read_csv('Test_set.csv')
real_stock_price=test_data.iloc[:,1:2].values
# we want only open column of the data in our test data and .values makes array
data_list = [training_data['Open'], test_data['Open']]
dataset_total = pd.concat(data_list, axis=0)
inputs=dataset_total[len(dataset_total)-len(test_data)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
x_test=[]
for i in range(60, 80):
    x_test.append(inputs[i-60:i,0])

x_test=np.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price=model.predict(x_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

print(predicted_stock_price[0], real_stock_price[0])
print(predicted_stock_price[5], real_stock_price[5])
print(predicted_stock_price[10], real_stock_price[10])

plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='green',
         label='Predicted Google Stock Price')
plt.title("Google Stock Price Prediction")
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()