import time
import numpy as np
import yfinance as yf
print("SBER.ME, AAPL, BTC-RUB, DOGE-RUB")
while True:
  time.sleep(0.04)
  data_0 = str(time.time())
  data_1 = str(round(time.time(),3))
  print(np.array([data_0,data_1,round(time.time(),1)-float(data_1),]))

  real_time = str(time.strftime("%Y-%m-%d"))
  real_timed = str(time.strftime("%Y-%m-01"))
  inspector = input("пропишите акции >>>> \n")
  data = yf.download(  inspector,start="1000-01-01",end=real_time,interval= "1d")
  print(data.values)
  print(data.shape)
  print(data)
  start_timer = time.time()


  """import requests  # Модуль для обработки URL
  from bs4 import BeautifulSoup  # Модуль для работы с HTML
  # Ссылка на нужную страницу
  url_btc_usd = "https://www.google.com/search?channel=fs&client=ubuntu&q="  + str(inspector)  # "https://www.investing.com/crypto/bitcoin/btc-usd"
  # Заголовки для передачи вместе с URL
  headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:96.0) Gecko/20100101 Firefox/96.0'}
  full_page = requests.get(url_btc_usd, headers=headers)
  # Разбираем через BeautifulSoup
  soup = BeautifulSoup(full_page.content, 'html.parser')
  # Получаем нужное для нас значение и возвращаем его
  convert = soup.findAll("span", {"class": "pclqee"})
  # print(str(convert[0].text))
  mattr = convert[0].text"""
  #print(mattr)
  #data_pay = [mattr[0] + mattr[1] + mattr[3] + mattr[4] + mattr[5] + "." + mattr[7] + mattr[8]]
  #data_pay = np.array(data_pay)
  #data_paye = data_pay[0]
  #data_pay = data_pay[0]
  #print(data_pay)

  real_date = str(time.strftime("%Y-%m-%d"))
  data_btc_usd = yf.download(inspector, start="2014-01-01", end=real_date, interval='1d')
  print(data_btc_usd)
  data = data_btc_usd.filter(['Close'])
  print(data)
  import matplotlib.pyplot as plt
  for i in range(0):
      x = [i for i in range(len(data))]
      y = data
      plt.plot(x, y)
      plt.scatter(x, y)
      plt.savefig(str(inspector) + ".png")

      # Конвертируем данные
  dataset = data.values
  print(dataset)
  # print(dataset)
  import math
  # получение цифровых строк для обучения модели
  training_data_len = math.ceil(len(dataset) * .8)
  print(training_data_len)
  from sklearn.preprocessing import MinMaxScaler
  from keras.models import Sequential
  from keras.layers import Dense, LSTM
  # масштабирование данных
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler.fit_transform(dataset)
  print(scaled_data)

  # создание обучающего набора данных
  train_data = scaled_data[0:training_data_len, :]

  # разделение данных на наборы данных x_train и y_train
  x_train = []
  y_train = []
  for i in range(60, len(train_data)):
      x_train.append(train_data[i - 60:i, 0])
      y_train.append(train_data[i, 0])
      if i <= 61:
          print(x_train)
          print(y_train)
          print()

  # преобразование x_train и y_train в массивы numpy
  x_train, y_train = np.array(x_train), np.array(y_train)
  # bot.send_message(CHANNEL_NAME, (x_train, y_train))

  # reshape the data
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
  print(x_train.shape)

  # biuld to LSTM model
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
  model.add(LSTM(101, return_sequences=False))
  model.add(Dense(50))
  model.add(Dense(25))
  model.add(Dense(1))

  # compale the model
  model.compile(optimizer='adam', loss='mean_squared_error')

  # train_the_model
  from tensorflow import keras

  #model = keras.models.load_model(inspector + ".h5")
  # model.load("/content/gdrive/MyDrive/8pro2023/Диплом/BTC-USD.h5")
  model.fit(x_train, y_train, batch_size=2, epochs=1)
  model.save(inspector + ".h5")

  # create the testing data set
  # create a new array containing scaled values from index 1713 to 2216
  test_data = scaled_data[training_data_len - 60:, :]

  # create the fata sets x_test and y_test
  x_test = []
  y_test = dataset[training_data_len:, :]
  for i in range(60, len(test_data)):
      x_test.append(test_data[i - 60:i, 0])

  # conert the data to numpy array
  x_test = np.array(x_test)

  # reshape the data
  X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

  # get the model predicted price values
  predictions = model.predict(X_test)
  predictions = scaler.inverse_transform(predictions)

  # get the root squared error (RMSE)
  rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
  print(rmse)

  # get the quate
  btc_quote = data_btc_usd
  # create a new dataframe

  new_df = btc_quote.filter(['Close'])

  # get teh last 60 days closing price values and convert the dataframe to an array
  last_60_days = new_df[-1:].values
  # scale the data to be values beatwet 0 and 1

  last_60_days_scaled = scaler.transform(last_60_days)

  # creAte an enemy list
  X_test = []
  # Append past 60 days
  X_test.append(last_60_days_scaled)

  # convert the x tesst dataset to numpy
  X_test = np.array(X_test)

  # Reshape the dataframe
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  # get predict scaled

  pred_price = model.predict(X_test)
  # undo the scaling
  pred_price = scaler.inverse_transform(pred_price)
  print(pred_price)
  pred_price = np.array(pred_price)
  pred_price = pred_price[0]
  pred_price = pred_price[0]
  # Отсылаем юзеру сообщение в его чат
  print(pred_price)

  end_timer = time.time() - start_timer

  answer = "Прогнозируемый результат BTC-USD " + str(pred_price) + "\n\nВремя обработки запроса сек.: " + str(
      int(end_timer))

  print(answer)

