import numpy  as np
import pandas as pd
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
import math
from import_data import arrange_time_series
import matplotlib.pyplot as plt

import seaborn

LOOK_BACK = 1
np.random.seed(1234)


def create_dataset(dataset):
    '''
    Looks shifts the time series back by LOOK_BACK steps
    '''
    dataX, dataY = [], []
    for i in range(len(dataset)-LOOK_BACK-1):
	a = dataset[i:(i+LOOK_BACK), 0]
	dataX.append(a)
	dataY.append(dataset[i + LOOK_BACK, 0])
    return np.array(dataX), np.array(dataY)


def create_train_test(dataset):
    '''
    Creates a train test split for time series data (univariate) 
    '''
    
    train_size = int( len(dataset) * 0.67 )
    test_size = 1 - train_size
    return  dataset[0:train_size,:], dataset[train_size:,:]


def define_model():
    '''
    Creates a model that 
    '''
    model = Sequential()
    model.add(Dense(8, input_dim=LOOK_BACK, activation='relu'))
    model.add(Dense(1))
    
    print "Compiling model"
    model.compile(loss='mean_squared_error', optimizer='adam')
    print "Compiled model"
    
    return model


if __name__ == '__main__':

    logins = arrange_time_series(rule='H')
    logins_df = pd.DataFrame(logins)
    dataset = logins_df.values.astype('float32')

    train, test = create_train_test(dataset)
    trainX, trainY = create_dataset(train)
    testX, testY = create_dataset(test)

    model = define_model()
    model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=2)

    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print 'Train Score: {:.2f} MSE ({:.2f} RMSE)'.format(trainScore, math.sqrt(trainScore))
    
    testScore = model.evaluate(testX, testY, verbose=0)
    print 'Test Score: {:.2f} MSE ({:.2f} RMSE)'.format(testScore, math.sqrt(testScore))

    # generate predictions for training
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Plots the predicted training
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[LOOK_BACK:len(trainPredict)+LOOK_BACK, :] = trainPredict

    # Plots the predicted test 
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(LOOK_BACK*2)+1:len(dataset)-1, :] = testPredict

    # Assembles plots and shows user 
    plt.plot(dataset, alpha=0.5)
    plt.plot(trainPredictPlot, alpha=0.5)
    plt.plot(testPredictPlot, alpha=0.5)
    plt.savefig('../images/nn_train_test_vs_dataset.png')
    plt.clf()
    
    
    
    train_size = int( len(dataset) * 0.67 )
    
    test_diff = testPredict - dataset[train_size+2:]
    
    plt.plot(test_diff)
    plt.show()
    plt.close()
