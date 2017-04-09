import numpy as np
from sklearn import preprocessing

csv = np.genfromtxt ('database.csv', delimiter=",")
#print(csv)
y=csv[1:,[8]]
#print(y) #magnitude
X=csv[1:,2:6]
X=np.delete(X, 2,axis=1)
#print(X[0:2]) #latitude, longitude, depth -- Not sure how to work with date time
X1_min = np.amin(X,0)
X1_max = np.amax(X,0)
print("Mininum values:",X1_min)
print("Maximum values:",X1_max)
InputX1_norm = (X-X1_min)/(X1_max-X1_min)
#print(InputX1_norm)

Y1_min = np.amin(y,0)
Y1_max = np.amax(y,0)
InputY1_norm = (y-Y1_min)/(Y1_max-Y1_min)
#print(InputY1_norm)
np.random.seed(1)
syn0 = 2 * np.random.random((3, 4)) - 1 #
syn1 = 2 * np.random.random((4, 1)) - 1 #InputX1_norm.shape[0]


def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(np.longfloat(-x)))


#Transforms features by scaling each feature to a given range
#min_max_scaler = preprocessing.MinMaxScaler()
#X = min_max_scaler.fit_transform(X)
#print(syn0)



for j in range(10000):
    I0 = InputX1_norm
    I1 = nonlin(np.dot(I0, syn0))
    I2 = nonlin(np.dot(I1, syn1))
    # print(I2)
    I2_error = y - I2

    I2_delta = I2_error * nonlin(I2, deriv=True)

    I1_error = I2_delta.dot(syn1.T)

    I1_delta = I1_error * nonlin(I1, deriv=True)

    syn1 += I1.T.dot(I2_delta)
    syn0 += I0.T.dot(I1_delta)



input_test=np.array([19.246,145.616,131.6])
min_max_scaler = preprocessing.MinMaxScaler()
X_test = min_max_scaler.fit_transform(input_test)
#print(X_test)
I1 = nonlin(np.dot(X_test, syn0))
I2 = nonlin(np.dot(I1, syn1))
print(I1)
print(I2)