import numpy as np
from numpy import concatenate

# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import numpy as np

import numpy as np

#Original
def create_dataset_antes(ts, dim ,h):
    look_back = dim + h -1
    # dataset = np.insert(dataset, [0] * look_back, 0)
    dataX, dataY = [], []
    for i in range(len(ts) - look_back):
        a = ts[i:(i + look_back)]
        dataX.append(a)
        dataY.append(ts[i + look_back])
    dataY = np.array(dataY)
    dataY = np.reshape(dataY, (dataY.shape[0], 1))
    dataset = np.concatenate((dataX, dataY), axis=1)
    return dataset

def rezago_sig(pacf, banda):
    rezago_elegido = None
    for i in range(len(pacf)-1, 1, -1):  
        if abs(pacf[i]) > banda and abs(pacf[i-1]) > banda and abs(pacf[i-2]) > banda:
            diferencia_porcentaje = abs(pacf[i]) / abs(pacf[i-1]) -1 
            if diferencia_porcentaje > 0.20:
                rezago_elegido = i
                break
    
    if rezago_elegido is None:
        for i in range(1, len(pacf)):  
            if abs(pacf[i]) > banda:
                rezago_elegido = i
                break
    
    return rezago_elegido


def create_dataset(data, dim, h, col):
    dataset = []
    look_back = dim + h - 1
    print(f"look_back: {look_back}")
    print(f"len(data): {len(data)}")

    for i in range((len(data) - look_back) // col + 1):
        start_ix = i * col
        end_ix = start_ix + look_back + col
        print(f"i: {i}, start_ix: {start_ix}, end_ix: {end_ix}")

        if end_ix > len(data):
            end_ix = len(data)  # Ajuste para incluir los últimos elementos de data

        window_data = data[start_ix:end_ix].flatten()
        dataset.append(window_data)
        print(f"window_data: {window_data}")

        if window_data[-1] == data[-1]:
            print("Last element of window_data matches the last element of data, breaking the loop.")
            break

    print(f"Final dataset length: {len(dataset)}")
    return np.array(dataset)

def create_dataset_rez(data, dim, h, col, rez):
    dataset = []
    look_back = rez
    

    for i in range(0, len(data)): 
        start_ix = i * col
        end_ix = start_ix + (look_back* col)+col
        

        if end_ix > len(data):
            end_ix = len(data)  # Ajuste para incluir los últimos elementos de data

        window_data = data[start_ix:end_ix].flatten()
        dataset.append(window_data)
        

        if window_data[-1] == data[-1]:
            print("Last element of window_data matches the last element of data, breaking the loop.")
            break

    print(f"Final dataset length: {len(dataset)}")
    return np.array(dataset)


def unpadding(y):
    a = y.copy()
    h = y.shape[1]
    s = np.empty(y.shape[0] + y.shape[1] -1)

    for i in range(s.shape[0]):
        s[i]=np.diagonal(np.flip(a,1), offset= -i + h-1,axis1=0,axis2=1).copy().mean()
    
    return s

def mape(y_true, y_pred): 
    y_true = unpadding(y_true)
    y_pred = unpadding(y_pred)

    mask =  y_true != 0.0
    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    N_metric =  (y_true[mask] - y_pred[mask])/y_true[mask]
    N_metric = np.fabs(N_metric)
    metric = N_metric.mean()

    return metric

def smape(y_true, y_pred): 
    y_true = unpadding(y_true)
    y_pred = unpadding(y_pred)

    mask =  y_true != 0.0
    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    N_metric =  (y_true[mask] - y_pred[mask])/(y_true[mask] + y_pred[mask])
    N_metric = np.fabs(N_metric)
    metric = N_metric.mean()

def rmse(y_true, y_pred):
    y_true = unpadding(y_true)
    y_pred = unpadding(y_pred)

    return np.sqrt(mean_squared_error(y_true,y_pred))



#### Original functions(uncomment if it is required)
#from sklearn.svm import SVR
#from sklearn.base import BaseEstimator, TransformerMixin

#class CustomMSVR(BaseEstimator, TransformerMixin):
#    def __init__(self, kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1):
#        self.kernel = kernel
#        self.degree = degree
#        self.gamma = gamma
#        self.coef0 = coef0
#        self.tol = tol
#        self.C = C
#        self.epsilon = epsilon
#        self.models = []  
#    def fit(self, X, y):
#        # Ajustar un modelo SVR para cada variable?
#        for i in range(y.shape[1]):
#            svr = SVR(kernel=self.kernel, degree=self.degree, gamma=self.gamma,
#                      coef0=self.coef0, tol=self.tol, C=self.C, epsilon=self.epsilon)
#            svr.fit(X, y[:, i])  
#            self.models.append(svr) 
#        return self
#
#    def predict(self, X):
#        predictions = np.column_stack([model.predict(X) for model in self.models])
#        return predictions
#
#    def get_params(self, deep=True):
#        return {
#            'kernel': self.kernel,
#            'degree': self.degree,
#            'gamma': self.gamma,
#            'coef0': self.coef0,
#            'tol': self.tol,
#            'C': self.C,
#            'epsilon': self.epsilon
#        }

###Modification por stopping based on number of iterations
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR
import numpy as np

class CustomMSVR(BaseEstimator, RegressorMixin):
    def __init__(self, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, max_iter=2000):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.max_iter = max_iter  # <--- Nuevo parámetro para forzar la finalización

    def fit(self, X, y):
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
            
        self.models_ = []  
        
        for i in range(y_arr.shape[1]):
            svr = SVR(
                kernel=self.kernel, 
                degree=self.degree, 
                gamma=self.gamma,
                coef0=self.coef0, 
                tol=self.tol, 
                C=self.C, 
                epsilon=self.epsilon,
                max_iter=self.max_iter  # <--- Pasa el límite a SVR
            )
            svr.fit(X, y_arr[:, i])  
            self.models_.append(svr)
            
        return self

    def predict(self, X):
        # Asegura la compatibilidad con el método predict
        return np.column_stack([model.predict(X) for model in self.models_])
