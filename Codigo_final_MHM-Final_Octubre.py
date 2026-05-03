# -*- coding: utf-8 -*-
"""
@author: huma1003

"""
import os
os.chdir("C:/Users/huma1003/OneDrive - NIQ/DOCUMENTOS IMPORTANTES/Mis_Cosas/MarlijarTM/msvr-master")

#Paquetes
import pandas as pd
import numpy as np
from model.MSVR import MSVR
from model.utility import create_dataset,create_dataset_antes, rmse, CustomMSVR,create_dataset_rez,rezago_sig
from model.Base import Base
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from scikeras.wrappers import  KerasRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
import math
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
import csv
import time
from sklearn.metrics import mean_squared_error
# from keras.models import KerasRegressor
import random
from sklearn.model_selection import train_test_split
from statsmodels.tsa.api import VAR
from scipy.linalg import orth
from scipy.stats import multivariate_normal
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.api import VAR
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))



#_Ejemplo de el paper_______________________________________________________#

# Construct x samples (input) and y samples (output)
# x: num_samples * inputDim
# y: num_smaples * outputH
ts = np.sin(np.arange(0, 9, 0.01)).reshape(-1)
segmentation = int(len(ts)*2/3)
dim = 50
h = 5

#Aqui como modifique el create_Dataset para que reciba salidas multivariadas
#el original es este:
dataset = create_dataset_antes(ts, dim, h)
dataset = create_dataset(ts, dim, h,1)
X, Y = dataset[:, :(0 - h)], dataset[:, (0-h):]
train_input = X[:segmentation, :]
train_target = Y[:segmentation].reshape(-1, h)
test_input = X[segmentation:, :]
test_target = Y[segmentation:].reshape(-1, h)

msvr = MSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1)
# Train
msvr.fit(train_input, train_target)

# Predict with train set
trainPred = msvr.predict(train_input)
# Predict with test set
testPred = msvr.predict(test_input)

trainMetric_paper = rmse(train_target,trainPred)
testMetric_paper = rmse(test_target,testPred)

print(trainMetric_paper, testMetric_paper)

####Ajustando los hiperpárametros

pipe = Pipeline([
    ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))

])

hyperparameters = {
    'MSVR__kernel': ['rbf','poly'],
    'MSVR__degree': range(1,10,1),
    'MSVR__gamma': [0, 1],
    'MSVR__coef0': [0, 1],
    'MSVR__C': range(1, 10),
    'MSVR__epsilon': [0, 1],   
}
  
# Búsqueda aleatoria de hiperparámetros
bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=50, scoring='neg_mean_squared_error', cv=10, verbose=1, error_score='raise')
best_model = bm.fit(train_input, train_target)
print("Best hyperparameters:", bm.best_params_)
print("Best score:", bm.best_score_)
 
msvr = MSVR(kernel = bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"), epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"), degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)
msvr.fit(train_input, train_target)

trainPred_hiper = msvr.predict(train_input)
testPred_hiper = msvr.predict(test_input)

trainMetric_hiper = rmse(train_target,trainPred_hiper)
testMetric_hiper = rmse(test_target,testPred_hiper)

print(trainMetric_hiper, testMetric_hiper)

##Ajustando la búsqueda de hiperparámetros, corrida 100 veces.
hiperparametros = []
train_RMSE = []
test_RMSE = []

for no in range(100):
    pipe = Pipeline([
        ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))
    ])

    hyperparameters = {
        # 'MSVR__kernel': ['rbf','poly'],
        'MSVR__kernel': ['rbf'],
        'MSVR__degree': range(1,10,1),
        'MSVR__gamma': [0,0.01,0.05,0.1,0.5, 1],
        'MSVR__coef0': [0,0.5,1,1.5,2, 5],
        'MSVR__C': range(1, 10),
        'MSVR__epsilon':[0,0.0001,0.001,0.01,0.05,0.1], 
    }
    
    # Búsqueda aleatoria de hiperparámetros
    bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=50, scoring='neg_mean_squared_error', cv=10, verbose=0, error_score='raise')
    best_model = bm.fit(train_input, train_target)
    print("Best hyperparameters:", bm.best_params_)
    print("Best score:", bm.best_score_)

    msvr = MSVR(kernel=bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"),
                epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"),
                degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)
    msvr.fit(train_input, train_target)

    trainPred_hiper = msvr.predict(train_input)
    testPred_hiper = msvr.predict(test_input)

    trainMetric_hiper = rmse(train_target,trainPred_hiper)
    testMetric_hiper = rmse(test_target,testPred_hiper)
    print(trainMetric_hiper, testMetric_hiper)
    
    hiperparametros.append(bm.best_params_)
    train_RMSE.append(trainMetric_hiper)
    test_RMSE.append(testMetric_hiper)


with open('resultados_3.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Hiperparametros', 'Train RMSE', 'Test RMSE'])
    for i in range(100):
        writer.writerow([hiperparametros[i], train_RMSE[i], test_RMSE[i]])

  

#Comparaciones
#Train
print(trainMetric_paper, trainMetric_hiper)
print("Modelo en Train          | rmse")
print("-----------------------------------------------")
print("Modelo de Paper:")
print(trainMetric_paper)
print("-----------------------------------------------")
print("Modelo con Hiperparámetros Ajustados:")
print(trainMetric_hiper)
#Test
print(testMetric_paper, testMetric_hiper)
print("Modelo en Test        | rmse")
print("-----------------------------------------------")
print("Modelo de Paper:")
print(testMetric_paper)
print("-----------------------------------------------")
print("Modelo con Hiperparámetros Ajustados:")
print(testMetric_hiper)

#Pares e impares

fechas = pd.DataFrame(list(range(1,51, 1)))
pares = pd.DataFrame(list(range(0, 100, 2)))
impares = pd.DataFrame(list(range(1, 100, 2)))
total = pd.concat([fechas,pares, impares], axis=1).values
dim=len(total)

#Construcción de la base de datos
data=Base(total)
data= data.base
col=2
h = 1
#Creamos la base de datos
dataset = create_dataset(data,dim,h,col)
X, Y = dataset[:, :(0 - h*2)], dataset[:, (0-h*2):]
#Train y test
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, shuffle=False)


##Ajustando la búsqueda de hiperparámetros, corrida 100 veces.
hiperparametros = []
train_RMSE = []
test_RMSE = []

for no in range(100):
    pipe = Pipeline([
        ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))
    ])

    hyperparameters = {
        # 'MSVR__kernel': ['rbf','poly'],
        'MSVR__kernel': ['poly'],
        'MSVR__degree': range(1,10,1),
        'MSVR__gamma': [0,0.01,0.05,0.1,0.5, 1],
        'MSVR__coef0': [0,0.5,1,1.5,2, 5],
        'MSVR__C': range(1, 10),
        'MSVR__epsilon':[0,0.0001,0.001,0.01,0.05,0.1], 
    }
    
    # Búsqueda aleatoria de hiperparámetros
    bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=50, scoring='neg_mean_squared_error', cv=10, verbose=0, error_score='raise')
    best_model = bm.fit(X_train,y_train)
    print("Best hyperparameters:", bm.best_params_)
    print("Best score:", bm.best_score_)

    msvr = MSVR(kernel=bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"),
                epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"),
                degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)
    
    msvr.fit(X_train, y_train)

    trainPred_hiper = msvr.predict(X_train)
    testPred_hiper = msvr.predict(X_test)

    trainMetric_hiper = rmse(y_train,trainPred_hiper)
    testMetric_hiper = rmse(y_test,testPred_hiper)
    print(trainMetric_hiper, testMetric_hiper)
    
    hiperparametros.append(bm.best_params_)
    train_RMSE.append(trainMetric_hiper)
    test_RMSE.append(testMetric_hiper)


with open('resultados_paresimpares_oly.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Hiperparametros', 'Train RMSE', 'Test RMSE'])
    for i in range(100):
        writer.writerow([hiperparametros[i], train_RMSE[i], test_RMSE[i]])

  


#------------------------------------------------------------------------------------------------------######

#--------------------------VAR BIVARIADO-Estacionario------------------------------------------------------------#
t=1000 # Longitud de la serie
k =2 # dimensión del vector Y
p = 1 # Número de retardos
h=1
col=2

# Almacenar resultados
hiperparametros_svr = []
vectores_soporte=[]
train_RMSE_svr = []
test_RMSE_svr = []
train_RMSE_var = []
test_RMSE_var = []
tiempo_var=[]
tiempo_msvr=[]
# Generación de la serie y ajuste de modelos
for no in range(100):
    
    print(f"-------------------------Iteration {no}--------------------------")
    # Generar series
    # A = np.array([[0.5, 0.4],
    #                [0.1, 0.8]])

    # initial = np.random.normal(size=(2,))
    # serie = np.zeros((2, t))
    # serie[:, :1] = initial[:, np.newaxis]

    # for i in range(1, t):
    #    retardo = serie[:, i-1:i]
    #    serie[:, i] = np.dot(A, retardo.flatten()) +np.random.normal(loc=0.0, scale=1.0, size=2)


    # series = pd.DataFrame(serie.T, columns=['Y1', 'Y2'])
    
    #Partición en train y test
    train_size = int(len(series) * 0.7)
    train, test = series.iloc[:train_size], series.iloc[train_size:]
    
    #PACF
    pacf_var1 = pacf(train['Y1'], nlags=16)
    pacf_var2 = pacf(train['Y2'], nlags=16)
    banda= 1.96 / np.sqrt(t)  
   
    rezago_elegido_1 = rezago_sig(pacf_var1, banda)
    rezago_elegido_2 = rezago_sig(pacf_var2, banda)
    enumerated_list = list(enumerate(pacf_var1))
    reversed_enumerated_list = list(reversed(enumerated_list))
    filtered_indices = [i for i, x in reversed_enumerated_list if abs(x) > banda]

    
    rez= int(min(rezago_elegido_1, rezago_elegido_2))
    #rez=1
    print(f"----------------Rezago {rez}--------------")
        
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
# # Plotear la PACF de Y1
# axes[0].stem(range(len(pacf_var1)), pacf_var1, basefmt=" ", use_line_collection=True)
# axes[0].axhline(y=banda, color='r', linestyle='--', label=f'Banda superior ({banda:.2f})')
# axes[0].axhline(y=-banda, color='r', linestyle='--', label=f'Banda inferior ({-banda:.2f})')
# axes[0].set_xlabel('Rezago')
# axes[0].set_ylabel('PACF Y1')
# axes[0].set_title('PACF de Y1 con Bandas de Confianza')
# axes[0].legend()

# # Plotear la PACF de Y2
# axes[1].stem(range(len(pacf_var2)), pacf_var2, basefmt=" ", use_line_collection=True)
# axes[1].axhline(y=banda, color='r', linestyle='--', label=f'Banda superior ({banda:.2f})')
# axes[1].axhline(y=-banda, color='r', linestyle='--', label=f'Banda inferior ({-banda:.2f})')
# axes[1].set_xlabel('Rezago')
# axes[1].set_ylabel('PACF Y2')
# axes[1].set_title('PACF de Y2 con Bandas de Confianza')
# axes[1].legend()

# # Mostrar los gráficos
# plt.tight_layout()
# plt.show()
    # 
    
    
    # Ajustar modelo VAR
    start_time = time.time()
    model_var = VAR(train)
    results_var = model_var.fit(maxlags=1, ic='aic')
    lag_order = results_var.k_ar
    
    modelo_var_train = []
    modelo_var_test = []
        
    #Predicciones para train
    train_pred = results_var.fittedvalues
    # Predicciones para test
    test_pred=[]
    input_data = train.values[-rez:]

    for i in range(len(test)):
        pred = results_var.forecast(y=input_data, steps=h)
        test_pred.append(pred[0])
        input_data = np.vstack([input_data[1:], test.values[i:i+1]])

    test_pred = np.array(test_pred)
          
     
    train_rmse_var = np.sqrt(mean_squared_error(train.values[rez:], train_pred))
    test_rmse_var = np.sqrt(mean_squared_error(test.values, test_pred))

    
    train_RMSE_var.append(train_rmse_var)
    test_RMSE_var.append(test_rmse_var)
    print("Termine de ajustar modelo VAR")
    end_time = time.time()
    execution_time = end_time - start_time
    tiempo_var.append(execution_time)
    
    # Ajustar modelo SVR
    start_time = time.time()
    fechas = pd.DataFrame(list(range(len(series))))
    total = pd.concat([fechas,series], axis=1).values
    dim=len(total)

    #Construcción de la base de datos
    data=Base(total)
    data= data.base
    #Creamos la base de datos
    dataset = create_dataset_rez(data,dim,h,col,rez)
    X, Y = dataset[:, :(0 - h*2)], dataset[:, (0-h*2):]
    #Train y test
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, shuffle=False)
    
    #normalizados
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    scaler_X.fit(X_train)
    scaler_y.fit(y_train)

   
    X_train_nor = scaler_X.transform(X_train)
    X_test_nor = scaler_X.transform(X_test)
    y_train_nor = scaler_y.transform(y_train)
    y_test_nor = scaler_y.transform(y_test)

    pipe = Pipeline([
        ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))
    ])

    hyperparameters = {
        #'MSVR__kernel': ['poly'],
        'MSVR__kernel': ['poly','rbf','linear'],
        'MSVR__degree': [2,5],
        #'MSVR__degree': [1],
        'MSVR__gamma': [0.5,1],
        'MSVR__coef0': [0.1,0.5,1],
        'MSVR__C': [5,9,11,13],
        'MSVR__epsilon':[1,2], 
    }
    
    bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=15, scoring='neg_mean_squared_error', cv=5, verbose=0, error_score='raise')
   
    best_model = bm.fit(X_train_nor, y_train_nor)
    best_params = bm.best_params_

    msvr = MSVR(kernel=bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"),
                epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"),
                degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)
    
    msvr.fit(X_train_nor, y_train_nor)
  
 
    trainPred_svr_nor = msvr.predict(X_train_nor)
    testPred_svr_nor = msvr.predict(X_test_nor)

    trainPred_svr  = scaler_y.inverse_transform(trainPred_svr_nor)
    testPred_svr  = scaler_y.inverse_transform(testPred_svr_nor)
    train_rmse_svr = rmse(y_train, trainPred_svr)
    test_rmse_svr = rmse(y_test, testPred_svr)
    
    hiperparametros_svr.append(best_params)
    vectores_soporte.append(msvr.NSV)
    print("SVR Best params:", msvr.NSV/t)
    train_RMSE_svr.append(train_rmse_svr)
    test_RMSE_svr.append(test_rmse_svr)
    
    end_time = time.time()
    execution_time = end_time - start_time
    tiempo_msvr.append(execution_time)

    
    print("SVR Best params:", best_params)
    print("VAR Train RMSE:", train_rmse_var, "Test RMSE:", test_rmse_var)
    print("SVR Train RMSE:", train_rmse_svr, "Test RMSE:", test_rmse_svr)



# Guardar resultados en un archivo CSV

with open('resultados_comparacion_setar_uni.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Modelo', 'Hiperparametros', 'Vectores_soporte', 'Train RMSE', 'Test RMSE', 'Tiempo'])
    for i in range(100):
        writer.writerow(['SVR', hiperparametros_svr[i],vectores_soporte[i], train_RMSE_svr[i], test_RMSE_svr[i],tiempo_msvr[i]])
        writer.writerow(['VAR', 'N/A','N/A', train_RMSE_var[i], test_RMSE_var[i],tiempo_var[i]])

#--------------------------VAR BIVARIADO-Estacionario- de orden 2------------------------------------------------------------#

t=1000 # Longitud de la serie
k =2 # dimensión del vector Y
p = 2 # Número de retardos
h=1
col=2
# Almacenar resultados
hiperparametros_svr = []
vectores_soporte=[]
train_RMSE_svr = []
test_RMSE_svr = []
train_RMSE_var = []
test_RMSE_var = []
tiempo_var=[]
tiempo_msvr=[]
# Generación de la serie y ajuste de modelos
for no in range(100):
    
    print(f"-------------------------Iteration {no}--------------------------")
    # Generar series
    A1 = np.array([[-0.55, -0.08],  #Rezago 1
               [-0.04, -0.28]])
    A2 = np.array([[0.02, -0.60],  #Rezago 2
               [-0.03, 0.32]])
    serie = np.zeros((k, t))
    initial = np.random.normal(size=(k,p))
    serie[:, :2] = initial
    
    for i in range(2, t):
        serie[:, i] = (np.dot(A1, serie[:, i-1]) + 
                   np.dot(A2, serie[:, i-2]) + 
                   np.random.normal(loc=0.0, scale=9, size=k))   

    series = pd.DataFrame(serie.T, columns=['Y1', 'Y2'])
    
    #Partición en train y test
    train_size = int(len(series) * 0.7)
    train, test = series.iloc[:train_size], series.iloc[train_size:]
    
    #PACF
    pacf_var1 = pacf(train['Y1'], nlags=16)
    pacf_var2 = pacf(train['Y2'], nlags=16)
    banda= 1.96 / np.sqrt(t)  
   
    rezago_elegido_1 = rezago_sig(pacf_var1, banda)
    rezago_elegido_2 = rezago_sig(pacf_var2, banda)
    enumerated_list = list(enumerate(pacf_var1))
    reversed_enumerated_list = list(reversed(enumerated_list))
    filtered_indices = [i for i, x in reversed_enumerated_list if abs(x) > banda]

    
    #rez= int(min(rezago_elegido_1, rezago_elegido_2))
    rez=p
    print(f"----------------Rezago {rez}--------------")
    
    
    # Ajustar modelo VAR
    start_time = time.time()
    model_var = VAR(train)
    results_var = model_var.fit(maxlags=p)
    lag_order = results_var.k_ar
    
    modelo_var_train = []
    modelo_var_test = []
        
    #Predicciones para train
    train_pred = results_var.fittedvalues
    # Predicciones para test
    test_pred=[]
    input_data = train.values[-rez:]

    for i in range(len(test)):
        pred = results_var.forecast(y=input_data, steps=h)
        test_pred.append(pred[0])
        input_data = np.vstack([input_data[1:], test.values[i:i+1]])

    test_pred = np.array(test_pred)
          
     
    train_rmse_var = np.sqrt(mean_squared_error(train.values[rez:], train_pred))
    test_rmse_var = np.sqrt(mean_squared_error(test.values, test_pred))

    
    train_RMSE_var.append(train_rmse_var)
    test_RMSE_var.append(test_rmse_var)
    print("Termine de ajustar modelo VAR")
    end_time = time.time()
    execution_time = end_time - start_time
    tiempo_var.append(execution_time)
    
    # Ajustar modelo SVR
    start_time = time.time()
    fechas = pd.DataFrame(list(range(len(series))))
    total = pd.concat([fechas,series], axis=1).values
    dim=len(total)

    #Construcción de la base de datos
    data=Base(total)
    data= data.base
    #Creamos la base de datos
    dataset = create_dataset_rez(data,dim,h,col,rez)
    X, Y = dataset[:, :(0 - h*2)], dataset[:, (0-h*2):]
    #Train y test
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, shuffle=False)
    
    #normalizados
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    scaler_X.fit(X_train)
    scaler_y.fit(y_train)

   
    X_train_nor = scaler_X.transform(X_train)
    X_test_nor = scaler_X.transform(X_test)
    y_train_nor = scaler_y.transform(y_train)
    y_test_nor = scaler_y.transform(y_test)

    pipe = Pipeline([
        ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))
    ])

    hyperparameters = {
        #'MSVR__kernel': ['poly'],
        'MSVR__kernel': ['poly','rbf','linear'],
        'MSVR__degree': [2,5],
        #'MSVR__degree': [1],
        'MSVR__gamma': [0.5,1],
        'MSVR__coef0': [0.1,0.5,1],
        'MSVR__C': [5,9,11,13],
        'MSVR__epsilon':[1,2], 
    }
    
    bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=15, scoring='neg_mean_squared_error', cv=5, verbose=0, error_score='raise')
   
    best_model = bm.fit(X_train_nor, y_train_nor)
    best_params = bm.best_params_

    msvr = MSVR(kernel=bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"),
                epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"),
                degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)
    
    msvr.fit(X_train_nor, y_train_nor)
  
 
    trainPred_svr_nor = msvr.predict(X_train_nor)
    testPred_svr_nor = msvr.predict(X_test_nor)

    trainPred_svr  = scaler_y.inverse_transform(trainPred_svr_nor)
    testPred_svr  = scaler_y.inverse_transform(testPred_svr_nor)
    train_rmse_svr = rmse(y_train, trainPred_svr)
    test_rmse_svr = rmse(y_test, testPred_svr)
    
    hiperparametros_svr.append(best_params)
    vectores_soporte.append(msvr.NSV)
    print("SVR Best params:", msvr.NSV/t)
    train_RMSE_svr.append(train_rmse_svr)
    test_RMSE_svr.append(test_rmse_svr)
    
    end_time = time.time()
    execution_time = end_time - start_time
    tiempo_msvr.append(execution_time)

    
    print("SVR Best params:", best_params)
    print("VAR Train RMSE:", train_rmse_var, "Test RMSE:", test_rmse_var)
    print("SVR Train RMSE:", train_rmse_svr, "Test RMSE:", test_rmse_svr)



# Guardar resultados en un archivo CSV
#CORRER PRQUE NO LO HE CORRIDO
with open('resultados_comparacion_varest_svr200_2_rez2_var9.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Modelo', 'Hiperparametros', 'Vectores_soporte', 'Train RMSE', 'Test RMSE', 'Tiempo'])
    for i in range(100):
        writer.writerow(['SVR', hiperparametros_svr[i],vectores_soporte[i], train_RMSE_svr[i], test_RMSE_svr[i],tiempo_msvr[i]])
        writer.writerow(['VAR', 'N/A','N/A', train_RMSE_var[i], test_RMSE_var[i],tiempo_var[i]])


#--------------------------Modelo VEC Bivariado con una relación de cointegración.------------------------------------------------------------#

def generate_phi(n, h):
    A = orth(np.random.randn(n, n))[:h, :]
    Phi = np.eye(n)
    if h == 1:
        Phi -= np.outer(A, A)
    elif h > 1:
        Phi -= A.T @ A
    return Phi

def cointegrated_vector(tt, n, h, C, Phi, burn=1000):
    """
    Simulate cointegrated time series data.
    
    Parameters:
    tt : int Time points to simulate.
    n : int Number of time series.
    h : int Number of cointegrating relations.
    C : array_like Covariance matrix for the innovations.
    burn : int, optional Number of initial points to discard (burn-in period)..
    """
    tot_t = tt + burn
    innov = multivariate_normal.rvs(mean=np.zeros(n),cov=C, size=tot_t) 
    X = np.zeros((1 + tot_t, n))
    print("Phi:\n", Phi) 
    for i in range(tot_t):
        X[i + 1, :] = Phi @ X[i, :] + innov[i, :]
    
    return X[burn + 1:, :]
#mantener phi quieto.
#modelos modificando las relaciones de cointegración.

t = [50,200,500,1000,5000]  # Longitud de la serie
k = 2  # Dimensión del vector Y
p = 1  # Número de retardos
c = 1  # Número de relaciones de cointegración
h = 1 #Número de pasos a predecir.
Phi = generate_phi(k, c)  

# Carpeta y archivo Excel
folder_path = r'C:\Users\huma1003\OneDrive - NIQ\DOCUMENTOS IMPORTANTES\Mis_Cosas\MarlijarTM\msvr-master\VEC Bivarido\Data_1'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
for size in t:
    excel_filename = os.path.join(folder_path, f'dataset_size_{size}.xlsx')
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        for no in range(100):
            print(f"-------------------------Tamaño {size}-------------------------")
            print(f"-------------------------Iteration {no}--------------------------")
            series = np.zeros((k, size))
            series = cointegrated_vector(size, k, no, np.eye(k), Phi)
            series = pd.DataFrame(series, columns=['Y1', 'Y2'])
            
            sheet_name = f"Iter_{no}"
            
            series.to_excel(writer, sheet_name=sheet_name, index=False)




# Almacenar resultados
rez=1
hiperparametros_svr = {size: [] for size in t}
vectores_soporte = {size: [] for size in t}
train_RMSE_svr = {size: [] for size in t}
test_RMSE_svr = {size: [] for size in t}
train_RMSE_var_dif = {size: [] for size in t}
test_RMSE_var_dif= {size: [] for size in t}
tiempo_var_dif = {size: [] for size in t}
tiempo_msvr = {size: [] for size in t}
resultados = []

data_folder = "C:/Users/huma1003/OneDrive - NIQ/DOCUMENTOS IMPORTANTES/Mis_Cosas/MarlijarTM/msvr-master/VEC Bivariado/Data"

def load_data(size, no):
    file_path = f"{data_folder}/dataset_size_{size}.xlsx"
    sheet_name = f"Iter_{no}"
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

# Generación de la serie y ajuste de modelos
for size in t:
      a=size
      for no in range(100):
        print(f"-------------------------Tamaño {a}-------------------------")
        print(f"-------------------------Iteration {no}--------------------------")
        #series = np.zeros((k, size))
        #series = cointegrated_vector(size, k, no, np.eye(k),Phi)
        #series = pd.DataFrame(series, columns=['Y1', 'Y2'])
        series = load_data(size, no)
      
        #Partición en train y test
        train_size = int(len(series) * 0.7)
        train, test = series.iloc[:train_size], series.iloc[train_size:]
        test = test.reset_index(drop=True) 
         # Ajustar modelo VAR DIFERENCIADO
        start_time = time.time()
         
        # Ver la serie diferenciada

        series_dif = series.diff().dropna()
        train_size_dif = int(len(series_dif) * 0.7)
        train_dif, test_dif = series_dif.iloc[:train_size_dif], series_dif.iloc[train_size_dif:]
        test_dif = test_dif.reset_index(drop=True) 
        model_var_dif = VAR(train_dif)
        results_var_dif = model_var_dif.fit(maxlags=1)
        lag_order = results_var_dif.k_ar
        
        modelo_var_train_dif = []
        modelo_var_test_dif = []
            
        #Predicciones para train
        train_pred_dif = results_var_dif.fittedvalues
        # Predicciones para test
        test_pred_dif=[]
        input_data = train_dif.values[-rez:]

        for i in range(len(test_dif)):
            pred = results_var_dif.forecast(y=input_data, steps=h)
            test_pred_dif.append(pred[0])
            print
            input_data = np.vstack([input_data[1:], test_dif.values[i:i+1]])

        test_pred_dif = pd.DataFrame(test_pred_dif, columns=['Y1', 'Y2'])
         
        last_value = train.iloc[-1, :]
        
        train_pred_levels = pd.DataFrame(train_pred_dif.cumsum() + train.iloc[lag_order], columns=['Y1', 'Y2'])
        test_pred_levels = pd.DataFrame(test_pred_dif.cumsum() + last_value.values, columns=['Y1', 'Y2'])

        #test_pred_levels = pd.DataFrame(test.iloc[:len(test_pred_dif), :] + test_pred_dif)
        
        
        train_rmse_var_dif = rmse(train.iloc[2:].values, train_pred_levels)
        test_rmse_var_dif = rmse(test.values, test_pred_levels)
        train_RMSE_var_dif[size].append(train_rmse_var_dif)
        test_RMSE_var_dif[size].append(test_rmse_var_dif)

        
       
         # vecm = VECM(train, k_ar_diff=0, coint_rank=h, deterministic="ci")
         # vecm_fit = vecm.fit()
        
       
        # #Predicciones para train
        # train_pred = vecm_fit.fittedvalues
        # # Predicciones para test
        # test_pred=[]
        # input_data = train.values[-p:]
        
    
        # for i in range(len(test)):
        #     print(i)  
        #     #pred = vecm_fit.predict(steps=1)  usando el vec no se puede dar la opcion de input_Data
        #     pred = vecm_fit.forecast(input_data, steps=1)
        #     print(pred)
        #     test_pred.append(pred[0]) 
        #     input_data = test.iloc[i]
        #     print(input_data)

        # test_pred = np.array(test_pred)
              
        # train_rmse_vec = np.sqrt(mean_squared_error(train.values[p:], train_pred))
        # test_rmse_vec = np.sqrt(mean_squared_error(test.values, test_pred))
        # train_RMSE_vecm[size].append(train_rmse_vec)
        # test_RMSE_vecm[size].append(test_rmse_vec)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print("Termine de ajustar modelo VEC")
        end_time = time.time()
        execution_time = end_time - start_time
        tiempo_var_dif[size].append(end_time - start_time)
        
        # Ajustar modelo SVR
        start_time = time.time()
        fechas = pd.DataFrame(list(range(len(series))))
        total = pd.concat([fechas,series], axis=1).values
        dim=len(total)
    
        #Construcción de la base de datos
        data=Base(total)
        data= data.base
        #Creamos la base de datos
        dataset = create_dataset_rez(data,dim,h,k,p)
        X, Y = dataset[:, :(0 - h*2)], dataset[:, (0-h*2):]
        #Train y test
        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, shuffle=False)
        
        #normalizados
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        scaler_X.fit(X_train)
        scaler_y.fit(y_train)
    
       
        X_train_nor = scaler_X.transform(X_train)
        X_test_nor = scaler_X.transform(X_test)
        y_train_nor = scaler_y.transform(y_train)
        y_test_nor = scaler_y.transform(y_test)
    
        pipe = Pipeline([
            ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))
        ])
    
        hyperparameters = {
            #'MSVR__kernel': ['poly'],
            'MSVR__kernel': ['linear'],
            #'MSVR__degree': [2,3,4,5],
            'MSVR__degree': [1],
            'MSVR__gamma': [0.5,1],
            'MSVR__coef0': [0.1,0.5,1],
            'MSVR__C': [1,3,5,9,11,13,20],
            'MSVR__epsilon':[0.5,1,2,5], 
        }
        
        bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=15, scoring='neg_mean_squared_error', cv=5, verbose=0, error_score='raise')
        
        best_model = bm.fit(X_train_nor, y_train_nor)
        best_params = bm.best_params_
    
        msvr = MSVR(kernel=bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"),
                    epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"),
                    degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)
        
        msvr.fit(X_train_nor, y_train_nor)
      
     
        trainPred_svr_nor = msvr.predict(X_train_nor)
        testPred_svr_nor = msvr.predict(X_test_nor)
    
        trainPred_svr  = pd.DataFrame(scaler_y.inverse_transform(trainPred_svr_nor))
        testPred_svr  = pd.DataFrame(scaler_y.inverse_transform(testPred_svr_nor))
        train_rmse_svr = rmse(y_train, trainPred_svr)
        test_rmse_svr = rmse(y_test, testPred_svr)
        
        hiperparametros_svr[size].append(best_params)
        vectores_soporte[size].append(msvr.NSV)
        train_RMSE_svr[size].append(train_rmse_svr)
        test_RMSE_svr[size].append(test_rmse_svr)
        
        end_time = time.time()
        execution_time = end_time - start_time
        tiempo_msvr[size].append(end_time - start_time)
    
        
        print("SVR Best params:", best_params)
        print("VECM Train RMSE:", train_rmse_var_dif, "Test RMS ")
        
    #     df_train = pd.DataFrame({
    #     'Iteracion': no,  
    #     'y1_real': train.iloc[:, 0],  
    #     'y2_real': train.iloc[:, 1],  
    #     'y1_pred_var': train_pred_levels.iloc[:, 0],  
    #     'y2_pred_var': train_pred_levels.iloc[:, 1],  
    #     'y1_pred_svr': trainPred_svr.iloc[:, 0],  
    #     'y2_pred_svr': trainPred_svr.iloc[:, 1],  
    #     'Tipo': 'Train'
    # })

    #     # Crear DataFrame para las pruebas
    #     df_test = pd.DataFrame({
    #         'Iteracion': no,  
    #         'y1_real': test.iloc[:, 0],  
    #         'y2_real': test.iloc[:, 1],  
    #         'y1_pred_var': test_pred_levels.iloc[:, 0],  
    #         'y2_pred_var': test_pred_levels.iloc[:, 1],  
    #         'y1_pred_svr': testPred_svr.iloc[:, 0],  
    #         'y2_pred_svr': testPred_svr.iloc[:, 1],  
    #         'Tipo': 'Test'
    #     })
    #     a = pd.concat([df_train, df_test], ignore_index=True)  
    #     resultados.append(a)


# Concatenar resultados corrida 200
#df_final = pd.concat(resultados, ignore_index=True)
#df_final.to_excel('resultados_completos.xlsx', index=False)

#Corridas en general
filename = f'resultados_comparacion_VEC_Bivariado_Python_1relacion_finalfinal.csv'


with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Tamaño', 'Modelo', 'Hiperparametros', 'Vectores_soporte', 'Train RMSE', 'Test RMSE', 'Tiempo'])
    for size in t:
        writer.writerow([f'Tamaño: {size}', '', '', '', '', '', ''])
        for i in range(100):  
            writer.writerow(['SVR', hiperparametros_svr[size][i], vectores_soporte[size][i], train_RMSE_svr[size][i], test_RMSE_svr[size][i], tiempo_msvr[size][i]])
        for i in range(100):  # Asumiendo que tienes 100 iteraciones
         writer.writerow(['VAR DIF', 'N/A', 'N/A', train_RMSE_var_dif[size][i], test_RMSE_var_dif[size][i], tiempo_var_dif[size][i]])

    print(f'Resultados guardados en {filename}')





#--------------------------Modelo VEC Trivariado con una y dos relaciones de cointegración.------------------------------------------------------------#

def generate_phi(n, h):
    A = orth(np.random.randn(n, n))[:h, :]
    Phi = np.eye(n)
    if h == 1:
        Phi -= np.outer(A, A)
    elif h > 1:
        Phi -= A.T @ A
    return Phi

def cointegrated_vector(tt, n, h, C, Phi, burn=1000):
    """
    Simulate cointegrated time series data.
    
    Parameters:
    tt : int Time points to simulate.
    n : int Number of time series.
    h : int Number of cointegrating relations.
    C : array_like Covariance matrix for the innovations.
    burn : int, optional Number of initial points to discard (burn-in period)..
    """
    tot_t = tt + burn
    innov = multivariate_normal.rvs(mean=np.zeros(n),cov=C, size=tot_t) 
    X = np.zeros((1 + tot_t, n))
    print("Phi:\n", Phi) 
    for i in range(tot_t):
        X[i + 1, :] = Phi @ X[i, :] + innov[i, :]
    
    return X[burn + 1:, :]
#mantener phi quieto.
#modelos modificando las relaciones de cointegración.


t = [50, 200, 500, 1000, 5000]  # Longitud de la serie
k = 3  # Dimensión del vector Y
p = 1  # Número de retardos
c = 2  # Número de relaciones de cointegración
h = 1 #Número de pasos a predecir.
Phi = generate_phi(k, c)  

# Carpeta y archivo Excel
folder_path = r'C:\Users\huma1003\OneDrive - NIQ\DOCUMENTOS IMPORTANTES\Mis_Cosas\MarlijarTM\msvr-master\VEC Trivarido\Data'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
for size in t:
    excel_filename = os.path.join(folder_path, f'dataset_size_{size}.xlsx')
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        for no in range(100):
            print(f"-------------------------Tamaño {size}-------------------------")
            print(f"-------------------------Iteration {no}--------------------------")
            series = np.zeros((k, size))
            series = cointegrated_vector(size, k, no, np.eye(k), Phi)
            series = pd.DataFrame(series, columns=['Y1', 'Y2', 'Y3'])
            
            sheet_name = f"Iter_{no}"
            
            series.to_excel(writer, sheet_name=sheet_name, index=False)




# Almacenar resultados
hiperparametros_svr = {size: [] for size in t}
vectores_soporte = {size: [] for size in t}
train_RMSE_svr = {size: [] for size in t}
test_RMSE_svr = {size: [] for size in t}
train_RMSE_var_dif = {size: [] for size in t}
test_RMSE_var_dif= {size: [] for size in t}
tiempo_var_dif = {size: [] for size in t}
tiempo_msvr = {size: [] for size in t}

data_folder = "C:/Users/huma1003/OneDrive - NIQ/DOCUMENTOS IMPORTANTES/Mis_Cosas/MarlijarTM/msvr-master/VEC Trivarido/Data"

def load_data(size, iteration):
    file_path = f"{data_folder}/dataset_size_{size}.xlsx"
    sheet_name = f"Iter_{iteration}"
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

#########


t=[50,200,500,1000,5000] # Longitud de la serie
k =3# dimensión del vector Y
p = 1 # Número de retardos
c=2 #Núnero de relaciones de cointegración
h=1
# Almacenar resultados
hiperparametros_svr = {size: [] for size in t}
vectores_soporte = {size: [] for size in t}
train_RMSE_svr = {size: [] for size in t}
test_RMSE_svr = {size: [] for size in t}
train_RMSE_var_dif = {size: [] for size in t}
test_RMSE_var_dif = {size: [] for size in t}
tiempo_var_dif = {size: [] for size in t}
tiempo_msvr = {size: [] for size in t}
Phi = generate_phi(k, c)

# Generación de la serie y ajuste de modelos
for size in t:
      a=size
      for no in range(100):
        print(f"-------------------------Tamaño {a}-------------------------")
        print(f"-------------------------Iteration {no}--------------------------")
        #series = np.zeros((k, size))
        #series = cointegrated_vector(size, k, no, np.eye(k),Phi)
        #series = pd.DataFrame(series, columns=['Y1', 'Y2','Y3'])
        series = load_data(size, no)
      
        #Partición en train y test
        train_size = int(len(series) * 0.7)
        train, test = series.iloc[:train_size], series.iloc[train_size:]
        test = test.reset_index(drop=True) 
         # Ajustar modelo VAR DIFERENCIADO
        start_time = time.time()
         
        # Ver la serie diferenciada
        series_dif = series.diff().dropna()
        train_size_dif = int(len(series_dif) * 0.7)
        train_dif, test_dif = series_dif.iloc[:train_size_dif], series_dif.iloc[train_size_dif:]
        test_dif = test_dif.reset_index(drop=True) 
        model_var_dif = VAR(train_dif)
        results_var_dif = model_var_dif.fit(maxlags=p)
        lag_order = results_var_dif.k_ar
        
        modelo_var_train_dif = []
        modelo_var_test_dif = []
            
        #Predicciones para train
        train_pred_dif = results_var_dif.fittedvalues
        # Predicciones para test
        test_pred_dif=[]
        input_data = train_dif.values[-p:]

        for i in range(len(test_dif)):
            pred = results_var_dif.forecast(y=input_data, steps=h)
            test_pred_dif.append(pred[0])
            input_data = np.vstack([input_data[1:], test_dif.values[i:i+1]])

        test_pred_dif = np.array(test_pred_dif)
              
         
        train_rmse_var_dif = np.sqrt(mean_squared_error(train_dif.values[p:], train_pred_dif))
        test_rmse_var_dif = np.sqrt(mean_squared_error(test_dif.values, test_pred_dif))

        
         
        train_RMSE_var_dif[size].append(train_rmse_var_dif)
        test_RMSE_var_dif[size].append(test_rmse_var_dif)

        
    #     #Partición en train y test
    #     train_size = int(len(series) * 0.7)
    #     train, test = series.iloc[:train_size], series.iloc[train_size:]
          
    #     # Ajustar modelo VEC
    #     start_time = time.time()
        
    #     #Sacar las diferencias  
    #     vecm = VECM(train, k_ar_diff=0, coint_rank=h, deterministic="ci")
    #     vecm_fit = vecm.fit()
        
       
    #     #Predicciones para train
    #     train_pred = vecm_fit.fittedvalues
    #     # Predicciones para test
    #     test_pred=[]
    #     input_data = train.values[p:]
    
    # #revisar imputar el dato real#    
    
    # for i in range(len(test)):
    #         pred = vecm_fit.predict(steps=1)  
    #         test_pred.append(pred[0])
    #         input_data = np.vstack([input_data[1:], pred[0]])
    
    #     test_pred = np.array(test_pred)
              
    #     train_rmse_vec = np.sqrt(mean_squared_error(train.values[p:], train_pred))
    #     test_rmse_vec = np.sqrt(mean_squared_error(test.values, test_pred))
    #     train_RMSE_vecm[size].append(train_rmse_vec)
    #     test_RMSE_vecm[size].append(test_rmse_vec)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print("Termine de ajustar modelo VEC")
        end_time = time.time()
        execution_time = end_time - start_time
        tiempo_var_dif[size].append(end_time - start_time)
        
        # Ajustar modelo SVR
        start_time = time.time()
        fechas = pd.DataFrame(list(range(len(series))))
        total = pd.concat([fechas,series], axis=1).values
        dim=len(total)
    
        #Construcción de la base de datos
        data=Base(total)
        data= data.base
        #Creamos la base de datos
        dataset = create_dataset_rez(data,dim,h,k,p)
        X, Y = dataset[:, :(0 - h*k)], dataset[:, (0-h*k):]
        #Train y test
        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, shuffle=False)
        
        #normalizados
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        scaler_X.fit(X_train)
        scaler_y.fit(y_train)
    
       
        X_train_nor = scaler_X.transform(X_train)
        X_test_nor = scaler_X.transform(X_test)
        y_train_nor = scaler_y.transform(y_train)
        y_test_nor = scaler_y.transform(y_test)
    
        pipe = Pipeline([
            ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))
        ])
    
        hyperparameters = {
            #'MSVR__kernel': ['poly'],
            'MSVR__kernel': ['poly','rbf', 'linear'],
            'MSVR__degree': [2,3,4,5],
            #'MSVR__degree': [1],
            'MSVR__gamma': [0.5,1],
            'MSVR__coef0': [0.1,0.5,1],
            'MSVR__C': [5,9,11,13],
            'MSVR__epsilon':[1,2], 
        }
        
        bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=15, scoring='neg_mean_squared_error', cv=5, verbose=0, error_score='raise')
       
        best_model = bm.fit(X_train_nor, y_train_nor)
        best_params = bm.best_params_
    
        msvr = MSVR(kernel=bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"),
                    epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"),
                    degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)
        
        msvr.fit(X_train_nor, y_train_nor)
      
     
        trainPred_svr_nor = msvr.predict(X_train_nor)
        testPred_svr_nor = msvr.predict(X_test_nor)
    
        trainPred_svr  = scaler_y.inverse_transform(trainPred_svr_nor)
        testPred_svr  = scaler_y.inverse_transform(testPred_svr_nor)
        train_rmse_svr = rmse(y_train, trainPred_svr)
        test_rmse_svr = rmse(y_test, testPred_svr)
        
        hiperparametros_svr[size].append(best_params)
        vectores_soporte[size].append(msvr.NSV)
        train_RMSE_svr[size].append(train_rmse_svr)
        test_RMSE_svr[size].append(test_rmse_svr)
        
        end_time = time.time()
        execution_time = end_time - start_time
        tiempo_msvr[size].append(end_time - start_time)
    
        
        print("SVR Best params:", best_params)
        # print("VECM Train RMSE:", train_rmse_vec, "Test RMSE:", test_rmse_vec)
        # print("SVR Train RMSE:", train_rmse_svr, "Test RMSE:", test_rmse_svr)


filename = f'resultados_comparacion_var3_co2__finallllllllllllll.csv'


with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Tamaño', 'Modelo', 'Hiperparametros', 'Vectores_soporte', 'Train RMSE', 'Test RMSE', 'Tiempo'])
    for size in t:
        writer.writerow([f'Tamaño: {size}', '', '', '', '', '', ''])
        for i in range(100):  
            writer.writerow(['SVR', hiperparametros_svr[size][i], vectores_soporte[size][i], train_RMSE_svr[size][i], test_RMSE_svr[size][i], tiempo_msvr[size][i]])
        for i in range(100):  # Asumiendo que tienes 100 iteraciones
            writer.writerow(['VAR DIF', 'N/A', 'N/A', train_RMSE_var_dif[size][i], test_RMSE_var_dif[size][i], tiempo_var_dif[size][i]])

    print(f'Resultados guardados en {filename}')


#--------------------------Modelo VEC Cincovariado con una,2,3,4 relaciones de cointegración.------------------------------------------------------------#



from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import select_coint_rank

t=[50,200,500,1000,5000] # Longitud de la serie
k =5# dimensión del vector Y
p = 1 # Número de retardos
c=4 #Núnero de relaciones de cointegración
h=1 #pasos para predecir
# Almacenar resultados
hiperparametros_svr = {size: [] for size in t}
vectores_soporte = {size: [] for size in t}
train_RMSE_svr = {size: [] for size in t}
test_RMSE_svr = {size: [] for size in t}
train_RMSE_vecm = {size: [] for size in t}
test_RMSE_vecm = {size: [] for size in t}
tiempo_vecm = {size: [] for size in t}
tiempo_msvr = {size: [] for size in t}
Phi = generate_phi(k, c)

# Generación de la serie y ajuste de modelos
for size in t:
      a=size
      for no in range(100):
        print(f"-------------------------Tamaño {a}-------------------------")
        print(f"-------------------------Iteration {no}--------------------------")
        series = np.zeros((k, size))
        series = cointegrated_vector(size, k, no, np.eye(k),Phi)
        series = pd.DataFrame(series, columns=['Y1', 'Y2','Y3','Y4','Y5'])
        
        #Partición en train y test
        train_size = int(len(series) * 0.7)
        train, test = series.iloc[:train_size], series.iloc[train_size:]
          
        # Ajustar modelo VEC
        start_time = time.time()
        
        #Sacar las diferencias  
        vecm = VECM(train, k_ar_diff=0, coint_rank=h, deterministic="ci")
        vecm_fit = vecm.fit()
        
       
        #Predicciones para train
        train_pred = vecm_fit.fittedvalues
        # Predicciones para test
        test_pred=[]
        input_data = train.values[p:]
    
        for i in range(len(test)):
            pred = vecm_fit.predict(steps=1)  
            test_pred.append(pred[0])
            input_data = np.vstack([input_data[1:], pred[0]])
    
        test_pred = np.array(test_pred)
              
        train_rmse_vec = np.sqrt(mean_squared_error(train.values[p:], train_pred))
        test_rmse_vec = np.sqrt(mean_squared_error(test.values, test_pred))
        train_RMSE_vecm[size].append(train_rmse_vec)
        test_RMSE_vecm[size].append(test_rmse_vec)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print("Termine de ajustar modelo VEC")
        end_time = time.time()
        execution_time = end_time - start_time
        tiempo_vecm[size].append(end_time - start_time)
        
        # Ajustar modelo SVR
        start_time = time.time()
        fechas = pd.DataFrame(list(range(len(series))))
        total = pd.concat([fechas,series], axis=1).values
        dim=len(total)
    
        #Construcción de la base de datos
        data=Base(total)
        data= data.base
        #Creamos la base de datos
        dataset = create_dataset_rez(data,dim,h,k,p)
        X, Y = dataset[:, :(0 - h*k)], dataset[:, (0-h*k):]
        #Train y test
        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, shuffle=False)
        
        #normalizados
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        scaler_X.fit(X_train)
        scaler_y.fit(y_train)
    
       
        X_train_nor = scaler_X.transform(X_train)
        X_test_nor = scaler_X.transform(X_test)
        y_train_nor = scaler_y.transform(y_train)
        y_test_nor = scaler_y.transform(y_test)
    
        pipe = Pipeline([
            ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))
        ])
    
        hyperparameters = {
            #'MSVR__kernel': ['poly'],
            'MSVR__kernel': ['poly','rbf'],
            'MSVR__degree': [2,3,4,5],
            #'MSVR__degree': [1],
            'MSVR__gamma': [0.5,1],
            'MSVR__coef0': [0.1,0.5,1],
            'MSVR__C': [5,9,11,13],
            'MSVR__epsilon':[1,2], 
        }
        
        bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=15, scoring='neg_mean_squared_error', cv=5, verbose=0, error_score='raise')
       
        best_model = bm.fit(X_train_nor, y_train_nor)
        best_params = bm.best_params_
    
        msvr = MSVR(kernel=bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"),
                    epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"),
                    degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)
        
        msvr.fit(X_train_nor, y_train_nor)
      
     
        trainPred_svr_nor = msvr.predict(X_train_nor)
        testPred_svr_nor = msvr.predict(X_test_nor)
    
        trainPred_svr  = scaler_y.inverse_transform(trainPred_svr_nor)
        testPred_svr  = scaler_y.inverse_transform(testPred_svr_nor)
        train_rmse_svr = rmse(y_train, trainPred_svr)
        test_rmse_svr = rmse(y_test, testPred_svr)
        
        hiperparametros_svr[size].append(best_params)
        vectores_soporte[size].append(msvr.NSV)
        train_RMSE_svr[size].append(train_rmse_svr)
        test_RMSE_svr[size].append(test_rmse_svr)
        
        end_time = time.time()
        execution_time = end_time - start_time
        tiempo_msvr[size].append(end_time - start_time)
    
        
        print("SVR Best params:", best_params)
        print("VECM Train RMSE:", train_rmse_vec, "Test RMSE:", test_rmse_vec)
        print("SVR Train RMSE:", train_rmse_svr, "Test RMSE:", test_rmse_svr)


filename = f'resultados_comparacion_var5_co4__sinlinear.csv'


with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Tamaño', 'Modelo', 'Hiperparametros', 'Vectores_soporte', 'Train RMSE', 'Test RMSE', 'Tiempo'])
    for size in t:
        writer.writerow([f'Tamaño: {size}', '', '', '', '', '', ''])
        for i in range(100):  
            writer.writerow(['SVR', hiperparametros_svr[size][i], vectores_soporte[size][i], train_RMSE_svr[size][i], test_RMSE_svr[size][i], tiempo_msvr[size][i]])
        for i in range(100):  # Asumiendo que tienes 100 iteraciones
            writer.writerow(['VEC', 'N/A', 'N/A', train_RMSE_vecm[size][i], test_RMSE_vecm[size][i], tiempo_vecm[size][i]])

    print(f'Resultados guardados en {filename}')


#------------------------------------------------------------------------------------------------------######
#--------------------------VAR TRIVARIADO-Estacionario------------------------------------------------------------#
t=500 # Longitud de la serie
k =3 # dimensión del vector Y
p = 1 # Número de retardos
h=1
col=3
# Almacenar resultados
hiperparametros_svr = []
vectores_soporte=[]
train_RMSE_svr = []
test_RMSE_svr = []
train_RMSE_var = []
test_RMSE_var = []
tiempo_var=[]
tiempo_msvr=[]
#Phi = generate_phi(k, h)
# Generación de la serie y ajuste de modelos
for no in range(100):
    
    print(f"-------------------------Iteration {no}--------------------------")
    # Generar series
    A = np.array([[0.5, 0,0],
                   [0.1, 0.1,0.3],
                   [0, 0.2,0.3]
                   ])

    initial = np.random.normal(size=(k,))
    serie = np.zeros((k, t))
    serie[:, :1] = initial[:, np.newaxis]

    for i in range(1, t):
       retardo = serie[:, i-1:i]
       serie[:, i] = np.dot(A, retardo.flatten()) +np.random.normal(loc=0.0, scale=9, size=k)


    series = pd.DataFrame(serie.T, columns=['Y1', 'Y2','Y3'])
    
    #Partición en train y test
    train_size = int(len(series) * 0.7)
    train, test = series.iloc[:train_size], series.iloc[train_size:]
    
    #PACF
    pacf_var1 = pacf(train['Y1'], nlags=16)
    pacf_var2 = pacf(train['Y2'], nlags=16)
    banda= 1.96 / np.sqrt(t)  
   
    rezago_elegido_1 = rezago_sig(pacf_var1, banda)
    rezago_elegido_2 = rezago_sig(pacf_var2, banda)
    enumerated_list = list(enumerate(pacf_var1))
    reversed_enumerated_list = list(reversed(enumerated_list))
    filtered_indices = [i for i, x in reversed_enumerated_list if abs(x) > banda]

    
    #rez= int(min(rezago_elegido_1, rezago_elegido_2))
    rez=1
    #print(f"----------------Rezago {rez}--------------")
        
    
    
    #fig, axes = plt.subplots(2, 1, figsize=(10, 10))
# # Plotear la PACF de Y1
# axes[0].stem(range(len(pacf_var1)), pacf_var1, basefmt=" ", use_line_collection=True)
# axes[0].axhline(y=banda, color='r', linestyle='--', label=f'Banda superior ({banda:.2f})')
# axes[0].axhline(y=-banda, color='r', linestyle='--', label=f'Banda inferior ({-banda:.2f})')
# axes[0].set_xlabel('Rezago')
# axes[0].set_ylabel('PACF Y1')
# axes[0].set_title('PACF de Y1 con Bandas de Confianza')
# axes[0].legend()

# # Plotear la PACF de Y2
# axes[1].stem(range(len(pacf_var2)), pacf_var2, basefmt=" ", use_line_collection=True)
# axes[1].axhline(y=banda, color='r', linestyle='--', label=f'Banda superior ({banda:.2f})')
# axes[1].axhline(y=-banda, color='r', linestyle='--', label=f'Banda inferior ({-banda:.2f})')
# axes[1].set_xlabel('Rezago')
# axes[1].set_ylabel('PACF Y2')
# axes[1].set_title('PACF de Y2 con Bandas de Confianza')
# axes[1].legend()

# # Mostrar los gráficos
# plt.tight_layout()
# plt.show()
    # 
    
    
    # Ajustar modelo VAR
    start_time = time.time()
    model_var = VAR(train)
    results_var = model_var.fit(maxlags=1)
    lag_order = results_var.k_ar
    
    modelo_var_train = []
    modelo_var_test = []
        
    #Predicciones para train
    train_pred = results_var.fittedvalues
    # Predicciones para test
    test_pred=[]
    input_data = train.values[-rez:]

    for i in range(len(test)):
        pred = results_var.forecast(y=input_data, steps=h)
        test_pred.append(pred[0])
        input_data = np.vstack([input_data[1:], test.values[i:i+1]])

    test_pred = np.array(test_pred)
          
     
    train_rmse_var = np.sqrt(mean_squared_error(train.values[rez:], train_pred))
    test_rmse_var = np.sqrt(mean_squared_error(test.values, test_pred))

    
    train_RMSE_var.append(train_rmse_var)
    test_RMSE_var.append(test_rmse_var)
    print("Termine de ajustar modelo VAR")
    end_time = time.time()
    execution_time = end_time - start_time
    tiempo_var.append(execution_time)
    
    # Ajustar modelo SVR
    start_time = time.time()
    fechas = pd.DataFrame(list(range(len(series))))
    total = pd.concat([fechas,series], axis=1).values
    dim=len(total)

    #Construcción de la base de datos
    data=Base(total)
    data= data.base
    #Creamos la base de datos
    dataset = create_dataset_rez(data,dim,h,col,rez)
    X, Y = dataset[:, :(0 - h*2)], dataset[:, (0-h*2):]
    #Train y test
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, shuffle=False)
    
    #normalizados
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    scaler_X.fit(X_train)
    scaler_y.fit(y_train)

   
    X_train_nor = scaler_X.transform(X_train)
    X_test_nor = scaler_X.transform(X_test)
    y_train_nor = scaler_y.transform(y_train)
    y_test_nor = scaler_y.transform(y_test)

    pipe = Pipeline([
        ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))
    ])

    hyperparameters = {
        #'MSVR__kernel': ['poly'],
        'MSVR__kernel': ['poly','rbf','linear'],
        'MSVR__degree': [2,5],
        #'MSVR__degree': [1],
        'MSVR__gamma': [0.5,1],
        'MSVR__coef0': [0.1,0.5,1],
        'MSVR__C': [5,9,11,13],
        'MSVR__epsilon':[1,2], 
    }
    
    bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=15, scoring='neg_mean_squared_error', cv=5, verbose=0, error_score='raise')
   
    best_model = bm.fit(X_train_nor, y_train_nor)
    best_params = bm.best_params_

    msvr = MSVR(kernel=bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"),
                epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"),
                degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)
    
    msvr.fit(X_train_nor, y_train_nor)
  
 
    trainPred_svr_nor = msvr.predict(X_train_nor)
    testPred_svr_nor = msvr.predict(X_test_nor)

    trainPred_svr  = scaler_y.inverse_transform(trainPred_svr_nor)
    testPred_svr  = scaler_y.inverse_transform(testPred_svr_nor)
    train_rmse_svr = rmse(y_train, trainPred_svr)
    test_rmse_svr = rmse(y_test, testPred_svr)
    
    hiperparametros_svr.append(best_params)
    vectores_soporte.append(msvr.NSV)
    print("SVR Best params:", msvr.NSV/t)
    train_RMSE_svr.append(train_rmse_svr)
    test_RMSE_svr.append(test_rmse_svr)
    
    end_time = time.time()
    execution_time = end_time - start_time
    tiempo_msvr.append(execution_time)

    
    print("SVR Best params:", best_params)
    print("VAR Train RMSE:", train_rmse_var, "Test RMSE:", test_rmse_var)
    print("SVR Train RMSE:", train_rmse_svr, "Test RMSE:", test_rmse_svr)



# Guardar resultados en un archivo CSV
with open('resultados_comparacion_varest_svr_tri_500_9.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Modelo', 'Hiperparametros', 'Vectores_soporte', 'Train RMSE', 'Test RMSE', 'Tiempo'])
    for i in range(100):
        writer.writerow(['SVR', hiperparametros_svr[i],vectores_soporte[i], train_RMSE_svr[i], test_RMSE_svr[i],tiempo_msvr[i]])
        writer.writerow(['VAR', 'N/A','N/A', train_RMSE_var[i], test_RMSE_var[i],tiempo_var[i]])

#------------------------------------------------------------------------------------------------------######
#--------------------------VAR TRIVARIADO-Estacionario rezago 2 ------------------------------------------------------------#
t=50 # Longitud de la serie
k =3 # dimensión del vector Y
p = 2 # Número de retardos
h=1
col=3
# Almacenar resultados
hiperparametros_svr = []
vectores_soporte=[]
train_RMSE_svr = []
test_RMSE_svr = []
train_RMSE_var = []
test_RMSE_var = []
tiempo_var=[]
tiempo_msvr=[]
# Generación de la serie y ajuste de modelos
for no in range(100):
    
    print(f"-------------------------Iteration {no}--------------------------")
    # Generar series
    A1 = np.array([[-0.23454989, -1.41986710,  0.1040019], #Rezago 1
                   [0.51455053 ,-0.79843409 ,-0.4671284],
                   [0.03754303, -0.06477289 ,-0.1226731]])
    A2 = np.array([[0.6685725, -0.5928383 ,-0.594248869], #Rezago 2
                   [0.3044929, -0.3008990, -0.009444197],
                   [0.3110371 , 0.5085483 , 0.170686206]])
    
    initial = np.random.normal(size=(k,p))
    serie = np.zeros((k, t))
    serie[:, :p] = initial

    for i in range(2, t):
        serie[:, i] = (np.dot(A1, serie[:, i-1]) + 
                   np.dot(A2, serie[:, i-2]) + 
                   np.random.normal(loc=0.0, scale=0.5, size=k)) 


    series = pd.DataFrame(serie.T, columns=['Y1', 'Y2','Y3'])
        
    #Partición en train y test
    train_size = int(len(series) * 0.7)
    train, test = series.iloc[:train_size], series.iloc[train_size:]
    
    #PACF
    pacf_var1 = pacf(train['Y1'], nlags=16)
    pacf_var2 = pacf(train['Y2'], nlags=16)
    banda= 1.96 / np.sqrt(t)  
   
    rezago_elegido_1 = rezago_sig(pacf_var1, banda)
    rezago_elegido_2 = rezago_sig(pacf_var2, banda)
    enumerated_list = list(enumerate(pacf_var1))
    reversed_enumerated_list = list(reversed(enumerated_list))
    filtered_indices = [i for i, x in reversed_enumerated_list if abs(x) > banda]

    
    #rez= int(min(rezago_elegido_1, rezago_elegido_2))
    rez=1
    #print(f"----------------Rezago {rez}--------------")
        
    
    
    #fig, axes = plt.subplots(2, 1, figsize=(10, 10))
# # Plotear la PACF de Y1
# axes[0].stem(range(len(pacf_var1)), pacf_var1, basefmt=" ", use_line_collection=True)
# axes[0].axhline(y=banda, color='r', linestyle='--', label=f'Banda superior ({banda:.2f})')
# axes[0].axhline(y=-banda, color='r', linestyle='--', label=f'Banda inferior ({-banda:.2f})')
# axes[0].set_xlabel('Rezago')
# axes[0].set_ylabel('PACF Y1')
# axes[0].set_title('PACF de Y1 con Bandas de Confianza')
# axes[0].legend()

# # Plotear la PACF de Y2
# axes[1].stem(range(len(pacf_var2)), pacf_var2, basefmt=" ", use_line_collection=True)
# axes[1].axhline(y=banda, color='r', linestyle='--', label=f'Banda superior ({banda:.2f})')
# axes[1].axhline(y=-banda, color='r', linestyle='--', label=f'Banda inferior ({-banda:.2f})')
# axes[1].set_xlabel('Rezago')
# axes[1].set_ylabel('PACF Y2')
# axes[1].set_title('PACF de Y2 con Bandas de Confianza')
# axes[1].legend()

# # Mostrar los gráficos
# plt.tight_layout()
# plt.show()
    # 
    
    
    # Ajustar modelo VAR
    start_time = time.time()
    model_var = VAR(train)
    results_var = model_var.fit(maxlags=1)
    lag_order = results_var.k_ar
    
    modelo_var_train = []
    modelo_var_test = []
        
    #Predicciones para train
    train_pred = results_var.fittedvalues
    # Predicciones para test
    test_pred=[]
    input_data = train.values[-rez:]

    for i in range(len(test)):
        pred = results_var.forecast(y=input_data, steps=h)
        test_pred.append(pred[0])
        input_data = np.vstack([input_data[1:], test.values[i:i+1]])

    test_pred = np.array(test_pred)
          
     
    train_rmse_var = np.sqrt(mean_squared_error(train.values[rez:], train_pred))
    test_rmse_var = np.sqrt(mean_squared_error(test.values, test_pred))

    
    train_RMSE_var.append(train_rmse_var)
    test_RMSE_var.append(test_rmse_var)
    print("Termine de ajustar modelo VAR")
    end_time = time.time()
    execution_time = end_time - start_time
    tiempo_var.append(execution_time)
    
    # Ajustar modelo SVR
    start_time = time.time()
    fechas = pd.DataFrame(list(range(len(series))))
    total = pd.concat([fechas,series], axis=1).values
    dim=len(total)

    #Construcción de la base de datos
    data=Base(total)
    data= data.base
    #Creamos la base de datos
    dataset = create_dataset_rez(data,dim,h,col,rez)
    X, Y = dataset[:, :(0 - h*2)], dataset[:, (0-h*2):]
    #Train y test
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, shuffle=False)
    
    #normalizados
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    scaler_X.fit(X_train)
    scaler_y.fit(y_train)

   
    X_train_nor = scaler_X.transform(X_train)
    X_test_nor = scaler_X.transform(X_test)
    y_train_nor = scaler_y.transform(y_train)
    y_test_nor = scaler_y.transform(y_test)

    pipe = Pipeline([
        ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))
    ])

    hyperparameters = {
        #'MSVR__kernel': ['poly'],
        'MSVR__kernel': ['poly','rbf','linear'],
        'MSVR__degree': [2,5],
        #'MSVR__degree': [1],
        'MSVR__gamma': [0.5,1],
        'MSVR__coef0': [0.1,0.5,1],
        'MSVR__C': [5,9,11,13],
        'MSVR__epsilon':[1,2], 
    }
    
    bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=15, scoring='neg_mean_squared_error', cv=5, verbose=0, error_score='raise')
   
    best_model = bm.fit(X_train_nor, y_train_nor)
    best_params = bm.best_params_

    msvr = MSVR(kernel=bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"),
                epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"),
                degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)
    
    msvr.fit(X_train_nor, y_train_nor)
  
 
    trainPred_svr_nor = msvr.predict(X_train_nor)
    testPred_svr_nor = msvr.predict(X_test_nor)

    trainPred_svr  = scaler_y.inverse_transform(trainPred_svr_nor)
    testPred_svr  = scaler_y.inverse_transform(testPred_svr_nor)
    train_rmse_svr = rmse(y_train, trainPred_svr)
    test_rmse_svr = rmse(y_test, testPred_svr)
    
    hiperparametros_svr.append(best_params)
    vectores_soporte.append(msvr.NSV)
    print("SVR Best params:", msvr.NSV/t)
    train_RMSE_svr.append(train_rmse_svr)
    test_RMSE_svr.append(test_rmse_svr)
    
    end_time = time.time()
    execution_time = end_time - start_time
    tiempo_msvr.append(execution_time)

    
    print("SVR Best params:", best_params)
    print("VAR Train RMSE:", train_rmse_var, "Test RMSE:", test_rmse_var)
    print("SVR Train RMSE:", train_rmse_svr, "Test RMSE:", test_rmse_svr)



# Guardar resultados en un archivo CSV
with open('resultados_comparacion_varest_svr_tri_rezago_2_50_0.5.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Modelo', 'Hiperparametros', 'Vectores_soporte', 'Train RMSE', 'Test RMSE', 'Tiempo'])
    for i in range(100):
        writer.writerow(['SVR', hiperparametros_svr[i],vectores_soporte[i], train_RMSE_svr[i], test_RMSE_svr[i],tiempo_msvr[i]])
        writer.writerow(['VAR', 'N/A','N/A', train_RMSE_var[i], test_RMSE_var[i],tiempo_var[i]])



#--------------------------VAR Puro integrado------------------------------------------------------------#
t=5000 # Longitud de la serie
k =2 # dimensión del vector Y
p = 1 # Número de retardos
h=1
col=2
# Almacenar resultados
hiperparametros_svr = []
vectores_soporte=[]
train_RMSE_svr = []
test_RMSE_svr = []
train_RMSE_var = []
test_RMSE_var = []
tiempo_var=[]
tiempo_msvr=[]
# Generación de la serie y ajuste de modelos
for no in range(100):
    
    print(f"-------------------------Iteration {no}--------------------------")
    # Generar series
    A = np.array([[1, 0],
                   [0, 1]])

    initial = np.random.normal(size=(2,))
    serie = np.zeros((2, t))
    serie[:, :1] = initial[:, np.newaxis]

    for i in range(1, t):
       retardo = serie[:, i-1:i]
       serie[:, i] = np.dot(A, retardo.flatten()) +np.random.normal(loc=0.0, scale=1.0, size=2)


    series = pd.DataFrame(serie.T, columns=['Y1', 'Y2'])
    
    #Partición en train y test
    train_size = int(len(series) * 0.7)
    train, test = series.iloc[:train_size], series.iloc[train_size:]
    
    #PACF
    pacf_var1 = pacf(train['Y1'], nlags=16)
    pacf_var2 = pacf(train['Y2'], nlags=16)
    banda= 1.96 / np.sqrt(t)  
   
    # #rezago_elegido_1 = rezago_sig(pacf_var1, banda)
    # #rezago_elegido_2 = rezago_sig(pacf_var2, banda)
    # enumerated_list = list(enumerate(pacf_var1))
    # reversed_enumerated_list = list(reversed(enumerated_list))
    # filtered_indices = [i for i, x in reversed_enumerated_list if abs(x) > banda]

    
    # rez= int(min(rezago_elegido_1, rezago_elegido_2))
    rez=1
    print(f"----------------Rezago {rez}--------------")
        
    
#     fig, axes = plt.subplots(2, 1, figsize=(10, 10))
# # Plotear la PACF de Y1
# axes[0].stem(range(len(pacf_var1)), pacf_var1, basefmt=" ", use_line_collection=True)
# axes[0].axhline(y=banda, color='r', linestyle='--', label=f'Banda superior ({banda:.2f})')
# axes[0].axhline(y=-banda, color='r', linestyle='--', label=f'Banda inferior ({-banda:.2f})')
# axes[0].set_xlabel('Rezago')
# axes[0].set_ylabel('PACF Y1')
# axes[0].set_title('PACF de Y1 con Bandas de Confianza')
# axes[0].legend()

# # Plotear la PACF de Y2
# axes[1].stem(range(len(pacf_var2)), pacf_var2, basefmt=" ", use_line_collection=True)
# axes[1].axhline(y=banda, color='r', linestyle='--', label=f'Banda superior ({banda:.2f})')
# axes[1].axhline(y=-banda, color='r', linestyle='--', label=f'Banda inferior ({-banda:.2f})')
# axes[1].set_xlabel('Rezago')
# axes[1].set_ylabel('PACF Y2')
# axes[1].set_title('PACF de Y2 con Bandas de Confianza')
# axes[1].legend()

# # Mostrar los gráficos
# plt.tight_layout()
# plt.show()
    
    
    
    # Ajustar modelo VAR
    start_time = time.time()
    model_var = VAR(train)
    results_var = model_var.fit(maxlags=1, ic='aic')
    lag_order = results_var.k_ar
    
    modelo_var_train = []
    modelo_var_test = []
        
    #Predicciones para train
    train_pred = results_var.fittedvalues
    # Predicciones para test
    test_pred=[]
    input_data = train.values[-rez:]

    for i in range(len(test)):
        pred = results_var.forecast(y=input_data, steps=h)
        test_pred.append(pred[0])
        input_data = np.vstack([input_data[1:], test.values[i:i+1]])

    test_pred = np.array(test_pred)
          
     
    train_rmse_var = np.sqrt(mean_squared_error(train.values[rez:], train_pred))
    test_rmse_var = np.sqrt(mean_squared_error(test.values, test_pred))

    
    train_RMSE_var.append(train_rmse_var)
    test_RMSE_var.append(test_rmse_var)
    print("Termine de ajustar modelo VAR")
    end_time = time.time()
    execution_time = end_time - start_time
    tiempo_var.append(execution_time)
    
    # Ajustar modelo SVR
    start_time = time.time()
    fechas = pd.DataFrame(list(range(len(series))))
    total = pd.concat([fechas,series], axis=1).values
    dim=len(total)

    #Construcción de la base de datos
    data=Base(total)
    data= data.base
    #Creamos la base de datos
    dataset = create_dataset_rez(data,dim,h,col,rez)
    X, Y = dataset[:, :(0 - h*2)], dataset[:, (0-h*2):]
    #Train y test
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, shuffle=False)
    
    #normalizados
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    scaler_X.fit(X_train)
    scaler_y.fit(y_train)

   
    X_train_nor = scaler_X.transform(X_train)
    X_test_nor = scaler_X.transform(X_test)
    y_train_nor = scaler_y.transform(y_train)
    y_test_nor = scaler_y.transform(y_test)

    pipe = Pipeline([
        ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))
    ])

    hyperparameters = {
        #'MSVR__kernel': ['poly'],
        'MSVR__kernel': ['poly','rbf','linear'],
        'MSVR__degree': [2,5],
        #'MSVR__degree': [1],
        'MSVR__gamma': [0.5,1],
        'MSVR__coef0': [0.1,0.5,1],
        'MSVR__C': [5,9,11,13],
        'MSVR__epsilon':[1,2], 
    }
    
    bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=15, scoring='neg_mean_squared_error', cv=5, verbose=0, error_score='raise')
   
    best_model = bm.fit(X_train_nor, y_train_nor)
    best_params = bm.best_params_

    msvr = MSVR(kernel=bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"),
                epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"),
                degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)
    
    msvr.fit(X_train_nor, y_train_nor)
  
 
    trainPred_svr_nor = msvr.predict(X_train_nor)
    testPred_svr_nor = msvr.predict(X_test_nor)

    trainPred_svr  = scaler_y.inverse_transform(trainPred_svr_nor)
    testPred_svr  = scaler_y.inverse_transform(testPred_svr_nor)
    train_rmse_svr = rmse(y_train, trainPred_svr)
    test_rmse_svr = rmse(y_test, testPred_svr)
    
    hiperparametros_svr.append(best_params)
    vectores_soporte.append(msvr.NSV)
    print("SVR Best params:", msvr.NSV/t)
    train_RMSE_svr.append(train_rmse_svr)
    test_RMSE_svr.append(test_rmse_svr)
    
    end_time = time.time()
    execution_time = end_time - start_time
    tiempo_msvr.append(execution_time)

    
    print("SVR Best params:", best_params)
    print("VAR Train RMSE:", train_rmse_var, "Test RMSE:", test_rmse_var)
    print("SVR Train RMSE:", train_rmse_svr, "Test RMSE:", test_rmse_svr)



# Guardar resultados en un archivo CSV
#CORRER PRQUE NO LO HE CORRIDO
with open('resultados_comparacion_varintegradopuro5000.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Modelo', 'Hiperparametros', 'Vectores_soporte', 'Train RMSE', 'Test RMSE', 'Tiempo'])
    for i in range(100):
        writer.writerow(['SVR', hiperparametros_svr[i],vectores_soporte[i], train_RMSE_svr[i], test_RMSE_svr[i],tiempo_msvr[i]])
        writer.writerow(['VAR', 'N/A','N/A', train_RMSE_var[i], test_RMSE_var[i],tiempo_var[i]])

##Polution

#--------------------------VAR Puro integrado------------------------------------------------------------#
t=1000 # Longitud de la serie
k =2 # dimensión del vector Y
p = 1 # Número de retardos
h=1
col=2
# Almacenar resultados
hiperparametros_svr = []
vectores_soporte=[]
train_RMSE_svr = []
test_RMSE_svr = []
train_RMSE_var = []
test_RMSE_var = []
tiempo_var=[]
tiempo_msvr=[]
# Generación de la serie y ajuste de modelos
for no in range(100):
    
    print(f"-------------------------Iteration {no}--------------------------")
    # Generar series
    A = np.array([[1, 0],
                   [0, 1]])

    initial = np.random.normal(size=(2,))
    serie = np.zeros((2, t))
    serie[:, :1] = initial[:, np.newaxis]

    for i in range(1, t):
       retardo = serie[:, i-1:i]
       serie[:, i] = np.dot(A, retardo.flatten()) +np.random.normal(loc=0.0, scale=1.0, size=2)


    series = pd.DataFrame(serie.T, columns=['Y1', 'Y2'])
    
    #Partición en train y test
    train_size = int(len(series) * 0.7)
    train, test = series.iloc[:train_size], series.iloc[train_size:]
    
    #PACF
    pacf_var1 = pacf(train['Y1'], nlags=16)
    pacf_var2 = pacf(train['Y2'], nlags=16)
    banda= 1.96 / np.sqrt(t)  
   
    # #rezago_elegido_1 = rezago_sig(pacf_var1, banda)
    # #rezago_elegido_2 = rezago_sig(pacf_var2, banda)
    # enumerated_list = list(enumerate(pacf_var1))
    # reversed_enumerated_list = list(reversed(enumerated_list))
    # filtered_indices = [i for i, x in reversed_enumerated_list if abs(x) > banda]

    
    # rez= int(min(rezago_elegido_1, rezago_elegido_2))
    rez=1
    print(f"----------------Rezago {rez}--------------")
        
    
#     fig, axes = plt.subplots(2, 1, figsize=(10, 10))
# # Plotear la PACF de Y1
# axes[0].stem(range(len(pacf_var1)), pacf_var1, basefmt=" ", use_line_collection=True)
# axes[0].axhline(y=banda, color='r', linestyle='--', label=f'Banda superior ({banda:.2f})')
# axes[0].axhline(y=-banda, color='r', linestyle='--', label=f'Banda inferior ({-banda:.2f})')
# axes[0].set_xlabel('Rezago')
# axes[0].set_ylabel('PACF Y1')
# axes[0].set_title('PACF de Y1 con Bandas de Confianza')
# axes[0].legend()

# # Plotear la PACF de Y2
# axes[1].stem(range(len(pacf_var2)), pacf_var2, basefmt=" ", use_line_collection=True)
# axes[1].axhline(y=banda, color='r', linestyle='--', label=f'Banda superior ({banda:.2f})')
# axes[1].axhline(y=-banda, color='r', linestyle='--', label=f'Banda inferior ({-banda:.2f})')
# axes[1].set_xlabel('Rezago')
# axes[1].set_ylabel('PACF Y2')
# axes[1].set_title('PACF de Y2 con Bandas de Confianza')
# axes[1].legend()

# # Mostrar los gráficos
# plt.tight_layout()
# plt.show()
    
    
    
    # Ajustar modelo VAR
    start_time = time.time()
    model_var = VAR(train)
    results_var = model_var.fit(maxlags=1, ic='aic')
    lag_order = results_var.k_ar
    
    modelo_var_train = []
    modelo_var_test = []
        
    #Predicciones para train
    train_pred = results_var.fittedvalues
    # Predicciones para test
    test_pred=[]
    input_data = train.values[-rez:]

    for i in range(len(test)):
        pred = results_var.forecast(y=input_data, steps=h)
        test_pred.append(pred[0])
        input_data = np.vstack([input_data[1:], test.values[i:i+1]])

    test_pred = np.array(test_pred)
          
     
    train_rmse_var = np.sqrt(mean_squared_error(train.values[rez:], train_pred))
    test_rmse_var = np.sqrt(mean_squared_error(test.values, test_pred))

    
    train_RMSE_var.append(train_rmse_var)
    test_RMSE_var.append(test_rmse_var)
    print("Termine de ajustar modelo VAR")
    end_time = time.time()
    execution_time = end_time - start_time
    tiempo_var.append(execution_time)
    
    # Ajustar modelo SVR
    start_time = time.time()
    fechas = pd.DataFrame(list(range(len(series))))
    total = pd.concat([fechas,series], axis=1).values
    dim=len(total)

    #Construcción de la base de datos
    data=Base(total)
    data= data.base
    #Creamos la base de datos
    dataset = create_dataset_rez(data,dim,h,col,rez)
    X, Y = dataset[:, :(0 - h*2)], dataset[:, (0-h*2):]
    #Train y test
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, shuffle=False)
    
    #normalizados
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    scaler_X.fit(X_train)
    scaler_y.fit(y_train)

   
    X_train_nor = scaler_X.transform(X_train)
    X_test_nor = scaler_X.transform(X_test)
    y_train_nor = scaler_y.transform(y_train)
    y_test_nor = scaler_y.transform(y_test)

    pipe = Pipeline([
        ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))
    ])

    hyperparameters = {
        #'MSVR__kernel': ['poly'],
        'MSVR__kernel': ['poly','rbf','linear'],
        'MSVR__degree': [2,5],
        #'MSVR__degree': [1],
        'MSVR__gamma': [0.5,1],
        'MSVR__coef0': [0.1,0.5,1],
        'MSVR__C': [5,9,11,13],
        'MSVR__epsilon':[1,2], 
    }
    
    bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=15, scoring='neg_mean_squared_error', cv=5, verbose=0, error_score='raise')
   
    best_model = bm.fit(X_train_nor, y_train_nor)
    best_params = bm.best_params_

    msvr = MSVR(kernel=bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"),
                epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"),
                degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)
    
    msvr.fit(X_train_nor, y_train_nor)
  
 
    trainPred_svr_nor = msvr.predict(X_train_nor)
    testPred_svr_nor = msvr.predict(X_test_nor)

    trainPred_svr  = scaler_y.inverse_transform(trainPred_svr_nor)
    testPred_svr  = scaler_y.inverse_transform(testPred_svr_nor)
    train_rmse_svr = rmse(y_train, trainPred_svr)
    test_rmse_svr = rmse(y_test, testPred_svr)
    
    hiperparametros_svr.append(best_params)
    vectores_soporte.append(msvr.NSV)
    print("SVR Best params:", msvr.NSV/t)
    train_RMSE_svr.append(train_rmse_svr)
    test_RMSE_svr.append(test_rmse_svr)
    
    end_time = time.time()
    execution_time = end_time - start_time
    tiempo_msvr.append(execution_time)

    
    print("SVR Best params:", best_params)
    print("VAR Train RMSE:", train_rmse_var, "Test RMSE:", test_rmse_var)
    print("SVR Train RMSE:", train_rmse_svr, "Test RMSE:", test_rmse_svr)



# Guardar resultados en un archivo CSV
#CORRER PRQUE NO LO HE CORRIDO
with open('resultados_comparacion_varintegradopuro1000.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Modelo', 'Hiperparametros', 'Vectores_soporte', 'Train RMSE', 'Test RMSE', 'Tiempo'])
    for i in range(100):
        writer.writerow(['SVR', hiperparametros_svr[i],vectores_soporte[i], train_RMSE_svr[i], test_RMSE_svr[i],tiempo_msvr[i]])
        writer.writerow(['VAR', 'N/A','N/A', train_RMSE_var[i], test_RMSE_var[i],tiempo_var[i]])

#--------------------------Modelo SETAR de orden 1------------------------------------------------------------#


t=200 # Longitud de la serie
k =2 # dimensión del vector Y
p = 1 # Número de retardos
h=1
col=2
threshold=0
# Almacenar resultados
hiperparametros_svr = []
vectores_soporte=[]
train_RMSE_svr = []
test_RMSE_svr = []
train_RMSE_setar = []
test_RMSE_setar = []
tiempo_setar=[]
tiempo_msvr=[]
# Generación de la serie y ajuste de modelos
for no in range(100):
    
    print(f"-------------------------Iteration {no}--------------------------")
    # Generar series
    t=200
    x1 = np.random.normal(size=t)
    x2 = np.random.normal(size=t)
    y1 = np.zeros(t)
    y2 = np.zeros(t)
    
    #Revisar para modificar la interacción con y2.
    
    for m in range(1, t):
        if y1[m-1] < 0:
           y1[m] = 0.5 * y1[m-1] + 0.3 * x1[m] + np.random.normal()
           y2[m] = 0.4 * y2[m-1] + 0.2 * x2[m] + np.random.normal()
        else:
           y1[m] = 0.7 * y1[m-1] + 0.1 * x1[m] + np.random.normal()
           y2[m] = 0.6 * y2[m-1] + 0.3 * x2[m] + np.random.normal()


    series = pd.DataFrame({'y1': y1, 'y2': y2})
                           
    #Graficar las series temporales.                       
    
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(series['y1'], label='y1')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Serie Temporal y1')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(series['y2'], label='y2')
    plt.title('Serie Temporal y2')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    #Partición en train y test
    train_size = int(len(series) * 0.7)
    train, test = series.iloc[:train_size], series.iloc[train_size:]
    
    #PACF
    #pacf_var1 = pacf(train['Y1'], nlags=16)
    #pacf_var2 = pacf(train['Y2'], nlags=16)
    #banda= 1.96 / np.sqrt(t)  
   
    # #rezago_elegido_1 = rezago_sig(pacf_var1, banda)
    # #rezago_elegido_2 = rezago_sig(pacf_var2, banda)
    # enumerated_list = list(enumerate(pacf_var1))
    # reversed_enumerated_list = list(reversed(enumerated_list))
    # filtered_indices = [i for i, x in reversed_enumerated_list if abs(x) > banda]

    
    # rez= int(min(rezago_elegido_1, rezago_elegido_2))
    rez=1
    print(f"----------------Rezago {rez}--------------")
        
    
#     fig, axes = plt.subplots(2, 1, figsize=(10, 10))
# # Plotear la PACF de Y1
# axes[0].stem(range(len(pacf_var1)), pacf_var1, basefmt=" ", use_line_collection=True)
# axes[0].axhline(y=banda, color='r', linestyle='--', label=f'Banda superior ({banda:.2f})')
# axes[0].axhline(y=-banda, color='r', linestyle='--', label=f'Banda inferior ({-banda:.2f})')
# axes[0].set_xlabel('Rezago')
# axes[0].set_ylabel('PACF Y1')
# axes[0].set_title('PACF de Y1 con Bandas de Confianza')
# axes[0].legend()

# # Plotear la PACF de Y2
# axes[1].stem(range(len(pacf_var2)), pacf_var2, basefmt=" ", use_line_collection=True)
# axes[1].axhline(y=banda, color='r', linestyle='--', label=f'Banda superior ({banda:.2f})')
# axes[1].axhline(y=-banda, color='r', linestyle='--', label=f'Banda inferior ({-banda:.2f})')
# axes[1].set_xlabel('Rezago')
# axes[1].set_ylabel('PACF Y2')
# axes[1].set_title('PACF de Y2 con Bandas de Confianza')
# axes[1].legend()

# # Mostrar los gráficos
# plt.tight_layout()
# plt.show()
    
    
    
    # Ajustar modelo SETAR
    start_time = time.time()
    def fit_setar(data, threshold_var, y_vars, threshold):
        regimes = data[threshold_var].shift(1) < threshold
        model1 = sm.OLS(data.loc[regimes, y_vars[0]], sm.add_constant(data.loc[regimes, y_vars[1]])).fit()
        model2 = sm.OLS(data.loc[~regimes, y_vars[0]], sm.add_constant(data.loc[~regimes, y_vars[1]])).fit()
        return model1, model2


    threshold_var = 'y1'
    y_vars = ['y1', 'y2']
    model1, model2 = fit_setar(train, threshold_var, y_vars, 0)

   # Función para predecir un paso adelante
    def predict_next_step(current_y1, current_y2, model1, model2, threshold):
        if current_y1 < threshold:
           next_y1 = model1.predict([1, current_y2])[0]
        else:
           next_y1 = model2.predict([1, current_y2])[0]
        next_y2 = next_y1  # asumiendo que y2 sigue el mismo comportamiento que y1 en este caso simplificado
        return next_y1, next_y2
     
    # Realizar predicciones en el conjunto de entrenamiento
    train_predictions_y1 = []
    train_predictions_y2 = []
    for m in range(1, len(train)):
        current_y1 = train.iloc[m - 1]['y1']
        current_y2 = train.iloc[m - 1]['y2']
        next_y1, next_y2 = predict_next_step(current_y1, current_y2, model1, model2, 0)
        train_predictions_y1.append(next_y1)
        train_predictions_y2.append(next_y2)

# Añadir las predicciones al DataFrame de entrenamiento
    train = train.iloc[1:]  # Eliminar la primera fila ya que no podemos predecir el primer valor
    train['predicted_y1'] = train_predictions_y1
    train['predicted_y2'] = train_predictions_y2

# Realizar predicciones en el conjunto de prueba
    test_predictions_y1 = []
    test_predictions_y2 = []
    for m in range(len(test)):
        current_y1 = test.iloc[m- 1]['y1'] if t > 0 else train.iloc[-1]['y1']
        current_y2 = test.iloc[m - 1]['y2'] if t > 0 else train.iloc[-1]['y2']
        next_y1, next_y2 = predict_next_step(current_y1, current_y2, model1, model2, 0)
        test_predictions_y1.append(next_y1)
        test_predictions_y2.append(next_y2)

# Añadir las predicciones al DataFrame de prueba
    test['predicted_y1'] = test_predictions_y1
    test['predicted_y2'] = test_predictions_y2

# Calcular los errores de predicción
    train_mse_y1 = mean_squared_error(train['y1'], train['predicted_y1'])
    train_mse_y2 = mean_squared_error(train['y2'], train['predicted_y2'])
    train_RMSE_setar_value = [np.sqrt((train_mse_y1 + train_mse_y2) / 2)]
    train_RMSE_setar.append(train_RMSE_setar_value)
    test_mse_y1 = mean_squared_error(test['y1'], test['predicted_y1'])
    test_mse_y2 = mean_squared_error(test['y2'], test['predicted_y2'])
    test_RMSE_setar_value = np.sqrt((test_mse_y1 + test_mse_y2) / 2)
    test_RMSE_setar.append(test_RMSE_setar_value)
    

 
    end_time = time.time()
    execution_time = end_time - start_time
    tiempo_setar.append(execution_time)
    
    
    
    
    
    
    print("Termine de ajustar modelo SETAR")
    end_time = time.time()
    execution_time = end_time - start_time
    tiempo_setar.append(execution_time)
    
    # Ajustar modelo SVR
    start_time = time.time()
    fechas = pd.DataFrame(list(range(len(series))))
    total = pd.concat([fechas,series], axis=1).values
    dim=len(total)

    #Construcción de la base de datos
    data=Base(total)
    data= data.base
    #Creamos la base de datos
    dataset = create_dataset_rez(data,dim,h,col,rez)
    X, Y = dataset[:, :(0 - h*2)], dataset[:, (0-h*2):]
    #Train y test
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, shuffle=False)
    
    #normalizados
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    scaler_X.fit(X_train)
    scaler_y.fit(y_train)

   
    X_train_nor = scaler_X.transform(X_train)
    X_test_nor = scaler_X.transform(X_test)
    y_train_nor = scaler_y.transform(y_train)
    y_test_nor = scaler_y.transform(y_test)

    pipe = Pipeline([
        ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))
    ])

    hyperparameters = {
        #'MSVR__kernel': ['poly'],
        'MSVR__kernel': ['poly','rbf','linear'],
        'MSVR__degree': [2,5],
        #'MSVR__degree': [1],
        'MSVR__gamma': [0.5,1],
        'MSVR__coef0': [0.1,0.5,1],
        'MSVR__C': [5,9,11,13],
        'MSVR__epsilon':[1,2], 
    }
    
    bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=15, scoring='neg_mean_squared_error', cv=2, verbose=0, error_score='raise')
   
    best_model = bm.fit(X_train_nor, y_train_nor)
    best_params = bm.best_params_

    msvr = MSVR(kernel=bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"),
                epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"),
                degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)
    
    msvr.fit(X_train_nor, y_train_nor)
  
 
    trainPred_svr_nor = msvr.predict(X_train_nor)
    testPred_svr_nor = msvr.predict(X_test_nor)

    trainPred_svr  = scaler_y.inverse_transform(trainPred_svr_nor)
    testPred_svr  = scaler_y.inverse_transform(testPred_svr_nor)
    train_rmse_svr = rmse(y_train, trainPred_svr)
    test_rmse_svr = rmse(y_test, testPred_svr)
    
    hiperparametros_svr.append(best_params)
    vectores_soporte.append(msvr.NSV)
    train_RMSE_svr.append(train_rmse_svr)
    test_RMSE_svr.append(test_rmse_svr)
    
    end_time = time.time()
    execution_time = end_time - start_time
    tiempo_msvr.append(execution_time)

    
    print("SVR Best params:", best_params)
    print("SETAR Train RMSE:", train_RMSE_setar, "Test RMSE:", test_RMSE_setar)
    print("SVR Train RMSE:", train_rmse_svr, "Test RMSE:", test_rmse_svr)


# Guardar resultados en un archivo CSV
#CORRER PRQUE NO LO HE CORRIDO
with open('resultados_comparacion_setar200.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Modelo', 'Hiperparametros', 'Vectores_soporte', 'Train RMSE', 'Test RMSE', 'Tiempo'])
    for i in range(100):
        writer.writerow(['SVR', hiperparametros_svr[i],vectores_soporte[i], train_RMSE_svr[i], test_RMSE_svr[i],tiempo_msvr[i]])
        writer.writerow(['SETAR', 'N/A','N/A', train_RMSE_setar[i], test_RMSE_setar[i],tiempo_setar[i]])


#--------------------------Modelo MSETAR.------------------------------------------------------------#

   
t = [200,500,1000]  # Longitud de la serie
k = 2  # Dimensión del vector Y
p = 1  # Número de retardos
h = 1 #Número de pasos a predecir.

# Almacenar resultados
rez=1
hiperparametros_svr = {size: [] for size in t}
vectores_soporte = {size: [] for size in t}
train_RMSE_var = {size: [] for size in t}
test_RMSE_var = {size: [] for size in t}
train_RMSE_svr = {size: [] for size in t}
test_RMSE_svr = {size: [] for size in t}
tiempo_msvr = {size: [] for size in t}
resultados = []

data_folder = "C:/Users/huma1003/OneDrive - NIQ/DOCUMENTOS IMPORTANTES/Mis_Cosas/MarlijarTM/msvr-master/MSETAR/Data_Sergio"

def load_data(size, no):
    file_path = f"{data_folder}/dataset_size_{size}.xlsx"
    sheet_name = f"Iter_{no}"
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

# Generación de la serie y ajuste de modelos
for size in t:
      a=size
      for no in range(100):
        print(f"-------------------------Tamaño {a}-------------------------")
        print(f"-------------------------Iteration {no}--------------------------")
        #series = np.zeros((k, size))
        #series = cointegrated_vector(size, k, no, np.eye(k),Phi)
        #series = pd.DataFrame(series, columns=['Y1', 'Y2'])
        series = load_data(size, no)
        scaler_X = StandardScaler()
        scaler_X.fit(series)
        series = pd.DataFrame(scaler_X.transform(series))
       
       
        train_size = int(len(series) * 0.7)
        train, test = series.iloc[:train_size], series.iloc[train_size:]
        test = test.reset_index(drop=True) 
        
        
        #Ajustar VAR
        start_time = time.time()
        model_var = VAR(train)
        results_var = model_var.fit(maxlags=1, ic='aic')
        print(results_var.summary())
        coefs = results_var.coefs
        if len(coefs) == 0 or len(coefs[0]) == 0:
         print("Coefs está vacío o no tiene elementos en el primer índice. Saltando esta iteración.")
        
         continue 
        #lag_order = results_var.k_ar
        
        modelo_var_train = []
        modelo_var_test = []
            
        #Predicciones para train
        trainPred = results_var.fittedvalues
        
        # Predicciones para test
        testPred=[]
        input_data = train.values[-rez:]

        for i in range(len(test)):
            pred = results_var.forecast(y=input_data, steps=h)
            testPred.append(pred[0])
            input_data = np.vstack([input_data[1:], test.values[i:i+1]])
            
        
        trainPred= np.array(trainPred)
        testPred= np.array(testPred)
        train=np.array(train)
        test=np.array(test)
              
        train_rmse_var = np.sqrt(np.mean((train[1:, 0] - trainPred[:,0])**2 + (train[1:, 1] - trainPred[:, 1])**2))

        #test_rmse_svr = rmse(y_test, testPred_svr)
        test_rmse_var = np.sqrt(np.mean((test[:, 0] - testPred[:, 0])**2 + (test[:, 1] - testPred[:, 1])**2))
        print(test_rmse_var)
        
        train_RMSE_var[size].append(train_rmse_var)
        test_RMSE_var[size].append(test_rmse_var)
    
        #Ajustar modelo SVR
        start_time = time.time()
        fechas = pd.DataFrame(list(range(len(series))))
        total = pd.concat([fechas,series], axis=1).values
        dim=len(total)
    
        #Construcción de la base de datos
        data=Base(total)
        data= data.base
        #Creamos la base de datos
        dataset = create_dataset_rez(data,dim,h,k,p)
        X, Y = dataset[:, :(0 - h*2)], dataset[:, (0-h*2):]
        #Train y test
        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, shuffle=False)
        
        #normalizados
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        scaler_X.fit(X_train)
        scaler_y.fit(y_train)
    
       
        X_train_nor = scaler_X.transform(X_train)
        X_test_nor = scaler_X.transform(X_test)
        y_train_nor = scaler_y.transform(y_train)
        y_test_nor = scaler_y.transform(y_test)
    
        pipe = Pipeline([
            ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))
        ])
    
        hyperparameters = {
            #'MSVR__kernel': ['poly'],
            'MSVR__kernel': ['linear','poly','rbf'],
            'MSVR__degree': [2,3,4,5],
            #'MSVR__degree': [1],
            'MSVR__gamma': [0.5,1],
            'MSVR__coef0': [0.1,0.5,1],
            'MSVR__C': [5,9,11,13],
            'MSVR__epsilon':[1,2], 
        }
        
        bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=15, scoring='neg_mean_squared_error', cv=5, verbose=0, error_score='raise')
        
        best_model = bm.fit(X_train_nor, y_train_nor)
        best_params = bm.best_params_
    
        msvr = MSVR(kernel=bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"),
                    epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"),
                    degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)
        
        msvr.fit(X_train_nor, y_train_nor)
      
    
    
        trainPred_svr_nor = msvr.predict(X_train_nor)
        testPred_svr_nor = msvr.predict(X_test_nor)

        trainPred_svr  = scaler_y.inverse_transform(trainPred_svr_nor)
        testPred_svr  = scaler_y.inverse_transform(testPred_svr_nor)
        # train_rmse_svr = rmse(y_train, trainPred_svr)
        train_rmse_svr = np.sqrt(np.mean((y_train[:, 0] - trainPred_svr[:, 0])**2 + (y_train[:, 1] - trainPred_svr[:, 1])**2))

        # test_rmse_svr = rmse(y_test, testPred_svr)
        test_rmse_svr = np.sqrt(np.mean((y_test[:, 0] - testPred_svr[:, 0])**2 + (y_test[:, 1] - testPred_svr[:, 1])**2))
        print(test_rmse_svr)
        # # #Revisión
        
       
        
        
        hiperparametros_svr[size].append(best_params)
        vectores_soporte[size].append(msvr.NSV)
        train_RMSE_svr[size].append(train_rmse_svr)
        test_RMSE_svr[size].append(test_rmse_svr)
        
        end_time = time.time()
        execution_time = end_time - start_time
        tiempo_msvr[size].append(end_time - start_time)
    
        
        #print("SVR Best params:", best_params)
        
        
    #     df_train = pd.DataFrame({
    #     'Iteracion': no,  
    #     'y1_real': train.iloc[:, 0],  
    #     'y2_real': train.iloc[:, 1],  
    #     'y1_pred_var': train_pred_levels.iloc[:, 0],  
    #     'y2_pred_var': train_pred_levels.iloc[:, 1],  
    #     'y1_pred_svr': trainPred_svr.iloc[:, 0],  
    #     'y2_pred_svr': trainPred_svr.iloc[:, 1],  
    #     'Tipo': 'Train'
    # })

    #     # Crear DataFrame para las pruebas
    #     df_test = pd.DataFrame({
    #         'Iteracion': no,  
    #         'y1_real': test.iloc[:, 0],  
    #         'y2_real': test.iloc[:, 1],  
    #         'y1_pred_var': test_pred_levels.iloc[:, 0],  
    #         'y2_pred_var': test_pred_levels.iloc[:, 1],  
    #         'y1_pred_svr': testPred_svr.iloc[:, 0],  
    #         'y2_pred_svr': testPred_svr.iloc[:, 1],  
    #         'Tipo': 'Test'
    #     })
    #     a = pd.concat([df_train, df_test], ignore_index=True)  
    #     resultados.append(a)


# Concatenar resultados corrida 200
#df_final = pd.concat(resultados, ignore_index=True)
#df_final.to_excel('resultados_completos.xlsx', index=False)

#Corridas en general
filename = f'resultados_comparacion_MSTAR_data_final.csv'


with open('resultados_comparacion_MSETAR_DATAsERGIO.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Modelo', 'Hiperparametros', 'Vectores_soporte', 'Train RMSE', 'Test RMSE', 'Tiempo'])
    for i in range(100):
        writer.writerow(['SVR', hiperparametros_svr[i],vectores_soporte[i], train_RMSE_svr[i], test_RMSE_svr[i],tiempo_msvr[i]])
        writer.writerow(['VAR', 'N/A','N/A', train_RMSE_var[i], test_RMSE_var[i],tiempo_var[i]])



with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Tamaño', 'Modelo', 'Hiperparametros', 'Vectores_soporte', 'Train RMSE', 'Test RMSE', 'Tiempo'])
    for size in t:
        writer.writerow([f'Tamaño: {size}', '', '', '', '', '', ''])
        for i in range(len(hiperparametros_svr[size])):
            writer.writerow(['SVR', hiperparametros_svr[size][i], vectores_soporte[size][i], train_RMSE_svr[size][i], test_RMSE_svr[size][i], tiempo_msvr[size][i]])
            writer.writerow(['VAR', 'N/A','N/A', train_RMSE_var[size][i], test_RMSE_var[size][i], 'N/A'])

print(f'Resultados guardados en {filename}')


with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Tamaño', 'Modelo',  'Train RMSE', 'Test RMSE', 'Tiempo'])
    for size in t:
        writer.writerow([f'Tamaño: {size}', '', '', '', '', '', ''])
        for i in range(len(tiempo_msvr[size])):
            writer.writerow(['SVR', train_RMSE_svr[size][i], test_RMSE_svr[size][i], tiempo_msvr[size][i]])


________________
print(f'Resultados guardados en {filename}')


with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Tamaño', 'Modelo',  'Train RMSE', 'Test RMSE', 'Tiempo'])
    for size in t:
        writer.writerow([f'Tamaño: {size}', '', '', '', '', '', ''])
        for i in range(len(tiempo_msvr[size])):
            writer.writerow(['SVR', train_RMSE_svr[size][i], test_RMSE_svr[size][i], tiempo_msvr[size][i]])


#
#--------------------------Modelo MSETAR univariado.------------------------------------------------------------#

   
t = [1000]  # Longitud de la serie
k = 2  # Dimensión del vector Y
p = 1  # Número de retardos
h = 1 #Número de pasos a predecir.

# Almacenar resultados
rez=1
hiperparametros_svr = {size: [] for size in t}
vectores_soporte = {size: [] for size in t}
train_RMSE_svr = {size: [] for size in t}
test_RMSE_svr = {size: [] for size in t}
tiempo_msvr = {size: [] for size in t}
resultados = []


# Generación de la serie y ajuste de modelos
for size in t:
      a=size
      for no in range(100):
        print(f"-------------------------Tamaño {a}-------------------------")
        print(f"-------------------------Iteration {no}--------------------------")
        #series = np.zeros((k, size))
        #series = cointegrated_vector(size, k, no, np.eye(k),Phi)
        #series = pd.DataFrame(series, columns=['Y1', 'Y2'])
        series = pd.read_excel("C:/Users/huma1003/OneDrive - NIQ/DOCUMENTOS IMPORTANTES/Mis_Cosas/MarlijarTM/msvr-master/MSETAR/Univariada/series_setar.xlsx")
      
        #Partición en train y test
        train_size = int(len(series) * 0.7)
        train, test = series.iloc[:train_size], series.iloc[train_size:]
        test = test.reset_index(drop=True) 
        
        
        #Ajustar VAR
        # start_time = time.time()
        # model_var = VAR(train)
        # results_var = model_var.fit(maxlags=1, ic='aic')
        # print(results_var.summary())
        # coefs = results_var.coefs
        # #lag_order = results_var.k_ar
        
        # modelo_var_train = []
        # modelo_var_test = []
            
        # #Predicciones para train
        # trainPred = results_var.fittedvalues
        
        # # Predicciones para test
        # testPred=[]
        # input_data = train.values[-rez:]

        # for i in range(len(test)):
        #     pred = results_var.forecast(y=input_data, steps=h)
        #     testPred.append(pred[0])
        #     input_data = np.vstack([input_data[1:], test.values[i:i+1]])
            
        
        # trainPred= np.array(trainPred)
        # testPred= np.array(testPred)
        # train=np.array(train)
        # test=np.array(test)
              
        # train_rmse_svr = np.sqrt(np.mean((train[1:, 0] - trainPred[:,0])**2 + (train[1:, 1] - trainPred[:, 1])**2))

        # #test_rmse_svr = rmse(y_test, testPred_svr)
        # test_rmse_svr = np.sqrt(np.mean((test[:, 0] - testPred[:, 0])**2 + (test[:, 1] - testPred[:, 1])**2))
        # print(test_rmse_svr)
        
    
        # Ajustar modelo SVR
        start_time = time.time()
        fechas = pd.DataFrame(list(range(len(series))))
        total = pd.concat([fechas,series], axis=1).values
        dim=len(total)
    
        #Construcción de la base de datos
        data=Base(total)
        data= data.base
        #Creamos la base de datos
        dataset = create_dataset_rez(data,dim,h,k,p)
        X, Y = dataset[:, :(0 - h*2)], dataset[:, (0-h*2):]
        #Train y test
        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, shuffle=False)
        
        #normalizados
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        scaler_X.fit(X_train)
        scaler_y.fit(y_train)
    
       
        X_train_nor = scaler_X.transform(X_train)
        X_test_nor = scaler_X.transform(X_test)
        y_train_nor = scaler_y.transform(y_train)
        y_test_nor = scaler_y.transform(y_test)
    
        pipe = Pipeline([
            ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))
        ])
    
        hyperparameters = {
            #'MSVR__kernel': ['poly'],
            'MSVR__kernel': ['linear','poly','rbf'],
            'MSVR__degree': [2,3,4,5],
            #'MSVR__degree': [1],
            'MSVR__gamma': [0.5,1],
            'MSVR__coef0': [0.1,0.5,1],
            'MSVR__C': [5,9,11,13],
            'MSVR__epsilon':[1,2], 
        }
        
        bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=15, scoring='neg_mean_squared_error', cv=5, verbose=0, error_score='raise')
        
        best_model = bm.fit(X_train_nor, y_train_nor)
        best_params = bm.best_params_
    
        msvr = MSVR(kernel=bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"),
                    epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"),
                    degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)
        
        msvr.fit(X_train_nor, y_train_nor)
      
    
    
        trainPred_svr_nor = msvr.predict(X_train_nor)
        testPred_svr_nor = msvr.predict(X_test_nor)

        trainPred_svr  = scaler_y.inverse_transform(trainPred_svr_nor)
        testPred_svr  = scaler_y.inverse_transform(testPred_svr_nor)
        # train_rmse_svr = rmse(y_train, trainPred_svr)
        train_rmse_svr = np.sqrt(np.mean((y_train[:, 0] - trainPred_svr[:, 0])**2 + (y_train[:, 1] - trainPred_svr[:, 1])**2))

        # test_rmse_svr = rmse(y_test, testPred_svr)
        test_rmse_svr = np.sqrt(np.mean((y_test[:, 0] - testPred_svr[:, 0])**2 + (y_test[:, 1] - testPred_svr[:, 1])**2))
        print(test_rmse_svr)
        # #Revisión
        
       
        
        
        hiperparametros_svr[size].append(best_params)
        vectores_soporte[size].append(msvr.NSV)
        train_RMSE_svr[size].append(train_rmse_svr)
        test_RMSE_svr[size].append(test_rmse_svr)
        
        end_time = time.time()
        execution_time = end_time - start_time
        tiempo_msvr[size].append(end_time - start_time)
    
        
        #print("SVR Best params:", best_params)
        
        
    #     df_train = pd.DataFrame({
    #     'Iteracion': no,  
    #     'y1_real': train.iloc[:, 0],  
    #     'y2_real': train.iloc[:, 1],  
    #     'y1_pred_var': train_pred_levels.iloc[:, 0],  
    #     'y2_pred_var': train_pred_levels.iloc[:, 1],  
    #     'y1_pred_svr': trainPred_svr.iloc[:, 0],  
    #     'y2_pred_svr': trainPred_svr.iloc[:, 1],  
    #     'Tipo': 'Train'
    # })

    #     # Crear DataFrame para las pruebas
    #     df_test = pd.DataFrame({
    #         'Iteracion': no,  
    #         'y1_real': test.iloc[:, 0],  
    #         'y2_real': test.iloc[:, 1],  
    #         'y1_pred_var': test_pred_levels.iloc[:, 0],  
    #         'y2_pred_var': test_pred_levels.iloc[:, 1],  
    #         'y1_pred_svr': testPred_svr.iloc[:, 0],  
    #         'y2_pred_svr': testPred_svr.iloc[:, 1],  
    #         'Tipo': 'Test'
    #     })
    #     a = pd.concat([df_train, df_test], ignore_index=True)  
    #     resultados.append(a)


# Concatenar resultados corrida 200
#df_final = pd.concat(resultados, ignore_index=True)
#df_final.to_excel('resultados_completos.xlsx', index=False)

#Corridas en general
filename = f'resultados_comparacion_MSTAR_uni.csv'


with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Tamaño', 'Modelo', 'Hiperparametros', 'Vectores_soporte', 'Train RMSE', 'Test RMSE', 'Tiempo'])
    for size in t:
        writer.writerow([f'Tamaño: {size}', '', '', '', '', '', ''])
        for i in range(len(hiperparametros_svr[size])):
            writer.writerow(['SVR', hiperparametros_svr[size][i], vectores_soporte[size][i], train_RMSE_svr[size][i], test_RMSE_svr[size][i], tiempo_msvr[size][i]])

print(f'Resultados guardados en {filename}')


with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Tamaño', 'Modelo',  'Train RMSE', 'Test RMSE', 'Tiempo'])
    for size in t:
        writer.writerow([f'Tamaño: {size}', '', '', '', '', '', ''])
        for i in range(len(tiempo_msvr[size])):
            writer.writerow(['SVR', train_RMSE_svr[size][i], test_RMSE_svr[size][i], tiempo_msvr[size][i]])


#

#--------------------------Estudio de caso.------------------------------------------------------------#

t=[3276]  
k = 3 # Dimensión del vector Y
p = 1  # Número de retardos
h = 1 #Número de pasos a predecir.

# Almacenar resultados
rez=1
hiperparametros_svr = {size: [] for size in t}
vectores_soporte = {size: [] for size in t}
train_RMSE_svr = {size: [] for size in t}
test_RMSE_svr = {size: [] for size in t}
tiempo_msvr = {size: [] for size in t}
resultados = []

data_folder = "C:/Users/huma1003/OneDrive - NIQ/DOCUMENTOS IMPORTANTES/Mis_Cosas/MarlijarTM/msvr-master/Datos_energia"

def load_data(size, no):
    file_path = f"{data_folder}/dataset_size_{size}.xlsx"
    sheet_name = f"Iter_{no}"
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

# Generación de la serie y ajuste de modelos
for size in t:
      a=size
      for no in range(100):
        print(f"-------------------------Tamaño {a}-------------------------")
        print(f"-------------------------Iteration {no}--------------------------")
        #series = np.zeros((k, size))
        #series = cointegrated_vector(size, k, no, np.eye(k),Phi)
        #series = pd.DataFrame(series, columns=['Y1', 'Y2'])
        #series = load_data(size, no)
        series=series
        #Partición en train y test
        train_size = int(len(series) * 0.7)
        train, test = series.iloc[:train_size], series.iloc[train_size:]
        test = test.reset_index(drop=True) 
        
        
        #Ajustar VAR
        start_time = time.time()
        model_var = VAR(train)
        results_var = model_var.fit(maxlags=1, ic='aic')
        print(results_var.summary())
        coefs = results_var.coefs
        #lag_order = results_var.k_ar
        
        modelo_var_train = []
        modelo_var_test = []
            
        #Predicciones para train
        trainPred = results_var.fittedvalues
        
        # Predicciones para test
        testPred=[]
        input_data = train.values[-rez:]

        for i in range(len(test)):
            pred = results_var.forecast(y=input_data, steps=h)
            testPred.append(pred[0])
            input_data = np.vstack([input_data[1:], test.values[i:i+1]])
            
        
        trainPred= np.array(trainPred)
        testPred= np.array(testPred)
        train=np.array(train)
        test=np.array(test)
              
        train_rmse_svr = np.sqrt(np.mean((train[1:, 0] - trainPred[:,0])**2 + (train[1:, 1] - trainPred[:, 1])**2))

        #test_rmse_svr = rmse(y_test, testPred_svr)
        test_rmse_svr = np.sqrt(np.mean((test[:, 0] - testPred[:, 0])**2 + (test[:, 1] - testPred[:, 1])**2))
        print(test_rmse_svr)
        
    
        # # Ajustar modelo SVR
        # start_time = time.time()
        # fechas = pd.DataFrame(list(range(len(series))))
        # total = pd.concat([fechas,series], axis=1).values
        # dim=len(total)
    
        # #Construcción de la base de datos
        # data=Base(total)
        # data= data.base
        # #Creamos la base de datos
        # dataset = create_dataset_rez(data,dim,h,k,p)
        # X, Y = dataset[:, :(0 - h*2)], dataset[:, (0-h*2):]
        # #Train y test
        # X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, shuffle=False)
        
        # #normalizados
        # scaler_X = StandardScaler()
        # scaler_y = StandardScaler()
        
        # scaler_X.fit(X_train)
        # scaler_y.fit(y_train)
    
       
        # X_train_nor = scaler_X.transform(X_train)
        # X_test_nor = scaler_X.transform(X_test)
        # y_train_nor = scaler_y.transform(y_train)
        # y_test_nor = scaler_y.transform(y_test)
    
        # pipe = Pipeline([
        #     ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))
        # ])
    
        # hyperparameters = {
        #     #'MSVR__kernel': ['poly'],
        #     'MSVR__kernel': ['linear','poly','rbf'],
        #     'MSVR__degree': [2,3,4,5],
        #     #'MSVR__degree': [1],
        #     'MSVR__gamma': [0.5,1],
        #     'MSVR__coef0': [0.1,0.5,1],
        #     'MSVR__C': [5,9,11,13],
        #     'MSVR__epsilon':[1,2], 
        # }
        
        # bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=15, scoring='neg_mean_squared_error', cv=5, verbose=0, error_score='raise')
        
        # best_model = bm.fit(X_train_nor, y_train_nor)
        # best_params = bm.best_params_
    
        # msvr = MSVR(kernel=bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"),
        #             epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"),
        #             degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)
        
        # msvr.fit(X_train_nor, y_train_nor)
      
    
    
        # trainPred_svr_nor = msvr.predict(X_train_nor)
        # testPred_svr_nor = msvr.predict(X_test_nor)

        # trainPred_svr  = scaler_y.inverse_transform(trainPred_svr_nor)
        # testPred_svr  = scaler_y.inverse_transform(testPred_svr_nor)
        #train_rmse_svr = rmse(y_train, trainPred_svr)
        #train_rmse_svr = np.sqrt(np.mean((y_train[:, 0] - trainPred_svr[:, 0])**2 + (y_train[:, 1] - trainPred_svr[:, 1])**2))

        #test_rmse_svr = rmse(y_test, testPred_svr)
        #test_rmse_svr = np.sqrt(np.mean((y_test[:, 0] - testPred_svr[:, 0])**2 + (y_test[:, 1] - testPred_svr[:, 1])**2))
        print(test_rmse_svr)
        #Revisión
        
       
        
        
        #hiperparametros_svr[size].append(best_params)
        #vectores_soporte[size].append(msvr.NSV)
        train_RMSE_svr[size].append(train_rmse_svr)
        test_RMSE_svr[size].append(test_rmse_svr)
        
        end_time = time.time()
        execution_time = end_time - start_time
        tiempo_msvr[size].append(end_time - start_time)
    
        
        #print("SVR Best params:", best_params)
        
        
    #     df_train = pd.DataFrame({
    #     'Iteracion': no,  
    #     'y1_real': train.iloc[:, 0],  
    #     'y2_real': train.iloc[:, 1],  
    #     'y1_pred_var': train_pred_levels.iloc[:, 0],  
    #     'y2_pred_var': train_pred_levels.iloc[:, 1],  
    #     'y1_pred_svr': trainPred_svr.iloc[:, 0],  
    #     'y2_pred_svr': trainPred_svr.iloc[:, 1],  
    #     'Tipo': 'Train'
    # })

    #     # Crear DataFrame para las pruebas
    #     df_test = pd.DataFrame({
    #         'Iteracion': no,  
    #         'y1_real': test.iloc[:, 0],  
    #         'y2_real': test.iloc[:, 1],  
    #         'y1_pred_var': test_pred_levels.iloc[:, 0],  
    #         'y2_pred_var': test_pred_levels.iloc[:, 1],  
    #         'y1_pred_svr': testPred_svr.iloc[:, 0],  
    #         'y2_pred_svr': testPred_svr.iloc[:, 1],  
    #         'Tipo': 'Test'
    #     })
    #     a = pd.concat([df_train, df_test], ignore_index=True)  
    #     resultados.append(a)


# Concatenar resultados corrida 200
#df_final = pd.concat(resultados, ignore_index=True)
#df_final.to_excel('resultados_completos.xlsx', index=False)

#Corridas en general
filename = f'resultados_comparacion_VAR_msetar_uni.csv'


with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Tamaño', 'Modelo', 'Hiperparametros', 'Vectores_soporte', 'Train RMSE', 'Test RMSE', 'Tiempo'])
    for size in t:
        writer.writerow([f'Tamaño: {size}', '', '', '', '', '', ''])
        for i in range(len(hiperparametros_svr[size])):
            writer.writerow(['SVR', hiperparametros_svr[size][i], vectores_soporte[size][i], train_RMSE_svr[size][i], test_RMSE_svr[size][i], tiempo_msvr[size][i]])

print(f'Resultados guardados en {filename}')


with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Tamaño', 'Modelo',  'Train RMSE', 'Test RMSE', 'Tiempo'])
    for size in t:
        writer.writerow([f'Tamaño: {size}', '', '', '', '', '', ''])
        for i in range(len(tiempo_msvr[size])):
            writer.writerow(['SVR', train_RMSE_svr[size][i], test_RMSE_svr[size][i], tiempo_msvr[size][i]])

#Caso estudio

#--------------------------VAR BIVARIADO-Estacionario------------------------------------------------------------#
file_path = "C:/Users/huma1003/OneDrive - NIQ/DOCUMENTOS IMPORTANTES/Mis_Cosas/MarlijarTM/msvr-master/Resultados_estudio de caso/Profe Sergio/Dats sets.xlsx"
data = pd.read_excel(file_path)
data=data

t=len(data) # Longitud de la serie
k =2 # dimensión del vector Y
p = 1 # Número de retardos
h=1
col=2

# Almacenar resultados
hiperparametros_svr = []
vectores_soporte=[]
train_RMSE_svr = []
test_RMSE_svr = []
train_RMSE_var = []
test_RMSE_var = []
tiempo_var=[]
tiempo_msvr=[]
# Generación de la serie y ajuste de modelos   
series = data.iloc[:, 1:3]
   
#Partición en train y test
train_size = int(len(series) * 0.7)
train, test = series.iloc[:train_size], series.iloc[train_size:]
#PACF
pacf_var1 = pacf(train['RTB3MSY1'], nlags=16)
pacf_var2 = pacf(train['RTN3YSY2'], nlags=16)
banda= 1.96 / np.sqrt(t)  
   

  

fig, axes = plt.subplots(2, 1, figsize=(10, 10))
# Plotear la PACF de Y1
axes[0].stem(range(len(pacf_var1)), pacf_var1, basefmt=" ", use_line_collection=True)
axes[0].axhline(y=banda, color='r', linestyle='--', label=f'Banda superior ({banda:.2f})')
axes[0].axhline(y=-banda, color='r', linestyle='--', label=f'Banda inferior ({-banda:.2f})')
axes[0].set_xlabel('Rezago')
axes[0].set_ylabel('PACF Y1')
axes[0].set_title('PACF de Y1 con Bandas de Confianza')
axes[0].legend()

# Plotear la PACF de Y2
axes[1].stem(range(len(pacf_var2)), pacf_var2, basefmt=" ", use_line_collection=True)
axes[1].axhline(y=banda, color='r', linestyle='--', label=f'Banda superior ({banda:.2f})')
axes[1].axhline(y=-banda, color='r', linestyle='--', label=f'Banda inferior ({-banda:.2f})')
axes[1].set_xlabel('Rezago')
axes[1].set_ylabel('PACF Y2')
axes[1].set_title('PACF de Y2 con Bandas de Confianza')
axes[1].legend()

# Plotear la PACF de Y3
axes[2].stem(range(len(pacf_var3)), pacf_var3, basefmt=" ", use_line_collection=True)
axes[2].axhline(y=banda, color='r', linestyle='--', label=f'Banda superior ({banda:.2f})')
axes[2].axhline(y=-banda, color='r', linestyle='--', label=f'Banda inferior ({-banda:.2f})')
axes[2].set_xlabel('Rezago')
axes[2].set_ylabel('PACF Y3')
axes[2].set_title('PACF de Y3 con Bandas de Confianza')
axes[2].legend()
# Mostrar los gráficos
plt.tight_layout()
plt.show()

    
rez=5   
# Ajustar modelo VAR
start_time = time.time()
model_var = VAR(train)
results_var = model_var.fit(maxlags=5, ic='aic')
lag_order = results_var.k_ar

modelo_var_train = []
modelo_var_test = []
    
#Predicciones para train
train_pred = results_var.fittedvalues
# Predicciones para test
test_pred=[]
input_data = train.values[-rez:]

for i in range(len(test)):
    pred = results_var.forecast(y=input_data, steps=h)
    test_pred.append(pred[0])
    input_data = np.vstack([input_data[1:], test.values[i:i+1]])

test_pred = pd.DataFrame(test_pred) 
#Sacar rmse     
#train_rmse_var = np.sqrt(np.mean((train.iloc[lag_order:, 0] - train_pred.iloc[:, 0])**2 + (train.iloc[lag_order:, 1] - train_pred.iloc[:, 1])**2))
train_rmse_var = np.sqrt(np.mean((train.iloc[lag_order:, 0] - train_pred.iloc[:, 0])**2 + (train.iloc[lag_order:, 1] - train_pred.iloc[:, 1])**2))


# Cálculo del RMSE para el conjunto de prueba
#test_rmse_var = np.sqrt(np.mean((test.iloc[:, 0] - test_pred.iloc[:, 0])**2 + (test.iloc[:, 1] - test_pred.iloc[:, 1])**2))
test_rmse_var = np.sqrt(np.mean((test.iloc[:, 0] - test_pred.iloc[:, 0])**2 + (test.iloc[:, 1] - test_pred.iloc[:, 1])**2))

print(test_RMSE_var)
import pandas as pd

# Supongamos que tienes tus datos originales en `train` y `test`
# y tus predicciones en `train_pred` y `test_pred`

# Asegúrate de que las predicciones son DataFrames y tienen el mismo número de filas que los datos originales
train_pred_df = pd.DataFrame(train_pred, columns=['Predicción_Var1', 'Predicción_Var2' ])
test_pred_df = pd.DataFrame(test_pred, columns=['Predicción_Var1', 'Predicción_Var2'])

# Concatenar los DataFrames
train_combined = pd.concat([train.reset_index(drop=True), train_pred.reset_index(drop=True)], axis=1)
test_combined = pd.concat([test.reset_index(drop=True), test_pred.reset_index(drop=True)], axis=1)

# Guardar en archivos CSV
train_combined.to_csv('train_with_predictions_sergio_Var.csv', index=False)
test_combined.to_csv('test_with_predictions_sergio_var.csv', index=False)






print("Termine de ajustar modelo VAR")
end_time = time.time()
execution_time = end_time - start_time
tiempo_var.append(execution_time)

# Ajustar modelo SVR
start_time = time.time()
fechas = pd.DataFrame(list(range(len(series))))
total = pd.concat([fechas,series], axis=1).values
dim=len(total)

#Construcción de la base de datos
data=Base(total)
data= data.base
#Creamos la base de datos
dataset = create_dataset_rez(data,dim,h,col,rez)
X, Y = dataset[:, :(0 - h*2)], dataset[:, (0-h*2):]
#Train y test
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, shuffle=False)

#normalizados
scaler_X = StandardScaler()
scaler_y = StandardScaler()

scaler_X.fit(X_train)
scaler_y.fit(y_train)

   
X_train_nor = scaler_X.transform(X_train)
X_test_nor = scaler_X.transform(X_test)
y_train_nor = scaler_y.transform(y_train)
y_test_nor = scaler_y.transform(y_test)

pipe = Pipeline([
    ('MSVR', CustomMSVR(kernel='rbf', degree=3, gamma=0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1))
])

hyperparameters = {
    #'MSVR__kernel': ['poly'],
    'MSVR__kernel': ['poly','rbf','linear'],
    'MSVR__degree': [2,5],
    #'MSVR__degree': [1],
    'MSVR__gamma': [0.5,1],
    'MSVR__coef0': [0.1,0.5,1],
    'MSVR__C': [5,9,11,13],
    'MSVR__epsilon':[1,2], 
}

bm = RandomizedSearchCV(pipe, hyperparameters, n_iter=15, scoring='neg_mean_squared_error', cv=5, verbose=0, error_score='raise')
   
best_model = bm.fit(X_train_nor, y_train_nor)
best_params = bm.best_params_

msvr = MSVR(kernel=bm.best_params_.get("MSVR__kernel"), gamma=bm.best_params_.get("MSVR__gamma"),
            epsilon=bm.best_params_.get("MSVR__epsilon"), C=bm.best_params_.get("MSVR__C"),
            degree=bm.best_params_.get("MSVR__degree"), coef0=bm.best_params_.get("MSVR__coef0"), tol=0.01)

msvr.fit(X_train_nor, y_train_nor)
  
 
trainPred_svr_nor = msvr.predict(X_train_nor)
testPred_svr_nor = msvr.predict(X_test_nor)

trainPred_svr  = scaler_y.inverse_transform(trainPred_svr_nor)
testPred_svr  = scaler_y.inverse_transform(testPred_svr_nor)
train_rmse_svr = rmse(y_train, trainPred_svr)
test_rmse_svr = rmse(y_test, testPred_svr)
train_pred_df = pd.DataFrame(trainPred_svr, columns=['Predicción_Var1', 'Predicción_Var2'])
test_pred_df = pd.DataFrame(testPred_svr, columns=['Predicción_Var1', 'Predicción_Var2'])

# Concatenar los DataFrames
train_combined = pd.concat([train.reset_index(drop=True), train_pred_df.reset_index(drop=True)], axis=1)
test_combined = pd.concat([test.reset_index(drop=True), test_pred_df.reset_index(drop=True)], axis=1)

# Guardar en archivos CSV
train_combined.to_csv('train_with_prediction_svr_1_sergio.csv', index=False)
test_combined.to_csv('test_with_predictions_svr_1_sergio.csv', index=False)

hiperparametros_svr.append(best_params)
vectores_soporte.append(msvr.NSV)
print("SVR Best params:", msvr.NSV/t)
train_RMSE_svr.append(train_rmse_svr)
test_RMSE_svr.append(test_rmse_svr)

end_time = time.time()
execution_time = end_time - start_time
tiempo_msvr.append(execution_time)


print("SVR Best params:", best_params)
print("VAR Train RMSE:", train_rmse_var, "Test RMSE:", test_rmse_var)
print("SVR Train RMSE:", train_rmse_svr, "Test RMSE:", test_rmse_svr)

#Pronóstico de persistencia


# Crear conjunto de entrenamiento y prueba
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Pronóstico de persistencia
def persistence_forecast(train, test):
    predictions = pd.concat([train.iloc[[-1]], test.iloc[:-1]])  # Persistencia para todas las columnas
    return predictions

# Generar predicciones
predictions = persistence_forecast(train, test)

# Evaluar el modelo
test_rmse_per = rmse(test.iloc[:, 1:3], predictions.iloc[:, 1:3])
print(test_rmse_per)

# Visualizar resultados
plt.figure(figsize=(12, 6))

# Serie 1 y Serie 2 en un solo gráfico
plt.plot(test.index, test['u_comp'], label='Serie 1 (Real)', color='blue')
plt.plot(test.index, predictions['u_comp'], label='u_comp', linestyle='dashed', color='orange')
plt.plot(test.index, test['v_comp'], label='Serie 2 (Real)', color='green')
plt.plot(test.index, predictions['v_comp'], label='Pronóstico Serie 2', linestyle='dashed', color='red')

plt.title('Series Originales y Pronóstico de Persistencia')
plt.legend()
plt.show()

import matplotlib.pyplot as plt

# Crear una figura con dos subgráficos (uno por serie)
plt.figure(figsize=(12, 6))

# Gráfico para Serie 1
plt.subplot(2, 1, 1)  # 2 filas, 1 columna, primer gráfico
plt.plot(test.index, test['u_comp'], label='Serie 1 (Real)', color='blue')
plt.plot(test.index, predictions['u_comp'], label='Pronóstico u_comp', linestyle='dashed', color='orange')
plt.title('Serie 1 - Pronóstico de Persistencia')
plt.legend()

# Gráfico para Serie 2
plt.subplot(2, 1, 2)  # 2 filas, 1 columna, segundo gráfico
plt.plot(test.index, test['v_comp'], label='v_comp (Real)', color='green')
plt.plot(test.index, predictions['v_comp'], label='Pronóstico v_comp', linestyle='dashed', color='red')
plt.title('Serie 2 - Pronóstico de Persistencia')
plt.legend()

# Ajuste del diseño para que no se superpongan los elementos
plt.tight_layout()

# Mostrar la gráfica
plt.show()
