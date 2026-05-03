library(NTS)
library(mtarm)
library(tsDyn)
library(Metrics)
library(openxlsx)
source("C:/Users/huma1003/OneDrive - NIQ/DOCUMENTOS IMPORTANTES/Mis_Cosas/MarlijarTM/Rolling MSETAR.R")


#Crear y guardar la serie

# Parámetros
long <- c( 200,500,1000)  # Longitud de la serie
h.ahead <-1 # Número de pasos a predecir
k <- 2     #Número de variables
iterations=300
p=1 #retardos
#Matrices de coeficientes del primer y segundo régimen
#ars <- list(p=c(1,1),q=c(0,0),d=c(0,0)) 
#phi1=matrix(c(-1.5,0,0,0.8),k,k)
#phi2=matrix(c(0.6,0,0,-1.6),k,k)
phi1=matrix(c(-1.5,0,0,0.8),k,k)
phi2=matrix(c(0.6,0,0,-1.6),k,k)

#Matrices de covarianza,
sigma1=matrix(c(1,0,0,1),2,2)
sigma2=matrix(c(1,0,0,1),2,2)
#Vectores de constantes
c1=c(0,0)
c2=c(0,0)
delay=c(0,0) # El retardo que se aplica para la variable umbral que define el paso de un régimen a otro.
Trim=c(0.2,0.8) #Los puntos de corte para definir el umbral (utilizados para asegurarse de que la simulación no pase entre regímenes muy frecuentemente).
umbral=0 # Umbral que determina cuándo se cambia de un régimen al otro

results_df <- data.frame()

# Carpeta y archivo Excel
folder_path <- "C:/Users/huma1003/OneDrive - NIQ/DOCUMENTOS IMPORTANTES/Mis_Cosas/MarlijarTM/msvr-master/MSETAR/Data_Mario_2"
if (!dir.exists(folder_path)) {
  dir.create(folder_path, recursive = TRUE)
}

# Generar series y guardar en Excel
for (size in long) {
  excel_filename <- paste0(folder_path, "/dataset_size_", size, ".xlsx")
  
  wb <- createWorkbook()
  
  for (no in 0:99) {
    print(paste("Tamaño:", size, "Iteración:", no))
    
    y=mTAR.sim(size+h.ahead,thr=umbral,phi1,phi2,sigma1,sigma2,c1,c2,delay,ini=500)
    series_df=y$series
    colnames(series_df) <- paste0("Y", 1:k)
    #plot(as.ts(datos[,1:k]))
    # Añadir los datos a una nueva hoja
    sheet_name <- paste0("Iter_", no)
    addWorksheet(wb, sheet_name)
    writeData(wb, sheet_name, series_df)
  }
  # Guardar el archivo Excel
  saveWorkbook(wb, excel_filename, overwrite = TRUE)
}

###Lectura y ajuste de modelo
wb <- createWorkbook()
addWorksheet(wb, "Resultados")
# Función para cargar datos
#long<-1000
load_data <- function(size, iteration) {
  file_path <- file.path(folder_path, paste0("dataset_size_", size, ".xlsx"))
  sheet_name <- paste0("Iter_", iteration)
  data <- read.xlsx(file_path, sheet = sheet_name)
  return(data)
}

train_RMSE_msetar <- vector("list", length(t))
test_RMSE_msetar<- vector("list", length(t))
tiempo_msetar <- vector("list", length(t))
cov_matrices <- list()
predicciones <- list()
predicciones_msetar <- list()
rmse_msetar <- list()

for (size in long) {
  # Inicializar listas para este tamaño de muestra
  predicciones_msetar[[as.character(size)]] <- vector("list", 100)
  rmse_msetar[[as.character(size)]] <- vector("list", 100)
  train_RMSE_msetar[[as.character(size)]] <- vector("list", 100)
  test_RMSE_msetar[[as.character(size)]] <- vector("list", 100)
  tiempo_msetar[[as.character(size)]] <- vector("list", 100)
  
  for (no in 0:99) {
    print(paste("Tamaño:", size))
    
    # Cargar la serie
    series_df <- load_data(size, no)
    data_ts<-datos
    # Dividir en train y test
    train_size <- floor(nrow(data_ts) * 0.7)
    train <- data_ts[1:train_size, ]
    test <- data_ts[(train_size + 1):nrow(data), ]
    
    start_time <- Sys.time()
    
    # Ajustar el modelo MSETAR
    threshold_mean <- mean(data_ts, na.rm = TRUE)
    threshold_median <- median(data_ts, na.rm = TRUE)
    table(ifelse(data_ts > -2-2, "Régimen 1", "Régimen 2"))  # Cambia "threshold" por el umbral que estés usando.
    
    
    est <- mTAR(train[1:train_size, ], 2, 2, 0, train[1:train_size, 1], delay, Trim, iterations, include.mean = TRUE, "AIC")
    # Extraer las matrices de coeficientes de los regímenes
    residuals <- est$residuals
    #cov_matrix <- cov(residuals)
    #cov_matrices[[paste("size", size, "iter", no + 1, sep = "_")]] <- cov_matrix
    
    # Predicciones para train
    #print("Predicciones para train")
    #train_pred <- numeric(nrow(train) - p) #Asi lo estaba haciendo antes
    #train_prediccion<-mTAR.pred.rolling(est,h = 1,iterations = 200,ci = 0.95,roll_steps = 1,test_set = as.matrix(train))
    #train_pred=train_prediccion$preds
    #valores_extraidos <- list()
    #for (i in seq_along(train_pred)) {
    # valores_extraidos[[i]] <- as.vector(train_pred[[i]])
    #}
    #valores_df <- do.call(rbind, valores_extraidos)
    #train_pred<- as.data.frame(valores_df)
    
    # Separar las predicciones por régimen basando en el umbral que es cero
    #regimen <- ifelse(train_pred[, 1] > umbral, 1, 2)
    
    #pred_regimen_1 <- train_pred[regimen == 1, ]  
    #pred_regimen_2 <- train_pred[regimen== 2, ]
    
    #predicciones_msetar[[as.character(size)]][[no + 1]] <- list(
    # regimen_1 = pred_regimen_1,
    #regimen_2 = pred_regimen_2
    #)
    
    # Valores reales por régimen
    #actual_regimen_1 <- train[regimen == 1, ]
    #actual_regimen_2 <- train[regimen == 2, ]
    
    
    # Calcular el RMSE para cada régimen
    #rmse <- function(y_actual, y_pred) {
    # sqrt(mean((y_actual - y_pred)^2))
    #}
    
    #rmse_regimen_1 <- rmse(actual_regimen_1[, 1], pred_regimen_1[, 1])
    #rmse_regimen_2 <- rmse(actual_regimen_2[, 1], pred_regimen_2[, 1])
    
    # Guardar el RMSE para cada régimen
    #rmse_msetar[[as.character(size)]][[no + 1]] <- list(
    # regimen_1 = rmse_regimen_1,
    #regimen_2 = rmse_regimen_2)
    
    #for (i in (p + 1):nrow(train)) {
    #pred <- mTAR.pred(est, orig = i, h = 1, iterations = 300, ci = 0.95, TRUE)
    #train_pred[i - p] <- pred$pred  # Extraer la predicción
    #}
    
    # Predicciones para test
    test_pred <- numeric(nrow(test))
    #origg <- tail(train, 1)  # Última fila de train
    test_pred<-mTAR.pred.rolling(est,h = 1,iterations = 50,ci = 0.95,roll_steps = 1,test_set = as.matrix(test))
    train_combined <- cbind(train_data, train_data+residuals)
    test_combined <- cbind(test_data, test_pred)
    
    
    # Calcular RMSE
    #rmse_var1 <- rmse(train[, 1], as.data.frame(train_pred)[, 1])
    #rmse_var2 <- rmse(train[, 2], as.data.frame(train_pred)[, 2]) 
    train_rmse_vec <- sqrt(mean((residuals[, 1])^2 + (residuals[,2])^2))
    
    #train_values <- as.vector(train[(p + 1):nrow(train), 1])  
    #train_rmse_vec <- rmse(train_values, train_pred)
    
    test_predd <- matrix(ncol = 2, nrow = nrow(test))
    for (i in 1:nrow(test)) {
      test_predd[i, 1] <- test_pred$preds[[i]][1]  
      test_predd[i, 2] <- test_pred$preds[[i]][2]  
    }
    test_rmse_vec <- sqrt(mean((test[, 1] - test_predd[, 1])^2 + (test[, 2] - test_predd[, 2])^2))
    #Guardar la predicción en test y train
    
    
    # Guardar los resultados de RMSE y tiempo
    train_RMSE_msetar[[as.character(size)]][[no + 1]] <- train_rmse_vec
    test_RMSE_msetar[[as.character(size)]][[no + 1]] <- test_rmse_vec
    
    end_time <- Sys.time()
    tiempo_msetar[[as.character(size)]][[no + 1]] <- as.numeric(difftime(end_time, start_time, units = "secs"))
    
    print("Terminé de ajustar el modelo MSETAR")
    
  }
  
  # Añadir resultados de este tamaño al dataframe general
  for (no in 1:100) {
    results_df <- rbind(results_df, data.frame(
      Size = size,
      Iteracion = no - 1,
      Train_RMSE = train_RMSE_msetar[[as.character(size)]][[no]],
      Test_RMSE = test_RMSE_msetar[[as.character(size)]][[no]],
      Tiempo = tiempo_msetar[[as.character(size)]][[no]]
    ))
  }
}

# Escribir el dataframe acumulado al Excel
writeData(wb, sheet = "Resultados", x = results_df)

# Guardar el libro de trabajo
output_file <- "C:/Users/huma1003/OneDrive - NIQ/DOCUMENTOS IMPORTANTES/Mis_Cosas/MarlijarTM/msvr-master/MSETAR/Resultados_msetar_data_3.xlsx"
saveWorkbook(wb, output_file, overwrite = TRUE)

print("Resultados guardados en el archivo Excel.")

#no lo use
#backTAR(est, orig=50, h = 1, iter = 1)

#Extraer la info a csv para hacer boxplot
rmse_df <- data.frame(iteracion = integer(), 
                      RMSE_variable_1 = numeric(), 
                      RMSE_variable_2 = numeric())

for (i in 1:100) {
  # Extraer RMSE para cada variable
  rmse_var1 <- rmse_msetar$`200`[[i]]$regimen_1
  rmse_var2 <- rmse_msetar$`200`[[i]]$regimen_2 
  
  # Añadir los valores al data frame
  rmse_df <- rbind(rmse_df, data.frame(iteracion = i, 
                                       RMSE_variable_1 = rmse_var1, 
                                       RMSE_variable_2 = rmse_var2))
}

write.csv(test_combined,"test_msetar.csv")

# Comprobar residuos
residuals <- residuals(est)

# Test de autocorrelación de residuos
library(tseries)
Box.test(residuals[,1], lag = 20, type = "Ljung-Box")
Box.test(residuals[,2], lag = 20, type = "Ljung-Box")

# Graficar residuos
plot(residuals, main = "Residuos del modelo MSETAR")
x_squared_value <- 35.44
df <- 20
p_value <- 1 - pchisq(x_squared_value, df)
print(p_value)



#Prueba Diebold Mariano
# Instalar y cargar el paquete forecast si no está instalado

