library(NTS)
library(mtarm)
library(tsDyn)
library(Metrics)
library(openxlsx)
# It is necessary to load the rolling MSETAR forecasting functions before running this script.
source("XXXX.R")

# Generate and save synthetic datasets

# Parameters

long <- c(200, 500, 1000, 5000)   # Series length
h.ahead <- 1                       # Forecast horizon
k <- 2                             # Number of variables
iterations <- 300                  # Number of simulation replications
p <- 1                             # Number of lags

# Coefficient matrices for the first and second regimes
phi1 <- matrix(c(0.5,0.7,0.3,0.2),k,k)
phi2 <- matrix(c(-0.4,-0.6,-0.5,0.5),k,k)

# Innovation covariance matrices
sigma1 <- matrix(c(1,0,0,1),2,2)
sigma2 <- matrix(c(1,0,0,1),2,2)

# Constant vectors
c1 <- c(0,0)
c2 <- c(0,0)

delay <- c(1,1)      # Delay applied to the threshold variable used for regime switching

Trim <- c(0.2,0.8)   # Trimming bounds used to prevent excessively frequent regime transitions

umbral <- 0          # Threshold value determining the active regime

results_df <- data.frame()

# Output folder
folder_path <- "..."

# Create folder if it does not exist
if (!dir.exists(folder_path)) {
  dir.create(folder_path, recursive = TRUE)
}

# Generate synthetic series and save them to Excel
for (size in long) {

  excel_filename <- paste0(folder_path, "/dataset_size_", size, ".xlsx")

  wb <- createWorkbook()

  for (no in 0:99) {

    print(paste("Sample Size:", size, "Iteration:", no))

    y <- mTAR.sim(size + h.ahead,
                  thr = umbral,
                  phi1,
                  phi2,
                  sigma1,
                  sigma2,
                  c1,
                  c2,
                  delay,
                  ini = 500)

    series_df <- y$series
    colnames(series_df) <- paste0("Y", 1:k)

    # Add data to a new worksheet
    sheet_name <- paste0("Iter_", no)

    addWorksheet(wb, sheet_name)
    writeData(wb, sheet_name, series_df)
  }

  # Save Excel workbook
  saveWorkbook(wb, excel_filename, overwrite = TRUE)
}

### Data Loading and Model Estimation

wb <- createWorkbook()
addWorksheet(wb, "Results")

# Function to load a dataset
load_data <- function(size, iteration) {
  file_path <- file.path(folder_path, paste0("dataset_size_", size, ".xlsx"))
  sheet_name <- paste0("Iter_", iteration)
  data <- read.xlsx(file_path, sheet = sheet_name)
  return(data)
}

# Run this part to obtain MSETAR forecasts.

train_RMSE_msetar <- vector("list", length(t))
test_RMSE_msetar<- vector("list", length(t))
tiempo_msetar <- vector("list", length(t))
cov_matrices <- list()
predicciones <- list()
predicciones_msetar <- list()
rmse_msetar <- list()



for (size in long) {
  predicciones_msetar[[as.character(size)]] <- vector("list", 100)
  rmse_msetar[[as.character(size)]] <- vector("list", 100)
  train_RMSE_msetar[[as.character(size)]] <- vector("list", 100)
  test_RMSE_msetar[[as.character(size)]] <- vector("list", 100)
  tiempo_msetar[[as.character(size)]] <- vector("list", 100)
  
  for (no in 0:99) {
    print(paste("Tamaño:", size))
    
   
    series_df <- load_data(size, no)
    
    
    train_size <- floor(nrow(data_ts) * 0.7)
    train <- data_ts[1:train_size, ]
    test <- data_ts[(train_size + 1):nrow(data), ]
    
    start_time <- Sys.time()
    
    
    threshold_mean <- mean(data_ts, na.rm = TRUE)
    threshold_median <- median(data_ts, na.rm = TRUE)
    table(ifelse(data_ts > -2-2, "Régimen 1", "Régimen 2"))  
    
    
        est <- mTAR(train[1:train_size, ], 2, 2, -2.2, train[1:train_size, 1], delay, Trim, iterations, include.mean = TRUE, "AIC")
        
        residuals <- est$residuals
        cov_matrix <- cov(residuals)
        cov_matrices[[paste("size", size, "iter", no + 1, sep = "_")]] <- cov_matrix
        
        #Training Set Predictions
        
        train_prediccion<-mTAR.pred.rolling(est,h = 1,iterations = 200,ci = 0.95,roll_steps = 1,test_set = as.matrix(train))
        train_pred=train_prediccion$preds
        valores_extraidos <- list()
        for (i in seq_along(train_pred)) {
          valores_extraidos[[i]] <- as.vector(train_pred[[i]])
        }
        valores_df <- do.call(rbind, valores_extraidos)
        train_pred<- as.data.frame(valores_df)
        
        regimen <- ifelse(train_pred[, 1] > umbral, 1, 2)
        pred_regimen_1 <- train_pred[regimen == 1, ]
        pred_regimen_2 <- train_pred[regimen == 2, ]
        actual_regimen_1 <- train[regimen == 1, ]
        actual_regimen_2 <- train[regimen == 2, ]
        
        rmse <- function(y_actual, y_pred) {
          sqrt(mean((y_actual - y_pred)^2))
          }
          rmse_regimen_1 <- rmse(actual_regimen_1[,1], pred_regimen_1[,1])
          rmse_regimen_2 <- rmse(actual_regimen_2[,1], pred_regimen_2[,1])
        
        
  
        # Guardar el RMSE para cada régimen
        rmse_msetar[[as.character(size)]][[no + 1]] <- list(
         regimen_1 = rmse_regimen_1,
         regimen_2 = rmse_regimen_2)
        
        for (i in (p + 1):nrow(train)) {
         pred <- mTAR.pred(est, orig = i, h = 1, iterations = 300, ci = 0.95, TRUE)
          train_pred[i - p] <- pred$pred  # Extraer la predicción
        }
        
        # Predicciones para test
        test_pred <- numeric(nrow(test))
        test_pred<-mTAR.pred.rolling(est,h = 1,iterations = 50,ci = 0.95,roll_steps = 1,test_set = as.matrix(test))
        train_combined <- cbind(train_data, train_data+residuals)
        test_combined <- cbind(test_data, test_pred)
        
        
        # Calcular RMSE
         train_rmse_vec <- sqrt(mean((residuals[, 1])^2 + (residuals[,2])^2))
    
        
        
        test_predd <- matrix(ncol = 2, nrow = nrow(test))
        for (i in 1:nrow(test)) {
          test_predd[i, 1] <- test_pred$preds[[i]][1]  
          test_predd[i, 2] <- test_pred$preds[[i]][2]  
        }
        test_rmse_vec <- sqrt(mean((test[, 1] - test_predd[, 1])^2 + (test[, 2] - test_predd[, 2])^2))
        
        train_RMSE_msetar[[as.character(size)]][[no + 1]] <- train_rmse_vec
        test_RMSE_msetar[[as.character(size)]][[no + 1]] <- test_rmse_vec
        
        end_time <- Sys.time()
        tiempo_msetar[[as.character(size)]][[no + 1]] <- as.numeric(difftime(end_time, start_time, units = "secs"))
        
        print("Terminé de ajustar el modelo MSETAR")
        
  }
  
  # Append results for the current sample size to the overall data frame
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



# Export the aggregated data frame to Excel
writeData(wb, sheet = "Resultados", x = results_df)



# Save
output_file <- "XXX.xlsx"
saveWorkbook(wb, output_file, overwrite = TRUE)
print("Finish")




