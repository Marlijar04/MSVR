library(mtarm)
library(NTS)
library(openxlsx)
library(mtarm)
library(tsDyn)
library(Metrics)

#-----------------------------
# Data path
#-----------------------------

folder_path <- "XXX"
file_path<-"XXX.xlsx"

mTAR.pred<-function (model, orig, h = 1, iterations = 3000, ci = 0.95, output = TRUE) 
{
  y <- model$data
  arorder <- model$arorder
  beta <- model$beta
  sigma <- model$sigma
  thr <- model$thr
  include.mean <- model$cnst
  delay <- model$delay
  p <- max(arorder)
  k <- length(arorder)
  d <- delay[2]
  nT <- nrow(y)
  ky <- ncol(y)
  if (orig < 1) 
    orig <- nT
  if (orig > nT) 
    orig <- nT
  if (h < 1) 
    h <- 1
  Sigh <- NULL
  for (j in 1:k) {
    sig <- sigma[, ((j - 1) * ky + 1):(j * ky)]
    m1 <- eigen(sig)
    P <- m1$vectors
    Di <- diag(sqrt(m1$values))
    sigh <- P %*% Di %*% t(P)
    Sigh <- cbind(Sigh, sigh)
  }
  Ysim <- array(0, dim = c(h, ky, iterations))
  for (it in 1:iterations) {
    yp <- y[1:orig, ]
    et <- matrix(rnorm(h * ky), h, ky)
    for (ii in 1:h) {
      t <- orig + ii
      thd <- yp[(t - d), delay[1]]
      JJ <- 1
      for (j in 1:(k - 1)) {
        if (thd > thr[j]) {
          JJ <- j + 1
        }
      }
      Jst <- (JJ - 1) * ky
      at <- matrix(et[ii, ], 1, ky) %*% Sigh[, (Jst + 1):(Jst + ky)]
      x <- NULL
      if (include.mean[JJ]) 
        x <- 1
      pJ <- arorder[JJ]
      phi <- beta[, (Jst + 1):(Jst + ky)]
      for (i in 1:pJ) {
        x <- c(x, yp[(t - i), ])
      }
      yhat <- matrix(x, 1, length(x)) %*% phi[1:length(x), ]
      yhat <- yhat + at
      yp <- rbind(yp, yhat)
      Ysim[ii, , it] <- yhat
    }
  }
  pred <- NULL
  upp <- NULL
  low <- NULL
  pr <- (1 - ci)/2
  pro <- c(pr, 1 - pr)
  for (ii in 1:h) {
    fst <- NULL
    lowb <- NULL
    uppb <- NULL
    for (j in 1:ky) {
      ave <- mean(Ysim[ii, j, ])
      quti <- quantile(Ysim[ii, j, ], prob = pro)
      fst <- c(fst, ave)
      lowb <- c(lowb, quti[1])
      uppb <- c(uppb, quti[2])
    }
    pred <- rbind(pred, fst)
    low <- rbind(low, lowb)
    upp <- rbind(upp, uppb)
  }
  if (output) {
    colnames(pred) <- colnames(y)
    cat("Forecast origin: ", orig, "\n")
    cat("Predictions: 1-step to ", h, "-step", "\n")
    print(pred)
    cat("Lower bounds of ", ci * 100, " % confident intervals", "\n")
    print(low)
    cat("Upper bounds: ", "\n")
    print(upp)
  }
  mTAR.pred <- list(data = y, pred = pred, Ysim = Ysim)
}

mTAR.pred.rolling <- function(model, h = 1, iterations = 3000, ci = 0.95, output = TRUE, roll_steps = 1, test_set=NULL) {

  model=est
  test_set=as.matrix(test)

  y <- rbind(model$data,test_set)
  y_test<-test_set
  y_train=model$data  # Original data

  arorder <- model$arorder
  beta <- model$beta
  sigma <- model$sigma
  thr <- model$thr
  include.mean <- model$cnst
  delay <- model$delay

  p <- max(arorder)
  k <- length(arorder)
  d <- delay[2]

  nT <- nrow(y_train)  # Training set size
  ky <- ncol(y_train)
  nTest <- nrow(y_test)
  orig<-nT

  # Ensure that orig is within the valid range
  h=1
  if (h < 1) h <- 1

  # Covariance matrices
  Sigh <- NULL
  for (j in 1:k) {
    sig <- sigma[, ((j - 1) * ky + 1):(j * ky)]
    m1 <- eigen(sig)
    P <- m1$vectors
    Di <- diag(sqrt(m1$values))
    sigh <- P %*% Di %*% t(P)
    Sigh <- cbind(Sigh, sigh)
  }

  # Matrices to store rolling forecasts
  all_preds <- list()  
  all_lows <- list()   
  all_upps <- list()   

  # Rolling forecast
  current_orig <- orig

  while (current_orig <= (nT+ nTest - h)) {

    Ysim <- array(0, dim = c(h, ky, iterations))

    # Monte Carlo simulation for each rolling origin
    for (it in 1:iterations) {

      yp <- y[1:current_orig, ]  # Initial values up to the current forecast origin
      et <- matrix(rnorm(h * ky), h, ky)  # Random errors for each forecast horizon

      for (ii in 1:h) {

        t <- current_orig + ii

        thd <- yp[(t - d), delay[1]]  # Determines the threshold for regime switching

        JJ <- 1
        for (j in 1:(k - 1)) {
          if (thd > thr[j]) {
            JJ <- j + 1
          }
        }

        Jst <- (JJ - 1) * ky

        at <- matrix(et[ii, ], 1, ky) %*% Sigh[, (Jst + 1):(Jst + ky)]  # Random innovation

        # AR variables, including the mean if specified
        x <- NULL
        if (include.mean[JJ]) x <- 1

        pJ <- arorder[JJ]
        phi <- beta[, (Jst + 1):(Jst + ky)]

        # Compute AR lag terms
        for (i in 1:pJ) {
          x <- c(x, yp[(t - i), ])
        }

        # Forecast
        yhat <- matrix(x, 1, length(x)) %*% phi[1:length(x), ]
        yhat <- yhat + at

        yp <- rbind(yp, y[current_orig+ii, ])  # Add the forecast to the historical data

        Ysim[ii, , it] <- yhat  # Store the simulated forecast
      }
    }

    # Compute the mean forecast and confidence intervals
    pred <- NULL
    upp <- NULL
    low <- NULL
    pr <- (1 - ci) / 2
    pro <- c(pr, 1 - pr)

    for (ii in 1:h) {
      fst <- NULL
      lowb <- NULL
      uppb <- NULL
      for (j in 1:ky) {
        ave <- mean(Ysim[ii, j, ])
        quti <- quantile(Ysim[ii, j, ], prob = pro)
        fst <- c(fst, ave)
        lowb <- c(lowb, quti[1])
        uppb <- c(uppb, quti[2])
      }
      pred <- rbind(pred, fst)
      low <- rbind(low, lowb)
      upp <- rbind(upp, uppb)
    }

    # Store results from the current rolling step
    all_preds[[length(all_preds) + 1]] <- pred
    all_lows[[length(all_lows) + 1]] <- low
    all_upps[[length(all_upps) + 1]] <- upp

    # Print results if output = TRUE
    if (output) {
      colnames(pred) <- colnames(y)
      cat("Rolling forecast origin: ", current_orig, "\n")
      cat("Predictions: 1-step to ", h, "-step\n")
      print(pred)
      cat("Lower bounds of ", ci * 100, " % confident intervals\n")
      print(low)
      cat("Upper bounds: \n")
      print(upp)
    }

    # Move the forecast origin according to roll_steps
    current_orig <- current_orig + roll_steps
  }

  # Return the cumulative results from all rolling forecasts
  mTAR.pred.rolling <- list(preds = all_preds, lows = all_lows, upps = all_upps)

  return(mTAR.pred.rolling)
}

library(readxl)
data= read_excel(file_path)
data= data[,-1]

# Folder to save forecasts
output_folder <- file.path(folder_path,"Predicciones_MSETAR")

train_size <- floor(0.70*nrow(data))
train <- data[1:train_size,]
test <- data[(train_size+1):nrow(data),]

delay = c(2,2)
Trim = c(0.2,0.8)

est <- mTAR(
      train,
      2,
      2,
      0,
      train[,1],
      delay,
      Trim,
      iterations,
      include.mean=TRUE,
      "AIC"
    )

# ==============================================================================
# 1. FORECASTING WITH A FIXED SEED (REPRODUCIBILITY)
# ==============================================================================

set.seed(42)  

pred <- mTAR.pred.rolling(
  model = est,
  h = 1,
  iterations = 50,
  ci = 0.95,
  output = FALSE,
  roll_steps = 1,
  test_set = as.matrix(test)
)

#-----------------------------
# Convert forecasts to a matrix
#-----------------------------

pred_matrix <- matrix(
  NA,
  nrow = length(pred$preds),
  ncol = ncol(test)
)

for(i in 1:length(pred$preds)){
  pred_matrix[i, ] <- pred$preds[[i]][1, ]
}

pred_df <- as.data.frame(pred_matrix)

# ==============================================================================
# 2. EXPORT RESULTS ALIGNED WITH THE PYTHON FORMAT
# ==============================================================================

# Use [[ ]] to extract the raw vector from the tibble and as.numeric for safety
resultado <- data.frame(
  Real_u_comp = as.numeric(test[[1]]),
  Real_v_comp = as.numeric(test[[2]]),
  Pred_u_comp = as.numeric(pred_df[[1]]),
  Pred_v_comp = as.numeric(pred_df[[2]])
)

# Export to Excel 
write.xlsx(
  resultado,
  file.path("~/Downloads/MSVR-main_Vec", "test_MSETAR_alineado.xlsx"),
  rowNames = FALSE
)

# ==============================================================================
# 3. METRIC CALCULATION (CONSISTENT WITH PYTHON)
# ==============================================================================

# RMSE for each component separately
rmse_u <- rmse(resultado$Real_u_comp, resultado$Pred_u_comp)
rmse_v <- rmse(resultado$Real_v_comp, resultado$Pred_v_comp)

# Global RMSE (same as scikit-learn: the square root of the average MSE across all elements)
matriz_real <- as.matrix(resultado[, c("Real_u