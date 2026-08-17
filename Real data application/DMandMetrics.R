# ==============================================================================
# COMPLETE FORECAST EVALUATION AND DIEBOLD-MARIANO TESTING PIPELINE IN R
# ==============================================================================

# Load and install required libraries
if (!require("readxl")) install.packages("readxl")
if (!require("forecast")) install.packages("forecast")
if (!require("dplyr")) install.packages("dplyr")

library(readxl)
library(forecast)
library(dplyr)

# ------------------------------------------------------------------------------
# Data Loading and Preparation from Excel
# ------------------------------------------------------------------------------
ruta_excel <- "consolidado_test_todos.xlsx"

df_raw <- read_excel(
  path = ruta_excel, 
  sheet = "Val_and_Preds_in_Test", 
  skip = 1, 
  col_names = FALSE
)

colnames(df_raw) <- c(
  "real_u", "real_v",
  "VAR_u",  "VAR_v",
  "MSVR_u", "MSVR_v",
  "PERS_u", "PERS_v",
  "MSETAR_u", "MSETAR_v"
)

# Character cleanup
df <- df_raw %>%
  mutate(across(everything(), ~ {
    x <- as.character(.)
    x <- gsub("'", "", x)
    x <- gsub(",", ".", x)
    as.numeric(trimws(x))
  }))

# VERIFICATION: Display the first loaded row
head(df, 1)


modelos <- c( "VAR","MSVR","PERS", "MSETAR" )
pares <- combn(modelos, 2, simplify = FALSE)  
# ------------------------------------------------------------------------------
#  STAGE 1: COMPUTATION OF FORECAST ERROR METRICS (MAE AND RMSE)
# ------------------------------------------------------------------------------
N <- nrow(df)

resultados_metricas <- list()

for (m in modelos) {

  e_u <- df$real_u - df[[paste0(m, "_u")]]
  e_v <- df$real_v - df[[paste0(m, "_v")]]
  
  # --- MAE CALCULATION ---
  mae_u <- mean(abs(e_u))
  mae_v <- mean(abs(e_v))
  mae_conjunto <- (mae_u + mae_v) / 2
  
  # --- RMSE CALCULATION ---
  rmse_u <- sqrt(mean(e_u^2))
  rmse_v <- sqrt(mean(e_v^2))
  rmse_conjunto <- sqrt((sum(e_u^2) + sum(e_v^2)) / (2 * N))
  

  resultados_metricas[[m]] <- data.frame(
    Modelo        = m,
    MAE_u         = round(mae_u, 5),
    MAE_v         = round(mae_v, 5),
    MAE_Conjunto  = round(mae_conjunto, 5),
    RMSE_u        = round(rmse_u, 5),
    RMSE_v        = round(rmse_v, 5),
    RMSE_Conjunto = round(rmse_conjunto, 5)
  )
}


tabla_metricas_final <- do.call(rbind, resultados_metricas)
# ------------------------------------------------------------------------------
# STAGE 2: JOINT DIEBOLD-MARIANO TEST (BIVARIATE EUCLIDEAN LOSS)
# ------------------------------------------------------------------------------
perdidas_conjuntas <- list()
for (m in modelos) {
  e_u <- df$real_u - df[[paste0(m, "_u")]]
  e_v <- df$real_v - df[[paste0(m, "_v")]]
  perdidas_conjuntas[[m]] <- e_u^2 + e_v^2
}

res_dm_conjunto <- data.frame()
for (par in pares) {
  m1 <- par[1]; m2 <- par[2]
  
  dm_out <- dm.test(e1 = perdidas_conjuntas[[m1]], e2 = perdidas_conjuntas[[m2]], 
                    h = 1, power = 1, alternative = "two.sided")
  
  res_dm_conjunto <- rbind(res_dm_conjunto, data.frame(
    Modelo_1 = m1, Modelo_2 = m2,
    DM_Stat = as.numeric(dm_out$statistic),
    P_Valor = as.numeric(dm_out$p.value)
  ))
}

res_dm_conjunto$P_Valor_Ajustado <- p.adjust(res_dm_conjunto$P_Valor, method = "BH")
res_dm_conjunto$Ganador <- ifelse(
  res_dm_conjunto$P_Valor_Ajustado < 0.05,
  ifelse(res_dm_conjunto$DM_Stat < 0, res_dm_conjunto$Modelo_1, res_dm_conjunto$Modelo_2),
  "Empate"
)

# ------------------------------------------------------------------------------
# STAGE 3: UNIVARIATE DIEBOLD-MARIANO TESTS (u_comp and v_comp)
# ------------------------------------------------------------------------------
evaluar_dm_componente <- function(comp) {
  res_comp <- data.frame()
  real_col <- paste0("real_", comp)
  
  for (par in pares) {
    m1 <- par[1]; m2 <- par[2]
    e1 <- df[[real_col]] - df[[paste0(m1, "_", comp)]]
    e2 <- df[[real_col]] - df[[paste0(m2, "_", comp)]]
    
    dm_out <- dm.test(e1 = e1, e2 = e2, h = 1, power = 2, alternative = "two.sided")
    
    res_comp <- rbind(res_comp, data.frame(
      Componente = comp, Modelo_1 = m1, Modelo_2 = m2,
      DM_Stat = as.numeric(dm_out$statistic),
      P_Valor = as.numeric(dm_out$p.value)
    ))
  }
  
  res_comp$P_Valor_Ajustado <- p.adjust(res_comp$P_Valor, method = "BH")
  res_comp$Ganador <- ifelse(
    res_comp$P_Valor_Ajustado < 0.05,
    ifelse(res_comp$DM_Stat < 0, res_comp$Modelo_1, res_comp$Modelo_2),
    "Empate"
  )
  return(res_comp)
}

res_dm_u <- evaluar_dm_componente("u")
res_dm_v <- evaluar_dm_componente("v")

# ------------------------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------------------------
cat("\n=======================================================\n")
cat(" 1. TABLA DE MÉTRICAS PUNTUALES DE ERROR (MSE Y MAPE)\n")
cat("=======================================================\n")
print(tabla_metricas_final)

cat("\n=======================================================\n")
cat(" 2. DIEBOLD-MARIANO: PRUEBA CONJUNTA (BIVARIADA)\n")
cat("=======================================================\n")
print(res_dm_conjunto)

cat("\n=======================================================\n")
cat(" 3. DIEBOLD-MARIANO: COMPONENTE U (u_comp)\n")
cat("=======================================================\n")
print(res_dm_u)

cat("\n=======================================================\n")
cat(" 4. DIEBOLD-MARIANO: COMPONENTE V (v_comp)\n")
cat("=======================================================\n")
print(res_dm_v)


library(dplyr)
library(tidyr)
library(ggplot2)


df_errores_long <- df %>%
  transmute(
    MSVR_u   = real_u - MSVR_u,
    MSVR_v   = real_v - MSVR_v,
    VAR_u    = real_u - VAR_u,
    VAR_v    = real_v - VAR_v,
    MSETAR_u = real_u - MSETAR_u,
    MSETAR_v = real_v - MSETAR_v,
    PERS_u   = real_u - PERS_u,
    PERS_v   = real_v - PERS_v
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = c("Modelo", "Componente"),
    names_sep = "_",
    values_to = "Error"
  )


library(dplyr)
library(tidyr)
library(ggplot2)


df_errores_long <- df %>%
  transmute(
    MSVR_u   = real_u - MSVR_u,
    MSVR_v   = real_v - MSVR_v,
    VAR_u    = real_u - VAR_u,
    VAR_v    = real_v - VAR_v,
    MSETAR_u = real_u - MSETAR_u,
    MSETAR_v = real_v - MSETAR_v,
    PERS_u   = real_u - PERS_u,
    PERS_v   = real_v - PERS_v
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = c("Modelo", "Componente"),
    names_sep = "_",
    values_to = "Error"
  )

df_errores_long$Modelo <- factor(df_errores_long$Modelo, levels = c("MSVR", "VAR", "MSETAR", "PERS"))
df_errores_long$Componente <- factor(df_errores_long$Componente, levels = c("u", "v"))


figura_boxplot <- ggplot(df_errores_long, aes(x = Modelo, y = Error, fill = Componente)) +
  geom_boxplot(outlier.size = 0.8, outlier.alpha = 0.25, alpha = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 0.7) +
  facet_wrap(~ Componente) +
  
  
  coord_cartesian(ylim = c(-25, 25)) +
  scale_y_continuous(breaks = seq(-25, 25, by = 5)) +
  
  scale_fill_manual(values = c("u" = "#2b5c8f", "v" = "#d95f02")) +
  theme_bw(base_size = 12) +
  labs(
    title = "Diagrama de Cajas (Box-Plot) de los Errores de Predicción",
    subtitle = "Comparación de distribución y dispersión por Modelo y Componente",
    x = "Modelo de Pronóstico",
    y = "Error de Predicción (Real - Predicho)",
    fill = "Componente"
  ) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11, color = "gray30"),
    strip.text = element_text(face = "bold", size = 12),
    strip.background = element_rect(fill = "gray90"),
    legend.position = "bottom"
  )


print(figura_boxplot)
