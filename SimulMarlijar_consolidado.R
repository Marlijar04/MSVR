library(mtarm)
library(GIGrvg)
library(Formula)
library(Rfast)
source("C:/Users/huma1003/OneDrive - NIQ/DOCUMENTOS IMPORTANTES/Mis_Cosas/MarlijarTM/Rolling MSETAR.R")

calen=100
long=500
h.ahead=10
Tlen = long+calen+h.ahead
inic<-calen

k = 1
v=0


dist <- "Gaussian"
#extra=6
#delta=2
delay <- 1
#ars <- list(p=c(1,2),q=c(1,0),d=c(0,0))
#ars_3reg <- list(p=c(1,2,1),q=c(1,0,0),d=c(0,0,0))
#ars_var<-list(p=c(1))

ars <- list(p=c(1,1),q=c(0,0),d=c(0,0))

#ars_2orders <- list(p=c(2,2),q=c(0,0),d=c(0,0))
#ars_3orders<-list(p=c(3,3),q=c(0,0),d=c(0,0))
Intercept <- TRUE
### R1 regimen ====
#Location_R1 = list(phi1 = matrix(c(0.1,0.6,0.4,-0.4,0.5,-0.7,0.2,0.6,-0.3),k,k,byrow = TRUE))
#Location_R1 = list(phi1 = matrix(c(0.1,0.6,0.4,-0.4,0.5,-0.7,0.2,0.6,-0.3),k,k,byrow = TRUE),beta1=matrix(c(0.6,-0.5,-0.4,0.6,0.1,0.3),k,v,byrow = TRUE),delta1=matrix(c(0.6,1,-0.4),k,1,byrow = TRUE))
Location_R1 = list(phi1 = matrix(c(-1.5),k,k,byrow = TRUE))

#Sigma_R1 = matrix(c(1,0.3,0.4,0.3,1,-0.5,0.4,-0.5,1),k,k,byrow = TRUE)
Sigma_R1 = matrix(c(1),k,k,byrow = TRUE)

#cs_1=matrix(c(1,-2,6),nrow=k)

cs_1=matrix(c(0),nrow=k)

R1 = list(orders = list(p = 1,q = 0,d = 0),Location = Location_R1,Sigma = Sigma_R1,cs=cs_1)

### R2 regimen ====
#Location_R2 = list(phi1 = matrix(c(0.3,0.5,-0.5,0.2,0.7,-0.1,0.3,-0.4,0.6),k,k,byrow = TRUE))
Location_R2 = list(phi1 = matrix(c(0.6),k,k,byrow = TRUE))

Sigma_R2 = matrix(c(1),k,k,byrow = TRUE)

cs_2=matrix(c(0),nrow=k)

R2 = list(orders = list(p = 1,q = 0,d = 0),
          Location = Location_R2,Sigma = Sigma_R2,cs=cs_2)## crea lista de objeto tipo Regime

Rg = list(R1 = R1,R2 = R2) # 2 reg
umbrales = 0 # 2 reg
params <- list()

for(i in 1:length(ars$p)){
  np <- Intercept + ars$p[i]*k + ars$q[i]*v + ars$d[i]
  params[[i]] <- list()
  params[[i]]$location <-rbind(t(Rg[[i]][[4]]),matrix(unlist(Rg[[i]][[2]]),ncol=k,byrow=TRUE))
  #params[[i]]$location <- matrix(c(rbeta(np*k,shape1=4,shape2=16)),np,k)
  # params[[i]]$scale <- diag(rgamma(k,shape=1,scale=1))
  params[[i]]$scale <- Rg[[i]][[3]]
  params[[i]]$scale2 <- chol(params[[i]]$scale)
}

params
Z <- as.matrix(arima.sim(n=Tlen+max(ars$p),list(ar=c(0.5))))

myseries <- simtar(n=Tlen,k=k,ars=ars,Intercept=Intercept,parms=params,thresholds = umbrales ,delay=delay,dist=dist,extra=1)
myseries <- simtar(n=Tlen,k=k,ars=ars,Intercept=Intercept,parms=params,thresholds = umbrales ,delay=delay,dist=dist,t.series=Z)

datos <- data.frame(myseries[(max(ars$p,ars$q,ars$d,delay)+inic+1):dim(myseries)[1],])

#Generar varias series
# Carpeta y archivo Excel
folder_path <- "C:/Users/huma1003/OneDrive - NIQ/DOCUMENTOS IMPORTANTES/Mis_Cosas/MarlijarTM/msvr-master/MSETAR/Data_Sergio"
if (!dir.exists(folder_path)) {
  dir.create(folder_path, recursive = TRUE)
}

# Generar series y guardar en Excel
long <- c( 200,500,1000)  # Longitud de la serie
for (size in long) {
  excel_filename <- paste0(folder_path, "/dataset_size_", size, ".xlsx")
  wb <- createWorkbook()
  for (no in 0:99) {
    
    datos <- NULL
    print(paste("Tamaño:", size, "Iteración:", no))
    
    myseries <- simtar(n=Tlen, k=k, ars=ars, Intercept=Intercept, parms=params, thresholds=umbrales, delay=delay, dist=dist, setar=1)
    extracted_data_1 <- myseries[(max(ars$p, ars$q, ars$d, delay) + inic + 1):dim(myseries)[1], ]
    myseries <- simtar(n=Tlen, k=k, ars=ars, Intercept=Intercept, parms=params, thresholds=umbrales, delay=delay, dist=dist, setar=1)
    extracted_data_2 <- myseries[(max(ars$p, ars$q, ars$d, delay) + inic + 1):dim(myseries)[1], ]
    datos <- data.frame(cbind(extracted_data_1, extracted_data_2))
    
    sheet_name <- paste0("Iter_", no)
    addWorksheet(wb, sheet_name)
    writeData(wb, sheet_name, datos)
    saveWorkbook(wb, excel_filename, overwrite = TRUE)
  }
  # Guardar el archivo Excel
 
}


#plot(as.ts(datos[,1:k]))

Fechas=seq(as.Date("2000/1/1"), by = "day", length.out = (Tlen-calen))
datos1=data.frame(datos,Fecha=Fechas)
colnames(datos1)<-c("Y1","Fecha")
fecha_final<-Fechas[long]
fit <- mtarm::mtar(~Y1|Y1,
                       data=datos1,
                       ars=ars,
                       dist=dist,
                       row.names=Fecha,
                       subset={Fecha<=fecha_final},
                       n.burnin=1000,
                       n.sim=2000,
                       n.thin=1,
                       Intercept=TRUE,prior=list(hmin=1))
summary(fit)
nano <- forecasting(fit,subset(datos1,Fecha > fecha_final),row.names=Fecha)
nano$summary ##Predicción
subset(datos1,Fecha > fecha_final) ##Verdaderos valores
