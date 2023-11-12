# Install and load necessary packages
library(dbscan)
library(imputeTS)

# Charger les données depuis le fichier CSV
data <- read.csv('morceau_1.csv', sep=';')

# Sélectionner les colonnes pertinentes pour la détection d'anomalies
selected_features <- c('lat', 'lon', 'RS_E_InAirTemp_PC1', 'RS_E_InAirTemp_PC2', 'RS_E_OilPress_PC1', 'RS_E_OilPress_PC2', 'RS_E_RPM_PC1', 'RS_E_RPM_PC2', 'RS_E_WatTemp_PC1', 'RS_E_WatTemp_PC2', 'RS_T_OilTemp_PC1', 'RS_T_OilTemp_PC2')

X <- data[selected_features]

# Mesurer le temps d'exécution
start_time <- system.time({
  # Imputer pour remplir les valeurs manquantes avec la moyenne
  X_imputed <- na_mean(X)
  
  # Standardiser les données (important pour DBSCAN)
  X_scaled <- scale(X_imputed)
  
  # Créer un modèle DBSCAN
  dbscan_model <- dbscan(X_scaled, eps=0.5, minPts=100)  # Vous pouvez ajuster les hyperparamètres selon vos besoins
  
  # Obtenir les étiquettes de clusters et les étiquettes des anomalies
  labels <- dbscan_model$cluster
  
  # Identifier les anomalies (les points ayant une étiquette -1)
  anomalies <- X[labels == -1, ]
  
  # Afficher les anomalies
  cat("Anomalies détectées :\n")
  print(anomalies)
})

# Afficher le temps d'exécution
cat("Temps d'exécution:", start_time["elapsed"], "secondes\n")
