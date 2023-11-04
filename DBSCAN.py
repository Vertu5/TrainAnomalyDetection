import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # Import the imputer

# Charger les données depuis le fichier CSV
data = pd.read_csv('C:\\Users\\azert\\Documents\\Data Mining\\ar41_for_ulb_mini.csv', delimiter=';')

# Sélectionner les colonnes pertinentes pour la détection d'anomalies
selected_features = ['lat', 'lon', 'RS_E_InAirTemp_PC1', 'RS_E_InAirTemp_PC2', 'RS_E_OilPress_PC1', 'RS_E_OilPress_PC2', 'RS_E_RPM_PC1', 'RS_E_RPM_PC2', 'RS_E_WatTemp_PC1', 'RS_E_WatTemp_PC2', 'RS_T_OilTemp_PC1', 'RS_T_OilTemp_PC2']

X = data[selected_features]

# Standardiser les données (important pour DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Imputer to fill in missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_scaled)

# Créer un modèle DBSCAN
dbscan = DBSCAN(eps=5, min_samples=5)  # Vous pouvez ajuster les hyperparamètres selon vos besoins

# Ajuster le modèle aux données
dbscan.fit(X_imputed)  # Use the imputed data

# Obtenir les étiquettes de clusters et les étiquettes des anomalies
labels = dbscan.labels_

# Identifier les anomalies (les points ayant une étiquette -1)
anomalies = X[labels == -1]

# Afficher les anomalies
print("Anomalies détectées :")
print(anomalies)
