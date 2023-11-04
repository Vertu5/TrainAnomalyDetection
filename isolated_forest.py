import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Charger les données depuis le fichier CSV
data = pd.read_csv('C:\\Users\\azert\\Documents\\Data Mining\\ar41_for_ulb.csv', delimiter=';')

# Sélectionner les colonnes pertinentes pour la détection d'anomalies
selected_features = ['lat', 'lon', 'RS_E_InAirTemp_PC1', 'RS_E_InAirTemp_PC2', 'RS_E_OilPress_PC1', 'RS_E_OilPress_PC2', 'RS_E_RPM_PC1', 'RS_E_RPM_PC2', 'RS_E_WatTemp_PC1', 'RS_E_WatTemp_PC2', 'RS_T_OilTemp_PC1', 'RS_T_OilTemp_PC2']

X = data[selected_features]

# Standardiser les données (important pour l'Isolation Forest)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Créer un modèle Isolation Forest
isolation_forest = IsolationForest(contamination=0.07)  # Vous pouvez ajuster le taux de contamination selon vos besoins

# Ajuster le modèle aux données
isolation_forest.fit(X_scaled)

# Prédire les anomalies
anomalies = isolation_forest.predict(X_scaled)

# Identifier les anomalies (les points ayant une prédiction -1)
anomalies_indices = anomalies == -1
anomalies_data = X[anomalies_indices]

# Afficher les anomalies
print("Anomalies détectées :")
print(anomalies_data)
