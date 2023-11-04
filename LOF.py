import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Charger les données depuis le fichier CSV
data = pd.read_csv('C:\\Users\\azert\\Documents\\Data Mining\\ar41_for_ulb_mini.csv', delimiter=';')

# Sélectionner les colonnes pertinentes pour la détection d'anomalies
selected_features = ['lat', 'lon', 'RS_E_InAirTemp_PC1', 'RS_E_InAirTemp_PC2', 'RS_E_OilPress_PC1', 'RS_E_OilPress_PC2', 'RS_E_RPM_PC1', 'RS_E_RPM_PC2', 'RS_E_WatTemp_PC1', 'RS_E_WatTemp_PC2', 'RS_T_OilTemp_PC1', 'RS_T_OilTemp_PC2']

X = data[selected_features]

# Standardiser les données (important pour le LOF)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Créer un modèle LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.07)  # Vous pouvez ajuster le nombre de voisins et le taux de contamination selon vos besoins

# Calculer les scores LOF
lof_scores = lof.fit_predict(X_scaled)

# Identifier les anomalies (les points ayant un score LOF négatif)
anomalies_indices = lof_scores == -1
anomalies_data = X[anomalies_indices]

# Afficher les anomalies
print("Anomalies détectées :")
print(anomalies_data)
