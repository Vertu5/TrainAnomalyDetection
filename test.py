import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer


# Charger les données CSV dans un DataFrame
df = pd.read_csv('C:\\Users\\azert\\Documents\\Data Mining\\ar41_for_ulb.csv', delimiter=';')

# Sélectionner les colonnes de données que vous souhaitez utiliser pour l'analyse
X = df[['lat', 'lon', 'RS_E_InAirTemp_PC1', 'RS_E_InAirTemp_PC2', 'RS_E_OilPress_PC1', 'RS_E_OilPress_PC2', 'RS_E_RPM_PC1', 'RS_E_RPM_PC2', 'RS_E_WatTemp_PC1', 'RS_E_WatTemp_PC2', 'RS_T_OilTemp_PC1', 'RS_T_OilTemp_PC2']]


# Imputer to fill in missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)


# Initialiser le modèle Nearest Neighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X_imputed)

# Trouver les voisins les plus proches
distances, indices = nbrs.kneighbors(X_imputed)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]

# Tracer les distances
plt.plot(distances)
plt.xlabel('Échantillons')
plt.ylabel('Distance au voisin le plus proche')
plt.title('Graphe de distances aux voisins les plus proches')
plt.show()


