import pandas as pd

# Charger le fichier CSV en utilisant pandas
fichier_csv = 'C:\\Users\\azert\\Documents\\Data Mining\\ar41_for_ulb.csv'
df = pd.read_csv(fichier_csv)

# Définir le nombre de lignes par morceau
taille_morceau = 500000  # Choisissez la taille appropriée

# Diviser le DataFrame en morceaux
morceaux = [df.iloc[i:i + taille_morceau] for i in range(0, len(df), taille_morceau)]

# Sauvegarder chaque morceau dans un nouveau fichier CSV
for i, morceau in enumerate(morceaux):
    nom_fichier_sortie = f"morceau_{i + 1}.csv"
    morceau.to_csv(nom_fichier_sortie, index=False)
    print(f"Morceau {i + 1} sauvegardé dans {nom_fichier_sortie}")
