COMPTE RENDU D’ANALYSE DES DONNÉES
Dataset : Instagram Analytics

Réalisé par : Douaa El Khayari – CAC2 – Apogée : 24010315

1. Introduction

Dans un contexte où Instagram est devenu un outil central de communication, la compréhension des performances des publications est essentielle. Les entreprises, influenceurs et créateurs de contenu doivent analyser leurs statistiques pour optimiser leur visibilité et améliorer l’engagement de leur audience.

Cette étude utilise un dataset Instagram comportant 29 999 publications et 15 variables, permettant une analyse complète de l’interaction des utilisateurs.

2. Problématique

La question centrale est :

Quels sont les facteurs principaux qui influencent l’engagement sur Instagram, et dans quelle mesure peut-on prédire ce niveau d’engagement ?

Pour y répondre, l’analyse s’est déroulée en plusieurs phases :
chargement des données, preprocessing, feature engineering, exploration statistique, visualisation, modélisation et évaluation.

3. Analyse détaillée cellule par cellule
🔵 Cellule 0 — Chargement du dataset et aperçu
Code :
import pandas as pd

file_path = "/content/Instagram_Analytics.csv"
df = pd.read_csv(file_path)

print("Shape:", df.shape)
df.head()

Explication du code

pd.read_csv() charge le dataset depuis un fichier CSV.

df.shape permet de connaître le nombre de lignes et de colonnes.

df.head() affiche les cinq premières lignes pour vérifier le format et les valeurs.

Sortie :
Shape: (29999, 15)

Interprétation

Nous avons un dataset très large (29 999 lignes) contenant 15 colonnes.
Cela assure une bonne diversité statistique et permet une modélisation de qualité.

🔵 Cellule 1 — Inspection des types + préparation date
Code :
df.info()

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

Explication

df.info() liste les types de données : int, float, object.

pd.to_datetime() convertit la colonne timestamp en un format date utilisable.

Sortie :

Affichage du nombre de colonnes, de leur type, et mémoire utilisée.

Interprétation

Beaucoup de colonnes sont numériques → bon pour l’analyse statistique.

Certaines colonnes object devront être transformées.

Conversion du timestamp est indispensable pour l’analyse temporelle.

🔵 Cellule 2 — Calcul de l’engagement
Code :
df['likes'] = df['likes'].fillna(0)
df['comments'] = df['comments'].fillna(0)

df['engagement'] = df['likes'] + df['comments']
df['engagement_rate'] = df['engagement'] / df['followers'] * 100

df[['engagement', 'engagement_rate']].head()

Explication

Remplacement des valeurs manquantes par 0 pour éviter les erreurs de calcul.

Calcul de l’engagement : somme des interactions directes.

Calcul de l’engagement_rate (%) : mesure clé sur Instagram.

Interprétation

L’engagement est proportionnel au nombre d’abonnés.
Un taux important signifie que la publication attire réellement l’attention du public.

🔵 Cellule 3 — Feature Engineering (Nouvelles variables)
Code :
df['day'] = df['timestamp'].dt.day_name()
df['is_weekend'] = df['day'].isin(['Saturday', 'Sunday'])

df['caption_length'] = df['caption'].astype(str).apply(len)
df['hashtags_count'] = df['hashtags'].astype(str).apply(lambda x: len(x.split()))

Explication

On crée des nouvelles variables utiles :

jour de la semaine

weekend ou non

longueur de la légende

nombre de hashtags

Sortie : affichage d’un tableau avec ces colonnes.
Interprétation

Ces variables permettent de tester des hypothèses comme :

Les posts du weekend performent-ils mieux ?

Les hashtags augmentent-ils l’engagement ?

Une légende longue attire-t-elle plus d’attention ?

Ces features enrichissent grandement l’analyse et les modèles ML.

🔵 Cellule 4 — Statistiques descriptives
Code :
df.describe()

Explication

Affiche des statistiques :
moyenne, médiane, minimum, maximum, quartiles…

Interprétation

Forte variance dans les likes → certaines publications sont virales.

Écarts extrêmes dans le reach → certaines publications ont explosé en visibilité.

Les commentaires sont plus faibles mais corrélés aux likes.

🔵 Cellule 5 — Visualisations (histogrammes)

Graphiques produits :

distribution des likes

distribution des followers

distribution des engagement_rates

Interprétation

Les likes sont asymétriques → beaucoup de posts faibles, quelques pics extraordinaires.

Les followers sont très concentrés → peu d’outliers.

L’engagement rate varie beaucoup, indiquant un public irrégulier.

🔵 Cellule 6 — Matrice de corrélation
Code :
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Corrélations")
plt.show()

Explication

Calcul des corrélations entre les variables numériques.

La heatmap aide à repérer les relations fortes.

Interprétation

likes ↔ engagement : corrélation très forte (logique).

followers ↔ reach : une base solide augmente la portée.

hashtags ↔ engagement_rate : faible corrélation → les hashtags n’aident pas toujours.

🔵 Cellule 7 — Sélection des variables pour le modèle
Code :
feature_cols = ['caption_length','hashtags_count','likes','comments','is_weekend']
X = df[feature_cols]
y = df['engagement_rate']

Explication

On choisit les variables qui serviront au modèle prédictif.
La variable cible (target) est engagement_rate.

Interprétation

Les features combinent texte, comportement utilisateur, et interactions.

🔵 Cellule 8 — Train-test split
Code :
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Explication

Séparation du dataset : 80% entraînement, 20% test.

Important pour éviter l’overfitting.

Interprétation

Le modèle sera évalué sur des données jamais vues, garantissant une performance fiable.

🔵 Cellule 9 — Modèle Ridge et métriques
Code :
model = Ridge()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R²:", r2)

Sortie :
RMSE: 49.72
R²: -0.0006

Interprétation

RMSE élevé (≈ 50) → le modèle ne parvient pas à prédire précisément le taux d’engagement.

R² négatif → le modèle fait pire qu’une prédiction constante.

Conclusion locale

Le modèle Ridge n’est pas adapté.
Les variables choisies ne suffisent pas à expliquer l’engagement.
Il faudra tester :
✔ Random Forest
✔ Gradient Boosting
✔ XGBoost
✔ non-linéarités et interactions

4. Conclusion générale

Cette étude montre que :

Le dataset est riche et permet une analyse détaillée.

L’engagement dépend fortement des likes, du reach et des interactions globales.

Les variables textuelles doivent être mieux exploitées (NLP).

Les modèles linéaires comme Ridge ne captent pas la complexité du phénomène.

Une approche non linéaire ou deep learning serait plus performante.
