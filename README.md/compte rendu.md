COMPTE RENDU D’ANALYSE DE DONNÉES
Instagram_Analytics.csv

DOUAA EL KHAYARI – CAC 2 – Apogée : 24010315

.SOMMAIRE

1. Introduction

2.Problématique

3.Méthodologie utilisée

4.Analyse et interprétation des résultats

   4.1 Chargement de la base de données
   
   4.2 Vérification des valeurs manquantes
   
   4.3 Statistiques descriptives
   
   4.4 Distribution des likes
   
   4.5 Évolution des likes dans le temps
   
   4.6 Relation entre Reach et Likes
   
   4.7 Calcul du taux d’engagement
   
   4.8 Évolution du taux d’engagement

5. Conclusion

   INTRODUCTION

Ce rapport présente une analyse détaillée du dataset Instagram_Analytics.csv, contenant des informations sur la performance d’un compte Instagram : likes, commentaires, impressions, reach, saves, shares, followers, etc.

L’objectif de cette étude est de comprendre les comportements des utilisateurs face aux publications, d’identifier les facteurs influençant la performance et d’évaluer la qualité de l’engagement.

2. PROBLEMATIQUE

Les entreprises, marques et créateurs de contenu utilisent Instagram comme outil stratégique pour développer leur visibilité et leur communauté. Cependant, comment mesurer efficacement la performance d’un compte Instagram et quels indicateurs influencent réellement l’engagement des abonnés ?

Ainsi, la problématique principale est :

« Quels sont les indicateurs qui influencent le plus la performance des publications Instagram, et comment évolue l'engagement au fil du temps ? »

Cette analyse vise à répondre à cette problématique en examinant les données à travers des statistiques, des visualisations et des indicateurs de performance.

3. METHODOLOGIE UTILISEE

Pour répondre à la problématique, la démarche suivante a été adoptée :

Importation et lecture du fichier CSV

Vérification de la qualité des données (valeurs manquantes, types)

Calcul des statistiques descriptives

Analyse graphique : histogrammes, courbes temporelles, nuages de points

Création et analyse du taux d’engagement

Interprétation des résultats obtenus

4. ANALYSE ET INTERPRETATION DES RESULTATS

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
<img width="713" height="393" alt="téléchargement (4)" src="https://github.com/user-attachments/assets/d575a607-2b90-4280-9a68-7249a53ded6b" />
<img width="544" height="385" alt="téléchargement (5)" src="https://github.com/user-attachments/assets/234b7198-efd0-421a-9609-17d403fe904c" />


🔵 Cellule 6 — Matrice de corrélation
Code :
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Corrélations")
plt.show()

Explication

Calcul des corrélations entre les variables numériques.

La heatmap aide à repérer les relations fortes.
<img width="735" height="528" alt="téléchargement (6)" src="https://github.com/user-attachments/assets/5cf770eb-08b9-40b2-acf6-68ed3d0f4035" />


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

5. CONCLUSION

Cette étude montre que :

Le dataset est riche et permet une analyse détaillée.

L’engagement dépend fortement des likes, du reach et des interactions globales.

Les variables textuelles doivent être mieux exploitées (NLP).

Les modèles linéaires comme Ridge ne captent pas la complexité du phénomène.

Une approche non linéaire ou deep learning serait plus performante.
