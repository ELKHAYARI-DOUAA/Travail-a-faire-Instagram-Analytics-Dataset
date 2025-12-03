Importer les données brutes est la première étape. On vérifie que le fichier se charge correctement.

  # Chargement du dataset Instagram depuis un fichier local.
# Le chemin /content/ est typique de Google Colab.
import pandas as pd

file_path = "/content/Instagram_Analytics.csv"
df = pd.read_csv(file_path)

print("Shape:", df.shape)   # Vérifie le nombre de lignes et de colonnes
df.head()                  # Aperçu des premières lignes
Avant d’appliquer des modèles, il faut comprendre la structure des données
→ identifier types, incohérences, NaN, doublons…
  # Affiche les types des colonnes et les valeurs manquantes.
df.info()

# Supprime les doublons pour éviter de biaiser le modèle.
df = df.drop_duplicates()

# Conversion timestamp : indispensable pour créer des features temporelles.
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
Le but du projet est de prédire l’engagement.
On crée une variable explicite que les modèles pourront prédire.
  # Remplace les valeurs manquantes dans likes/comments par 0 (valeur logique)
df['likes'] = df['likes'].fillna(0)
df['comments'] = df['comments'].fillna(0)

# Calculer l'engagement brut
df['engagement'] = df['likes'] + df['comments']

# Followers = base de calcul du taux d'engagement → imputation nécessaire
if 'followers' in df.columns:
    df['followers'] = df['followers'].replace(0, pd.NA)   # 0 followers = incohérent
    df['followers'] = df['followers'].fillna(df['followers'].median())
else:
    print("⚠️ La colonne 'followers' n'existe pas.")

# Calcul du taux d'engagement (métrique standard en marketing digital)
df['engagement_rate'] = df['engagement'] / (df['followers'] + 1e-9)

df[['engagement', 'engagement_rate']].head()
Créer des variables dérivées (FE) permet d’améliorer la performance du modèle.
Chaque variable apporte une information utile sur un post.
# Remplace caption manquante par chaîne vide pour éviter erreurs NLP
df['caption'] = df['caption'].fillna('')

# Longueur de caption → souvent corrélée à l'engagement
df['caption_length'] = df['caption'].apply(len)

# Compte des hashtags → très pertinent pour visibilité organique
df['num_hashtags'] = df['caption'].str.count(r'#\w+')

# Compte des mentions → impact sur reach
df['num_mentions'] = df['caption'].str.count(r'@\w+')

# Extraction temporelle → l'heure du post influence l'engagement
if 'timestamp' in df.columns:
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
Visualiser permet de comprendre le comportement des données, détecter anomalies, outliers, tendances.
# Histogramme de la distribution pour détecter asymétrie, outliers
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.hist(df['engagement_rate'], bins=50)
plt.title("Distribution du Engagement Rate")
plt.xlabel("engagement_rate")
plt.ylabel("count")
plt.show()
# Boxplot pour comparer l'engagement selon media_type (photo, vidéo, carousel)
if 'media_type' in df.columns:
    df.boxplot(column='engagement_rate', by='media_type', figsize=(6,4))
    plt.title("Engagement Rate par Media Type")
    plt.suptitle("")
    plt.show()
# Heatmap pour identifier quelles variables corrèlent avec engagement_rate
import seaborn as sns

plt.figure(figsize=(8,6))
num_cols = ['engagement_rate','followers','caption_length','num_hashtags','num_mentions']
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Corrélations")
plt.show()
Pour entraîner un modèle, on sépare X (features) et y (target), puis on crée un dataset de test.
  # Variables pertinentes choisies pour la prédiction
feature_cols = [
    'caption_length', 'num_hashtags', 'num_mentions',
    'followers', 'hour', 'weekday', 'is_weekend'
]

X = df[feature_cols]
y = df['engagement_rate']
# Sépare 80% entraînement / 20% test pour évaluer le modèle correctement
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
Modèle linéaire régularisé → simple, rapide, bon point de comparaison avant les modèles avancés.
  from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Pipeline pour standardiser + entraîner le modèle linéaire
model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Métriques clés : RMSE (erreur) et R² (qualité d'explication)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R²:", r2)
