import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
sns.set(style="whitegrid")

# --- CELL ---

# Adatok betöltése
df = pd.read_csv('Map-Crime_Incidents-Previous_Three_Months.csv')

# 1. Az IncidntNum hiányzó értékeinek kezelése
df['IncidntNum'] = df['IncidntNum'].fillna(0).astype(int)

# 2. Konvertálás dátum/idő típusra
df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time

# 3. Adattípus-optimalizálás
df['Category'] = df['Category'].astype('category')
df['PdDistrict'] = df['PdDistrict'].astype('category')
df['DayOfWeek'] = df['DayOfWeek'].astype('category')

# Érvénytelen koordináták szűrése
df = df[(df['X'] < -120) & (df['Y'] > 30)]

df.info()
df.head()

# --- CELL ---

# Top 10 bűncselekmény kategória
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='Category', order=df['Category'].value_counts().iloc[:10].index, palette='viridis')
plt.title('Top 10 bűncselekmény kategória')
plt.show()

# --- CELL ---

# Csoportosítás megoldott és megoldatlan kategóriákba
def categorize_resolution(res):
    if 'ARREST' in str(res).upper():
        return 'Megoldott (Letartóztatás)'
    elif 'NONE' in str(res).upper():
        return 'Megoldatlan'
    else:
        return 'Egyéb'

df['Resolution_Status'] = df['Resolution'].apply(categorize_resolution)

# Top 10 kategória kiválasztása
top_10_cats = df['Category'].value_counts().iloc[:10].index
df_top_10 = df[df['Category'].isin(top_10_cats)]

# Stacked bar chart készítése
resolution_counts = df_top_10.groupby(['Category', 'Resolution_Status']).size().unstack().fillna(0)
# Csak a Megoldott és Megoldatlan oszlopok arányát nézzük
resolution_counts = resolution_counts[['Megoldott (Letartóztatás)', 'Megoldatlan']]

# Arányok kiszámítása (százalékosítva)
resolution_pct = resolution_counts.div(resolution_counts.sum(axis=1), axis=0) * 100

plt.figure(figsize=(12, 8))
resolution_pct.plot(kind='barh', stacked=True, color=['#2ecc71', '#e74c3c'], ax=plt.gca())
plt.title('Megoldott vs Megoldatlan ügyek aránya (Top 10 kategória)')
plt.xlabel('Arány (%)')
plt.ylabel('Bűncselekmény típusa')
plt.legend(title='Státusz', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- CELL ---

map_df = df.sample(min(1000, len(df)))
sf_map = folium.Map(location=[37.77, -122.42], zoom_start=12)
heat_data = [[row['Y'], row['X']] for index, row in df.sample(min(2000, len(df))).iterrows()]
HeatMap(heat_data).add_to(sf_map)
marker_cluster = MarkerCluster().add_to(sf_map)
for index, row in map_df.iterrows():
    folium.Marker(location=[row['Y'], row['X']], popup=f"{row['Category']}: {row['Descript']}").add_to(marker_cluster)
sf_map

# --- CELL ---

# Óra kinyerése
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour

# Napszakok kategorizálása (Éjszaka: 0-5, Reggel: 6-11, Délután: 12-17, Este: 18-23)
def get_time_of_day_refined(hour):
    if 0 <= hour <= 5: return 'Éjszaka'
    elif 6 <= hour <= 11: return 'Reggel'
    elif 12 <= hour <= 17: return 'Délután'
    elif 18 <= hour <= 23: return 'Este'

df['Time_of_Day'] = df['Time_of_Day'] = df['Hour'].apply(get_time_of_day_refined)

plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='Time_of_Day', order=['Éjszaka', 'Reggel', 'Délután', 'Este'], palette='coolwarm')
plt.title('Bűncselekmények napszakok szerint')
plt.show()

# --- CELL ---

coords = df[['X', 'Y']].sample(min(2000, len(df)))
db = DBSCAN(eps=0.001, min_samples=10).fit(coords)
coords['Cluster'] = db.labels_
plt.figure(figsize=(10, 8))
plt.scatter(coords['X'], coords['Y'], c=coords['Cluster'], cmap='tab20', s=10)
plt.title('Bűncselekmények térbeli klaszterezése (DBSCAN)')
plt.show()

text = " ".join(df['Descript'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# --- CELL ---

le_cat = LabelEncoder()
le_day = LabelEncoder()
le_dist = LabelEncoder()
le_tod = LabelEncoder()

ml_df = df.copy()
ml_df['Category_Encoded'] = le_cat.fit_transform(ml_df['Category'])
ml_df['Day_Encoded'] = le_day.fit_transform(ml_df['DayOfWeek'])
ml_df['District_Encoded'] = le_dist.fit_transform(ml_df['PdDistrict'])
ml_df['TOD_Encoded'] = le_tod.fit_transform(ml_df['Time_of_Day'])

# Új jellemzők listája (belevéve a TOD_Encoded-ot)
features = ['Day_Encoded', 'District_Encoded', 'Hour', 'X', 'Y', 'TOD_Encoded']
X = ml_df[features]
y = ml_df['Category_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le_cat.classes_[:len(np.unique(y_test))], zero_division=0))