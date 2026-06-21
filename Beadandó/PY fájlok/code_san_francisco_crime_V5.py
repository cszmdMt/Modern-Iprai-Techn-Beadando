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
import re
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
sns.set(style="whitegrid")

# --- CELL ---

# 1. Adatbetöltés és NaN kezelés
df = pd.read_csv('Map-Crime_Incidents-Previous_Three_Months.csv')
df['IncidntNum'] = df['IncidntNum'].fillna(0).astype(int)

# 2. Dátum és Idő kezelés (pd.to_datetime)
df['Date'] = pd.to_datetime(df['Date'])
df['Time_DT'] = pd.to_datetime(df['Time'], format='%H:%M')
df['Hour'] = df['Time_DT'].dt.hour

# 3. Típusoptimalizálás (Category dtype)
categorical_cols = ['Category', 'PdDistrict', 'DayOfWeek', 'Resolution']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Irreális koordináták szűrése
df = df[(df['X'] < -120) & (df['Y'] > 30)]

df.info()

# --- CELL ---

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.countplot(data=df, y='Category', order=df['Category'].value_counts().iloc[:10].index, palette='magma')
plt.title('Top 10 Bűncselekmény Típus')

plt.subplot(1, 2, 2)
sns.countplot(data=df, x='DayOfWeek', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], palette='viridis')
plt.title('Bűncselekmények eloszlása a hét napjai szerint')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- CELL ---

sf_map = folium.Map(location=[37.77, -122.42], zoom_start=12)

# HeatMap réteg
heat_data = df[['Y', 'X']].dropna().values[:5000] # Limit a teljesítmény miatt
HeatMap(heat_data, radius=15).add_to(sf_map)

# MarkerCluster réteg
marker_cluster = MarkerCluster().add_to(sf_map)
for idx, row in df.head(500).iterrows():
    folium.Marker([row['Y'], row['X']], popup=f"{row['Category']}: {row['Descript']}").add_to(marker_cluster)

sf_map

# --- CELL ---

# 6. Time_of_Day létrehozása
def get_time_of_day(hour):
    if 5 <= hour < 12: return 'Morning'
    elif 12 <= hour < 17: return 'Afternoon'
    elif 17 <= hour < 21: return 'Evening'
    else: return 'Night'

df['Time_of_Day'] = df['Hour'].apply(get_time_of_day).astype('category')

# 7. DBSCAN Térbeli Klaszterezés
coords = df[['X', 'Y']].values
db = DBSCAN(eps=0.001, min_samples=15).fit(coords)
df['Crime_Hotspot_Label'] = db.labels_

plt.figure(figsize=(10, 6))
plt.scatter(df['X'], df['Y'], c=df['Crime_Hotspot_Label'], cmap='tab20', s=1, alpha=0.5)
plt.title('Bűnügyi Gócpontok Automatikus Azonosítása (DBSCAN)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# --- CELL ---

# 8. Szövegtisztítás és WordCloud
text = " ".join(str(val).lower() for val in df['Descript'])
text = re.sub(r'[^a-z\s]', '', text)

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='inferno').generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Bűncselekmény Leírások Kulcsszó-felhője')
plt.show()

# --- CELL ---

# 9. RandomForest Modell
le = LabelEncoder()
df['Day_Code'] = le.fit_transform(df['DayOfWeek'])
df['TOD_Code'] = le.fit_transform(df['Time_of_Day'])
df['Cat_Code'] = le.fit_transform(df['Category'])

X = df[['X', 'Y', 'Day_Code', 'TOD_Code', 'Hour']]
y = df['Cat_Code']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Modell Teljesítmény Jelentés:")
print(classification_report(y_test, y_pred, target_names=le.classes_))