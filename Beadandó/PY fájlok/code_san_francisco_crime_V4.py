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
plt.scatter(coords['X'], coords['Y'], c=coords['Cluster'], cmap='tab10', s=10)
plt.title('Bűncselekmények térbeli klaszterezése (DBSCAN)')
plt.show()

text = " ".join(df['Descript'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# --- CELL ---

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Adatok előkészítése
# Megtartjuk a szűrt adatokat a V3-ból (df_filtered)
cat_counts = df['Category'].value_counts()
valid_cats = cat_counts[cat_counts >= 10].index
df_filtered = df[df['Category'].isin(valid_cats)].copy()

le_cat = LabelEncoder()
le_day = LabelEncoder()
le_dist = LabelEncoder()
le_tod = LabelEncoder()

y_raw = le_cat.fit_transform(df_filtered['Category'])
num_classes = len(le_cat.classes_)

# Jellemzők kódolása
df_filtered['Day_Encoded'] = le_day.fit_transform(df_filtered['DayOfWeek'])
df_filtered['District_Encoded'] = le_dist.fit_transform(df_filtered['PdDistrict'])
df_filtered['TOD_Encoded'] = le_tod.fit_transform(df_filtered['Time_of_Day'])

features = ['Day_Encoded', 'District_Encoded', 'Hour', 'X', 'Y', 'TOD_Encoded']
X = df_filtered[features].values

# Skálázás (Neurális hálóknál kritikus!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encoding a kimenethez
y_onehot = to_categorical(y_raw, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42, stratify=y_raw)

# 2. Neurális Háló Felépítése
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Tanítás (Epoch-okkal)
print("Neurális háló tanítása...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 4. Kiértékelés
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTeszt pontosság: {accuracy:.4f}")

# 5. Tanulási görbék vizualizációja
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Tanítási pontosság')
plt.plot(history.history['val_accuracy'], label='Validációs pontosság')
plt.title('Modell pontossága')
plt.xlabel('Epoch')
plt.ylabel('Pontosság')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Tanítási veszteség')
plt.plot(history.history['val_loss'], label='Validációs veszteség')
plt.title('Modell vesztesége (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Veszteség')
plt.legend()

plt.tight_layout()
plt.show()