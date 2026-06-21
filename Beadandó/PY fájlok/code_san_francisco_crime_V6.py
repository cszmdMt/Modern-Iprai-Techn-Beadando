import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

# --- CELL ---

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Tisztítás
    df['IncidntNum'] = df['IncidntNum'].fillna(0).astype(int)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
    
    # Típusok
    for col in ['Category', 'PdDistrict', 'DayOfWeek', 'Resolution']:
        df[col] = df[col].astype('category')
    
    # Szűrés
    df = df[(df['X'] < -120) & (df['Y'] > 30)]
    
    # Feature Engineering
    df['Is_Weekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)
    return df

df = load_and_clean_data('Map-Crime_Incidents-Previous_Three_Months.csv')
df.head()

# --- CELL ---

# Bűncselekmények óránkénti eloszlása (Interaktív)
hourly_crime = df.groupby('Hour').size().reset_index(name='Count')
fig = px.line(hourly_crime, x='Hour', y='Count', title='Bűncselekmények alakulása a nap folyamán', 
              markers=True, template='plotly_dark')
fig.show()

# Körzetek szerinti megoszlás (TreeMap)
district_crime = df.groupby(['PdDistrict', 'Category']).size().reset_index(name='Count')
fig2 = px.treemap(district_crime, path=['PdDistrict', 'Category'], values='Count', 
                  title='Bűnügyi Térkép (Körzet -> Típus)')
fig2.show()

# --- CELL ---

cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = cv.fit_transform(df['Descript'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(dtm)

print("A bűncselekmény leírásokból kinyert 5 fő téma (Kulcsszavak):")
for index, topic in enumerate(lda.components_):
    print(f'TÉMA #{index}:')
    print([cv.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print('\n')

# --- CELL ---

le = LabelEncoder()
X = df[['X', 'Y', 'Hour', 'Is_Weekend']]
y = le.fit_transform(df['Category'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paraméter rács a kereséshez
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=3, random_state=42)
rand_search.fit(X_train, y_train)

print(f"Legjobb paraméterek: {rand_search.best_params_}")
best_model = rand_search.best_estimator_
print(classification_report(y_test, best_model.predict(X_test), target_names=le.classes_))

# --- CELL ---

m = folium.Map(location=[37.77, -122.42], zoom_start=12, tiles='Stamen Toner')

# Klaszterezés (DBSCAN)
db = DBSCAN(eps=0.001, min_samples=10).fit(df[['X', 'Y']])
df['Cluster'] = db.labels_

HeatMap(df[df['Cluster'] != -1][['Y', 'X']]).add_to(m)
m