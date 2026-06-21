import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

print("Adatok betöltése és előkészítése...")
# 1. ADATELŐKÉSZÍTÉS
df = pd.read_csv('Map-Crime_Incidents-Previous_Three_Months.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
df = df[(df['X'] < -120) & (df['Y'] > 30)]

# --- 2. REKURRENS NEURÁLIS HÁLÓ (RNN) ---
# A holnapi bűncselekmények TELJES SZÁMÁNAK becslése az idősoros adatok alapján
print("1/3 Lépés: Rekurrens Neurális Háló (RNN) tanítása a napi esetszámokra...")
daily_crimes = df.groupby('Date').size().reset_index(name='Count')
daily_crimes = daily_crimes.sort_values('Date')

seq_length = 7 # Az elmúlt 1 hét alapján jósoljuk a következőt
data = daily_crimes['Count'].values
X_rnn, y_rnn = [], []
for i in range(len(data) - seq_length):
    X_rnn.append(data[i:i+seq_length])
    y_rnn.append(data[i+seq_length])
    
X_rnn, y_rnn = np.array(X_rnn), np.array(y_rnn)
X_rnn = np.reshape(X_rnn, (X_rnn.shape[0], X_rnn.shape[1], 1))

# Skálázás
max_val = np.max(data)
X_rnn_scaled = X_rnn / max_val
y_rnn_scaled = y_rnn / max_val

# RNN Architektúra építése
tf.random.set_seed(42)
rnn_model = Sequential([
    SimpleRNN(64, activation='relu', input_shape=(seq_length, 1)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
rnn_model.compile(optimizer='adam', loss='mse')
rnn_model.fit(X_rnn_scaled, y_rnn_scaled, epochs=50, batch_size=4, verbose=0)

# Holnapi esetszám predikciója
last_7_days = data[-seq_length:]
last_7_days_scaled = (last_7_days / max_val).reshape(1, seq_length, 1)
predicted_count_scaled = rnn_model.predict(last_7_days_scaled, verbose=0)[0][0]
predicted_crimes_tomorrow = int(predicted_count_scaled * max_val)
print(f"-> RNN Eredmény: Holnap várhatóan {predicted_crimes_tomorrow} bűncselekmény történik összesen.")


# --- 3. MARKOV-LÁNC ---
# A bűncselekmények KATEGÓRIÁJÁNAK időbeli átmeneti valószínűségei
print("\n2/3 Lépés: Markov-lánc átmeneti valószínűségi mátrixának számítása...")
df_sorted = df.sort_values(by=['Date', 'Time'])
df_sorted['Next_Category'] = df_sorted['Category'].shift(-1)

# Átmeneti mátrix kiszámítása: P(C_t | C_{t-1})
transitions = pd.crosstab(df_sorted['Category'], df_sorted['Next_Category'], normalize='index')


# --- 4. BAYES-TÉTEL ÉS TELJES VALÓSZÍNŰSÉG TÉTELE ---
# Térbeli eloszlás meghatározása adott kategória és időpont esetén
# P(District | Category, Hour) kiszámítása
print("3/3 Lépés: Bayes-tétel alkalmazása a helyszínek predikciójához...")

# Prior: P(District)
p_district = df['PdDistrict'].value_counts(normalize=True).to_dict()

# Likelihood: P(Category, Hour | District)
likelihoods = {}
for dist in df['PdDistrict'].unique():
    dist_data = df[df['PdDistrict'] == dist]
    # Sima gyakoriság
    dist_cat_hour = dist_data.groupby(['Category', 'Hour']).size() / len(dist_data)
    likelihoods[dist] = dist_cat_hour.to_dict()

def predict_district_bayes(category, hour):
    posteriors = {}
    
    # Teljes valószínűség tétele a nevezőhöz (Marginal Likelihood):
    # Sum(P(C,H | D) * P(D)) minden D körzetre
    marginal = 0
    for dist in p_district.keys():
        like = likelihoods[dist].get((category, hour), 1e-6) # Laplace smoothing szerű pici esély
        marginal += like * p_district[dist]
        
    for dist in p_district.keys():
        like = likelihoods[dist].get((category, hour), 1e-6)
        prior = p_district[dist]
        
        # Bayes tétel: P(D | C, H) = P(C, H | D) * P(D) / P(C, H)
        posteriors[dist] = (like * prior) / marginal
        
    # A legvalószínűbb körzet visszaadása
    return max(posteriors, key=posteriors.get)


# --- 5. SZIMULÁCIÓ A HOLNAPI NAPRA (Generatív folyamat) ---
print("\n[!] Holnapi események szimulációja folyamatban...")
current_cat = df_sorted['Category'].iloc[-1]
predicted_events = []

for _ in range(predicted_crimes_tomorrow):
    # 1. Milyen bűncselekmény? (Markov-lánc lépés)
    if current_cat in transitions.index:
        next_cat_probs = transitions.loc[current_cat]
        # Kategória sorsolása a valószínűségek alapján
        next_cat = np.random.choice(next_cat_probs.index, p=next_cat_probs.values)
    else:
        next_cat = np.random.choice(df['Category'].unique())
    
    # 2. Mikor? (Történelmi eloszlás alapján az adott kategóriára)
    cat_data = df[df['Category'] == next_cat]
    if len(cat_data) > 0:
        hour = np.random.choice(cat_data['Hour'])
    else:
        hour = np.random.randint(0, 24)
        
    # 3. Hol? (Körzet predikció Bayes-tétellel)
    predicted_dist = predict_district_bayes(next_cat, hour)
    
    # 4. Pontos koordináta szimulálása a körzeten belül (kis zajjal, hogy ne pont egyezzen meg a régiekkel)
    dist_coords = df[df['PdDistrict'] == predicted_dist][['Y', 'X']].dropna().values
    if len(dist_coords) > 0:
        base_coord = dist_coords[np.random.randint(0, len(dist_coords))]
        # Gauss-zaj hozzáadása (~200 méter szórás)
        lat = base_coord[0] + np.random.normal(0, 0.002)
        lon = base_coord[1] + np.random.normal(0, 0.002)
    else:
        lat, lon = 37.77, -122.42
        
    predicted_events.append({
        'Category': next_cat, 
        'Hour': hour, 
        'PdDistrict': predicted_dist, 
        'Y': lat, 
        'X': lon
    })
    
    # Állapot frissítése a Markov lánchoz
    current_cat = next_cat

pred_df = pd.DataFrame(predicted_events)


# --- 6. KÜLÖN TÉRKÉP GENERÁLÁSA A HOLNAPI PREDIKCIÓKRA ---
print("Predikciós térkép generálása...")
m_tomorrow = folium.Map(location=[37.77, -122.42], zoom_start=12, tiles='CartoDB Dark_Matter')

# Hőtérkép a prediktált sűrűségre
HeatMap(
    pred_df[['Y', 'X']].values, 
    radius=14, 
    blur=10, 
    gradient={0.2: 'blue', 0.5: 'yellow', 1.0: 'red'}
).add_to(m_tomorrow)

# Markerek az "Erőszakos / Súlyos" predikciókhoz (például)
severe_crimes = ['ROBBERY', 'ASSAULT', 'BURGLARY', 'WEAPON LAWS']
severe_preds = pred_df[pred_df['Category'].isin(severe_crimes)].head(30) # Max 30-at mutatunk

for idx, row in severe_preds.iterrows():
    folium.Marker(
        location=[row['Y'], row['X']],
        popup=folium.Popup(f"<b>VÁRHATÓ: {row['Category']}</b><br>Idő: {row['Hour']}:00 - {row['Hour']+1}:00<br>Körzet: {row['PdDistrict']}", max_width=250),
        icon=folium.Icon(color='red', icon='warning-sign')
    ).add_to(m_tomorrow)

# Térkép mentése
output_html = 'predicted_crimes_tomorrow_V7.html'
m_tomorrow.save(output_html)
print(f"Kész! A predikciós térképet elmentettem ide: {output_html}")
