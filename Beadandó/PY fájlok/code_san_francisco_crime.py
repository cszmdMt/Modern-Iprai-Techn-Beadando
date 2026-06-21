#pip install folium

# --- CELL ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium

# --- CELL ---

dataset_path = "Map-Crime_Incidents-Previous_Three_Months.csv"
SF = pd.read_csv(dataset_path)

# --- CELL ---

SF.head()

# --- CELL ---

pd.set_option('display.max_rows', 8)
SF

# --- CELL ---

SF.columns

# --- CELL ---

print(len(SF))
SF.head(2)

# --- CELL ---

SF['Date']

# --- CELL ---

SF['Month'] = SF['Date'].apply(lambda row: int(row[0:2]))
SF['Day'] = SF['Date'].apply(lambda row: int(row[3:5]))
SF.head(5)

# --- CELL ---

print(SF['Month'][0:2])
print(SF['Day'][0:2])

# --- CELL ---

#del SF['IncidntNum']
SF.head(1)
SF.to_csv('test.csv')

# --- CELL ---

SF.drop('Location', axis=1, inplace=True)

# --- CELL ---

SF

# --- CELL ---

CountCategory = SF['Category'].value_counts()
print(CountCategory)

# --- CELL ---

SF['Category'].value_counts(ascending=True)

# --- CELL ---

SF['PdDistrict'].value_counts(ascending=False)


# --- CELL ---

AugustCrimes = SF[SF['Month'] == 8]
AugustCrimes

# --- CELL ---

AugustCrimesBurglary = AugustCrimes[AugustCrimes['Category'] == "BURGLARY"]
AugustCrimesBurglary

# --- CELL ---

Crime0704 = SF.query('Month == 7 and Day == 4')
Crime0704

# --- CELL ---

plt.plot(SF['X'], SF['Y'], 'ro')
plt.show()

# --- CELL ---

pd_districts = np.unique(SF['PdDistrict'])
print(pd_districts)
pd_districts_levels = dict(zip(pd_districts, range(len(pd_districts))))
print(pd_districts_levels)

# --- CELL ---

SF['PdDistrictCode'] = SF['PdDistrict'].apply(lambda row: pd_districts_levels[row])

# --- CELL ---

plt.scatter(SF['X'], SF['Y'], c=SF['PdDistrictCode'])
plt.show()

# --- CELL ---

from matplotlib import colors

districts = np.unique(SF['PdDistrict'])
print(districts)
print(colors.cnames.values())
print(colors.cnames)
print(list(colors.cnames.values())[0:len(districts)])

# --- CELL ---

color_dict = dict(zip(districts, list(colors.cnames.values())[0:-1:len(districts)]))
print(color_dict)

# --- CELL ---

map_osm = folium.Map(tiles="OpenStreetMap", zoom_start=10,
                     max_zoom=23, control_scale=True,
                     location=[
                         SF['Y'].mean(),
                         SF['X'].mean()
                     ])

plotEvery = 5
obs = list(zip(SF['Y'], SF['X'], SF['PdDistrict']))

for el in obs[0:-1:plotEvery]:
    folium.CircleMarker(
        el[0:2], color=color_dict[el[2]],
        fill_color=el[2], radius=10
    ).add_to(map_osm)

map_osm