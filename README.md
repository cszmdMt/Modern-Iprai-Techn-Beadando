# San Francisco Bűnözési Elemzés Projekt

Ez a projekt a San Francisco-i rendőrség háromhavi bűnözési adatait dolgozza fel és elemzi Python környezetben, Jupyter Notebookok segítségével. A projekt célja a bűnügyi adatok vizualizálása, a gócpontok azonosítása és a különböző verziók során a kódminőség és az elemzési mélység fejlesztése.

## Projekt Struktúra
A `Beadandó` mappa tartalmazza a következő kulcsfontosságú elemeket:
- **Adatok:** `Map-Crime_Incidents-Previous_Three_Months.csv` (alapadatbázis).
- **Elemzések:** Három különböző fázisú Jupyter Notebook (`.ipynb`).
- **Dokumentáció:** `javaslatok.txt`, amely a fejlesztési irányokat fekteti le.

## Adatforrás és Dataframe Vizsgálat
Az elemzés alapját képező dataframe (`SF` vagy `df`) az alábbi oszlopokat tartalmazza:
- `IncidntNum`: Az incidens egyedi azonosítója (sok esetben hiányzó érték az alapverzióban).
- `Category`: A bűncselekmény kategóriája (pl. LARCENY/THEFT, DRUG/NARCOTIC).
- `Descript`: Részletes leírás az esetről.
- `DayOfWeek`: A hét napja.
- `Date` és `Time`: Az elkövetés pontos ideje.
- `PdDistrict`: A rendőrségi körzet megnevezése.
- `Resolution`: Az ügy kimenetele (pl. Letartóztatás vagy lezáratlan).
- `Address`: Az elkövetés helyszíne.
- `X` és `Y`: Földrajzi koordináták (Longitudinális és Latitudinális adatok).
- `Location`: A koordináták tuple formátumban.

### Elemzési lehetőségek az adatokból:
1.  **Időbeli elemzés:** Bűncselekmények eloszlása napszakok, napok és hónapok szerint (szezonalitás vizsgálata).
2.  **Kategória alapú elemzés:** A leggyakoribb bűncselekmény-típusok azonosítása.
3.  **Térbeli elemzés:** Interaktív hőtérképek (Heatmap) és klaszterezés (DBSCAN) a veszélyesebb környékek behatárolására.
4.  **Hatékonysági elemzés:** A megoldott vs. megoldatlan ügyek aránya körzetenként vagy kategóriánként.

## Verziók Összehasonlítása

### 1. `san_francisco_crime.ipynb` (Alapverzió)
Ez a verzió szolgál kiindulópontként.
- **Jellemzők:** Alapszintű adatbetöltés, egyszerű Matplotlib grafikonok és Folium térképek.
- **Hiányosságok:** Nincs érdemi adattisztítás (hiányzó értékek kezelése, adattípusok optimalizálása), a dátumokat stringként kezeli, ami korlátozza az idősoros elemzést. A vizualizációk statikusak és kevésbé esztétikusak.

### 2. `san_francisco_crime_V2.ipynb` (Refaktorált kód)
A `javaslatok.txt` alapján készült technikai fejlesztés.
- **Fejlődés az eredetihez képest:** 
    - Bevezeti a professzionális adattisztítást (NaN értékek kitöltése, `pd.to_datetime` használata).
    - Memóriaoptimalizálás a kategória típusú oszlopok használatával.
    - Seaborn könyvtár használata a modernebb vizualizációkhoz.
    - Gépi tanulási modellek (RandomForest, DBSCAN) importálása és az elemzési logika előkészítése.
- **Különlegesség:** Ez a fájl elsősorban a tiszta kódot és a logikai struktúrát tartalmazza (kisebb fájlméret az elmentett kimenetek hiánya miatt).

### 3. `san_francisco_crime_V3.ipynb` (Teljes jelentés)
A projekt legfejlettebb és legátfogóbb verziója.
- **Fejlődés a V2-höz képest:** 
    - Tartalmazza az összes futtatott eredményt, vizualizációt és interaktív térképet (HeatMap, MarkerCluster).
    - **DBSCAN Klaszterezés:** Aktív térbeli klaszterezést hajt végre a bűncselekmények sűrűsége alapján.
    - **Megoldási arány elemzés:** Részletes, százalékos kimutatás a "Resolution" státuszokról kategóriákra lebontva.
    - Szöveges magyarázatokkal és Markdown fejezetekkel ellátott, publikálható formátumú elemzés.

**Összegzés:** Míg a V2 a technikai refaktorálásra és a kódminőségre fókuszál, addig a V3 egy komplett, adatokkal alátámasztott elemzési dokumentum, amely mélyebb statisztikai és gépi tanulási betekintést nyújt San Francisco közbiztonsági helyzetébe.
