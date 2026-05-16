# San Francisco Bűnözési Elemzés Projekt

Ez a projekt a San Francisco-i rendőrség bűnügyi adatainak átfogó elemzését mutatja be, az alapvető vizualizációtól a komplex gépi tanulási modellekig.

## Projekt Struktúra
- **Adatok:** `Map-Crime_Incidents-Previous_Three_Months.csv`
- **Notebookok:** Hat fejlődési szakasz (Alap -> V6).
- **Dokumentáció:** `javaslatok.txt` és ez a `README.md`.

## Adatforrás és Dataframe Részletes Elemzése
A dataframe az alábbi kulcsfontosságú attribútumokkal rendelkezik:
- `IncidntNum`: Egyedi azonosító. A modern verziókban (V5+) 0-val pótolva.
- `Category` & `PdDistrict`: Kategóriális adatok, 'category' típusúak a V5/V6-ban.
- `Date` & `Time`: Időbeli adatok professzionális `pd.to_datetime` formátumban.
- `X` & `Y`: Földrajzi koordináták, hibás adatok szűrve.
- `Resolution`: Az ügy kimenetele, alapja a megoldási arány elemzésnek.

## Verziók Összehasonlító Elemzése

### 1-3. fázis: Az alapoktól a prezentálható jelentésig
- **Alap verzió:** Kezdeti próbálkozás, statikus ábrákkal és lassú térképpel.
- **V2 verzió:** Technikai refaktorálás, tiszta kód alapok.
- **V3 verzió:** Adatvezérelt prezentáció, megoldási arányok vizualizációja.

### 4. V5 verzió: Professzionális Data Science projekt
- **Jellemzők:** DBSCAN klaszterezés, WordCloud, alap RandomForest predikció.
- **Fejlődés:** Bevezeti a gépi tanulást és a térbeli gócpont-azonosítást.

### 5. V6 verzió: Grandmaster Edition - Az Abszolút Csúcs
A V6 verzió a projekt legkifinomultabb változata, amely a modern adattudomány minden eszközét felvonultatja:

#### **A. Haladó Gépi Tanulás és Optimalizálás**
- **Hiperparaméter hangolás:** A V5 egyszerű modelljével ellentétben itt `RandomizedSearchCV` segítségével optimalizáljuk a Random Forest paramétereit (n_estimators, max_depth), jelentősen növelve a megbízhatóságot.
- **Feature Engineering:** Új prediktív változóként megjelenik az `Is_Weekend`, felismerve, hogy a hétvégi bűnözési mintázatok eltérnek a hétköznapiaktól.

#### **B. Interaktív Adatvizualizáció (Dashboard-szint)**
- **Plotly Express & Objects:** A statikus Seaborn ábrákat interaktív **Plotly** grafikonok váltják fel. 
- **Óránkénti trendvonal:** Dinamikusan nagyítható görbe a bűnözés napi hullámzásáról.
- **TreeMap:** Hierarchikus vizualizáció, amely egyetlen ábrán mutatja meg a rendőrségi körzeteket és az ott jellemző bűncselekmény típusok arányát.

#### **C. Szemantikai NLP (Téma-modellezés)**
- **LDA (Latent Dirichlet Allocation):** A WordCloudon túlmutatva, a V6 algoritmus szinten azonosít 5 fő "témát" a bűncselekmények leírásaiban. Ez lehetővé teszi, hogy ne csak szavakat, hanem összefüggő eset-típusokat (pl. kábítószer-birtoklás vs. gépjármű-feltörés) azonosítsunk automatikusan.

#### **D. Térinformatika 2.0**
- **Dinamikus Hőtérkép:** A Folium térkép `Stamen Toner` stílussal és dinamikus rétegekkel rendelkezik, ahol a DBSCAN klaszterek közvetlenül befolyásolják a hőtérkép intenzitását.

#### **E. Szoftverfejlesztési Minőség**
- **Moduláris kód:** Az adatbetöltés és tisztítás különálló, robusztus függvényekbe került, így a kód könnyen karbantartható és skálázható.

**Összegzés:** A V6 verzió nem csupán elemzi az adatokat, hanem rejtett összefüggéseket tár fel és optimalizált predikciós motorral szolgál, reprezentálva az ipari szintű Data Science munkafolyamatokat.

## Verziók Összehasonlító Táblázata

| Név | Verzió | Működési elv | Különbség az előzőhöz képest | Fejlődés az előzőhöz képest | Miben kiemelkedő |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Alap Elemzés** | Alap | Lineáris szkriptelés, alapvető adatmegjelenítés. | Nincs (kiindulópont). | Nincs (kiindulópont). | Egyszerűség és gyors átláthatóság. |
| **Refaktorált Váz** | V2 | Struktúra-alapú kódolás, memóriakezelés. | Tiszta kód, osztályok/függvények előkészítése. | 'Category' típusok és professzionális dátumkezelés. | Kódminőség és futási sebesség. |
| **Adatvizuális Jelentés** | V3 | Eredményközpontú elemzés, vizuális prezentáció. | Elmentett kimenetek és interaktív térképi rétegek. | Megoldási arányok (Resolution) statisztikai elemzése. | Prezentálható, üzleti kész állapot. |
| **AI & NLP Alapok** | V5 | Gépi tanulás és szövegbányászat integrációja. | Prediktív modellek és térbeli klaszterezés bevezetése. | RandomForest predikció és DBSCAN gócpont azonosítás. | Adatvezérelt előrejelzés. |
| **Grandmaster Edition** | V6 | Produkciós szintű Data Science keretrendszer. | Hiperparaméter hangolás és interaktív dashboard elemek. | Plotly vizualizáció és LDA téma-modellezés (NLP). | Teljes körű technikai és elemzési fölény. |
