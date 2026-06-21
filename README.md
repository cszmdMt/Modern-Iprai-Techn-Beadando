# San Francisco Bűnözési Elemzés Projekt

Ez a projekt a San Francisco-i rendőrség bűnügyi adatainak átfogó elemzését mutatja be, az alapvető vizualizációtól a komplex gépi tanulási modellekig.

## Projekt Struktúra
- **Adatok:** `Map-Crime_Incidents-Previous_Three_Months.csv`
- **Notebookok:** Hét fejlődési szakasz (Alap -> V7), a legújabb fájlok a `Verziók` mappában találhatók.
- **Dokumentáció:** `javaslatok.txt` és ez a `README.md`.

## Adatforrás és Dataframe Részletes Elemzése
A dataframe az alábbi kulcsfontosságú attribútumokkal rendelkezik:
- `IncidntNum`: Egyedi azonosító. A modern verziókban (V5+) 0-val pótolva.
- `Category` & `PdDistrict`: Kategóriális adatok, 'category' típusúak a V5/V6/V7-ben.
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
- **Hiperparaméter hangolás:** `RandomizedSearchCV` használata.
- **Feature Engineering:** `Is_Weekend` bevezetése.
- **Interaktív Adatvizualizáció:** Plotly Express & Objects használata.
- **Szemantikai NLP:** LDA (Latent Dirichlet Allocation) bevezetése a témák detektálására.

### 6. V7 verzió: Ipar 4.0 és Generatív Szimuláció (Modern Ipari Technológiák)
A V7 verzió (`Verziók/san_francisco_crime_V7.ipynb`) a projekt legújabb, modern ipari (Smart City, Ipar 4.0) sztenderdekre emelt kiadása. Ez a verzió nem csak a múltat elemzi, hanem egy szimulált "jövőt" is generál (Digital Twin koncepció), emellett valós időben érzékeli az anomáliákat.

#### **A. Miért használunk Rekurrens Neurális Hálót (RNN)?**
Az RNN technológiát azért vezettük be, mert a hagyományos algoritmusok (pl. a V5/V6 Random Forestje) egymástól függetlenként kezelik az egyes sorokat. Ezzel szemben a bűnözési statisztika egy **idősor (Time-Series)**, ahol a mai esetszám összefügg a tegnapival és az azelőttivel. Az RNN (SimpleRNN) "belső memóriával" (rejtett állapottal) rendelkezik, így képes megtanulni az egymást követő napok sorrendiségét és a heti ciklikus mintázatokat. Ezáltal lényegesen pontosabb előrejelzést ad a másnapi összesített esetszámra, mint a standard regressziós modellek.

#### **B. Matematikai Metodikák és Paraméterek a Szimulációban**

**1. Markov-láncok (Kategóriaváltás Valószínűsége)**
- **Képlet:** $P(C_t | C_{t-1})$
- **Metodika:** Meghatározza, hogy egy adott bűncselekmény kategória (pl. rablás) után mekkora valószínűséggel következik egy másik kategória (pl. testi sértés).
- **Paraméterek:** Egy Átmeneti Valószínűségi Mátrix (Transition Matrix), melyet a korábbi esetek gyakoriságából építünk fel (`normalize='index'`).
- **Miért használjuk:** Így a szimulált jövőben generált események sorrendje nem teljesen véletlenszerű, hanem statisztikailag visszatükrözi a város valós, megfigyelt eseményláncolatait.

**2. Bayes-tétel (Térbeli Eloszlás Predikciója)**
- **Képlet:** $P(D | C, H) = \frac{P(C, H | D) \cdot P(D)}{P(C, H)}$
- **Metodika:** Kiszámítja, hogy ha tudjuk egy bűncselekmény kategóriáját ($C$) és a megtörténésének óráját ($H$), akkor melyik rendőrségi kerület ($D$) a legvalószínűbb helyszín.
- **Paraméterek:**
  - **$P(D)$ (Prior):** A kerületek történelmi súlya (melyik mennyire "forgalmas").
  - **$P(C, H | D)$ (Likelihood):** Az adott kerületben milyen gyakori az a specifikus kategória abban az adott órában. Ide egy nagyon pici, un. *Laplace smoothing* ($10^{-6}$) zajt adunk, hogy a matematikai képletben elkerüljük a 0-val való szorzást a sosem látott extrém eseteknél.
- **Miért használjuk:** Különböző kerületek teljesen más profilúak (pl. éjszakai rablás a külső kerületben, nappali zsebmetszés a turistás belvárosban). A Bayes-tétel matematikai pontossággal képezi le ezeket a hely- és időfüggő összefüggéseket.

**3. Statisztikai Anomália Észlelés (Z-Score & Mozgóátlag)**
- **Képlet:** $Z = \frac{X - \mu}{\sigma}$ (Riasztás: ha $X > \mu + 2\sigma$)
- **Metodika:** Ipari folyamatszabályozási eljárás (SPC). A napi esetszámot összevetjük a megelőző időszak mozgóátlagával és mozgószórásával.
- **Paraméterek:**
  - **Ablakméret (Rolling Window) = 7 nap**: Azért pont ennyi, hogy eltüntesse a heti szezonalitás okozta fals riasztásokat.
  - **Küszöb (Threshold) = 2 szórás**: Ha a napi adat ennél magasabb, az normál eloszlás esetén 95%-os biztonsággal kiugró (szélsőséges) értéknek számít.
- **Miért használjuk:** Ez a statisztikai módszer rendkívül gyors és megbízható módja annak, hogy azonosítsuk a "rendkívüli" napokat/kerületeket (pl. egy váratlan zavargás napját) anélkül, hogy indokolatlanul komplex modellekhez (pl. Autoencoder) nyúlnánk.

#### **C. Animált Vizualizáció (HeatMapWithTime)**
A statikus grafikonokat a V7-ben kiegészíti egy idősíkon futó, interaktív videós hőtérkép a notebookban, amely bemutatja a bűnözési gócpontok órás vándorlását.

## Verziók Összehasonlító Táblázata

| Név | Verzió | Működési elv | Különbség az előzőhöz képest | Fejlődés az előzőhöz képest | Miben kiemelkedő |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Alap Elemzés** | Alap | Lineáris szkriptelés. | Nincs (kiindulópont). | Nincs (kiindulópont). | Egyszerűség. |
| **Refaktorált Váz** | V2 | Struktúra-alapú kódolás. | Tiszta kód, objektumok. | 'Category' és dátumkezelés. | Kódminőség. |
| **Adatvizuális Jelentés** | V3 | Eredményközpontú elemzés. | Elmentett kimenetek, térképek. | Resolution statisztikai elemzése. | Prezentálhatóság. |
| **AI & NLP Alapok** | V5 | ML és szövegbányászat. | Klaszterezés és Random Forest. | DBSCAN gócpont azonosítás. | Adatvezérelt predikció. |
| **Grandmaster Edition** | V6 | Produkciós Data Science. | Hiperparaméter hangolás. | Plotly vizualizáció és LDA (NLP). | Technikai fölény. |
| **Ipar 4.0 & Szimuláció** | V7 | Valószínűségi Generatív Modell. | Szimuláció és animált időtérkép. | Z-Score anomália és RNN predikció. | Matematikai komplexitás. |
