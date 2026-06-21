# San Francisco Bűnözési Elemzés Projekt

Ebben a projektben a San Francisco-i rendőrség bűnügyi adatainak átfogó elemzését mutatom be, az alapvető vizualizációtól a komplex gépi tanulási modellekig.

## Projekt Struktúra
- **Adatok:** `Map-Crime_Incidents-Previous_Three_Months.csv`
- **Notebookok:** Hét fejlődési szakaszt készítettem (Alap -> V7), a legújabb fájlokat a `Verziók` mappában találod.
- **Dokumentáció:** `javaslatok.txt` és ez a `README.md`.

## Adatforrás és Dataframe Részletes Elemzése
A dataframe-et az alábbi kulcsfontosságú attribútumokkal dolgoztam fel:
- `IncidntNum`: Egyedi azonosító. A modern verzióimban (V5+) 0-val pótoltam a hiányzókat.
- `Category` & `PdDistrict`: Kategóriális adatok, 'category' típusúvá alakítottam őket a V5/V6/V7-ben.
- `Date` & `Time`: Időbeli adatok, amelyeket professzionális `pd.to_datetime` formátumba konvertáltam.
- `X` & `Y`: Földrajzi koordináták, a hibás adatokat kiszűrtem.
- `Resolution`: Az ügy kimenetele, amit a megoldási arány elemzésem alapjaként használtam.

## Verziók Összehasonlító Elemzése

### 1-3. fázis: Az alapoktól a prezentálható jelentésig
- **Alap verzió:** Kezdeti próbálkozásom statikus ábrákkal és lassú térképpel.
- **V2 verzió:** Technikai refaktorálást végeztem a tiszta kód alapokért.
- **V3 verzió:** Adatvezérelt prezentációt készítettem a megoldási arányok vizualizációjával.

### 4. V5 verzió: Professzionális Data Science projekt
- **Jellemzők:** DBSCAN klaszterezést, WordCloud-ot és alap RandomForest predikciót alkalmaztam.
- **Fejlődés:** Bevezettem a gépi tanulást és a térbeli gócpont-azonosítást.

### 5. V6 verzió: Grandmaster Edition - Az Abszolút Csúcs
A V6 verzió a projektem legkifinomultabb változata, amelyben a modern adattudomány minden eszközét felvonultatom:
- **Hiperparaméter hangolás:** `RandomizedSearchCV`-t használtam.
- **Feature Engineering:** Bevezettem az `Is_Weekend` változót.
- **Interaktív Adatvizualizáció:** Plotly Express & Objects-et használtam.
- **Szemantikai NLP:** Bevezettem az LDA (Latent Dirichlet Allocation) modellt a témák detektálására.

### 6. V7 verzió: Ipar 4.0 és Generatív Szimuláció (Modern Ipari Technológiák)
A V7 verzió (`Verziók/san_francisco_crime_V7.ipynb`) a projektem legújabb, modern ipari (Smart City, Ipar 4.0) sztenderdekre emelt kiadása. Ebben a verzióban nem csak a múltat elemzem, hanem egy szimulált "jövőt" is generálok (Digital Twin koncepció), emellett valós időben érzékelem az anomáliákat.

#### **A. Miért használok Rekurrens Neurális Hálót (RNN)?**
Az RNN technológiát azért vezettem be, mert a hagyományos algoritmusok (pl. a V5/V6 Random Forestje) egymástól függetlenként kezelik az egyes sorokat. Ezzel szemben a bűnözési statisztikát **idősorként (Time-Series)** fogom fel, ahol a mai esetszám összefügg a tegnapival és az azelőttivel. Az RNN (SimpleRNN) "belső memóriájának" (rejtett állapotának) köszönhetően a modellem képes megtanulni az egymást követő napok sorrendiségét és a heti ciklikus mintázatokat. Ezáltal lényegesen pontosabb előrejelzést tudok adni a másnapi összesített esetszámra, mint a standard regressziós modellekkel.

#### **B. Matematikai Metodikák és Paraméterek a Szimulációban**

**1. Markov-láncok (Kategóriaváltás Valószínűsége)**
- **Képlet:** $P(C_t | C_{t-1})$
- **Metodika:** Meghatározom, hogy egy adott bűncselekmény kategória (pl. rablás) után mekkora valószínűséggel következik egy másik kategória (pl. testi sértés).
- **Paraméterek:** Egy Átmeneti Valószínűségi Mátrixot (Transition Matrix) használok, melyet a korábbi esetek gyakoriságából építek fel (`normalize='index'`).
- **Miért használom:** Így a szimulált jövőmben generált események sorrendje nem teljesen véletlenszerű, hanem statisztikailag visszatükrözi a város valós, megfigyelt eseményláncolatait.

**2. Bayes-tétel (Térbeli Eloszlás Predikciója)**
- **Képlet:** $P(D | C, H) = \frac{P(C, H | D) \cdot P(D)}{P(C, H)}$
- **Metodika:** Kiszámítom, hogy ha tudom egy bűncselekmény kategóriáját ($C$) és a megtörténésének óráját ($H$), akkor melyik rendőrségi kerület ($D$) a legvalószínűbb helyszín.
- **Paraméterek:**
  - **$P(D)$ (Prior):** A kerületek történelmi súlya (melyik mennyire "forgalmas").
  - **$P(C, H | D)$ (Likelihood):** Az adott kerületben milyen gyakori az a specifikus kategória abban az adott órában. Ide egy nagyon pici, un. *Laplace smoothing* ($10^{-6}$) zajt adok, hogy a matematikai képletben elkerüljem a 0-val való szorzást a sosem látott extrém eseteknél.
- **Miért használom:** Különböző kerületek teljesen más profilúak (pl. éjszakai rablás a külső kerületben, nappali zsebmetszés a turistás belvárosban). A Bayes-tétellel matematikai pontossággal képezem le ezeket a hely- és időfüggő összefüggéseket.

**3. Statisztikai Anomália Észlelés (Z-Score & Mozgóátlag)**
- **Képlet:** $Z = \frac{X - \mu}{\sigma}$ (Riasztás: ha $X > \mu + 2\sigma$)
- **Metodika:** Ipari folyamatszabályozási eljárást (SPC) alkalmazok. A napi esetszámot összevetem a megelőző időszak mozgóátlagával és mozgószórásával.
- **Paraméterek:**
  - **Ablakméret (Rolling Window) = 7 nap**: Azért pont ennyit állítottam be, hogy eltüntessem a heti szezonalitás okozta fals riasztásokat.
  - **Küszöb (Threshold) = 2 szórás**: Ha a napi adat ennél magasabb, azt normál eloszlás esetén 95%-os biztonsággal kiugró (szélsőséges) értéknek tekintem.
- **Miért használom:** Ezzel a statisztikai módszerrel rendkívül gyorsan és megbízhatóan azonosítom a "rendkívüli" napokat/kerületeket (pl. egy váratlan zavargás napját) anélkül, hogy indokolatlanul komplex modellekhez (pl. Autoencoder) kéne nyúlnom.

#### **C. Animált Vizualizáció (HeatMapWithTime)**
A statikus grafikonokat a V7-ben kiegészítettem egy idősíkon futó, interaktív videós hőtérképpel a notebookban, amellyel bemutatom a bűnözési gócpontok órás vándorlását.

## Verziók Összehasonlító Táblázata

| Név | Verzió | Működési elv | Különbség az előzőhöz képest | Fejlődés az előzőhöz képest | Miben kiemelkedő |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Alap Elemzés** | Alap | Lineáris szkriptelés. | Nincs (kiindulópont). | Nincs (kiindulópont). | Egyszerűség. |
| **Refaktorált Váz** | V2 | Struktúra-alapú kódolás. | Tiszta kódom, objektumok. | 'Category' és dátumkezelés bevezetése. | Kódminőség. |
| **Adatvizuális Jelentés** | V3 | Eredményközpontú elemzés. | Elmentett kimenetek, térképek. | Resolution statisztikai elemzése. | Prezentálhatóság. |
| **AI & NLP Alapok** | V5 | ML és szövegbányászat. | Klaszterezés és Random Forest. | DBSCAN gócpont azonosítás. | Adatvezérelt predikció. |
| **Grandmaster Edition** | V6 | Produkciós Data Science. | Hiperparaméter hangolásom. | Plotly vizualizáció és LDA (NLP). | Technikai fölény. |
| **Ipar 4.0 & Szimuláció** | V7 | Valószínűségi Generatív Modell. | Szimuláció és animált időtérkép. | Z-Score anomália és RNN predikció. | Matematikai komplexitás. |
