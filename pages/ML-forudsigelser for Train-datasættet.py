import streamlit as st
import pandas as pd
import pickle
import glob
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import Input_Generator_Train as igt

# Streamlit side-konfiguration
st.set_page_config(
    page_title="ML-forudsigelser for Train-datasættet",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session-state check for nødvendige dataframes 
if 'dfTrain' not in st.session_state or 'dfTrain_numeric' not in st.session_state:
    st.error("Data mangler i session state: 'dfTrain' og 'dfTrain_numeric'. Sørg for at indlæse data først.")
    st.stop() # Stop appen hvis data ikke er klar

# Loader dataframes fra session state
df = st.session_state['dfTrain']
dfNumeric = st.session_state['dfTrain_numeric']

# Fjerner 'price' fra data til clustering og klassifikation 
dfCluster = dfNumeric.drop(['price'], axis=1)
dfClassification = dfNumeric.copy().drop(['price'], axis=1)

# Opret pris-klasser vha. qcut for klassifikation 
try:
    price_classes = pd.qcut(dfNumeric['price'], q=5, duplicates='drop') # Deler prisen i 5 kvantiler (pris-klasser)
    price_intervals = price_classes.cat.categories # Gemmer interval-bounds for visning
except Exception as e:
    st.error(f"Fejl ved oprettelse af pris-klasser: {e}")
    st.stop()

# Initialisering af model-objekter
regression = None
classification = None
kmeans = None

try:
    st.warning("Træner/indlæser modeller...")

    # Regression model træning/indlæsning 
    if glob.glob("regression_Train.pkl"):

        # Hvis model findes, load den
        regression = pickle.load(open("regression_Train.pkl", "rb"))

        # Hvis testdata ikke findes i session, opret dem
        if 'X_test_reg_Train' not in st.session_state or 'y_test_reg_Train' not in st.session_state:
            X, y = dfNumeric.drop('price', axis=1), dfNumeric['price']
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=83)
            st.session_state['X_test_reg_Train'] = X_test
            st.session_state['y_test_reg_Train'] = y_test
    else:
        # Hvis ingen model, split data og træn ny RandomForestRegressor
        X, y = dfNumeric.drop('price', axis=1), dfNumeric['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=83)
        regression = RandomForestRegressor(n_estimators=50, random_state=116)
        regression.fit(X_train, y_train)
        pickle.dump(regression, open("regression_Train.pkl", "wb"))
        st.session_state['X_test_reg_Train'] = X_test
        st.session_state['y_test_reg_Train'] = y_test

    # Clustering model træning/indlæsning
    if glob.glob("cluster_Train.pkl") and glob.glob("cluster_Train.csv"):

        # Loader kmeans model og tilhørende cluster labels hvis findes
        kmeans = pickle.load(open("cluster_Train.pkl", "rb"))
        rowCluster = pd.read_csv("cluster_Train.csv")
    else:
        # Ellers træn kmeans med valgt antal klynger (9)
        antal_klynger = 9
        kmeans = KMeans(init='k-means++', n_clusters=antal_klynger, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(dfCluster)
        rowCluster = pd.DataFrame(clusters, columns=['cluster'])
        rowCluster.to_csv("cluster_Train.csv", index=False)  # Gemmer labels
        pickle.dump(kmeans, open("cluster_Train.pkl", "wb"))  # Gemmer model

    # Klassifikation model træning/indlæsning ---
    # Tilføjer cluster kolonnen til klassifikationsdata
    dfClassification['cluster'] = rowCluster['cluster']

    if glob.glob("classification_Train.pkl"):
        classification = pickle.load(open("classification_Train.pkl", "rb"))
        if 'Xc_test_Train' not in st.session_state or 'yc_test_Train' not in st.session_state:
            Xc, yc = dfClassification, price_classes.cat.codes
            _, Xc_test, _, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=88)
            st.session_state['Xc_test_Train'] = Xc_test
            st.session_state['yc_test_Train'] = yc_test
    else:
        Xc, yc = dfClassification, price_classes.cat.codes
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=88)
        classification = DecisionTreeClassifier(random_state=10)
        classification.fit(Xc_train, yc_train)
        pickle.dump(classification, open("classification_Train.pkl", "wb"))
        st.session_state['Xc_test_Train'] = Xc_test
        st.session_state['yc_test_Train'] = yc_test

except Exception as e:
    st.error(f"Model-fejl: {e}")
    st.stop()

# Bruges i clustering silhouette plot
X = dfCluster.copy()

# Streamlit UI med faner 
tab1, tab2, tab3, tab4 = st.tabs(["Om", "Regression", "Clustering", "Classification"])

with tab1:
    st.title("Om")
    st.write("Hver fane indeholder en trænet model klar til at lave forudsigelser.")
    st.write("Nedenfor vises et lille uddrag af dataet med den beregnede klynge:")
    df['cluster'] = rowCluster['cluster']
    kolonner = ['airline', 'class', 'price', 'journey_month', 'journey_week', 'journey_day', 'is_weekend', 'cluster']
    st.dataframe(df[kolonner].head())

with tab2:
    st.title("Regression")

    st.write(
        "Denne Random Forest Regression model anvender rejsemåned, uge, dag, weekendstatus, flyselskab og rejseklasse som input for at forudsige "
        "prisen på en rejse. Modellen estimerer en konkret prisværdi, der kan bruges til at danne en forventning om rejseomkostningerne."
    )

    st.write(
    "Vi anvender Random Forest Regressor fremfor andre regressorer, da det er en robust og nøjagtig algoritme. "
    "Algoritmen bygger på mange træer, som sammen bidrager til en mere præcis og pålidelig forudsigelse end mange andre modeller."
    )


    # Brugervalg til inputvariable
    jm = st.selectbox("Rejsemåned", sorted(df['journey_month'].unique()))
    jw = st.selectbox("Rejseuge", sorted(df['journey_week'].unique()))
    jd = st.selectbox("Rejsedag", sorted(df['journey_day'].unique()))
    iw = st.selectbox("Er det weekend?", [0, 1])
    al = st.selectbox("Flyselskab", df['airline'].unique())
    fc = st.selectbox("Rejseklasse", df['class'].unique())

    # Knappen til at lave forudsigelse
    if st.button("Forudsig pris", key="regression_button"):
        inp = igt.create_input_row(jm, jw, jd, iw, al, fc, dfNumeric.drop('price', axis=1).columns)
        pred = regression.predict(inp)[0]
        st.success(f"Forudsagt pris: {pred:.2f} kr.")


    # Viser modelperformance metrics på testdata
    if 'X_test_reg_Train' in st.session_state and 'y_test_reg_Train' in st.session_state:
        y_pred = regression.predict(st.session_state['X_test_reg_Train'])
        mse = mean_squared_error(st.session_state['y_test_reg_Train'], y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(st.session_state['y_test_reg_Train'], y_pred)
        r2 = r2_score(st.session_state['y_test_reg_Train'], y_pred)

        st.subheader("Model Performance Metrics")
        st.write(f"MSE: {mse:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAE: {mae:.2f}")
        st.write(f"R²: {r2:.2f}")

        st.title("Regression analyse")

        # Forklaring af metrics
        st.write(f"""

        - **MSE (Mean Squared Error):** Måler den gennemsnitlige kvadrerede fejl mellem de forudsagte og faktiske værdier. En MSE på **{mse:,.2f}** indikerer, at der stadig er betydelige fejl især store afvigelser vægtes tungt.

        - **RMSE (Root Mean Squared Error):** Kvadratroden af MSE og udtrykt i samme enhed som målet (pris). En RMSE på **{rmse:,.2f}** betyder, at de gennemsnitlige afvigelser fra faktiske priser er cirka **{rmse:,.0f}**

        - **MAE (Mean Absolute Error):** Giver gennemsnittet af de absolutte fejl uden at forstærke ekstreme outliers. En MAE på **{mae:,.2f}** viser, at modellen i gennemsnit afviger med ca. **{mae:,.0f}**, hvilket er ret præcist.

        - **R² (Determinationskoefficient):** Viser hvor stor en andel af variationen i data modellen kan forklare. En værdi på **{r2:.2f}** betyder, at modellen forklarer **{r2*100:.0f}%** af prisvariationen hvilket er en god forklaringsgrad.
        """)

        st.write(f"Modellen har en solid præcision med en RMSE på **{rmse:,.2f}** og forklarer **{r2*100:.0f}%** af variationen i priserne.")


with tab3:
    st.title("Clustering")

    st.write(
        "Denne clustering model bruger K-Means clustering til at gruppere rejser baseret på rejsemåned, uge, dag, weekendstatus, "
        "flyselskab, rejseklasse og pris. Formålet er at identificere naturlige grupperinger i data for bedre indsigt."
    )

    st.write(
    "Vi har valgt K-Means, fordi det er en enkel og effektiv algoritme, som samtidig gør det let at forstå resultaterne. "
    "Algoritmen deler dataene op i klynger, hvor hver klynge har sit eget centrum. "
    "Selvom datasættet ikke er særligt stort, kunne vi også have brugt Hierarkisk Klyngedannelse. "
    "Men en stor fordel ved K-Means er, at vi selv kan bestemme antallet af klynger."
    )

    # Brugervalg til inputvariabler inkl. pris for clustering
    jm = st.selectbox("Rejsemåned", sorted(df['journey_month'].unique()), key="cjm")
    jw = st.selectbox("Rejseuge", sorted(df['journey_week'].unique()), key="cjw")
    jd = st.selectbox("Rejsedag", sorted(df['journey_day'].unique()), key="cjd")
    iw = st.selectbox("Er det weekend?", [0, 1], key="ciw")
    al = st.selectbox("Flyselskab", df['airline'].unique(), key="cal")
    fc = st.selectbox("Rejseklasse", df['class'].unique(), key="cfc")
    price = st.number_input("Pris", min_value=0.0, format="%.2f", key="cprice")

    if st.button("Forudsig klynge", key="cluster_button"):
        inp = igt.createNewRow(jm, jw, jd, iw, al, fc, price, dfCluster)
        label = kmeans.predict(inp)[0]
        st.success(f"Klynge: {label}")

    # Visualisering af clusters via silhouette plot

    st.subheader("Silhouette Plot")

    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    fig, ax1 = plt.subplots(figsize=(8, 6))

    for i in range(kmeans.n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        farve = cm.nipy_spectral(float(i) / kmeans.n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                          facecolor=farve, edgecolor=farve, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax1.set_title("Silhouette plot for de forskellige klynger")
    ax1.set_xlabel("Silhouette-koefficient")
    ax1.set_ylabel("Klynge label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks(np.arange(-0.1, 1.1, 0.2))

    st.pyplot(fig)
    st.write(f"Silhouette score: **{silhouette_avg:.2f}**")

    st.write(
        """
        Silhouette scoren er et mål for, hvor godt data er opdelt i klynger. Den varierer fra -1 til 1, hvor:

        - Tæt på 1 betyder, at datapunkterne er godt placeret i deres egen klynge og klart adskilt fra andre klynger.
        - Tæt på 0 indikerer, at datapunkterne ligger tæt på grænsen mellem to klynger, og klyngeopdelingen derfor er mindre tydelig.
        - Under 0 betyder, at datapunkterne muligvis er forkert klassificeret.

        En score på omkring 0.41 tyder på, at klyngerne har en nogenlunde klar adskillelse, men der er stadig overlap og mulighed for forbedring. Det kan være acceptabelt i komplekse eller virkelighedsnære data, hvor klare grænser mellem klynger ikke altid findes.
        """
    )

    st.subheader("Analyse af antal klynger (K)")

    række = st.columns(2)
    krækkevidde = range(2, 12)
    with række[0]:
        scores = []
        for k in krækkevidde:
            model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
            score = silhouette_score(X, model.labels_)
            scores.append(score)

        plot = plt.figure()
        plt.plot(krækkevidde, scores, 'bx-')
        plt.xlabel('Antal klynger (K)')
        plt.ylabel('Silhouette Score')
        plt.title("Silhouette Score pr. K")
        st.pyplot(plot)

    with række[1]:
        distortions = []
        for k in krækkevidde:
            model = KMeans(n_clusters=k, n_init=10).fit(X)
            distortions.append(model.inertia_)

        plot2 = plt.figure()
        plt.plot(krækkevidde, distortions, 'bx-')
        plt.xlabel('Antal klynger (K)')
        plt.ylabel('Distortion')
        plt.title("Elbow Metode")
        st.pyplot(plot2)

    st.title("Clustering analyse")

    # Dynamisk analyse tekst baseret på Silhouette score og Elbow metode
    optimal_k = krækkevidde[np.argmax(scores)]
    max_score = max(scores)
    chosen_k = kmeans.n_clusters  # Antager dette er det antal klynger du har valgt

    st.write("Antallet af testede klynger er begrænset til 11, da betydningen af klynger mindskes ved større antal.")

    st.write(
        f"Ud fra grafikkerne kan vi se, at det optimale antal klynger er {optimal_k}. "
        f"Dette skyldes, at Silhouette scoren er højest ved {optimal_k}, "
        "og at Elbow grafen viser et knæk (inflection point) omkring dette antal."
    )

    if chosen_k != optimal_k:
        score_diff = abs(max_score - scores[chosen_k - krækkevidde.start])
        st.write(
            f"På trods af dette er antallet af klynger valgt til {chosen_k} for at muliggøre en mere detaljeret analyse. "
            f"Forskellen i Silhouette score mellem {optimal_k} og {chosen_k} er kun {score_diff:.2f}, hvilket er acceptabelt."
        )

    st.write(
        "En god beskrivelsesgrad af klyngerne er vigtig, da resultatet bruges til klassificeringsmodellen. "
        "Jo mere detaljerede klyngerne er, desto mere præcis bliver klassificeringen, når man forudsiger nye datapunkter baseret på klyngetilhørsforhold."
    )

with tab4:
    st.title("Classification")

    st.write(
        "Denne Classificationmodel baseret på DecisionTreeClassifier bruges til at forudsige pris-klassen for en ny observation baseret på den klynge, "
        "den tilhører. Modellen er trænet med prisen som afhængig variabel, hvor alle andre datafelter fungerer som uafhængige variable. "
        "Formålet er at estimere et sandsynligt prisinterval for den givne kombination af rejseparametre."
    )

    st.write(
        "Vi bruger DecisionTreeClassifier, fordi det er en simpel og effektiv algoritme, som både håndterer komplekse data og giver let forståelige resultater. "
        "Den kan håndtere både numeriske og kategoriske variable uden behov for omfattende dataforberedelse, og træerne kan visualiseres for bedre indsigt."
    )

    # Brugervalg til klassifikation (samme som regression + cluster)
    jm = st.selectbox("Rejsemåned", sorted(df['journey_month'].unique()), key="cl_jm")
    jw = st.selectbox("Rejseuge", sorted(df['journey_week'].unique()), key="cl_jw")
    jd = st.selectbox("Rejsedag", sorted(df['journey_day'].unique()), key="cl_jd")
    iw = st.selectbox("Er det weekend?", [0, 1], key="cl_iw")
    al = st.selectbox("Flyselskab", df['airline'].unique(), key="cl_al")
    fc = st.selectbox("Rejseklasse", df['class'].unique(), key="cl_fc")
    ci = st.selectbox("Klynge", sorted(dfClassification['cluster'].unique()), key="cl_ci")

    if st.button("Forudsig pris-klasse", key="classification_button"):
        inp = igt.createNewClassRow(jm, jw, jd, iw, al, fc, ci, dfClassification)
        
        # Tilføjer cluster info til input til klassifikationsmodel
        inp['cluster'] = ci

        # Sørger for at input har samme kolonner som træningsdata til klassifikation
        inp_class = inp[dfClassification.columns]

        prediction = classification.predict(inp_class)[0]

        try:
            interval_for_class = price_intervals[prediction]
            interval_str = f"[{interval_for_class.left:.2f}, {interval_for_class.right:.2f}]"
        except Exception:
            interval_str = "Ukendt interval"

        st.success(f"Forudsagt pris-klasse: {prediction} svarer til intervallet {interval_str}")

        # Evalueringsmatrix
        if 'Xc_test_Train' in st.session_state and 'yc_test_Train' in st.session_state:
            y_test_pred = classification.predict(st.session_state['Xc_test_Train'])
            acc = accuracy_score(st.session_state['yc_test_Train'], y_test_pred)
            cmatrix = confusion_matrix(st.session_state['yc_test_Train'], y_test_pred)

            st.write(f"Model Accuracy: {acc:.2f}")
            fig, ax = plt.subplots()
            sns.heatmap(cmatrix, annot=True, fmt='d', cmap='Blues', linewidths=0.5, linecolor='gray', ax=ax)
            ax.set_xlabel('Forudsagt')
            ax.set_ylabel('Sand')
            st.pyplot(fig)

            st.title("Classification analyse")

            st.write(
                f"Modellen opnår en accuracy på {acc:.2f}, hvilket betyder, at den korrekt klassificerer prisgrupperne lidt under halvdelen af gangene. Dette indikerer, at der stadig er plads til forbedringer, især når prisintervallerne ligger tæt på hinanden. "
                "Dette indikerer, at der stadig er plads til forbedringer, især når prisintervallerne ligger tæt på hinanden."
            )

            st.write(
            """
            **Confusion matrixen** viser, hvordan klassifikationsmodellen præsterer på testdata ved at sammenligne de *sande pris-klasser* med de *forudsagte*.

            - **Diagonalværdierne** (fx 191 for klasse 0, 136 for klasse 1 osv.) viser antallet af korrekt klassificerede observationer i hver pris-klasse.  
            - **Tal uden for diagonalen** viser fejlklassifikationer, dvs. observationer der blev forudsagt til en forkert pris-klasse.

            For eksempel bliver **83 observationer**, som i virkeligheden tilhører klasse 0, fejlagtigt forudsagt som klasse 1.

            Modellen klarer sig bedst for **klasse 0** og **klasse 4**, hvor flest observationer klassificeres korrekt.  
            For klasser midt i skalaen (**klasse 2** og **klasse 3**) ses flere fejlklassifikationer, hvilket tyder på, at modellen har sværere ved at skelne mellem nærtliggende prisintervaller.
            """
            )

            

