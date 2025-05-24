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

st.set_page_config(
    page_title="Prediction Med ML",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session-state dataframes ---------------------------------------
if 'dfTrain' not in st.session_state or 'dfTrain_numeric' not in st.session_state:
    st.error("❗ Data mangler i session state: 'dfTrain' og 'dfTrain_numeric'. Sørg for at indlæse data først.")
    st.stop()

df = st.session_state['dfTrain']
dfNumeric = st.session_state['dfTrain_numeric']
dfCluster = dfNumeric.drop(['price'], axis=1)
dfClassification = dfNumeric.copy().drop(['price'], axis=1)

# --- Opret pris-klasser med qcut til klassifikationens labels ---
try:
    price_classes = pd.qcut(dfNumeric['price'], q=5, duplicates='drop')
    price_intervals = price_classes.cat.categories
except Exception as e:
    st.error(f"Fejl ved oprettelse af pris-klasser: {e}")
    st.stop()

regression = None
classification = None
kmeans = None

try:
    st.warning("🔄 Træner/indlæser modeller...")

    # --- Regression --------------------------------------------------
    if glob.glob("regression.pkl"):
        regression = pickle.load(open("regression.pkl", "rb"))
        if 'X_test_reg' not in st.session_state or 'y_test_reg' not in st.session_state:
            X, y = dfNumeric.drop('price', axis=1), dfNumeric['price']
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=83)
            st.session_state['X_test_reg'] = X_test
            st.session_state['y_test_reg'] = y_test
    else:
        X, y = dfNumeric.drop('price', axis=1), dfNumeric['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=83)
        regression = RandomForestRegressor(n_estimators=50, random_state=116)
        regression.fit(X_train, y_train)
        pickle.dump(regression, open("regression.pkl", "wb"))
        st.session_state['X_test_reg'] = X_test
        st.session_state['y_test_reg'] = y_test

    # --- Clustering --------------------------------------------------
    if glob.glob("cluster.pkl") and glob.glob("data/cluster.csv"):
        kmeans = pickle.load(open("cluster.pkl", "rb"))
        rowCluster = pd.read_csv("data/cluster.csv")
    else:
        antal_klynger = 9
        kmeans = KMeans(init='k-means++', n_clusters=antal_klynger, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(dfCluster)
        rowCluster = pd.DataFrame(clusters, columns=['cluster'])
        rowCluster.to_csv("data/cluster.csv", index=False)
        pickle.dump(kmeans, open("cluster.pkl", "wb"))

    # --- Klassifikation ----------------------------------------------
    dfClassification['cluster'] = rowCluster['cluster']

    if glob.glob("classification.pkl"):
        classification = pickle.load(open("classification.pkl", "rb"))
        if 'Xc_test' not in st.session_state or 'yc_test' not in st.session_state:
            Xc, yc = dfClassification, price_classes.cat.codes
            _, Xc_test, _, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=88)
            st.session_state['Xc_test'] = Xc_test
            st.session_state['yc_test'] = yc_test
    else:
        Xc, yc = dfClassification, price_classes.cat.codes
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=88)
        classification = DecisionTreeClassifier(random_state=10)
        classification.fit(Xc_train, yc_train)
        pickle.dump(classification, open("classification.pkl", "wb"))
        st.session_state['Xc_test'] = Xc_test
        st.session_state['yc_test'] = yc_test

except Exception as e:
    st.error(f"❌ Model-fejl: {e}")
    st.stop()

# Bruges til clustering og score plots
X = dfCluster.copy()

# --- UI Tabs --------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Om", "Regression", "Klyngedannelse", "Klassifikation"])

with tab1:
    st.title("Om")
    st.write("Hver fane indeholder en trænet model klar til at lave forudsigelser.")
    st.write("Nedenfor vises et lille uddrag af dataet med den beregnede klynge:")
    df['cluster'] = rowCluster['cluster']
    kolonner = ['airline', 'class', 'price', 'journey_month', 'journey_week', 'journey_day', 'is_weekend', 'cluster']
    st.dataframe(df[kolonner].head())

with tab2:
    st.title("Regression med Random Forest")

    st.write(
        "Denne regressionsmodel anvender rejsemåned, uge, dag, weekendstatus, flyselskab og rejseklasse som input for at forudsige "
        "prisen på en rejse. Modellen estimerer en konkret prisværdi, der kan bruges til at danne en forventning om rejseomkostningerne."
    )

    jm = st.selectbox("Rejsemåned", sorted(df['journey_month'].unique()))
    jw = st.selectbox("Rejseuge", sorted(df['journey_week'].unique()))
    jd = st.selectbox("Rejsedag", sorted(df['journey_day'].unique()))
    iw = st.selectbox("Er det weekend?", [0, 1])
    al = st.selectbox("Flyselskab", df['airline'].unique())
    fc = st.selectbox("Rejseklasse", df['class'].unique())

    if st.button("Forudsig pris", key="regression_button"):
        inp = igt.create_input_row(jm, jw, jd, iw, al, fc, dfNumeric.drop('price', axis=1).columns)
        pred = regression.predict(inp)[0]
        st.success(f"Forudsagt pris: {pred:.2f} kr.")


    if 'X_test_reg' in st.session_state and 'y_test_reg' in st.session_state:
        y_pred = regression.predict(st.session_state['X_test_reg'])
        mse = mean_squared_error(st.session_state['y_test_reg'], y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(st.session_state['y_test_reg'], y_pred)
        r2 = r2_score(st.session_state['y_test_reg'], y_pred)

        st.subheader("Model Performance Metrics")
        st.write(f"MSE: {mse:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAE: {mae:.2f}")
        st.write(f"R²: {r2:.2f}")

        st.title("Regression analyse")

        st.write(f"""

        - **MSE (Mean Squared Error):** Måler den gennemsnitlige kvadrerede fejl mellem de forudsagte og faktiske værdier. En MSE på **{mse:,.2f}** indikerer, at der stadig er betydelige fejl især store afvigelser vægtes tungt.

        - **RMSE (Root Mean Squared Error):** Kvadratroden af MSE og udtrykt i samme enhed som målet (pris). En RMSE på **{rmse:,.2f}** betyder, at de gennemsnitlige afvigelser fra faktiske priser er cirka **{rmse:,.0f}**

        - **MAE (Mean Absolute Error):** Giver gennemsnittet af de absolutte fejl uden at forstærke ekstreme outliers. En MAE på **{mae:,.2f}** viser, at modellen i gennemsnit afviger med ca. **{mae:,.0f}**, hvilket er ret præcist.

        - **R² (Determinationskoefficient):** Viser hvor stor en andel af variationen i data modellen kan forklare. En værdi på **{r2:.2f}** betyder, at modellen forklarer **{r2*100:.0f}%** af prisvariationen hvilket er en god forklaringsgrad.
        """)

        st.write(f"Modellen har en solid præcision med en RMSE på **{rmse:,.2f}** og forklarer **{r2*100:.0f}%** af variationen i priserne.")


with tab3:
    st.title("Klyngedannelse med K-Means")

    st.write(
        "Denne klyngemodel bruger K-Means clustering til at gruppere rejser baseret på rejsemåned, uge, dag, weekendstatus, "
        "flyselskab, rejseklasse og pris. Formålet er at identificere naturlige grupperinger i data for bedre indsigt."
    )

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

        En score på omkring 0.40 tyder på, at klyngerne har en nogenlunde klar adskillelse, men der er stadig overlap og mulighed for forbedring. Det kan være acceptabelt i komplekse eller virkelighedsnære data, hvor klare grænser mellem klynger ikke altid findes.
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

    st.title("Klynge analyse")

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
    st.title("Beslutningstræ Klassifikation")

    st.write(
        "Denne klassifikationsmodel bruges til at forudsige pris-klassen for en ny observation baseret på den klynge, "
        "den tilhører. Modellen er trænet med prisen som afhængig variabel, hvor alle andre datafelter fungerer som uafhængige variable. "
        "Formålet er at estimere et sandsynligt prisinterval for den givne kombination af rejseparametre."
    )

    jm = st.selectbox("Rejsemåned", sorted(df['journey_month'].unique()), key="cl_jm")
    jw = st.selectbox("Rejseuge", sorted(df['journey_week'].unique()), key="cl_jw")
    jd = st.selectbox("Rejsedag", sorted(df['journey_day'].unique()), key="cl_jd")
    iw = st.selectbox("Er det weekend?", [0, 1], key="cl_iw")
    al = st.selectbox("Flyselskab", df['airline'].unique(), key="cl_al")
    fc = st.selectbox("Rejseklasse", df['class'].unique(), key="cl_fc")
    ci = st.selectbox("Klynge", sorted(dfClassification['cluster'].unique()), key="cl_ci")

    if st.button("Forudsig pris-klasse", key="classification_button"):
        inp = igt.create_input_row(jm, jw, jd, iw, al, fc, dfNumeric.drop('price', axis=1).columns)
        
        # Tilføj cluster info til input til klassifikationsmodel
        inp['cluster'] = ci

        # Sørg for at input har samme kolonner som træningsdata til klassifikation
        inp_class = inp[dfClassification.columns]

        prediction = classification.predict(inp_class)[0]

        try:
            interval_for_class = price_intervals[prediction]
            interval_str = f"[{interval_for_class.left:.2f}, {interval_for_class.right:.2f}]"
        except Exception:
            interval_str = "Ukendt interval"

        st.success(f"Forudsagt pris-klasse: {prediction} svarer til intervallet {interval_str}")

        if 'Xc_test' in st.session_state and 'yc_test' in st.session_state:
            y_test_pred = classification.predict(st.session_state['Xc_test'])
            acc = accuracy_score(st.session_state['yc_test'], y_test_pred)
            cmatrix = confusion_matrix(st.session_state['yc_test'], y_test_pred)

            st.write(f"Model Accuracy: {acc:.2f}")
            fig, ax = plt.subplots()
            sns.heatmap(cmatrix, annot=True, fmt='d', cmap='Blues', linewidths=0.5, linecolor='gray', ax=ax)
            ax.set_xlabel('Forudsagt')
            ax.set_ylabel('Sand')
            st.pyplot(fig)

            st.title("Klassifikation analyse")

            st.write(
                f"Modellen opnår en accuracy på {acc:.2f}, hvilket betyder, at den korrekt klassificerer prisgrupperne lidt mere end halvdelen af gangene. "
                "Dette indikerer, at der stadig er plads til forbedringer, især når prisintervallerne ligger tæt på hinanden."
            )

            st.write(
            """
            Confusion matrixen viser, hvordan klassifikationsmodellen præsterer på testdata ved at sammenligne de sande pris-klasser med de forudsagte.

            - Diagonalværdierne (fx 263 for klasse 0, 193 for klasse 1 osv.) viser antallet af korrekt klassificerede observationer i hver pris-klasse.
            - Tal uden for diagonalen viser fejlklassifikationer, dvs. observationer, der blev forudsagt til en forkert pris-klasse.
            - For eksempel bliver 51 observationer, som i virkeligheden tilhører klasse 0, fejlagtigt forudsagt som klasse 1.
            - Modellen klarer sig bedst for klasse 0 og klasse 4, hvor mange observationer klassificeres korrekt.
            - For klasser midt i skalaen (2 og 3) ses flere fejlklassifikationer, hvilket tyder på, at modellen har sværere ved at skelne mellem nærtliggende prisintervaller.
            """
            )

