import streamlit as st
import pandas as pd
import pickle
import glob
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
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

regression = None
classification = None
kmeans = None

try:
    st.warning("🔄 Træner/indlæser modeller...")

    # --- Regression --------------------------------------------------
    if glob.glob("regression.pkl"):
        regression = pickle.load(open("regression.pkl", "rb"))
    else:
        X, y = dfNumeric.drop('price', axis=1), dfNumeric['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=83)
        regression = RandomForestRegressor(n_estimators=50, random_state=116)
        regression.fit(X_train, y_train)
        pickle.dump(regression, open("regression.pkl", "wb"))

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
    else:
        Xc, yc = dfClassification, dfNumeric['price']
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=88)
        classification = DecisionTreeClassifier(random_state=10)
        classification.fit(Xc_train, yc_train)
        pickle.dump(classification, open("classification.pkl", "wb"))

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
    st.write("Nedenfor vises et lille uddrag af dine data med den beregnede klynge:")
    df['cluster'] = rowCluster['cluster']
    kolonner = ['airline', 'class', 'price', 'journey_month', 'journey_week', 'journey_day', 'is_weekend', 'cluster']
    st.dataframe(df[kolonner].head())

with tab2:
    st.title("Regression med Random Forest")
    jm = st.selectbox("Rejsemåned", sorted(df['journey_month'].unique()))
    jw = st.selectbox("Rejseuge",  sorted(df['journey_week'].unique()))
    jd = st.selectbox("Rejsedag",   sorted(df['journey_day'].unique()))
    iw = st.selectbox("Er det weekend?",   [0, 1])
    al = st.selectbox("Flyselskab",       df['airline'].unique())
    fc = st.selectbox("Rejseklasse",         df['class'].unique())

    if st.button("Forudsig pris", key="regression_button"):
        inp = igt.create_input_row(jm, jw, jd, iw, al, fc, dfNumeric.drop('price', axis=1).columns)
        pred = regression.predict(inp)[0]
        st.success(f"Forudsagt pris: {pred:.2f} kr.")

with tab3:
    st.title("Klyngedannelse med K-Means")
    jm = st.selectbox("Rejsemåned", sorted(df['journey_month'].unique()), key="cjm")
    jw = st.selectbox("Rejseuge",  sorted(df['journey_week'].unique()), key="cjw")
    jd = st.selectbox("Rejsedag",   sorted(df['journey_day'].unique()), key="cjd")
    iw = st.selectbox("Er det weekend?",   [0, 1], key="ciw")
    al = st.selectbox("Flyselskab",       df['airline'].unique(), key="cal")
    fc = st.selectbox("Rejseklasse",         df['class'].unique(), key="cfc")

    if st.button("Forudsig klynge", key="cluster_button"):
        inp = igt.create_input_row(jm, jw, jd, iw, al, fc, dfCluster.columns)
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

        y_lower = y_upper + 10  # 10 for afstand mellem klynger

    ax1.set_title("Silhouette plot for de forskellige klynger")
    ax1.set_xlabel("Silhouette-koefficient")
    ax1.set_ylabel("Klynge label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Fjern y-aksens labels / ticks
    ax1.set_xticks(np.arange(-0.1, 1.1, 0.2))

    st.pyplot(fig)
    st.write(f"🔎 Silhouette score: **{silhouette_avg:.2f}**")

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

with tab4:
    st.title("Beslutningstræ Klassifikation")
    jm = st.selectbox("Rejsemåned", sorted(df['journey_month'].unique()), key="cl_jm")
    jw = st.selectbox("Rejseuge",  sorted(df['journey_week'].unique()), key="cl_jw")
    jd = st.selectbox("Rejsedag",   sorted(df['journey_day'].unique()), key="cl_jd")
    iw = st.selectbox("Er det weekend?",   [0, 1], key="cl_iw")
    al = st.selectbox("Flyselskab",       df['airline'].unique(), key="cl_al")
    fc = st.selectbox("Rejseklasse", df['class'].unique(), key="cl_fc")
    ci = st.selectbox("Klynge", sorted(dfClassification['cluster'].unique()), key="cl_ci") 

    if st.button("Forudsig pris", key="classification_button"):
        inp = igt.createNewClassRow(jm, jw, jd, iw, al, fc, ci, dfClassification)
        pris = classification.predict(inp.drop(columns=['price'], errors='ignore'))[0]
        st.success(f"Forudsagt pris: {pris:.2f} kr.")
