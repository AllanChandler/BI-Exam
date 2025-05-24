import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import skew, kurtosis

st.set_page_config(page_title="Flight Price Prediction", layout="wide")

# Title
st.title("✈️ Spg. 2 Hvordan påvirker antallet af stop prisen?")

#Brug dfClean fra session_state
if 'dfClean' in st.session_state:
    df = st.session_state['dfClean']
else:
    st.error("❌ Data ikke tilgængelig i session_state. Sørg for at 'dfClean' er indlæst korrekt.")
    st.stop()

# --------------------------
# VISNING OG VISUALISERING
# --------------------------
st.subheader("🔍 Datavisning")
if st.checkbox("Vis et udsnit af data"):
    st.dataframe(df.sample(5))

if st.checkbox("Grundlæggende statistik"):

    st.write(df[["price", "stops_numb"]].describe())

   
st.subheader("📊 Visualiseringer fra analysen")


if st.checkbox("Vis histogram med skævhed og kurtosis"):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['price'], kde=True, bins=50, ax=ax)
    ax.set_title("Fordeling af Flypriser")
    ax.set_xlabel("Pris")
    ax.set_ylabel("Antal")
    st.pyplot(fig)

    price_skew = skew(df['price'].dropna())
    price_kurt = kurtosis(df['price'].dropna())

    st.markdown("### 📈 Statistik for fordeling af priser")
    st.write(f"**Skævhed (Skewness):** {price_skew:.3f}")
    st.write(f"**Kurtosis:** {price_kurt:.3f}")

    if price_skew > 0:
        st.info("📌 Fordelingen er højreskæv (positiv skævhed).")
    elif price_skew < 0:
        st.info("📌 Fordelingen er venstreskæv (negativ skævhed).")
    else:
        st.info("📌 Fordelingen er symmetrisk.")

    if price_kurt > 3:
        st.info("📌 Fordelingen er spidsere end en normalfordeling (høj kurtosis).")
    elif price_kurt < 3:
        st.info("📌 Fordelingen er fladere end en normalfordeling (lav kurtosis).")
    else:
        st.info("📌 Fordelingen har samme spidshed som en normalfordeling.")


# Sektion: Korrelation mellem antal stop og pris
if st.checkbox("Vis korrelation mellem antal stop og pris"):
    if 'stops_numb' in df.columns and 'price' in df.columns:

        st.subheader("📈 Korrelation: Antal Stop vs. Pris")

        # Beregn korrelation
        corr = df[['stops_numb', 'price']].corr()

        # Vis korrelationstabel
        st.write("Korrelationsmatrix:")
        st.dataframe(corr)

        # Tegn heatmap
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Korrelation mellem Antal Stop og Pris')
        st.pyplot(fig)




if st.checkbox("Vis scatter plot: Antal stop vs. Pris"):
    fig2, ax2 = plt.subplots()
    ax2.scatter(df['stops_numb'], df['price'], alpha=0.3)
    ax2.set_title("Sammenhæng mellem antal stop og pris")
    ax2.set_xlabel("Antal stop")
    ax2.set_ylabel("Pris")
    st.pyplot(fig2)

if st.checkbox("Vis boxplot: Pris fordelt på antal stop"):
    df_sorted = df.sort_values('stops_numb')
    fig3, ax3 = plt.subplots()
    df_sorted.boxplot(column='price', by='stops_numb', ax=ax3)
    plt.title("Pris fordelt på antal stop")
    plt.suptitle("")
    plt.xlabel("Antal stop")
    plt.ylabel("Pris")
    st.pyplot(fig3)

if st.checkbox("Vis barplot: Gennemsnitspris pr. antal stop"):
    st.subheader("📊 Gennemsnitspriser i forhold til antal stop")

    average_prices = df.groupby('stops_numb')['price'].mean().round(2)

    st.write("Gennemsnitspris pr. antal stop:")
    st.dataframe(average_prices.reset_index().rename(columns={'stops_numb': 'Antal stop', 'price': 'Gennemsnitspris'}))

    # Visualisering
    fig4, ax4 = plt.subplots()
    average_prices.plot(kind='bar', color='skyblue', ax=ax4)
    ax4.set_title("Gennemsnitspris i forhold til antal stop")
    ax4.set_xlabel("Antal stop")
    ax4.set_ylabel("Pris (i gennemsnit)")
    ax4.grid(axis='y')
    st.pyplot(fig4)



