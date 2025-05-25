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
st.title("✈️ Hvordan påvirker antallet af dage prisen?")

# Brug dfClean fra session_state
if 'dfClean' in st.session_state:
    df = st.session_state['dfClean']
else:
    st.error("❌ Data ikke tilgængelig i session_state. Sørg for at 'dfClean' er indlæst korrekt.")
    st.stop()

# Input: Hvor mange dage til afrejse
days_until_departure = st.slider("Hvor mange dage er der til afrejse?", min_value=0, max_value=365, value=30)

# VISNING OG VISUALISERING
st.subheader("🔍 Datavisning")
if st.checkbox("Vis et udsnit af data"):
    st.dataframe(df.sample(5))

if st.checkbox("Grundlæggende statistik"):
    st.write(df[["price", "days_left"]].describe())

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

if st.checkbox("Vis korrelation mellem antal dage til afrejse og pris"):
    if 'days_left' in df.columns and 'price' in df.columns:
        st.subheader("📈 Korrelation: Antal dage til afrejse vs. Pris")

        # Beregn korrelation
        corr = df[['days_left', 'price']].corr()

        # Vis korrelationstabel
        st.write("Korrelationsmatrix:")
        st.dataframe(corr)

        # Tegn heatmap
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Korrelation mellem Antal Dage til Afrejse og Pris')
        st.pyplot(fig)

if st.checkbox("Vis scatter plot: Antal dage til afrejse vs. Pris"):
    fig2, ax2 = plt.subplots()
    ax2.scatter(df['days_left'], df['price'], alpha=0.3)
    ax2.set_title("Sammenhæng mellem antal dage til afrejse og pris")
    ax2.set_xlabel("Antal dage til afrejse")
    ax2.set_ylabel("Pris")
    st.pyplot(fig2)
