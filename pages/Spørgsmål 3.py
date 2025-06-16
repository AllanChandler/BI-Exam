import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from datarenser import get_no_outliers_df_train

# Titel
st.title(" ✈️ Spg. 3 Hvordan varierer priserne afhængigt af rejsemåneden?")

# --- Session-state dataframes ---------------------------------------
if 'dfTrain' not in st.session_state:
    st.error("Data mangler i session state: 'dfTrain'. Sørg for at indlæse data først.")
    st.stop()



df = st.session_state['dfTrain']
df_clean = get_no_outliers_df_train(df)


# --------------------------
# VISNING OG VISUALISERING
# --------------------------
st.subheader("🔍 Datavisning")
if st.checkbox("Vis et udsnit af data"):
    st.dataframe(df.sample(5))

if st.checkbox("Grundlæggende statistik"):
     st.write(df[["price", "journey_month"]].describe())

st.subheader("Visualiseringer fra analysen")


# 1. Prisfordeling for udvalgt måned
if 'journey_month' in df.columns and st.checkbox("Vis prisfordeling for valgt måned", value=False):
    selected_month = st.selectbox('Vælg rejsemåned', sorted(df['journey_month'].unique()))
    df_month = df[df['journey_month'] == selected_month]

    st.write(f"Prisfordeling for måned: {selected_month}")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.histplot(df_month['price'], bins=50, kde=True, ax=ax2)
    ax2.set_title(f'Prisfordeling i måned {selected_month}')
    ax2.set_xlabel('Pris')
    ax2.set_ylabel('Antal')
    st.pyplot(fig2)


# 2. Gennemsnitspriser pr. rejsemåned
if st.checkbox("Vis gennemsnitspriser pr. rejsemåned", value=False):
    mean_prices = df.groupby('journey_month')['price'].mean().sort_index()
    st.write("Gennemsnitspriser pr. rejsemåned:")
    st.bar_chart(mean_prices)

st.subheader("Prisvisualisering")

show_distribution = st.checkbox("Vis prisfordeling")

if show_distribution:
    st.header("Fordeling af priser")

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Histogram med KDE
    sns.histplot(df['price'], kde=True, bins=50, ax=ax[0])
    ax[0].set_title('Fordeling af Priser - histogram')
    ax[0].set_ylabel('Antal')
    ax[0].set_xlabel('Pris')

    # Boxplot
    sns.boxplot(x=df['price'], ax=ax[1])
    ax[1].set_title('Boxplot af Pris')

    plt.tight_layout()
    st.pyplot(fig)

    # Beregning af skævhed og kurtosis
    skewness = skew(df['price'].dropna())
    kurt = kurtosis(df['price'].dropna())

    st.write(f"**Skævhed (Skewness):** {skewness:.2f}")
    st.write(f"**Kurtosis:** {kurt:.2f}")

    if skewness > 0:
        st.write("Fordelingen er højreskæv (positiv skævhed).")
    elif skewness < 0:
        st.write("Fordelingen er venstreskæv (negativ skævhed).")
    else:
        st.write("Fordelingen er symmetrisk.")

    if kurt > 3:
        st.write("Fordelingen er spidsere end en normalfordeling (høj kurtosis).")
    elif kurt < 3:
        st.write("Fordelingen er fladere end en normalfordeling (lav kurtosis).")
    else:
        st.write("Fordelingen har samme spidshed som en normalfordeling.")




st.subheader("Pris pr. rejsemåned")

show_scatter = st.checkbox("Vis scatterplot: Pris pr. rejsemåned")
show_box = st.checkbox("Vis boxplot: Pris pr. rejsemåned")


# 3. Scatterplot: Pris vs. Rejsemåned
if show_scatter:
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.stripplot(x='journey_month', y='price', data=df, jitter=True, ax=ax3, alpha=0.6)
    ax3.set_title('Scatterplot: Pris pr. rejsemåned')
    ax3.set_xlabel('Rejsemåned')
    ax3.set_ylabel('Pris')
    st.pyplot(fig3)


# 5. Boxplot: Pris pr. rejsemåned
if show_box:
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x='journey_month', y='price', ax=ax5)
    ax5.set_title('Boxplot: Pris pr. rejsemåned')
    ax5.set_xlabel('Rejsemåned')
    ax5.set_ylabel('Pris')
    st.pyplot(fig5)

# Vis gennemsnitspriser pr. rejsemåned hvis ét eller flere diagrammer vises
if show_scatter or show_box:
    mean_prices_by_month = df.groupby('journey_month')['price'].mean().sort_index()
    st.subheader("Gennemsnitspriser pr. rejsemåned")
    for month, price in mean_prices_by_month.items():
        st.write(f"Måned {month}: {price:.2f} DKK")


st.subheader("Korrelationsanalyse: Pris vs. Rejsemåned")

# 6. Korrelationsmatrix
if st.checkbox("Vis korrelation mellem pris og rejsemåned", value=False):
    corr = df[['price', 'journey_month']].corr()
    st.dataframe(corr)
    fig6, ax6 = plt.subplots(figsize=(4, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax6)
    ax6.set_title("Korrelationsmatrix: Pris vs. Rejsemåned")
    st.pyplot(fig6)










