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
if 'dfTrain' not in st.session_state or 'dfTrain_numeric' not in st.session_state:
    st.error("❗ Data mangler i session state: 'dfTrain' og 'dfTrain_numeric'. Sørg for at indlæse data først.")
    st.stop()


dfNumeric = st.session_state['dfTrain_numeric']
df = st.session_state['dfTrain']
df_clean = get_no_outliers_df_train(df)

# --------------------------
# VISNING OG VISUALISERING
# --------------------------
st.subheader("🔍 Datavisning")
if st.checkbox("Vis et udsnit af data"):
    st.dataframe(df.sample(5))

if st.checkbox("Grundlæggende statistik"):
    st.write(df.describe())

st.subheader("📊 Visualiseringer fra analysen")

# 1. Prisfordeling: Weekend vs Hverdag
if st.checkbox('Vis prisfordeling: Weekend vs. Hverdag', value=False):
    if 'is_weekend' in df.columns and 'price' in df.columns:
        fig, ax = plt.subplots()
        sns.boxplot(x='is_weekend', y='price', data=df, ax=ax)
        ax.set_title('Prisfordeling: Weekend vs. Hverdag')
        ax.set_xlabel('Er det weekend?')
        ax.set_ylabel('Pris')
        st.pyplot(fig)

# 2. Prisfordeling for udvalgt måned
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


    # Skævhed og kurtosis
    skewness = skew(df['price'])
    kurt_val = kurtosis(df['price'])
    st.write(f"Skævhed (Skewness): {skewness:.2f}")
    st.write(f"Kurtosis: {kurt_val:.2f}")

    if skewness > 0:
        st.write("Fordelingen er højreskæv (positiv skævhed).")
    elif skewness < 0:
        st.write("Fordelingen er venstreskæv (negativ skævhed).")
    else:
        st.write("Fordelingen er symmetrisk.")

    if kurt_val > 3:
        st.write("Fordelingen er spidsere end en normalfordeling (høj kurtosis).")
    elif kurt_val < 3:
        st.write("Fordelingen er fladere end en normalfordeling (lav kurtosis).")
    else:
        st.write("Fordelingen har samme spidshed som en normalfordeling.")

# 4. Gennemsnitspriser pr. rejsemåned
if st.checkbox("Vis gennemsnitspriser pr. rejsemåned", value=False):
    mean_prices = df.groupby('journey_month')['price'].mean().sort_index()
    st.write("Gennemsnitspriser pr. rejsemåned:")
    st.bar_chart(mean_prices)

# 5. Scatterplot: Pris vs. Rejsemåned
if st.checkbox("Vis scatterplot: Pris pr. rejsemåned", value=False):
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.stripplot(x='journey_month', y='price', data=df, jitter=True, ax=ax3, alpha=0.6)
    ax3.set_title('Scatterplot: Pris pr. rejsemåned')
    ax3.set_xlabel('Rejsemåned')
    ax3.set_ylabel('Pris')
    st.pyplot(fig3)

# 6. Histogram: Prisfordeling pr. rejsemåned
if st.checkbox("Vis histogram: Prisfordeling pr. rejsemåned", value=False):
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x='price', hue='journey_month', bins=50, kde=True, multiple='stack', ax=ax4)
    ax4.set_title('Histogram: Prisfordeling pr. rejsemåned')
    ax4.set_xlabel('Pris')
    ax4.set_ylabel('Antal')
    st.pyplot(fig4)

    # Beregn og vis skævhed og kurtosis
    skewness = skew(df['price'])
    kurt_val = kurtosis(df['price'])

    st.markdown("### 📈 Statistisk analyse af prisfordelingen")
    st.write(f"**Skævhed (Skewness):** {skewness:.2f}")
    st.write(f"**Kurtosis:** {kurt_val:.2f}")

    if skewness > 0:
        st.success("📌 Fordelingen er **højreskæv** (positiv skævhed).")
    elif skewness < 0:
        st.warning("📌 Fordelingen er **venstreskæv** (negativ skævhed).")
    else:
        st.info("📌 Fordelingen er **symmetrisk**.")

    if kurt_val > 3:
        st.success("📌 Fordelingen er **spidsere** end en normalfordeling (**høj kurtosis**).")
    elif kurt_val < 3:
        st.warning("📌 Fordelingen er **fladere** end en normalfordeling (**lav kurtosis**).")
    else:
        st.info("📌 Fordelingen har **samme spidshed** som en normalfordeling.")

# 7. Boxplot: Pris pr. rejsemåned
if st.checkbox("Vis boxplot: Pris pr. rejsemåned", value=False):
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x='journey_month', y='price', ax=ax5)
    ax5.set_title('Boxplot: Pris pr. rejsemåned')
    ax5.set_xlabel('Rejsemåned')
    ax5.set_ylabel('Pris')
    st.pyplot(fig5)

# 8. Korrelationsmatrix
if st.checkbox("Vis korrelation mellem pris og rejsemåned", value=False):
    corr = df[['price', 'journey_month']].corr()
    st.write("Korrelationsmatrix mellem pris og rejsemåned:")
    st.dataframe(corr)
    fig6, ax6 = plt.subplots(figsize=(4, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax6)
    ax6.set_title("Korrelationsmatrix: Pris vs. Rejsemåned")
    st.pyplot(fig6)

# --- MANGLENDE VISUALISERINGER ---

# 9. Samlet prisfordeling (hele datasættet)
if st.checkbox("Vis samlet prisfordeling (hele datasættet)", value=False):
    fig_all, ax_all = plt.subplots()
    sns.histplot(df['price'], kde=True, bins=50, ax=ax_all)
    ax_all.set_title('Samlet prisfordeling')
    ax_all.set_xlabel('Pris')
    ax_all.set_ylabel('Antal')
    st.pyplot(fig_all)

# 10. Samlet boxplot af priser
if st.checkbox("Vis samlet boxplot af priser", value=False):
    fig_box, ax_box = plt.subplots()
    sns.boxplot(x=df['price'], ax=ax_box)
    ax_box.set_title('Boxplot af alle priser')
    ax_box.set_xlabel('Pris')
    st.pyplot(fig_box)

# 11. Scatterplot uden outliers
if st.checkbox("Vis scatterplot uden outliers", value=False):
    fig_strip_clean, ax_strip_clean = plt.subplots(figsize=(8, 4))
    sns.stripplot(x='journey_month', y='price', data=df_clean, jitter=True, alpha=0.6, ax=ax_strip_clean)
    ax_strip_clean.set_title('Scatterplot uden outliers')
    ax_strip_clean.set_xlabel('Rejsemåned')
    ax_strip_clean.set_ylabel('Pris')
    st.pyplot(fig_strip_clean)

# 12. Histogram uden outliers (pr. måned) + statistisk analyse
if st.checkbox("Vis histogram uden outliers (pr. måned)", value=False):
    fig_hist_clean, ax_hist_clean = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df_clean, x='price', hue='journey_month', bins=50, kde=True, multiple='stack', ax=ax_hist_clean)
    ax_hist_clean.set_title('Histogram uden outliers')
    ax_hist_clean.set_xlabel('Pris')
    ax_hist_clean.set_ylabel('Antal')
    st.pyplot(fig_hist_clean)

    # Beregn skævhed og kurtosis på det rensede datasæt (uden outliers)
    skewness_clean = skew(df_clean['price'])
    kurt_val_clean = kurtosis(df_clean['price'])

    st.markdown("### 📈 Statistisk analyse af prisfordelingen (uden outliers)")
    st.write(f"**Skævhed (Skewness):** {skewness_clean:.2f}")
    st.write(f"**Kurtosis:** {kurt_val_clean:.2f}")

    if skewness_clean > 0:
        st.success("📌 Fordelingen er **højreskæv** (positiv skævhed).")
    elif skewness_clean < 0:
        st.warning("📌 Fordelingen er **venstreskæv** (negativ skævhed).")
    else:
        st.info("📌 Fordelingen er **symmetrisk**.")

    if kurt_val_clean > 3:
        st.success("📌 Fordelingen er **spidsere** end en normalfordeling (**høj kurtosis**).")
    elif kurt_val_clean < 3:
        st.warning("📌 Fordelingen er **fladere** end en normalfordeling (**lav kurtosis**).")
    else:
        st.info("📌 Fordelingen har **samme spidshed** som en normalfordeling.")
