
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from datarenser import get_no_outliers_df_train

st.set_page_config(layout="wide")


st.title("✈️ Hvordan kan flyselskaber kategoriseres i grupper baseret på prisvariationer over rejsemåneden og forskelle mellem standard og premium-versioner?")

# --- Session-state dataframes ---------------------------------------
if 'dfTrain' not in st.session_state:
    st.error("❗ Data mangler i session state: 'dfTrain'. Sørg for at indlæse data først.")
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
    st.write(df[["price", "journey_month", "class_numb", "airline_numb"]].describe())



st.subheader("📊 Visualiseringer fra analysen")  

st.subheader("Analyse af Prisfordeling")

# Histogram med KDE
if st.checkbox("Vis histogram med KDE", key="hist_kde"):
    fig1, ax1 = plt.subplots()
    sns.histplot(df['price'], kde=True, bins=50, ax=ax1)
    ax1.set_title('Fordeling af Priser - histogram')
    ax1.set_xlabel('Pris')
    ax1.set_ylabel('Antal')
    st.pyplot(fig1)

# Boxplot
if st.checkbox("Vis boxplot af pris", key="boxplot_price"):
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df['price'], ax=ax2)
    ax2.set_title('Boxplot af Pris')
    st.pyplot(fig2)

# Beregning og fortolkning
if st.checkbox("Vis skævhed og kurtosis", key="show_skew_kurtosis"):
    skewness = skew(df['price'].dropna())
    kurt = kurtosis(df['price'].dropna())

    st.subheader("Statistiske mål:")
    st.write(f"**Skævhed (Skewness):** {skewness:.2f}")
    st.write(f"**Kurtosis:** {kurt:.2f}")

    if skewness > 0:
        st.info("📈 Fordelingen er **højreskæv** (positiv skævhed).")
    elif skewness < 0:
        st.info("📉 Fordelingen er **venstreskæv** (negativ skævhed).")
    else:
        st.info("✅ Fordelingen er **symmetrisk**.")

    if kurt > 3:
        st.info("📌 Fordelingen er **spidsere** end en normalfordeling (høj kurtosis).")
    elif kurt < 3:
        st.info("📌 Fordelingen er **fladere** end en normalfordeling (lav kurtosis).")
    else:
        st.info("✅ Fordelingen har **samme spidshed** som en normalfordeling.")

st.subheader("Analyse af Pris (efter outlier-fjernelse)")

# Histogram
if st.checkbox("Vis histogram (uden outliers)", key="hist_clean"):
    fig1, ax1 = plt.subplots()
    sns.histplot(df_clean['price'], kde=True, bins=50, ax=ax1)
    ax1.set_title('Histogram af Pris (uden outliers)')
    ax1.set_xlabel('Pris')
    ax1.set_ylabel('Antal')
    st.pyplot(fig1)

# Boxplot
if st.checkbox("Vis boxplot (uden outliers)", key="boxplot_clean"):
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df_clean['price'], ax=ax2)
    ax2.set_title('Boxplot af Pris (uden outliers)')
    st.pyplot(fig2)

# Skævhed og kurtosis
if st.checkbox("Vis skævhed og kurtosis (uden outliers)", key="skew_kurt_clean"):
    price_clean = df_clean['price'].dropna()
    skewness_after = skew(price_clean)
    kurt_after = kurtosis(price_clean)

    st.subheader("Statistiske mål (uden outliers):")
    st.write(f"**Skævhed (Skewness):** {skewness_after:.2f}")
    st.write(f"**Kurtosis:** {kurt_after:.2f}")

    if skewness_after > 0:
        st.info("📈 Fordelingen er **højreskæv** (positiv skævhed).")
    elif skewness_after < 0:
        st.info("📉 Fordelingen er **venstreskæv** (negativ skævhed).")
    else:
        st.info("✅ Fordelingen er **symmetrisk**.")

    if kurt_after > 3:
        st.info("📌 Fordelingen er **spidsere** end en normalfordeling (høj kurtosis).")
    elif kurt_after < 3:
        st.info("📌 Fordelingen er **fladere** end en normalfordeling (lav kurtosis).")
    else:
        st.info("✅ Fordelingen har **samme spidshed** som en normalfordeling.")

st.subheader("Analyse af Pris pr. Klasse (med outliers)")

# Scatterplot / stripplot
if st.checkbox("Vis scatterplot (pris pr. klasse) - med outliers", key="scatter_outliers"):
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.stripplot(data=df, x='class', y='price', jitter=True, alpha=0.6, ax=ax1)
    ax1.set_title('Scatterplot: Pris pr. klasse (med outliers)')
    ax1.set_xlabel('Billetklasse')
    ax1.set_ylabel('Pris')
    st.pyplot(fig1)

# Histogram
if st.checkbox("Vis histogram (prisfordeling pr. klasse) - med outliers", key="hist_outliers"):
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='price', hue='class', bins=50, kde=True, multiple='stack', ax=ax2)
    ax2.set_title('Histogram: Prisfordeling pr. klasse (med outliers)')
    ax2.set_xlabel('Pris')
    ax2.set_ylabel('Antal')
    st.pyplot(fig2)

# Boxplot
if st.checkbox("Vis boxplot (pris pr. klasse) - med outliers", key="boxplot_outliers"):
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x='class', y='price', ax=ax3)
    ax3.set_title('Boxplot: Pris pr. klasse (med outliers)')
    ax3.set_xlabel('Billetklasse')
    ax3.set_ylabel('Pris')
    st.pyplot(fig3)

# Gennemsnitspriser
if st.checkbox("Vis gennemsnitspriser pr. klasse - med outliers", key="mean_prices_outliers"):
    mean_prices = df.groupby('class')['price'].mean().sort_values()

    st.subheader("📊 Gennemsnitspriser pr. klasse (med outliers):")
    for klasse, pris in mean_prices.items():
        st.write(f"**{klasse}:** {pris:.2f} kr.")

st.subheader("Analyse af Pris pr. Klasse (uden outliers)")

# Scatterplot / stripplot
if st.checkbox("Vis scatterplot (pris pr. klasse) - uden outliers", key="scatter_clean"):
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.stripplot(data=df_clean, x='class', y='price', jitter=True, alpha=0.6, ax=ax1)
    ax1.set_title('Scatterplot: Pris pr. klasse (uden outliers)')
    ax1.set_xlabel('Billetklasse')
    ax1.set_ylabel('Pris')
    st.pyplot(fig1)

# Histogram
if st.checkbox("Vis histogram (prisfordeling pr. klasse) - uden outliers", key="hist_clean_klasse"):
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_clean, x='price', hue='class', bins=50, kde=True, multiple='stack', ax=ax2)
    ax2.set_title('Histogram: Prisfordeling pr. klasse (uden outliers)')
    ax2.set_xlabel('Pris')
    ax2.set_ylabel('Antal')
    st.pyplot(fig2)

# Boxplot
if st.checkbox("Vis boxplot (pris pr. klasse) - uden outliers", key="boxplot_clean_klasse"):
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_clean, x='class', y='price', ax=ax3)
    ax3.set_title('Boxplot: Pris pr. klasse (uden outliers)')
    ax3.set_xlabel('Billetklasse')
    ax3.set_ylabel('Pris')
    st.pyplot(fig3)

# Gennemsnitspriser
if st.checkbox("Vis gennemsnitspriser pr. klasse - uden outliers", key="mean_prices_clean_klasse"):
    mean_prices = df_clean.groupby('class')['price'].mean().sort_values()

    st.subheader("📊 Gennemsnitspriser pr. klasse (uden outliers):")
    for klasse, pris in mean_prices.items():
        st.write(f"**{klasse}:** {pris:.2f} kr.")


st.subheader("Analyse af Pris pr. Rejsemåned (med outliers)")

# Scatterplot: price vs journey_month
if st.checkbox("Vis scatterplot: Pris pr. rejsemåned (med outliers)", key="scatter_journey_month"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.stripplot(data=df, x='journey_month', y='price', jitter=True, alpha=0.6, ax=ax)
    ax.set_title('Scatterplot: Pris pr. rejsemåned (med outliers)')
    ax.set_xlabel('Rejsemåned')
    ax.set_ylabel('Pris')
    plt.tight_layout()
    st.pyplot(fig)

# Histogram: prisfordeling pr. rejsemåned
if st.checkbox("Vis histogram: Prisfordeling pr. rejsemåned (med outliers)", key="hist_journey_month"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='price', hue='journey_month', bins=50, kde=True, multiple='stack', ax=ax)
    ax.set_title('Histogram: Prisfordeling pr. rejsemåned (med outliers)')
    ax.set_xlabel('Pris')
    ax.set_ylabel('Antal')
    plt.tight_layout()
    st.pyplot(fig)

# Boxplot: Pris pr. rejsemåned
if st.checkbox("Vis boxplot: Pris pr. rejsemåned (med outliers)", key="boxplot_journey_month"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x='journey_month', y='price', ax=ax)
    ax.set_title('Boxplot: Pris pr. rejsemåned (med outliers)')
    ax.set_xlabel('Rejsemåned')
    ax.set_ylabel('Pris')
    plt.tight_layout()
    st.pyplot(fig)

# Gennemsnitspriser pr. rejsemåned
if st.checkbox("Vis gennemsnitspriser pr. rejsemåned (med outliers)", key="mean_prices_journey_month"):
    mean_prices = df.groupby('journey_month')['price'].mean().sort_index()
    st.subheader("Gennemsnitspriser pr. rejsemåned (med outliers):")
    for month, pris in mean_prices.items():
        st.write(f"**Måned {month}:** {pris:.2f} kr.")


st.subheader("Analyse af Pris pr. Rejsemåned (uden outliers)")

# Scatterplot: price vs journey_month
if st.checkbox("Vis scatterplot: Pris pr. rejsemåned (uden outliers)", key="scatter_journey_month_clean"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.stripplot(data=df_clean, x='journey_month', y='price', jitter=True, alpha=0.6, ax=ax)
    ax.set_title('Scatterplot: Pris pr. rejsemåned (uden outliers)')
    ax.set_xlabel('Rejsemåned')
    ax.set_ylabel('Pris')
    plt.tight_layout()
    st.pyplot(fig)

# Histogram: prisfordeling pr. rejsemåned
if st.checkbox("Vis histogram: Prisfordeling pr. rejsemåned (uden outliers)", key="hist_journey_month_clean"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_clean, x='price', hue='journey_month', bins=50, kde=True, multiple='stack', ax=ax)
    ax.set_title('Histogram: Prisfordeling pr. rejsemåned (uden outliers)')
    ax.set_xlabel('Pris')
    ax.set_ylabel('Antal')
    plt.tight_layout()
    st.pyplot(fig)

# Boxplot: Pris pr. rejsemåned
if st.checkbox("Vis boxplot: Pris pr. rejsemåned (uden outliers)", key="boxplot_journey_month_clean"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_clean, x='journey_month', y='price', ax=ax)
    ax.set_title('Boxplot: Pris pr. rejsemåned (uden outliers)')
    ax.set_xlabel('Rejsemåned')
    ax.set_ylabel('Pris')
    plt.tight_layout()
    st.pyplot(fig)

# Gennemsnitspriser pr. rejsemåned
if st.checkbox("Vis gennemsnitspriser pr. rejsemåned (uden outliers)", key="mean_prices_journey_month_clean"):
    mean_prices = df_clean.groupby('journey_month')['price'].mean().sort_index()
    st.subheader("Gennemsnitspriser pr. rejsemåned (uden outliers):")
    for month, pris in mean_prices.items():
        st.write(f"**Måned {month}:** {pris:.2f} kr.")


st.subheader("Prisudvikling pr. flyselskab over rejsemåneder (med outliers)")

if st.checkbox("Vis lineplot: Gennemsnitspris pr. Flyselskab pr. Måned (med outliers)", key="lineplot_airline_month_outliers"):
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=df, x='journey_month', y='price', hue='airline', marker='o', ax=ax)
    ax.set_title('Lineplot: Gennemsnitspris pr. Flyselskab pr. Måned (med outliers)')
    ax.set_xlabel('Rejsemåned')
    ax.set_ylabel('Pris')
    ax.legend(title='Flyselskab', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

st.subheader("Prisudvikling pr. flyselskab over rejsemåneder (uden outliers)")

if st.checkbox("Vis lineplot: Gennemsnitspris pr. Flyselskab pr. Måned (uden outliers)", key="lineplot_airline_month_no_outliers"):
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=df_clean, x='journey_month', y='price', hue='airline', marker='o', ax=ax)
    ax.set_title('Lineplot: Gennemsnitspris pr. Flyselskab pr. Måned (uden outliers)')
    ax.set_xlabel('Rejsemåned')
    ax.set_ylabel('Pris')
    ax.legend(title='Flyselskab', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)


st.subheader(" Gennemsnitspris pr. Klasse og Flyselskab")

if st.checkbox("Vis barplot: Gennemsnitspris pr. klasse og flyselskab"):
    

    # Beregn gennemsnitspris
    mean_prices_per_class = df.groupby(['airline', 'class'])['price'].mean().reset_index()

    # Visualisering
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=mean_prices_per_class, x='airline', y='price', hue='class', ax=ax)
    ax.set_title('Barplot: Gennemsnitspris pr. Klasse og Flyselskab')
    ax.set_ylabel('Pris')
    ax.set_xlabel('Flyselskab')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid(axis='y')
    st.pyplot(fig)

st.subheader(" Boxplot: Prisvariation pr. Flyselskab (med outliers)")


if st.checkbox("Vis boxplot: Prisvariation pr. flyselskab (med outliers)"):
        
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=df, x='airline', y='price', ax=ax)
    ax.set_title('Boxplot: Prisvariation pr. Flyselskab (med outliers)')
    ax.set_xlabel('Flyselskab')
    ax.set_ylabel('Pris')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

st.subheader("Boxplot: Prisvariation pr. Flyselskab (uden outliers)")

if st.checkbox("Vis boxplot: Prisvariation pr. flyselskab (uden outliers)"):
    

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=df_clean, x='airline', y='price', ax=ax)
    ax.set_title('Boxplot: Prisvariation pr. Flyselskab (uden outliers)')
    ax.set_xlabel('Flyselskab')
    ax.set_ylabel('Pris')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

st.subheader(" Boxplot: Prisvariation pr. Flyselskab og Klasse (med outliers)")

if st.checkbox("Vis boxplot: Prisvariation pr. flyselskab og klasse (med outliers)"):
    
    
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=df, x='airline', y='price', hue='class', ax=ax)
    ax.set_title('Prisvariation pr. Flyselskab og Klasse (Boxplot)')
    ax.set_xlabel('Flyselskab')
    ax.set_ylabel('Pris')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

st.subheader("Korrelationsanalyse mellem pris og klasse")

if st.checkbox("Vis korrelationsmatrix: Pris vs. Klasse"):
        

    # Vælg relevante kolonner til korrelationsanalyse
    cols_to_corr = ['price', 'class_numb']
    st.write("Valgte kolonner til korrelationsanalyse:", cols_to_corr)

    # Beregn korrelationsmatrix
    corr = df[cols_to_corr].corr()
    st.write("Beregnet korrelationsmatrix:")
    st.dataframe(corr.style.format("{:.2f}"))

    # Visualiser korrelationsmatrix som heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title("Korrelationsmatrix: Pris vs. Klasse")
    st.pyplot(fig)

st.subheader("Korrelationsanalyse mellem pris og rejsemåned")

if st.checkbox("Vis korrelationsmatrix: pris vs. rejsemåned"):
        

    # Udvælg kolonner til korrelationsanalyse
    cols_to_corr = ['price', 'journey_month']

    # Beregn korrelationsmatrix
    corr = df[cols_to_corr].corr()

    # Vis korrelationsmatrix som dataframe
    st.write("Korrelationsmatrix mellem pris og rejsemåned:")
    st.dataframe(corr.style.format("{:.2f}"))

    # Visualiser korrelationsmatrix som heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title("Korrelationsmatrix: Pris vs. Rejsemåned")
    st.pyplot(fig)


st.subheader("Korrelationsmatrix – Pris, Måned, Klasse og Flyselskab")

if st.checkbox("Vis korrelationsmatrix for pris, måned, klasse og flyselskab"):
        

    # Udvælger relevante numeriske kolonner
    corr_df = df[['price', 'class_numb', 'journey_month', 'airline_numb']]

    # Beregner korrelationer mellem de udvalgte variabler
    corr = corr_df.corr()

    # Vis korrelationsmatrix som dataframe
    st.write("Korrelationsmatrix:")
    st.dataframe(corr.style.format("{:.2f}"))

    # Visualiserer korrelationerne med heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title('Korrelationsmatrix – Pris, Måned, Klasse og Flyselskab')
    plt.tight_layout()
    st.pyplot(fig)


