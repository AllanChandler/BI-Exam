import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from datarenser import get_no_outliers_df_train

st.set_page_config(layout="wide")
st.title("✈️ Hvordan varierer priserne mellem standard og premium-versioner af samme flyselskab?")

# --- Session-state dataframes ---------------------------------------
if 'dfTrain' not in st.session_state or 'dfTrain_numeric' not in st.session_state:
    st.error("❗ Data mangler i session state: 'dfTrain' og 'dfTrain_numeric'. Sørg for at indlæse data først.")
    st.stop()

df = st.session_state['dfTrain']
dfNumeric = st.session_state['dfTrain_numeric']
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


st.subheader("Prisvisualisering (uden outliers)")

# Checkbox for histogram og boxplot
show_hist = st.checkbox("Vis histogram af pris")
show_box = st.checkbox("Vis boxplot af pris")

# Histogram med skævhed og kurtosis
if show_hist:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df_clean['price'], kde=True, bins=50, ax=ax)
    ax.set_title("Histogram af Pris (uden outliers)")
    ax.set_xlabel("Pris")
    ax.set_ylabel("Antal")
    st.pyplot(fig)

    # Beregning af skævhed og kurtosis
    skew_val = skew(df_clean['price'].dropna())
    kurt_val = kurtosis(df_clean['price'].dropna())

    st.subheader("📈 Statistisk beskrivelse af prisfordelingen")
    st.write(f"**Skævhed (Skewness):** {skew_val:.2f}")
    if skew_val > 0:
        st.write("Fordelingen er højreskæv (positiv skævhed).")
    elif skew_val < 0:
        st.write("Fordelingen er venstreskæv (negativ skævhed).")
    else:
        st.write("Fordelingen er symmetrisk.")

    st.write(f"**Kurtosis:** {kurt_val:.2f}")
    if kurt_val > 3:
        st.write("Fordelingen er spidsere end en normalfordeling (høj kurtosis).")
    elif kurt_val < 3:
        st.write("Fordelingen er fladere end en normalfordeling (lav kurtosis).")
    else:
        st.write("Fordelingen har samme spidshed som en normalfordeling.")

# Boxplot
if show_box:
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.boxplot(x=df_clean['price'], ax=ax)
    ax.set_title("Boxplot af Pris (uden outliers)")
    st.pyplot(fig)


st.subheader("Pris pr. Klasse (med outliers)")    

# Checkboxes for hvert diagram
show_scatter = st.checkbox("Vis scatterplot: Pris pr. klasse")
show_hist = st.checkbox("Vis histogram: Prisfordeling pr. klasse")
show_box = st.checkbox("Vis boxplot: Pris pr. klasse")



# Scatterplot
if show_scatter:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.stripplot(data=df, x='class', y='price', jitter=True, alpha=0.6, ax=ax)
    ax.set_title('Scatterplot: Pris pr. klasse (med outliers)')
    ax.set_xlabel('Billetklasse')
    ax.set_ylabel('Pris')
    st.pyplot(fig)

# Histogram
if show_hist:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='price', hue='class', bins=50, kde=True, multiple='stack', ax=ax)
    ax.set_title('Histogram: Prisfordeling pr. klasse (med outliers)')
    ax.set_xlabel('Pris')
    ax.set_ylabel('Antal')
    st.pyplot(fig)

# Boxplot
if show_box:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x='class', y='price', ax=ax)
    ax.set_title('Boxplot: Pris pr. klasse (med outliers)')
    ax.set_xlabel('Billetklasse')
    ax.set_ylabel('Pris')
    st.pyplot(fig)

# Vis gennemsnitspriser hvis ét eller flere diagrammer vises
if show_scatter or show_hist or show_box:
    mean_prices = df.groupby('class')['price'].mean().sort_values()
    st.subheader("📈 Gennemsnitspriser pr. klasse (med outliers)")
    for klasse, pris in mean_prices.items():
        st.write(f"{klasse}: {pris:.2f}")



st.subheader("Pris pr. Klasse (uden outliers)")

# Checkboxes med unikke keys
show_scatter = st.checkbox("Vis scatterplot: Pris pr. klasse", key="scatter_no_outliers")
show_hist = st.checkbox("Vis histogram: Prisfordeling pr. klasse", key="hist_no_outliers")
show_box = st.checkbox("Vis boxplot: Pris pr. klasse", key="box_no_outliers")

# Scatterplot
if show_scatter:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.stripplot(data=df_clean, x='class', y='price', jitter=True, alpha=0.6, ax=ax)
    ax.set_title('Scatterplot: Pris pr. klasse (uden outliers)')
    ax.set_xlabel('Billetklasse')
    ax.set_ylabel('Pris')
    st.pyplot(fig)

# Histogram
if show_hist:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_clean, x='price', hue='class', bins=50, kde=True, multiple='stack', ax=ax)
    ax.set_title('Histogram: Prisfordeling pr. klasse (uden outliers)')
    ax.set_xlabel('Pris')
    ax.set_ylabel('Antal')
    st.pyplot(fig)

# Boxplot
if show_box:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_clean, x='class', y='price', ax=ax)
    ax.set_title('Boxplot: Pris pr. klasse (uden outliers)')
    ax.set_xlabel('Billetklasse')
    ax.set_ylabel('Pris')
    st.pyplot(fig)

# Vis gennemsnitspriser hvis mindst ét diagram er valgt
if show_scatter or show_hist or show_box:
    mean_prices = df_clean.groupby('class')['price'].mean().sort_values()
    st.subheader("📈 Gennemsnitspriser pr. klasse (uden outliers)")
    for klasse, pris in mean_prices.items():
        st.write(f"{klasse}: {pris:.2f}")


st.subheader("Korrelationsanalyse: Pris vs. Klasse")

# Checkbox for at vise korrelationsmatrix
show_corr = st.checkbox("Vis korrelationsmatrix (Pris og Klasse)", key="show_corr_matrix")

if show_corr:
    cols_to_corr = ['price', 'class_numb']
    st.write("Valgte kolonner til korrelationsanalyse:", cols_to_corr)

    corr = df[cols_to_corr].corr()
    st.write("**Beregnet korrelationsmatrix:**")
    st.dataframe(corr)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title("Korrelationsmatrix: Pris vs. Klasse")
    st.pyplot(fig)