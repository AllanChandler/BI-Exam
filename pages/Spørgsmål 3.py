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

st.subheader("Prisvisualisering (med outliers)")

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

if st.checkbox("Vis prisfordeling", value=False, key="dist_uden_outliers"):
    fig_no_outliers, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Histogram
    sns.histplot(df_clean['price'], kde=True, bins=50, ax=ax1)
    ax1.set_title('Histogram af Pris (uden outliers)')
    ax1.set_xlabel('Pris')
    ax1.set_ylabel('Antal')

    # Boxplot
    sns.boxplot(x=df_clean['price'], ax=ax2)
    ax2.set_title('Boxplot af Pris (uden outliers)')
    ax2.set_xlabel('Pris')

    plt.tight_layout()
    st.pyplot(fig_no_outliers)

    # Statistisk analyse
    skewness_after = skew(df_clean['price'].dropna())
    kurt_after = kurtosis(df_clean['price'].dropna())

    st.markdown("### 📊 Statistisk analyse af pris (uden outliers)")
    st.write(f"**Skævhed (Skewness):** {skewness_after:.2f}")
    st.write(f"**Kurtosis:** {kurt_after:.2f}")

    if skewness_after > 0:
        st.success("📌 Fordelingen er **højreskæv** (positiv skævhed).")
    elif skewness_after < 0:
        st.warning("📌 Fordelingen er **venstreskæv** (negativ skævhed).")
    else:
        st.info("📌 Fordelingen er **symmetrisk**.")

    if kurt_after > 3:
        st.success("📌 Fordelingen er **spidsere** end en normalfordeling (**høj kurtosis**).")
    elif kurt_after < 3:
        st.warning("📌 Fordelingen er **fladere** end en normalfordeling (**lav kurtosis**).")
    else:
        st.info("📌 Fordelingen har **samme spidshed** som en normalfordeling.")



st.subheader("Pris pr. rejsemåned (Med outliers)")

show_scatter = st.checkbox("Vis scatterplot: Pris pr. rejsemåned")
show_hist = st.checkbox("Vis histogram: Prisfordeling pr. rejsemåned")
show_box = st.checkbox("Vis boxplot: Pris pr. rejsemåned")


# 3. Scatterplot: Pris vs. Rejsemåned
if show_scatter:
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.stripplot(x='journey_month', y='price', data=df, jitter=True, ax=ax3, alpha=0.6)
    ax3.set_title('Scatterplot: Pris pr. rejsemåned')
    ax3.set_xlabel('Rejsemåned')
    ax3.set_ylabel('Pris')
    st.pyplot(fig3)

# 4. Histogram: Prisfordeling pr. rejsemåned (med outliers)
if show_hist:
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x='price', hue='journey_month', bins=50, kde=True, multiple='stack', ax=ax4)
    ax4.set_title('Histogram: Prisfordeling pr. rejsemåned (med outliers)')
    ax4.set_xlabel('Pris')
    ax4.set_ylabel('Antal')
    st.pyplot(fig4)

    # Beregn skævhed og kurtosis på det fulde datasæt (med outliers)
    skewness = skew(df['price'].dropna())
    kurt_val = kurtosis(df['price'].dropna())

    st.markdown("### 📊 Statistisk analyse af prisfordelingen (med outliers)")
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

# 5. Boxplot: Pris pr. rejsemåned
if show_box:
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x='journey_month', y='price', ax=ax5)
    ax5.set_title('Boxplot: Pris pr. rejsemåned')
    ax5.set_xlabel('Rejsemåned')
    ax5.set_ylabel('Pris')
    st.pyplot(fig5)

# Vis gennemsnitspriser pr. rejsemåned hvis ét eller flere diagrammer vises
if show_scatter or show_hist or show_box:
    mean_prices_by_month = df.groupby('journey_month')['price'].mean().sort_index()
    st.subheader("📈 Gennemsnitspriser pr. rejsemåned (med outliers)")
    for month, price in mean_prices_by_month.items():
        st.write(f"Måned {month}: {price:.2f} DKK")



st.subheader("Pris pr. rejsemåned (Uden outliers)")

show_scatter_clean = st.checkbox("Vis scatterplot: Pris pr. rejsemåned (uden outliers)")
show_hist_clean = st.checkbox("Vis histogram: Prisfordeling pr. rejsemåned (uden outliers)")
show_box_clean = st.checkbox("Vis boxplot: Pris pr. rejsemåned (uden outliers)")


# Scatterplot (uden outliers)
if show_scatter_clean:
    fig_strip_clean, ax_strip_clean = plt.subplots(figsize=(8, 4))
    sns.stripplot(x='journey_month', y='price', data=df_clean, jitter=True, alpha=0.6, ax=ax_strip_clean)
    ax_strip_clean.set_title('Scatterplot uden outliers')
    ax_strip_clean.set_xlabel('Rejsemåned')
    ax_strip_clean.set_ylabel('Pris')
    st.pyplot(fig_strip_clean)


# Histogram + Statistik (uden outliers)
if show_hist_clean:
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


# Boxplot (uden outliers)
if show_box_clean:
    fig_box_clean, ax_box_clean = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_clean, x='journey_month', y='price', ax=ax_box_clean)
    ax_box_clean.set_title('Boxplot: Pris pr. rejsemåned (uden outliers)')
    ax_box_clean.set_xlabel('Rejsemåned')
    ax_box_clean.set_ylabel('Pris')
    st.pyplot(fig_box_clean)


# Vis gennemsnitspriser pr. rejsemåned (uden outliers) hvis ét eller flere diagrammer vises
if show_scatter_clean or show_hist_clean or show_box_clean:
    mean_prices_by_month_clean = df_clean.groupby('journey_month')['price'].mean().sort_index()
    st.subheader("Gennemsnitspriser pr. rejsemåned (uden outliers)")
    for month, price in mean_prices_by_month_clean.items():
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



st.subheader("Prisforskelle mellem Weekend og Hverdag (Med outliners)")

# Sørg for at 'date_of_journey' findes
if 'date_of_journey' not in df_clean.columns:
    df_clean['date_of_journey'] = pd.to_datetime(
        df_clean['journey_day'].astype(str).str.zfill(2) + '-' +
        df_clean['journey_month'].astype(str).str.zfill(2) + '-2019',
        format='%d-%m-%Y'
    )

with st.expander("Vis dato-relaterede kolonner"):
    st.dataframe(df_clean[['date_of_journey', 'journey_day', 'journey_month', 'is_weekend']].head(10))


# 7. Prisfordeling: Weekend vs Hverdag
if st.checkbox('Vis prisfordeling: Weekend vs. Hverdag', value=False):
    if 'is_weekend' in df.columns and 'price' in df.columns:
        fig, ax = plt.subplots()
        sns.boxplot(x='is_weekend', y='price', data=df, ax=ax)
        ax.set_title('Prisfordeling: Weekend vs. Hverdag')
        ax.set_xlabel('Er det weekend?')
        ax.set_ylabel('Pris')
        st.pyplot(fig)



st.subheader("Datoanalyse: Dag, Uge og Måned (med outliers)")

# Optæl antal rejser pr. dag i måneden
daily_counts_by_month = df.groupby(['journey_month', 'journey_day']).size().reset_index(name='antal_rejser')

# Optæl antal rejser pr. uge i måneden
weekly_counts_by_month = df.groupby(['journey_month', 'journey_week']).size().reset_index(name='antal_rejser')

# Vis i Streamlit
with st.expander("Rejser pr. dag i hver måned"):
    st.dataframe(daily_counts_by_month)

with st.expander("Rejser pr. uge i hver måned"):
    st.dataframe(weekly_counts_by_month)



# Checkbox for at vise boxplot med outliers pr. dag i måneden
show_box_day = st.checkbox("Vis boxplot: Pris pr. dag i måneden (med outliers)")

# Checkbox for at vise boxplot med outliers pr. ugenummer
show_box_week = st.checkbox("Vis boxplot: Pris pr. ugenummer (med outliers)")

# Plot 1: Dag i måneden med outliers
if show_box_day:
    fig_day, ax_day = plt.subplots(figsize=(16, 6))
    sns.boxplot(
        data=df,
        x='journey_day',
        y='price',
        hue='journey_month',
        palette='Set2',
        ax=ax_day
    )
    ax_day.set_title('Pris pr. dag i måneden, opdelt pr. måned (med outliers)', fontsize=14)
    ax_day.set_xlabel('Dag i måneden', fontsize=12)
    ax_day.set_ylabel('Pris (DKK)', fontsize=12)
    ax_day.legend(title='Måned', loc='upper right')
    plt.tight_layout()
    st.pyplot(fig_day)
    plt.close(fig_day)

# Plot 2: Ugenummer med outliers
if show_box_week:
    fig_week, ax_week = plt.subplots(figsize=(16, 6))
    sns.boxplot(
        data=df,
        x='journey_week',
        y='price',
        hue='journey_month',
        palette='Set2',
        ax=ax_week
    )
    ax_week.set_title('Pris pr. ugenummer, opdelt pr. måned (med outliers)', fontsize=14)
    ax_week.set_xlabel('Ugenummer', fontsize=12)
    ax_week.set_ylabel('Pris (DKK)', fontsize=12)
    ax_week.legend(title='Måned', loc='upper right')
    plt.tight_layout()
    st.pyplot(fig_week)
    plt.close(fig_week)





st.subheader("Datoanalyse: Dag, Uge og Måned (uden outliers)")

daily_counts_by_month = df_clean.groupby(['journey_month', 'journey_day']).size().reset_index(name='antal_rejser')
weekly_counts_by_month = df_clean.groupby(['journey_month', 'journey_week']).size().reset_index(name='antal_rejser')

with st.expander("Rejser pr. dag i hver måned (uden outliers)"):
    st.dataframe(daily_counts_by_month)

with st.expander("Rejser pr. uge i hver måned (uden outliers)"):
    st.dataframe(weekly_counts_by_month)


show_box_day_clean = st.checkbox("Vis boxplot: Pris pr. dag i måneden (uden outliers)")
show_box_week_clean = st.checkbox("Vis boxplot: Pris pr. ugenummer (uden outliers)")

if show_box_day_clean:
    fig_day_clean, ax_day_clean = plt.subplots(figsize=(16, 6))
    sns.boxplot(
        data=df_clean,
        x='journey_day',
        y='price',
        hue='journey_month',
        palette='Set2',
        showfliers=False,
        ax=ax_day_clean
    )
    ax_day_clean.set_title('Pris pr. dag i måneden, opdelt pr. måned (uden outliers)', fontsize=14)
    ax_day_clean.set_xlabel('Dag i måneden', fontsize=12)
    ax_day_clean.set_ylabel('Pris (DKK)', fontsize=12)
    ax_day_clean.legend(title='Måned', loc='upper right')
    plt.tight_layout()
    st.pyplot(fig_day_clean)
    plt.close(fig_day_clean)

if show_box_week_clean:
    fig_week_clean, ax_week_clean = plt.subplots(figsize=(16, 6))
    sns.boxplot(
        data=df_clean,
        x='journey_week',
        y='price',
        hue='journey_month',
        palette='Set2',
        showfliers=False,
        ax=ax_week_clean
    )
    ax_week_clean.set_title('Pris pr. ugenummer, opdelt pr. måned (uden outliers)', fontsize=14)
    ax_week_clean.set_xlabel('Ugenummer', fontsize=12)
    ax_week_clean.set_ylabel('Pris (DKK)', fontsize=12)
    ax_week_clean.legend(title='Måned', loc='upper right')
    plt.tight_layout()
    st.pyplot(fig_week_clean)
    plt.close(fig_week_clean)




