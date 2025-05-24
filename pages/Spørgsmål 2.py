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
    df = st.session_state['dfClean'].copy()
    df['stops_numb'] = df['stops'].map({'zero': 0, 'one': 1, 'two_or_more': 2})
    df_encoded = pd.get_dummies(df, columns=['airline', 'source_city', 'stops', 'destination_city', 'class'], dtype=int)
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

if st.checkbox("Vis søjlediagram: Gennemsnitspris pr. antal stop"):
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


# --------------------------
# MODELTRÆNING
# --------------------------

st.subheader("Modellering")

st.subheader("📉RandomForestRegressor")

st.subheader("⚙️ Træn Model og Evaluer")

if st.button("Træn Random Forest Model"):
    X = df_encoded.drop('price', axis=1)
    y = df_encoded['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=30, random_state=42)
    model.fit(X_train, y_train)

    st.session_state.model = model

    y_pred = model.predict(X_test)

    st.success("✅ Model trænet!")
    st.write(f"**MSE**: {mean_squared_error(y_test, y_pred):,.0f}")
    st.write(f"**RMSE**: {np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
    st.write(f"**MAE**: {mean_absolute_error(y_test, y_pred):,.0f}")
    st.write(f"**R²**: {r2_score(y_test, y_pred):.2f}")

    importances = model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[-10:]

    fig, ax = plt.subplots()
    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Vigtighed')
    ax.set_title('Top 10 vigtigste features')
    st.pyplot(fig)

# --------------------------
# FORUDSIGELSE
# --------------------------
st.subheader("📈 Forudsig Pris")

if 'model' not in st.session_state:
    st.warning("⚠️ Du skal først træne modellen.")
else:
    model = st.session_state.model

    airlines = df['airline'].unique()
    sources = df['source_city'].unique()
    destinations = df['destination_city'].unique()
    stops_options = ['zero', 'one', 'two_or_more']
    classes = ['Economy', 'Business']

    with st.form("prediction_form"):
        st.markdown("### 🔧 Indtast flyoplysninger:")

        selected_airline = st.selectbox("Fly-selskab", airlines)
        selected_source = st.selectbox("Afrejseby", sources)
        selected_destination = st.selectbox("Destination", destinations)
        selected_stops = st.selectbox("Antal stop", stops_options)
        selected_class = st.radio("Rejseklasse", classes)
        duration = st.slider("Varighed (timer)", 1, 50, 10)
        days_left = st.slider("Dage til afrejse", 1, 60, 30)

        submitted = st.form_submit_button("🔮 Forudsig pris")

        if submitted:
            input_dict = {
                'duration': duration,
                'days_left': days_left,
                'stops_numb': {'zero': 0, 'one': 1, 'two_or_more': 2}[selected_stops],
                'class_Business': 1 if selected_class == 'Business' else 0,
                f'airline_{selected_airline}': 1,
                f'source_city_{selected_source}': 1,
                f'destination_city_{selected_destination}': 1,
            }

            for col in model.feature_names_in_:
                if col not in input_dict:
                    input_dict[col] = 0

            input_df = pd.DataFrame([input_dict])[model.feature_names_in_]
            pred_price = model.predict(input_df)[0]
            st.success(f"💰 Forudsagt pris: {pred_price:,.0f}")

# --------------------------
# MULTIPLE LINEAR REGRESSION
# --------------------------
st.subheader("📉 Multiple Linear Regression Model")

if st.checkbox("Træn og vis Multiple Linear Regression"):
    feature_cols = ['class_Business', 'stops_two_or_more', 'stops_one']
    X_mlr = df_encoded[feature_cols]
    y_mlr = df_encoded['price']

    X_train_mlr, X_test_mlr, y_train_mlr, y_test_mlr = train_test_split(X_mlr, y_mlr, test_size=0.2, random_state=1)

    linreg = LinearRegression()
    linreg.fit(X_train_mlr, y_train_mlr)

    y_pred_mlr = linreg.predict(X_test_mlr)

    st.markdown("**🔢 Modelkoefficienter**")
    coef_df = pd.DataFrame({
        'Feature': feature_cols,
        'Koefficient': linreg.coef_
    })
    st.table(coef_df)

    st.write(f"**Intercept (b₀):** {linreg.intercept_:.2f}")
    st.write(f"**MAE:** {mean_absolute_error(y_test_mlr, y_pred_mlr):,.2f}")
    st.write(f"**MSE:** {mean_squared_error(y_test_mlr, y_pred_mlr):,.2f}")
    st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test_mlr, y_pred_mlr)):.2f}")
    st.write(f"**R² Score:** {r2_score(y_test_mlr, y_pred_mlr):.2f}")

    fig, ax = plt.subplots()
    ax.scatter(y_test_mlr, y_pred_mlr, alpha=0.3)
    ax.set_xlabel("Sande priser")
    ax.set_ylabel("Forudsagte priser")
    ax.set_title("Multiple Linear Regression: Sande vs. Forudsagte Priser")
    st.pyplot(fig)

st.subheader("📉 Multiple Linear Regression med Log-Transform")

if st.checkbox("Træn log-transformeret Multiple Linear Regression"):
    df_log = df_encoded[df_encoded['price'] > 0].copy()
    df_log['log_price'] = np.log(df_log['price'])

    feature_cols = ['class_Business', 'stops_two_or_more', 'stops_one']
    X_log = df_log[feature_cols]
    y_log = df_log['log_price']

    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, random_state=1)

    linreg_log = LinearRegression()
    linreg_log.fit(X_train_log, y_train_log)
    y_pred_log = linreg_log.predict(X_test_log)

    mae = mean_absolute_error(y_test_log, y_pred_log)
    mse = mean_squared_error(y_test_log, y_pred_log)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_log, y_pred_log)

    st.write(f"**Intercept (b₀):** {linreg_log.intercept_:.3f}")
    st.write(f"**MAE (log):** {mae:.3f}")
    st.write(f"**MSE (log):** {mse:.3f}")
    st.write(f"**RMSE (log):** {rmse:.3f}")
    st.write(f"**R² Score:** {r2:.2f}")

    coef_table = pd.DataFrame({
        'Feature': feature_cols,
        'Koefficient (log)': linreg_log.coef_,
        'Multiplikator (e^b)': np.exp(linreg_log.coef_),
        'Tillægspris (ca. kr)': np.exp(linreg_log.coef_) * np.exp(linreg_log.intercept_) - np.exp(linreg_log.intercept_)
    })
    st.markdown("**📊 Modelkoefficienter og fortolkning:**")
    st.table(coef_table)

    fig, ax = plt.subplots()
    ax.scatter(y_test_log, y_pred_log, alpha=0.3)
    ax.set_xlabel("Sande log(priser)")
    ax.set_ylabel("Forudsagte log(priser)")
    ax.set_title("Log-Regression: Sande vs. Forudsagte Log(Priser)")
    st.pyplot(fig)

    log_price_example = linreg_log.intercept_ + linreg_log.coef_[0] * 1
    price_example = np.exp(log_price_example)
    st.write(f"Eksempel: Business uden stop ≈ {price_example:,.0f} kr.")

