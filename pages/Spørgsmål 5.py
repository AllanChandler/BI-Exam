
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