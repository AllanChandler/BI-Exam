import streamlit as st
from streamlit_option_menu import option_menu
import glob

# Konfigurerer Streamlit siden med titel, ikon, layout og sidebar startstatus
st.set_page_config(
    page_title="BI Exam Projekt",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# HTML-banner med farver og tekst øverst på siden
banner = """
    <body style="background-color:yellow;">
        <div style="background-image: linear-gradient(90deg, rgb(255, 75, 75), rgb(28, 3, 204)); padding:10px">
            <h2 style="color:white;text-align:center;"> BI Exam Project </h2>
            <h3 style="color:white;text-align:center;"> Lavet af: Allan, Malene, Marwa og Nicklas </h3>
            <div style="text-align:center">
            <span style="color:white;text-align:center;"> Dette projekt indeholder algoritmer til at udforske flyprisdata. </span>
            </div>
        </div>
    </body>
    <br>
"""

# Viser det definerede HTML-banner i Streamlit-appen
st.markdown(banner, unsafe_allow_html=True)

try:
    # Importerer alle nødvendige funktioner til dataindlæsning og forbehandling fra ekstern fil 'datarenser.py'
    from datarenser import (
        load_data_clean,
        load_data_train,
        get_numeric_df_clean,
        get_numeric_df_train,
        get_no_outliers_df_clean,
        get_no_outliers_df_train
    )

    # Tjekker om Clean datasæt findes i data mappen og indlæser det i session_state
    if glob.glob("data/Clean_Dataset.csv"):
        st.session_state['dfClean'] = load_data_clean("data/Clean_Dataset.csv")
    else:
        # Fejlmeddelelse hvis filen ikke findes
        raise FileNotFoundError("Clean data file not found")

    # Tjekker om Train datasæt findes og indlæser det i session_state
    if glob.glob("data/Data_Train.csv"):
        st.session_state['dfTrain'] = load_data_train("data/Data_Train.csv")
    else:
        # Fejlmeddelelse hvis filen ikke findes
        raise FileNotFoundError("Train data file not found")

    # Konverterer Clean datasættet til en version kun med numeriske kolonner 
    st.session_state['dfClean_numeric'] = get_numeric_df_clean(st.session_state['dfClean'])

    # Konverterer Train datasættet til en version kun med numeriske kolonner
    st.session_state['dfTrain_numeric'] = get_numeric_df_train(st.session_state['dfTrain'])

    # Fjerner outliers fra Clean datasættet og gemmer den rensede version
    st.session_state['dfClean_no_outliers'] = get_no_outliers_df_clean(st.session_state['dfClean'])

    # Fjerner outliers fra Train datasættet og gemmer den rensede version
    st.session_state['dfTrain_no_outliers'] = get_no_outliers_df_train(st.session_state['dfTrain'])

# Fanger alle fejl der opstår under import, dataindlæsning eller forbehandling
except Exception as e:
    # Viser fejlmeddelelse i appen, fx hvis datafiler mangler eller funktioner fejler
    st.error("Fejl ved indlæsning eller behandling af data. Sørg for, at datafilerne ligger i mappen 'data'.")
    st.error(str(e))
