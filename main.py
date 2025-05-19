import streamlit as st
from streamlit_option_menu import option_menu

import glob

st.set_page_config(
    page_title="BI Exam Project",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.header("Choose a page!", divider='rainbow')

banner = """
    <body style="background-color:yellow;">
        <div style="background-image: linear-gradient(90deg, rgb(255, 75, 75), rgb(28, 3, 204)); padding:10px">
            <h2 style="color:white;text-align:center;"> BI Exam Project </h2>
            <h3 style="color:white;text-align:center;"> By Magnus, Peter and Yusuf </h3>
            <div style="text-align:center">
                <span style="color:white;text-align:center;"> This project contains algorithms for exploring flight price data. </span>
            </div>
        </div>
    </body>
    <br>
"""

st.markdown(banner, unsafe_allow_html=True)

try:
    from datarenser import (
        load_data_clean,
        load_data_train,
        get_numeric_df_clean,
        get_numeric_df_train,
        get_no_outliers_df_clean,
        get_no_outliers_df_train
    )

    # Indlæs datasæt
    if glob.glob("data/Clean_Dataset.csv"):
        st.session_state['dfClean'] = load_data_clean("data/Clean_Dataset.csv")
    else:
        raise FileNotFoundError("Clean data file not found")

    if glob.glob("data/Data_Train.csv"):
        st.session_state['dfTrain'] = load_data_train("data/Data_Train.csv")
    else:
        raise FileNotFoundError("Train data file not found")

    # Først numeriske versioner af rå data (inkl. outliers)
    st.session_state['dfClean_numeric'] = get_numeric_df_clean(st.session_state['dfClean'])
    st.session_state['dfTrain_numeric'] = get_numeric_df_train(st.session_state['dfTrain'])

    # Dernæst fjern outliers fra rå data
    st.session_state['dfClean_no_outliers'] = get_no_outliers_df_clean(st.session_state['dfClean'])
    st.session_state['dfTrain_no_outliers'] = get_no_outliers_df_train(st.session_state['dfTrain'])

except Exception as e:
    st.error("Error loading or processing data. Please ensure the data files are in the 'Data' folder.")
    st.error(str(e))
    st.error("If these errors persist, please contact the developers at cph-mk797@cphbusiness.dk")