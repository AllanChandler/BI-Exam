import streamlit as st
import plotly.express as px
import pandas as pd

# Denne page bliver brugt som interaktiv dataudforskning af Clean datasættet
# Brugeren kan vælge om outliers skal medtages eller ej
# Visualiseringstyper inkluderer histogram, boxplot, scatterplot,
# korrelationsheatmap og barplot.
# Koden sikrer at kun passende kolonner bruges til de forskellige plots,
# med brugervenlige advarsler ved uegnede valg.

# Sideopsætning
st.set_page_config(
    page_title="Dataudforskning_Clean",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Titel med farvet baggrund
st.markdown("""
    <div style="background-image: linear-gradient(90deg, rgb(255, 75, 75), rgb(28, 3, 204)); padding:10px">
        <h2 style="color:white;text-align:center;">Udforskning af Train datasættet</h2>
    </div><br>
""", unsafe_allow_html=True)

# Valg om outliers skal medtages eller ej
df_option = st.radio("Vil du inkludere outliers?", ["Ja", "Nej"], horizontal=True)

# Henter datasæt fra session_state
try:
    if df_option == "Ja":
        df = st.session_state['dfClean']
    else:
        df = st.session_state['dfClean_no_outliers']
except KeyError:
    st.error("Datasæt ikke indlæst. Gå tilbage til startsiden og indlæs data først.")
    st.stop()

# Funktionen Vælg kolonner til visualisering
def column_picker(data):
    st.header('Vælg kolonner til diagrammer')
    # X: Kategori eller gruppe (typisk kategorisk)
    x = st.selectbox('**Vælg X (kategori/gruppe)**', data.columns)
    # Y: Måling (typisk numerisk)
    y = st.selectbox('**Vælg Y (måling)**', data.columns)
    # Z: Farveopdeling (kategori for farver)
    z = st.selectbox('**Vælg Z (farveopdeling)**', data.columns)
    return x, y, z

x, y, z = column_picker(df)

# Funktionen Tjek om kolonne er numerisk
def is_numeric(col):
    return pd.api.types.is_numeric_dtype(col)

# Funktionen Plotter histogram
def plot_histogram(data, x, y, z):
    if not is_numeric(data[y]):
        st.warning("Histogram kræver at Y-kolonnen er numerisk.")
        return
    try:
        fig = px.histogram(data, x=x, y=y, color=z, title='Histogram', histfunc='avg')
        st.plotly_chart(fig)
    except Exception:
        st.warning("Kan ikke lave histogram med de valgte kolonner.")

# Funktionen Plotter boxplot
def plot_boxplot(data, x, y, z):
    if not is_numeric(data[y]):
        st.warning("Boxplot kræver at Y-kolonnen er numerisk.")
        return
    try:
        fig = px.box(data, x=x, y=y, color=z, title='Boxplot')
        st.plotly_chart(fig)
    except Exception:
        st.warning("Kan ikke lave boxplot med de valgte kolonner.")

# Funktionen Plotter scatterplot
def plot_scatterplot(data, x, y, z):
    if not is_numeric(data[y]):
        st.warning("Scatterplot kræver at Y-kolonnen er numerisk.")
        return
    try:
        fig = px.scatter(data, x=x, y=y, color=z, title='Scatterplot')
        st.plotly_chart(fig)
    except Exception:
        st.warning("Kan ikke lave scatterplot med de valgte kolonner.")

# Funktionen Plotter korrelations-heatmap
def plot_correlation_heatmap(data, x, y):
    if not (is_numeric(data[x]) and is_numeric(data[y])):
        st.warning("Korrelations-heatmap kræver, at både X og Y er numeriske kolonner.")
        return
    try:
        corr_matrix = data[[x, y]].corr()
        fig = px.imshow(corr_matrix, title='Correlation Heatmap', text_auto=True)
        st.plotly_chart(fig)
    except Exception:
        st.warning("Kan ikke lave korrelations-heatmap med de valgte kolonner.")

# Funktionen Plotter barplot
def plot_barplot(data, x, y):
    if not is_numeric(data[y]):
        st.warning("Barplot kræver at Y-kolonnen er numerisk.")
        return
    try:
        df_avg = data.groupby(x)[y].mean().reset_index()
        fig = px.bar(df_avg, x=x, y=y, title='Barplot (gennemsnit pr. kategori)')
        st.plotly_chart(fig)
    except Exception:
        st.warning("Kan ikke lave barplot med de valgte kolonner.")

# Funktionen Plotter Visualiseringer i faner
def show_charts(data, x, y, z):
    # Opretter faner til visualiseringerne samt Om fane
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ['Om', 'Histogram', 'Boxplot', 'Scatterplot', 'Correlation Heatmap', 'Barplot']
    )

    with tab1:
        st.write("### Om datasættet og kolonner")
        st.write("Her vises oversigt over kolonner og deres datatype i det valgte datasæt.")
        col_info = pd.DataFrame({
            'Kolonne': data.columns,
            'Datatype': [str(data[col].dtype) for col in data.columns]
        })
        st.table(col_info)

    with tab2:
        st.write("**Histogram:** Viser fordelingen af værdier samt gennemsnit pr. kategori.")
        st.write("God til at analysere mønstre og hyppighed i data.")
        plot_histogram(data, x, y, z)

    with tab3:
        st.write("**Boxplot:** Viser minimum, Q1, median, Q3, maksimum og outliers.")
        st.write("Bruges til at analysere variation og finde ekstreme værdier.")
        plot_boxplot(data, x, y, z)

    with tab4:
        st.write("**Scatterplot:** Visualiserer forholdet mellem to variabler.")
        st.write("Kan bruges til at opdage mønstre, tendenser og sammenhænge i data.")
        plot_scatterplot(data, x, y, z)


    with tab5:
        st.write("**Korrelations-heatmap:** Viser styrken af lineære sammenhænge.")
        st.write("Bruges til at finde variabler der hænger sammen (positivt/negativt).")
        plot_correlation_heatmap(data, x, y)

    with tab6:
        st.write("**Barplot:** Viser gennemsnitlig værdi (Y) for hver kategori (X).")
        st.write("Giver overblik over forskelle mellem grupper på en nem måde.")
        plot_barplot(data, x, y)

# Knappen til at starte visualiseringerne
if st.button(":green[Explore]"):
    show_charts(df, x, y, z)
else:
    # Hvis ikke udforsket endnu, vises info i fanerne
    tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ['Histogram', 'Boxplot', 'Scatterplot', 'Correlation Heatmap', 'Barplot']
    )
    with tab2:
        st.info("Tryk på Explore for at se histogrammet.")
    with tab3:
        st.info("Tryk på Explore for at se boxplot.")
    with tab4:
        st.info("Tryk på Explore for at se scatterplot.")
    with tab5:
        st.info("Tryk på Explore for at se korrelations-heatmap.")
    with tab6:
        st.info("Tryk på Explore for at se barplot.")
