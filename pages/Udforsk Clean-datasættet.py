import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Dataudforskning_Clean",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <div style="background-image: linear-gradient(90deg, rgb(255, 75, 75), rgb(28, 3, 204)); padding:10px">
        <h2 style="color:white;text-align:center;">Udforskning af Clean datasættet</h2>
    </div>
    <br>
""", unsafe_allow_html=True)

df_option = st.radio("Vil du inkludere outliers?", ["Ja", "Nej"], horizontal=True)

try:
    if df_option == "Ja":
        df = st.session_state['dfClean']
    else:
        df = st.session_state['dfClean_no_outliers']
except KeyError:
    st.error("Datasæt ikke indlæst. Gå tilbage til startsiden og indlæs data først.")
    st.stop()

def column_picker(df):
    st.header('Vælg kolonner til diagrammer')
    x = st.selectbox('**Vælg X (kategori/gruppe)**', df.columns)
    y = st.selectbox('**Vælg Y (måling)**', df.columns)
    z = st.selectbox('**Vælg Z (farveopdeling)**', df.columns)
    return x, y, z

def charts():
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Histogram', 'Boxplot', 'Scatterplot', 'Lineplot', 'Correlation Heatmap', "Barplot"])
    with tab1:
        fig = px.histogram(df, x=x, y=y, color=z, title='Histogram', histfunc='avg')
        st.plotly_chart(fig)
    with tab2:
        fig = px.box(df, x=x, y=y, color=z, title='Boxplot')
        st.plotly_chart(fig)
    with tab3:
        fig = px.scatter(df, x=x, y=y, color=z, title='Scatterplot')
        st.plotly_chart(fig)
    with tab4:
        fig = px.line(df, x=x, y=y, color=z, title='Lineplot')
        st.plotly_chart(fig)
    with tab5:
        try:
            fig = px.imshow(df[[x, y]].corr(), title='Correlation Heatmap')
            st.plotly_chart(fig)
        except:
            st.warning("Vælg to numeriske kolonner til heatmap.")
    with tab6:
        df_avg = df.groupby(x)[y].mean().reset_index()
        fig = px.bar(df_avg, x=x, y=y, color=None, title='Barplot (gennemsnit pr. kategori)')
        st.plotly_chart(fig)

x, y, z = column_picker(df)

if st.button(":green[Explore]"):
    st.subheader("Udforsk data med interaktive diagrammer")
    st.write('Skift mellem fanerne for at analysere forskellige visualiseringer')
    charts()
