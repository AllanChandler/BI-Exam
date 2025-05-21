import pandas as pd
import streamlit as st

# Denne fil bruges til at generere input-rækker (DataFrames med en række)
# der er struktureret i overensstemmelse med Train datasættet, som anvendes i ML-modeller.
# Formålet er at omsætte brugerinput til et format, der passer til modellerne,
# herunder håndtering af både numeriske værdier og one-hot encoded kategoriske variable,
# så input kan anvendes direkte til regression, clustering eller klassifikation.

# Funktioner:
# - create_input_row: Returnerer en inputrække uden 'price' eller 'cluster' – til regression (prisforudsigelse).
# - createNewRow: Genererer en inputrække inklusive 'price' – til clustering.
# - createNewClassRow: Returnerer en inputrække med 'cluster' – til klassifikation.

# Funktion 1: create_input_row
# Genererer en inputrække til prisforudsigelse via regression – uden pris og cluster i input.
def create_input_row(journey_month, journey_week, journey_day, is_weekend, airline, flight_class, df_columns):
    # Opretter en ordbog med de numeriske inputfelter
    inputs = {
        'journey_month': journey_month,
        'journey_week': journey_week,
        'journey_day': journey_day,
        'is_weekend': is_weekend,
    }

    # Kategoriske variable der skal one-hot encodes
    prefixes = ['airline_', 'class_']
    values = [airline, flight_class]

    # Tilføjer one-hot encoding for hver kategorisk variabel
    for prefix, value in zip(prefixes, values):
        col_name = f"{prefix}{value}"  # F.eks. "airline_IndiGo" eller "class_Economy"
        
        if col_name in df_columns:
            # Hvis den valgte kategori findes som kolonne, sæt den til 1 og alle andre til 0
            # - 1 = valgte kategori (brugeren har valgt denne)
            # - 0 = ikke valgte kategorier (de resterende muligheder)
            inputs[col_name] = 1
            for col in df_columns:  
                if col.startswith(prefix) and col != col_name:
                    inputs[col] = 0
        else:
            # Hvis den valgte kategori ikke findes som kolonne (fx pga. fejl), sæt alle til 0
            for col in df_columns:
                if col.startswith(prefix):
                    inputs[col] = 0

    # Konverterer ordbog til DataFrame og sikrer korrekt kolonneorden og udfyldning
    input_row = pd.DataFrame([inputs])
    input_row = input_row.reindex(columns=df_columns, fill_value=0)

    # Viser input-row i Streamlit til kontrol 
    st.write(input_row)
    return input_row

# Funktion 2: createNewRow
# Bruges til at generere en række til clustering – inkluderer både pris og kategorier.
def createNewRow(journey_month, journey_week, journey_day, is_weekend, airline, flight_class, price, dfCluster):
    # Initialiserer ordbog med numeriske værdier og pris
    inputs = {
        'journey_month': journey_month,
        'journey_week': journey_week,
        'journey_day': journey_day,
        'is_weekend': is_weekend,
        'price': price,
    }

    prefixes = ['airline_', 'class_']
    values = [airline, flight_class]

    # One-hot encoding for 'airline' og 'class' – samme fremgangsmåde som i create_input_row
    for prefix, value in zip(prefixes, values):
        col_name = f"{prefix}{value}"

        if col_name in dfCluster.columns:
        # 1 = brugeren har valgt denne kategori, 0 = resten
            inputs[col_name] = 1
            for col in dfCluster.columns:
                if col.startswith(prefix) and col != col_name:
                    inputs[col] = 0
        else:
            # Hvis ukendt kategori – sættes alle kategorier med samme prefix til 0
            for col in dfCluster.columns:
                if col.startswith(prefix):
                    inputs[col] = 0

    # Konverterer ordbog til DataFrame og reordner kolonnerne
    input_row = pd.DataFrame([inputs])
    input_row = input_row.reindex(columns=dfCluster.columns, fill_value=0)

    # Viser input-row i Streamlit til kontrol
    st.write(input_row)
    return input_row

# Funktion 3: createNewClassRow
# Bruges til klassifikation – inkluderer cluster og kategorier.
def createNewClassRow(journey_month, journey_week, journey_day, is_weekend, airline, flight_class, cluster_input, dfClassification):
    # Initialiserer ordbog med numeriske inputfelter + cluster
    inputs = {
        'journey_month': journey_month,
        'journey_week': journey_week,
        'journey_day': journey_day,
        'is_weekend': is_weekend,
        'cluster': cluster_input,
    }

    # One-hot encoding for 'airline' og 'class'
    prefixes = ['airline_', 'class_']
    values = [airline, flight_class]

    for prefix, value in zip(prefixes, values):
        col_name = f"{prefix}{value}"

        if col_name in dfClassification.columns:
            # Brugervalgt kategori = 1, andre = 0
            inputs[col_name] = 1
            for col in dfClassification.columns:
                if col.startswith(prefix) and col != col_name:
                    inputs[col] = 0
        else:
            # Ukendt kategori – sæt alle til 0
            for col in dfClassification.columns:
                if col.startswith(prefix):
                    inputs[col] = 0

    # Konverterer ordbog til DataFrame og reordner kolonner
    input_row = pd.DataFrame([inputs])
    input_row = input_row.reindex(columns=dfClassification.columns, fill_value=0)

    # Viser input-row i Streamlit til kontrol
    st.write(input_row)
    return input_row