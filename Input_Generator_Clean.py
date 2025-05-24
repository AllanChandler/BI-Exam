import pandas as pd
import streamlit as st

# Denne fil bruges til at generere input-rækker (DataFrames med en række)
# der er struktureret i overensstemmelse med Clean datasættet, som anvendes i ML-modeller.
# Formålet er at omsætte brugerinput til et format, der passer til modellerne,
# herunder håndtering af både numeriske værdier og one-hot encoded kategoriske variable,
# så input kan anvendes direkte til regression, clustering eller klassifikation.

# Funktioner:
# - encode_categorical: Hjælpefunktion til one-hot encoding af kategoriske variabler, så der undgås gentagelser i koden.
# - create_input_row: Returnerer en inputrække uden 'price' eller 'cluster' – til regression (prisforudsigelse).
# - createNewRow: Genererer en inputrække inklusive 'price' – til clustering.
# - createNewClassRow: Returnerer en inputrække med 'cluster' – til klassifikation.

# Funktion 1: encode_categorical
# Hjælpefunktion der tilføjer one-hot encoding til en input-ordbog (inputs)
def encode_categorical(inputs, df_columns, prefixes, values):
    # Itererer over par af prefix (f.eks. "airline_") og valgte værdier (f.eks. "IndiGo")
    for prefix, value in zip(prefixes, values):
        col_name = f"{prefix}{value}"  # Danner det forventede kolonnenavn, f.eks. "airline_IndiGo"

        if col_name in df_columns:
            # Hvis den specifikke kategori eksisterer i kolonnerne:
            inputs[col_name] = 1  # Sæt den valgte kategori til 1 (aktiv)

            # Sæt alle andre kategorier med samme prefix til 0
            for col in df_columns:
                if col.startswith(prefix) and col != col_name:
                    inputs[col] = 0
        else:
            # Hvis den valgte kategori ikke findes som kolonne, sæt alle kategorier med samme prefix til 0
            for col in df_columns:
                if col.startswith(prefix):
                    inputs[col] = 0

    return inputs  # Returner den opdaterede input-dictionary


# Funktion 2: create_input_row
# Genererer en inputrække til prisforudsigelse via regression – uden pris og cluster i input.
def create_input_row(airline, stops, days_left, df_columns):
    # Opretter en ordbog med numeriske felter
    inputs = {
        'days_left': days_left,
    }


    # Tilføjer one-hot encodede kategorier via hjælpefunktionen
    prefixes = ['airline_', 'stops_']
    values = [airline, stops]
    inputs = encode_categorical(inputs, df_columns, prefixes, values)

    # Konverterer ordbog til DataFrame og tilpasser kolonneorden
    input_row = pd.DataFrame([inputs])
    input_row = input_row.reindex(columns=df_columns, fill_value=0)

    # Viser input-row i Streamlit til kontrol
    st.write(input_row)
    return input_row

# Funktion 3: createNewRow
# Bruges til at generere en række til clustering – inkluderer både pris og kategorier.
def createNewRow(airline, stops, days_left, price, dfCluster):
    # Numeriske felter + pris
    inputs = {
        'days_left': days_left,
        'price': price,
    }

    # One-hot encodede kategorier
    prefixes = ['airline_', 'stops_']
    values = [airline, stops]
    inputs = encode_categorical(inputs, dfCluster.columns, prefixes, values)

    # Konverterer ordbog til DataFrame og tilpasser kolonner
    input_row = pd.DataFrame([inputs])
    input_row = input_row.reindex(columns=dfCluster.columns, fill_value=0)

    # Viser input-row i Streamlit til kontrol
    st.write(input_row)
    return input_row


# Funktion 4: createNewClassRow
# Bruges til klassifikation – inkluderer cluster og kategorier.
def createNewClassRow(airline, stops, days_left, cluster_input, dfClassification):
    # Numeriske felter + cluster
    inputs = {
        'days_left': days_left,
        'cluster': cluster_input,
    }

    prefixes = ['airline_', 'stops_']
    values = [airline, stops]
    inputs = encode_categorical(inputs, dfClassification.columns, prefixes, values)

    # Konverter til DataFrame med korrekt kolonneorden
    input_row = pd.DataFrame([inputs])
    input_row = input_row.reindex(columns=dfClassification.columns, fill_value=0)

    # Viser input-row i Streamlit
    st.write(input_row)
    return input_row