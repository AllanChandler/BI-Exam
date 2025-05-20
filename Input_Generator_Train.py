import pandas as pd
import streamlit as st

def create_input_row(journey_month, journey_week, journey_day, is_weekend, airline, flight_class, df_columns):
    inputs = {
        'journey_month': journey_month,
        'journey_week': journey_week,
        'journey_day': journey_day,
        'is_weekend': is_weekend,
    }

    prefixes = ['airline_', 'class_']
    values = [airline, flight_class]

    for prefix, value in zip(prefixes, values):
        col_name = f"{prefix}{value}"
        if col_name in df_columns:
            inputs[col_name] = 1
            for col in df_columns:
                if col.startswith(prefix) and col != col_name:
                    inputs[col] = 0
        else:
            for col in df_columns:
                if col.startswith(prefix):
                    inputs[col] = 0

    input_row = pd.DataFrame([inputs])
    input_row = input_row.reindex(columns=df_columns, fill_value=0)

    st.write(input_row)
    return input_row

def createNewRow(journey_month, journey_week, journey_day, is_weekend, airline, flight_class, price, dfCluster):
    inputs = {
        'journey_month': journey_month,
        'journey_week': journey_week,
        'journey_day': journey_day,
        'is_weekend': is_weekend,
        'price': price,
    }

    prefixes = ['airline_', 'class_']
    values = [airline, flight_class]

    for prefix, value in zip(prefixes, values):
        col_name = f"{prefix}{value}"
        if col_name in dfCluster.columns:
            inputs[col_name] = 1
            for col in dfCluster.columns:
                if col.startswith(prefix) and col != col_name:
                    inputs[col] = 0
        else:
            for col in dfCluster.columns:
                if col.startswith(prefix):
                    inputs[col] = 0

    input_row = pd.DataFrame([inputs])
    input_row = input_row.reindex(columns=dfCluster.columns, fill_value=0)

    st.write(input_row)
    return input_row

def createNewClassRow(journey_month, journey_week, journey_day, is_weekend, airline, flight_class, cluster_input, dfClassification):
    inputs = {
        'journey_month': journey_month,
        'journey_week': journey_week,
        'journey_day': journey_day,
        'is_weekend': is_weekend,
        'cluster': cluster_input,
    }

    prefixes = ['airline_', 'class_']
    values = [airline, flight_class]

    for prefix, value in zip(prefixes, values):
        col_name = f"{prefix}{value}"
        if col_name in dfClassification.columns:
            inputs[col_name] = 1
            for col in dfClassification.columns:
                if col.startswith(prefix) and col != col_name:
                    inputs[col] = 0
        else:
            for col in dfClassification.columns:
                if col.startswith(prefix):
                    inputs[col] = 0

    input_row = pd.DataFrame([inputs])
    input_row = input_row.reindex(columns=dfClassification.columns, fill_value=0)

    st.write(input_row)
    return input_row
