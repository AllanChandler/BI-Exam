import pandas as pd

def load_data_clean(path):
    # Læser datasættet
    df = pd.read_csv(path, index_col=0)

    # Fjerner irrelevante kolonner
    df.drop(['flight', 'departure_time', 'arrival_time'], axis=1, inplace=True)

    # Udfører One-Hot Encoding på de kategoriske kolonner
    df = pd.get_dummies(df, columns=['airline', 'source_city', 'stops', 'destination_city', 'class'], dtype=pd.Int64Dtype())

    # Konverterer alle 'object'-kolonner til 'string'-type
    for col in df:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('string')

    return df

def load_data_train(path):
    # Læs datasættet
    df = pd.read_csv(path)

    # Fjerner irrelevante kolonner
    df.drop(['Dep_Time'], axis=1, inplace=True)

    # Konverter 'Date_of_Journey' til datetime format
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')

    # Tilføjer 'Journey_month' og 'days_until_flight' kolonner
    df['Journey_month'] = df['Date_of_Journey'].dt.month
    df['Journey_month'] = df['Journey_month'].astype('int64')  # Skift datatype til int64

    # Opretter 'Class' kolonne baseret på 'Airline' kolonnen
    df['Class'] = df['Airline'].apply(lambda x: 'Business' if 'Business' in x else ('Premium economy' if 'Premium economy' in x else 'Standard'))

    # Udfører One-Hot Encoding på de kategoriske kolonner
    df = pd.get_dummies(df, columns=['Airline', 'Source', 'Destination', 'Date_of_Journey'], dtype=pd.Int64Dtype())

    # Konverterer alle 'object'-kolonner til 'string'-type
    for col in df:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('string')

    return df
