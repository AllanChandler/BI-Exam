import pandas as pd

def load_data_clean(path):
    # Læser datasættet
    df = pd.read_csv(path, index_col=0)

    # Fjerner 'flight' kolonnen, da den kun er en identifikator og ikke bidrager til analysen
    df.drop(['flight'], axis=1, inplace=True)  

    # Fjerner 'departure_time' kolonnen, da den ikke er relevant for vores forskningsspørgsmål om dage før afrejse og antal stop
    df.drop(['departure_time'], axis=1, inplace=True)  

    # Fjerner 'arrival_time' kolonnen, da den ikke er relevant for vores forskningsspørgsmål om dage før afrejse og antal stop
    df.drop(['arrival_time'], axis=1, inplace=True)  

    # Udfører one-hot encoding på de kategoriske kolonner for at gøre dem numeriske og lettere at regressere på
    dfNumeric = pd.get_dummies(df, columns=['airline', 'source_city', 'stops', 'destination_city', 'class'], dtype=pd.Int64Dtype())

    # Konverterer alle 'object'-kolonner til 'string'-type
    for col in dfNumeric:
        if dfNumeric[col].dtype == 'object':
            dfNumeric[col] = dfNumeric[col].astype('string')

    return dfNumeric

def load_data_train(path):
    # Læser datasættet
    df = pd.read_csv(path)

    # Fjerner duplikater
    df = df.drop_duplicates()

    # Fjerner 'Dep_Time' kolonnen, da den ikke er relevant for analysen
    df.drop(['Dep_Time'], axis=1, inplace=True) 
    
    # Konverterer 'Date_of_Journey' til datetime format
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y') 

    # Opretter en ny kolonne 'Journey_month' som indeholder måneden fra 'Date_of_Journey'
    df['Journey_month'] = df['Date_of_Journey'].dt.month
    df['Journey_month'] = df['Journey_month'].astype('int64')  # Skifter datatypen til int64

    # Opdaterer den eksisterende 'Class' kolonne baseret på værdier i 'Airline' da den gamle kun havde 0 værdier
    df['Class'] = df['Airline'].apply(lambda x: 'Business' if 'Business' in x else ('Premium economy' if 'Premium economy' in x else 'Standard'))

    # Udfører one-hot encoding på de kategoriske kolonner for at gøre dem numeriske og lettere at regressere på
    dfNumeric = pd.get_dummies(df, columns=['Airline', 'Source', 'Destination', 'Date_of_Journey'], dtype=pd.Int64Dtype())
    
    # Konverterer alle 'object'-kolonner til 'string'-type
    for col in dfNumeric:
        if dfNumeric[col].dtype == 'object':
            dfNumeric[col] = dfNumeric[col].astype('string')

    # Gør kolonnenavne til små bogstaver
    dfNumeric.columns = [col.lower() for col in dfNumeric.columns]

    return dfNumeric
