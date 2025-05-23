import pandas as pd
from z_score import beregnListe, beregnDataFrame, hentOutliers

# Denne fil er blevet lavet fordi det er vigtigt med ensartet og pålidelig databehandling
# til analyse og maskinlæring. Den håndterer datarensning og gør dataforberedelsen hurtig,
# genanvendelig og struktureret.

# Funktioner:
# - load_data_clean(path): Læser og renser data til analyse (fjerner irrelevante kolonner, konverterer typer osv.).
# - load_data_train(path): Læser og renser data til analyse (fjerner irrelevante kolonner, konverterer typer osv.).
# - get_numeric_df_clean(df): One-hot encodning af kategoriske variabler i 'clean' datasættet.
# - get_numeric_df_train(df): One-hot encodning af kategoriske variabler i 'train' datasættet.
# - get_no_outliers_df_clean(df): Fjerner outliers i 'price' vha. z-score i 'clean' datasættet.
# - get_no_outliers_df_train(df): Kombineret IQR og z-score filtrering af outliers i 'train' datasættet.

def load_data_clean(path):
    # Læser datasættet
    df = pd.read_csv(path, index_col=0)

    # Fjerner kolonner, der ikke er relevante for analysen
    df.drop(['flight', 'departure_time', 'arrival_time'], axis=1, inplace=True)

    # Sikrer ens datatype ved at konvertere 'object'-kolonner til 'string'-kolonner
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('string')


    # Gør 'stops' numerisk for brug i f.eks. korrelationsanalyse
    df['stops_numb'] = df['stops'].map({
        'zero': 0,
        'one': 1,
        'two_or_more': 2,
    })

    return df

def load_data_train(path):
    # Læser datasættet
    df = pd.read_csv(path)

    # Fjerner duplikater for at undgå overvægt af gentagne data
    df = df.drop_duplicates()

    # Fjerner 'Dep_Time' kolonnen, da den ikke skal bruges i analysen
    df.drop(['Dep_Time'], axis=1, inplace=True)

    # Konverterer dato til datetime-type for nem ekstraktion
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')

    # Ekstraherer måned, dag, ugenummer og tilføjer en kolonne, der angiver om rejsen er i weekenden (lørdag eller søndag) fra 'Date_of_Journey' som nye kolonner
    df['Journey_month'] = df['Date_of_Journey'].dt.month.astype('int64')
    df['Journey_day'] = df['Date_of_Journey'].dt.day.astype('int64')
    df['Journey_week'] = df['Date_of_Journey'].dt.isocalendar().week.astype('int64')
    df['Is_weekend'] = (df['Date_of_Journey'].dt.dayofweek >= 5).astype('int64')

    # Fjerner 'Date_of_Journey', da nødvendige data nu er ekstraheret
    df.drop('Date_of_Journey', axis=1, inplace=True)

    # Opdaterer 'Class'-kolonnen ud fra indholdet i 'Airline' for at kategorisere klassen
    # (dvs. grupperer klasserne i typer som Business, Premium economy eller Standard)
    df['Class'] = df['Airline'].apply(
        lambda x: 'Business' if 'Business' in x else (
            'Premium economy' if 'Premium economy' in x else 'Standard'))

    # Sikrer ens datatype ved at konvertere 'object'-kolonner til 'string'-kolonner
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('string')

    # Gør 'Class' numerisk for brug i f.eks. korrelationsanalyse
    df['class_numb'] = df['Class'].map({
        'Standard': 0,
        'Premium economy': 1,
        'Business': 2,
    })

    # Omdanner 'Airline' fra tekst til numerisk værdi i en ny kolonne 'airline_numb'
    unique_airlines = df['Airline'].unique()

    # Laver en dictionary til mapping, f.eks. 'airline_numb': 0, 1, 2 osv.
    airline_mapping = {airline: i for i, airline in enumerate(unique_airlines)}

    # Anvender mapping til at lave en ny numerisk kolonne
    df['airline_numb'] = df['Airline'].map(airline_mapping)

    # Gør kolonnenavne små for at sikre ensartethed, gøre dem nemmere at arbejde med og forbedre læsbarheden
    df.columns = [col.lower() for col in df.columns]

    return df

def get_numeric_df_clean(df):
    # One-hot encoder kategoriske kolonner så de kan bruges i modeller
    df_numeric = pd.get_dummies(df, columns=['airline', 'source_city', 'stops', 'destination_city', 'class'], dtype=pd.Int64Dtype())
    return df_numeric

def get_numeric_df_train(df):
    # One-hot encoder relevante kategoriske kolonner
    df_numeric = pd.get_dummies(df, columns=['airline', 'source', 'destination', 'class'], dtype=pd.Int64Dtype())
    return df_numeric

def get_no_outliers_df_clean(df):
    # Kopierer data for ikke at ændre den originale DataFrame
    df_copy = df.copy()

    # Fjerner outliers i 'price' baseret på z-score-metoden (tærskel 3)
    clean_price_list = beregnListe(df_copy['price'].tolist(), tærskel=3.0, fjern=True)

    # Udvælger kun rækker med 'price'-værdier, der ikke er outliers
    df_no_outliers_cleaned = df_copy[df_copy['price'].isin(clean_price_list)].copy()

    # Returnerer DataFrame uden outliers og uden rækker med manglende 'price'-værdier
    return df_no_outliers_cleaned.dropna(subset=['price'])

def get_no_outliers_df_train(df):
    # Kombinerer IQR- og z-score-metoder for en mere præcis og robust outlier-detektion

    # Beregner grænser baseret på interkvartilafstand (IQR) for at fjerne ekstreme værdier
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Fjerner outliers uden for IQR-intervallet
    df_iqr_filtered = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)].copy()

    # Anvender z-score på det IQR-filtrerede datasæt for at identificere yderligere outliers
    beregnDataFrame(df_iqr_filtered[['price']], tærskel=3.0, fjern=False)

    # Henter listen over z-score-baserede outliers
    price_outliers = hentOutliers()

    # Fjerner de identificerede z-score outliers
    df_no_outliers_train = df_iqr_filtered[~df_iqr_filtered['price'].isin(price_outliers)].copy()

    # Returnerer renset DataFrame uden outliers og uden manglende 'price'-værdier
    return df_no_outliers_train.dropna(subset=['price'])