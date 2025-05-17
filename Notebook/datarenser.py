import pandas as pd
from z_score import beregnListe, beregnDataFrame, hentOutliers

def load_data_clean(path):
    # Læser datasættet
    df = pd.read_csv(path, index_col=0)

    # Fjerner irrelevante kolonner
    df.drop(['flight', 'departure_time', 'arrival_time'], axis=1, inplace=True)

    # Konverterer alle 'object'-kolonner til 'string'-type
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('string')

    return df

def load_data_train(path):
    # Læser datasættet
    df = pd.read_csv(path)

    # Fjerner duplikater
    df = df.drop_duplicates()

    # Fjerner 'Dep_Time' kolonnen
    df.drop(['Dep_Time'], axis=1, inplace=True)

    # Konverterer 'Date_of_Journey' til datetime
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')

    # Ekstraherer måned
    df['Journey_month'] = df['Date_of_Journey'].dt.month.astype('int64')

    # Fjerner den oprindelige dato, da vi ikke skal bruge den mere
    df.drop('Date_of_Journey', axis=1, inplace=True)

    # Opdaterer 'Class'-kolonnen
    df['Class'] = df['Airline'].apply(
        lambda x: 'Business' if 'Business' in x else (
            'Premium economy' if 'Premium economy' in x else 'Standard'))

    # Konverterer 'object'-kolonner til 'string'
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('string')

    # Gør kolonnenavne små
    df.columns = [col.lower() for col in df.columns]

    return df

def get_numeric_df_clean(df):
    # One-hot encoding af de kategoriske kolonner
    df_numeric = pd.get_dummies(df, columns=['airline', 'source_city', 'stops', 'destination_city', 'class'], dtype=pd.Int64Dtype())
    return df_numeric

def get_numeric_df_train(df):
    # One-hot encoding af de kategoriske kolonner
    df_numeric = pd.get_dummies(df, columns=['airline', 'source', 'destination', 'class'], dtype=pd.Int64Dtype())
    return df_numeric

def get_no_outliers_df_clean(df):
    # Laver en kopi af DataFrame for at undgå at ændre originalen
    df_copy = df.copy()

    # Anvender z-score metoden på 'price' og fjerner outliers fra listen
    clean_price_list = beregnListe(df_copy['price'].tolist(), tærskel=3.0, fjern=True)

    # Filtrerer DataFrame så kun rækker med 'price' i clean_price_list inkluderes (ikke outliers)
    df_no_outliers_cleaned = df_copy[df_copy['price'].isin(clean_price_list)].copy()

    return df_no_outliers_cleaned.dropna(subset=['price'])

def get_no_outliers_df_train(df):
    # Beregner IQR-grænser
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrerer efter IQR
    df_iqr_filtered = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)].copy()

    # Beregner z-score (men fjerner ikke outliers endnu)
    beregnDataFrame(df_iqr_filtered[['price']], tærskel=3.0, fjern=False)

    # Henter de identificerede outliers fra z-score analysen
    price_outliers = hentOutliers()

    # Fjerner z-score outliers fra IQR-filtreret data
    df_no_outliers_train = df_iqr_filtered[~df_iqr_filtered['price'].isin(price_outliers)].copy()

    return df_no_outliers_train.dropna(subset=['price'])
