import numpy as np
import pandas as pd

# En brugerdefineret funktion til at beregne z-scores og identificere outliers (afvigende værdier)
# Beregningsfunktionen forventer enten en pandas DataFrame eller en liste som input

# Liste til at gemme outliers
outliers = []

def beregnListe(data, tærskel=3.0, fjern=False):

    # Rydder outliers listen før hver beregning
    outliers.clear()

    # Beregner gennemsnit og standardafvigelse for dataen
    gennemsnit = np.mean(data)
    std = np.std(data)

    # Midlertidig liste til output
    ny_data = []

    # Iterer gennem hver værdi i listen
    for punkt in data:
        z = (punkt - gennemsnit) / std
        if np.abs(z) > tærskel:
            outliers.append(punkt)
        else:
            ny_data.append(punkt)

    # Returner enten filtreret eller original liste
    return ny_data if fjern else data

def beregnDataFrame(data, tærskel=3.0, fjern=False):

    # Rydder outliers listen før hver beregning
    outliers.clear()

    # Hvis data ikke allerede er en DataFrame, konverter til det
    data = pd.DataFrame(data)

    # Hvis fjern=True, opret en kopi som vi kan ændre
    df_filtered = data.copy() if fjern else data

    # Iterer gennem hver kolonne i DataFrame
    for kolonne in data.columns:
        gennemsnit = data[kolonne].mean()
        std = data[kolonne].std()

        # Udregn z-scores
        z_scores = (data[kolonne] - gennemsnit) / std

        # Identificér outliers
        kolonne_outliers = data[kolonne][np.abs(z_scores) > tærskel]
        outliers.extend(kolonne_outliers.tolist())

        # Hvis fjern, filtrer dem ud
        if fjern:
            df_filtered = df_filtered[np.abs(z_scores) <= tærskel]

    return df_filtered

def hentOutliers():
    # Returnerer en liste med de fundne outliers.
    return outliers
