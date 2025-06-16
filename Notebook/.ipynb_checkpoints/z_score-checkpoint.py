import numpy as np
import pandas as pd

# Denne fil er udviklet for at sikre ensartet og pålidelig håndtering af datarensning 
# og outlier-identifikation, som er afgørende for effektiv dataanalyse og maskinlæring.

# Funktioner:
# - beregnListe: Arbejder med lister af numeriske værdier, beregner z-scores, identificerer og kan filtrere outliers.
# - beregnDataFrame: Arbejder med pandas DataFrames, beregner z-scores kolonnevis, identificerer og kan filtrere outliers.
# - hentOutliers: Returnerer en samlet liste over de identificerede outliers fra den seneste analyse.

# Liste til at gemme outliers (nulstilles ved hver beregning)
outliers = []

def beregnListe(data, tærskel=3.0, fjern=False):

    # Rydder outliers listen før hver beregning
    outliers.clear()

    # Beregner gennemsnit og standardafvigelse for dataen
    gennemsnit = np.mean(data)
    std = np.std(data)

    # Midlertidig liste til output
    ny_data = []

    # Itererer gennem hver værdi i listen
    for punkt in data:
        z = (punkt - gennemsnit) / std
        if np.abs(z) > tærskel:
            outliers.append(punkt)
        else:
            ny_data.append(punkt)

    # Returnerer enten filtreret eller original liste
    return ny_data if fjern else data

def beregnDataFrame(data, tærskel=3.0, fjern=False):

    # Rydder outliers listen før hver beregning
    outliers.clear()

    # Hvis data ikke allerede er en DataFrame, konverter til det
    data = pd.DataFrame(data)

    # Hvis fjern=True, oprettes en kopi som kan ændres
    df_filtered = data.copy() if fjern else data

    # Itererer gennem hver kolonne i DataFramen
    for kolonne in data.columns:
        gennemsnit = data[kolonne].mean()
        std = data[kolonne].std()

        # Udregner z-scores
        z_scores = (data[kolonne] - gennemsnit) / std

        # Identificerer outliers
        kolonne_outliers = data[kolonne][np.abs(z_scores) > tærskel]
        outliers.extend(kolonne_outliers.tolist())

        # Hvis fjern=True, fjernes outliers fra datasættet ved at filtrere dem væk
        if fjern:
            df_filtered = df_filtered[np.abs(z_scores) <= tærskel]

    return df_filtered

def hentOutliers():
    # Returnerer en liste med de fundne outliers.
    return outliers
