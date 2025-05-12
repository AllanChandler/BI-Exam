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

    # Iterer gennem hver værdi i listen
    for punkt in data:
        z = (punkt - gennemsnit) / std
        if np.abs(z) > tærskel:
            outliers.append(punkt)
            if fjern:
                data = [x for x in data if x != punkt]

    return data

def beregnDataFrame(data, tærskel=3.0, fjern=False):

    # Rydder outliers listen før hver beregning
    outliers.clear()

    # Hvis data ikke allerede er en DataFrame, konverter til det
    data = pd.DataFrame(data)

    # Iterer gennem hver kolonne i DataFrame
    for kolonne in data.columns:
        gennemsnit = data[kolonne].mean()
        std = data[kolonne].std()

        # Konverter kolonnens data til en liste for at iterere gennem værdierne
        kolonneData = data[kolonne].tolist()

        # Iterer gennem hver værdi i kolonnen
        for punkt in kolonneData:
            # Sørg for, at punktet er en float værdi
            if not isinstance(punkt, float):
                punkt = float(punkt)
            z = (punkt - gennemsnit) / std

            # Hvis z-scoren er større end tærsklen, er det en outlier
            if np.abs(z) > tærskel:
                outliers.append(punkt)
                if fjern:
                    data = data[data[kolonne] != punkt]

    return data

def hentOutliers():

    # Returnerer en liste med de fundne outliers.

    return outliers
