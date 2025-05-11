# BI-Exam

# Gruppe

- Allan
- Marlene
- Marwa
- Niklas

# Problemformulering:

## Annotation
Vi adresserer udfordringen med uforudsigelige flybilletpriser, som skaber usikkerhed for både rejsende og flyselskaber. 
Dette er et vigtigt forskningsmål, da en dybere forståelse af, hvad der driver prisvariationer, kan hjælpe med at optimere prisstrategier og forbedre kundetilfredsheden. 
Vores løsning vil anvende business intelligence og maskinlæring til at analysere historiske data og rejseparametre for at finde mønstre i prisvariationerne.
Denne løsning vil gavne både rejsende, som får bedre prisoverblik, og flyselskaber, der kan tilpasse deres prisstrategier mere effektivt.

## Introduktion
Flybilletpriser svinger meget og kan være svære at forudse, hvilket kan skabe frustration både for rejsende og flyselskaber.
Priserne ændrer sig afhængigt af flere faktorer, såsom hvornår man booker, hvilken rute man tager, om der er stop undervejs, hvilken tid på året det er, og hvilken type billet eller flyselskab man vælger. 
Dette gør det svært for både rejsende at finde de billigste billetter og for flyselskaber at sætte de rigtige priser. 
I dette projekt vil vi bruge business intelligence og maskinlæring til at analysere historiske data og identificere, hvordan faktorer som afgangs og ankomstby, flyselskab, bookingtidspunkt, sæson og antal stop påvirker prisvariationer.
På den måde kan både rejsende få bedre overblik, og flyselskaberne kan tilpasse deres prisstrategier mere effektivt.

## Kontekst
Flypriser påvirkes af mange forskellige faktorer, hvilket gør det svært at gennemskue, hvordan priserne fastsættes. 
For at kunne forstå, hvordan priserne varierer, er det nødvendigt at analysere historiske data og identificere faktorer som afgangs og ankomstby, flyselskab, bookingtidspunkt, sæson og antal stop. 
Ved at finde mønstre i disse data kan vi få et bedre billede af, hvordan priserne ændrer sig under forskellige forhold.

## Mål
Projektets formål er at undersøge, hvordan flypriser varierer baseret på historiske data og relevante faktorer. 
Dette vil gavne både rejsende og flyselskaber i deres planlægning af rejser og prisstrategier.

## Forskningsspørgsmål

1. Hvordan varierer flypriserne afhængigt af antallet af dage, der er tilbage før afrejse?
2. Hvordan påvirker antallet af stop flypriserne?
3. Hvordan varierer flypriserne afhængigt af rejsemåneden?
4. Hvordan varierer billetpriser mellem standard og premium-versioner af samme flyselskab?

## Hypoteser
1. Jo færre dage der er tilbage før afrejse, desto højere vil flypriserne være.
2. Flyvninger med flere stop er generelt billigere end direkte flyvninger.
3. Flypriserne varierer systematisk med rejsemåneden, hvor højsæson (f.eks. juli og december) medfører højere priser end lavsæson.
4. Premium-versioner (f.eks. Business eller Premium Economy) af samme flyselskab koster væsentligt mere end standard.

## Datasæt:

https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

https://www.kaggle.com/datasets/anshuman0427/flight-price-dataset

## Implementeringsvejledning

Start applikationen med denne kommando:

```bash
streamlit run main.py
```
