# BI-Exam

# Gruppe

- Allan
- Marlene
- Marwa
- Nicklas

# Problemformulering:

## Annotation
Vi adresserer udfordringen med uforudsigelige flybilletpriser, som skaber usikkerhed for både rejsende og flyselskaber. 
Dette er et vigtigt forskningsmål, da en dybere forståelse af, hvad der driver prisvariationer, kan hjælpe med at optimere prisstrategier og forbedre kundetilfredsheden. 
Vores løsning vil anvende business intelligence og maskinlæring til at analysere historiske data og rejseparametre for at finde mønstre i prisvariationerne.
Denne løsning vil gavne både rejsende, som får bedre prisoverblik, og flyselskaber, der kan tilpasse deres prisstrategier mere effektivt.

## Introduktion
Flybilletpriser varierer betydeligt og kan være svære at gennemskue for både rejsende og flyselskaber. 
Priserne påvirkes af flere faktorer som afgangs og ankomstby, flyselskab, bookingtidspunkt, sæson og antal stop. Dette skaber usikkerhed i planlægningen og udfordringer i prisstrategien. 
I dette projekt vil vi analysere historiske data for at afdække, hvordan disse faktorer påvirker prisdannelsen, med det formål at skabe bedre indsigt for både forbrugere og udbydere.

## Kontekst
Flypriser påvirkes af mange forskellige faktorer, hvilket gør det svært at gennemskue, hvordan priserne fastsættes. 
For at kunne forstå, hvordan priserne varierer, er det nødvendigt at analysere historiske data og identificere faktorer som afgangs og ankomstby, flyselskab, bookingtidspunkt, sæson og antal stop. 
Ved at finde mønstre i disse data kan vi få et bedre billede af, hvordan priserne ændrer sig under forskellige forhold.

## Mål
Projektets formål er at undersøge, hvordan flypriser varierer baseret på historiske data og relevante faktorer. 
Ved hjælp af business intelligence og maskinlæring vil vi analysere disse data for at identificere mønstre i prisvariationer. 
Resultaterne vil kunne bruges af rejsende til at træffe bedre beslutninger og af flyselskaber til at optimere deres prisstrategier.

## Forskningsspørgsmål

1. Hvordan varierer priserne afhængigt af antallet af dage, der er tilbage før afrejse?
2. Hvordan påvirker antallet af stop prisen?
3. Hvordan varierer priserne afhængigt af rejsemåneden?
4. Hvordan varierer priserne mellem standard og premium-versioner af samme flyselskab?
5. Hvordan kan flyselskaber kategoriseres i grupper baseret på prisvariationer over rejsemåneden og forskelle mellem standard og premium-versioner?

## Hypoteser
1. Prisen stiger, jo færre dage der er tilbage før afrejse.
2. Flyvninger med flere stop er generelt billigere end direkte flyvninger.
3. Priserne varierer med rejsemåneden, hvor højsæson (f.eks. juli) medfører højere priser end lavsæson.
4. Premium-versioner som Business og Premium Economy af samme flyselskab har en højere pris end standard-versioner.
5. Flyselskaber kan kategoriseres i grupper baseret på prisvariationer over rejsemåneden, hvor priserne generelt stiger i højsæsonen, samt forskelle mellem standard og premium-versioner, hvor premium-versioner forventes at have markant højere priser end standardklassen.

## User Testing
Vi har gennemført en brugertest med en bruger, som gav os konstruktiv feedback til forbedring af vores applikation. For eksempel kommenterede brugeren på, hvordan vi beskrev vores modeller, og gav også gode forslag til, hvordan vi kunne omplacere elementer i applikationen, så den bliver mere brugervenlig.
   
## Datasæt:

https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

https://www.kaggle.com/datasets/anshuman0427/flight-price-dataset

## Implementeringsvejledning

Start applikationen med denne kommando:

```bash
streamlit run app.py
```
