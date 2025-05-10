# BI-Exam

# Gruppe

- Allan
- Marlene
- Marwa
- Niklas

# Problemformulering:

## Annotation
Vi adresserer udfordringen med uforudsigelige flybilletpriser, som skaber usikkerhed for både rejsende og flyselskaber. 
Dette er et vigtigt forskningsmål, da præcise forudsigelser kan hjælpe med at optimere prisstrategier og forbedre kundetilfredsheden. 
Vores løsning vil udvikle en model baseret på business intelligence og maskinlæring, der forudsiger flybilletpriser ud fra historiske data og rejseparametre. 
Denne løsning vil gavne både rejsende, som får bedre prisoverblik, og flyselskaber, der kan justere deres prisstrategier

## Introduktion
Flybilletpriser udgør en stor udfordring for både passagerer og flyselskaber, da priserne er påvirket af en række faktorer, der konstant ændrer sig. For at kunne forudsige disse priser mere præcist er det nødvendigt at forstå, hvordan forskellige faktorer som ruten, tidspunktet for booking, flyselskabet og afgangs-/ankomstbyerne bidrager til prisvariationerne. I dette projekt vil vi bruge business intelligence (BI) og maskinlæring (ML) til at udvikle modeller, der kan forudsige flybilletpriser baseret på de historiske data om flyrejser og de tilknyttede variable, som findes i de valgte datasæt.

## Kontekst
Flybilletpriser er blevet en udfordring for både passagerer og flyselskaber på grund af de komplekse og svingende faktorer, der påvirker priserne. For at kunne forudsige disse priser præcist er det vigtigt at forstå, hvilke faktorer der driver prisvariationerne. Dette projekt sigter mod at udvikle modeller, der kan forudsige billetpriser ved at analysere historiske flydata og faktorer som ruter, flyselskaber, afgangs-/ankomstbyer og bookingtidspunkt.

## Mål
Formålet med dette projekt er at anvende maskinlæring og business intelligence-teknikker til at forudsige flybilletpriser. Vi vil analysere historiske billetpriser sammen med faktorer som flyselskab, afgangs- og ankomstbyer samt datoer for at skabe en model, der præcist forudsiger prisen på flybilletter for forskellige ruter og tidspunkter.

## Forskningsspørgsmål

Dem som jeg tænker vi skal have:

datasæt 1:
1. Hvordan varierer flypriserne afhængigt af antallet af dage, der er tilbage før afrejse?
2. Hvordan påvirker antallet af stop flypriserne?

datasæt 2:
1. Hvordan varierer flypriserne afhængigt af rejsemåneden?
2. Hvordan varierer billetpriser mellem standard og premium-versioner af samme flyselskab?

dem vi havde i forvejen:
1. Hvilke faktorer (som rute, flyselskab, tidspunkt) påvirker flybilletpriser mest?
2. Hvordan påvirker flyselskab, rute og tidspunkt flybilletpriserne?
3. Hvordan kan maskinlæringsmodeller forudsige flybilletpriser præcist ud fra de tilgængelige faktorer?
4. Hvilke mønstre kan vi identificere i prisudviklingen baseret på rute og tidspunkt?

## Hypoteser
1. Jo færre dage der er tilbage før afrejse, desto højere vil flypriserne være.
2. Flyvninger med flere stop er generelt billigere end direkte flyvninger.
3. Flypriserne varierer systematisk med rejsemåneden, hvor højsæson (f.eks. juli og december) medfører højere priser end lavsæson.
4. Premium-versioner (f.eks. Business eller Premium Economy) af samme flyselskab koster væsentligt mere end standard.

## Udfordringer
Den største udfordring er at identificere de faktorer, der bidrager til prisfluktuationer, herunder sæsonbestemte variationer, efterspørgsel og variationer baseret på flyselskab og rute. En anden udfordring er at integrere de to datasæt og bygge en effektiv prædiktiv model, der tager højde for disse faktorer.

## Datasæt:

https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

https://www.kaggle.com/datasets/anshuman0427/flight-price-dataset

## Implementeringsvejledning

Start applikationen med denne kommando:

```bash
streamlit run main.py
```
