# Strøm - regne ut sin egen pris!

Dette prosjektet er en applikasjon som har som hovedmål å vise brukerne hvor mye strøm de har brukt frem til dette punktet (17.12.2022 - 07.01.2022), hvor mye de kan forvente å bruke innen utgangen av måneden (opptil 31.01.2023) og hva som er den billigste strømleverandøren for dem, gitt deres forbruksmønster. Husk at dette er et lekeeksempel basert på oppgitte data. 

1) Forbruks data brukes til å kalkulere forventet forbruk for resten av januar 2023 med SARIMAX (Time Series for seasonal data)
2) Spot-monthly og spot-hourly priser simuleres, for det er ingen historie tilgjengelig for hvordan de endrer seg over tiden
3) Brukeren får en anbefalt strømleverandør som har den billigste avtalen over hele perioden (17.12.2022-31.01.2023)
4) Brukeren får en estimert månedelig kostnad hos den anbefalte leverandøren

## Start applikasjonen
> Forutsetninger: sørg for at du har Docker installert
1) Clone the repository
2) Navigate to your project directory
3) Build the Docker image
> docker build -t app-name .
5) Run the Docker container
> docker run -p 4000:5000 app-name
6) Access the application at http://localhost:4000

## Kommentarer
*  Fastpris og variabel pris regnes ut på den samme måten. Discord admin ga beskjed at fixedPricePeriod og variablePricePeriod gir et antall måneder når pris varer, så det er samme kostnad for både fastpris og variabel pris. For å unngå tilfellen når den er billigste avtalen og en feil oppstår fordi koden får to verdier i stedet for en, legger jeg til 0.001 til kalkulert kostnad over hele perioden for variabel pris.
* Når applikasjonen starter, tar det circa 5 minutter for SARIMAX å forutsi strøm forbruk, så vennligst vent litt og ikke oppdater siden.
* Som jeg sa tidligere, jeg måtte simulere spot-monthly og spot-hourly priser, derfor bruker jeg uniform distribution for å gjøre det. Hvis det var en produkt for kunder, bør vi be dem om å laste opp sin egen data for siste år, for eksempel. Prisene kan da bli tatt fra ekstertnt API.
* SARIMAX gir ganske dårlig estimering av forventet forbruk fordi det er for lite data tilgjengelig og modellen tar in hyperparameters fra Grid Search som er en begrensende metode.

