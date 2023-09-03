# Strøm - regne ut sin egen pris!

Dette prosjektet er en applikasjon som har som hovedmål å vise brukerne hvor mye strøm de har brukt frem til dette punktet (17.12.2022 - 07.01.2022), hvor mye de kan forvente å bruke innen utgangen av måneden (opptil 31.01.2023) og hva som er den billigste strømleverandøren for dem, gitt deres forbruksmønster. Husk at dette er et lekeeksempel basert på oppgitte data. 

## Start applikasjon
> Forutsetninger: sørg for at du har Docker installert
1) Clone the repository
2) Navigate to your project directory
3) Build the Docker image
> docker build -t app-name .
5) Run the Docker container
> docker run -p 4000:5000 app-name
6) Access the application at http://localhost:4000

## Kommentarer
1) adding + 0.0001 to variable total cost
2) it will take about five minutes for SARIMAX to run
3) had to simulate spot prices
4) SARIMAX provides poor estimation of values
5) if we did that, we would need more data to forecast forbruk for future
6) 
7) + we would need to use external APIs to access data on how forbruk changes between seasons

