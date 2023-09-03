# Strøm - regne ut sin egen pris!

## Task description
This project is an application that aims to show users how much electricity they have used up to this point, how much they can expect to use by the end of the month and what is the cheapest provider for them, given their consumption pattern.
> Please provide a short description of your project.

## How to run
> Pre-requisites: make sure you have Docker installed
1) Clone the repository
2) Navigate to your project directory
3) Build the Docker image
> docker build -t app-name .
5) Run the Docker container
> docker run -p 4000:5000 app-name
6) Access the application at http://localhost:4000

## Comments
Comments regarding design choices, decisions, or anything at all.
1) adding + 0.0001 to variable total cost
2) it will take about five minutes for SARIMAX to run
3) had to simulate spot prices
4) SARIMAX provides poor estimation of values
5) if we did that, we would need more data to forecast forbruk for future
6) 
7) + we would need to use external APIs to access data on how forbruk changes between seasons

