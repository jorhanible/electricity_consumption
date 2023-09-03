import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
from sklearn.metrics import mean_squared_error
import random
import logging
import os
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import kaleido

logging.basicConfig(level = logging.DEBUG)
logging.info("Start logging")

# Loading data in .json format
def load_json_data(file_path):
    with open(file_path) as json_file:
        logging.info("Data loaded")
        return json.load(json_file)
    
# Pre-process consumption dataset to be used further for forecasting
def preprocess_consumption_data(consumption_data):
    consumption_df = pd.DataFrame(consumption_data)
    consumption_df['from'] = pd.to_datetime(consumption_df['from'])
    consumption_df['to'] = pd.to_datetime(consumption_df['to'])
    consumption_df['date'] = consumption_df['from'].dt.date
    consumption_df['date'] = pd.to_datetime(consumption_df['date'])
    consumption_df['startHour'] = consumption_df['from'].dt.hour
    consumption_df['endHour'] = consumption_df['to'].dt.hour
    consumption_df['weekDay'] = consumption_df['date'].dt.dayofweek
    consumption_df['dayTime'] = np.where((consumption_df["startHour"] >= 6) & (consumption_df["startHour"] <= 22), 1, 0)
    logging.info("Data pre-processed")
    return consumption_df

# Grid Search hyperparameter tuning for SARIMAX time series model
def sarimax_hyperparameter_tuning(train_data, test_data):
    p_range = [0, 1, 2]
    d_range = [0, 1]
    q_range = [0, 1, 2]
    P_range = [0, 1]
    D_range = [0, 1]
    Q_range = [0, 1]
    s = 24  # 24 hours a day

    best_rmse = float('inf')
    best_params = None

    for p, d, q, P, D, Q in itertools.product(p_range, d_range, q_range, P_range, D_range, Q_range):
        model = sm.tsa.SARIMAX(train_data['consumption'], order=(p, d, q), seasonal_order=(P, D, Q, s))
        results = model.fit()
        forecast = results.get_forecast(steps=len(test_data))
        forecasted_values = forecast.predicted_mean
        rmse = mean_squared_error(test_data['consumption'], forecasted_values, squared=False)

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = (p, d, q, P, D, Q)
        
    logging.info("Hyperparameters tuned")
    return best_params, best_rmse

# Calculate monthly costs separately: only works for December and January
def calculate_spot_monthly_cost(consumption_data, prices):
    # Split the consumption data into December and January
    december_data = consumption_data[consumption_data["from"].dt.month == 12]
    january_data = consumption_data[consumption_data["from"].dt.month == 1]

    # Calculate the cost for December and January using the provided prices
    december_cost = (december_data["consumption"] * prices[0]).sum()
    january_cost = (january_data["consumption"] * prices[1]).sum()
    overall_cost = january_cost + december_cost
    logging.info("Spot-montly cost calculated")
    return overall_cost

# Simulating spot-hourly prices for each hour in the dataset with original and forecasted data
def simulate_spot_hourly_prices(combined_df):
    return [random.uniform(0.3, 0.7) for _ in range(len(combined_df))]

# Simulating spot-monthly prices for two months: December and January
def simulate_spot_monthly_prices():
    return [random.uniform(0.3, 0.7) for _ in range(2)]

# Calculating costs for each provider for the whole period
def calculate_provider_costs(providers, consumption, spot_hourly_prices, spot_monthly_prices):
    total_costs = []
    for index, provider in providers.iterrows():
        if provider["pricingModel"] == "fixed":
            total_cost = provider["fixedPrice"] * sum(consumption["consumption"])
        elif provider["pricingModel"] == "variable":
            total_cost = provider["variablePrice"] * sum(consumption["consumption"]) + 0.001
        elif provider["pricingModel"] == "spot-hourly":
            total_cost = sum(spot_hourly_prices * consumption["consumption"])
        elif provider["pricingModel"] == "spot-monthly":
            total_cost = calculate_spot_monthly_cost(consumption, spot_monthly_prices)
    
        total_costs.append(total_cost)

    providers["total_cost"] = total_costs
    logging.info("Provider costs calculated")

# SARIMAX time series
def sarimax_forecast(train_data, test_data, best_params):
    p = best_params[0]  # Autoregressive order
    d = best_params[1]  # Non-seasonal differencing
    q = best_params[2]  # Moving average order
    P = best_params[3]  # Seasonal autoregressive order
    D = best_params[4]  # Seasonal differencing
    Q = best_params[5]  # Seasonal moving average order
    s = 24  # 24 hours per day

    model = sm.tsa.SARIMAX(train_data['consumption'], exog=train_data["dayTime"], order=(p, d, q), seasonal_order=(P, D, Q, s))
    results = model.fit()
    forecast = results.get_forecast(steps=len(test_data), exog=test_data['dayTime'].values.reshape(-1, 1))
    forecasted_values = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()
    logging.info("Predictions forecasted")
    return forecasted_values, confidence_intervals

# Creating the timestamps for the rest of the Janury, starting at the last hour available in consumption.json
def generate_january_dataset(consumption_df):
    start_date = consumption_df["to"].max() + pd.Timedelta(hours=1)  # Start from the next hour
    end_date = pd.Timestamp('2023-01-31 23:00:00+01:00')
    january_date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Create a DataFrame with the timestamp and day/night information used in SARIMAX
    january_dataset = pd.DataFrame({
        'timestamp': january_date_range,
        'dayTime': np.where((january_date_range.hour >= 6) & (january_date_range.hour <= 22), 1, 0)
    })
    logging.info("Prediction dates set up")
    return january_dataset

def calculate_cheapest_provider(providers, combined_df):
    min_cost = providers["total_cost"].min()
    provider_name = providers[providers["total_cost"] == min_cost].iloc[0][0]
    num_of_hours = len(combined_df)
    pricing_model = providers[providers["total_cost"] == min_cost].iloc[0][1]
    average_cost = ((min_cost / num_of_hours) * 30*24)

    message = f"Den billigste avtalen er hos {provider_name} med {pricing_model} pris. I gjennomsnitt kan du forvente å betale {average_cost: .2f} NOK per måned for strøm."
    logging.info("Cheapest provider found")
    return message

def original_electricity_consumption(consumption_df):
    fig = px.line(consumption_df, x='from', y='consumption', labels={'from': 'Dato', 'consumption': 'Forbruk'},
                  title='Strøm forbruk i siste 500 timer', line_shape='linear')

    image_path = '/app/static/electricity_consumption_plot.png'
    pio.write_image(fig, image_path, format="png")

    return image_path

import plotly.graph_objects as go

def original_vs_predicted_electricity_consumption(combined_df):
    combined_df["from"] = pd.to_datetime(combined_df["from"])
    combined_df["date"] = combined_df["from"].dt.date
    
    original_trace = go.Scatter(x=combined_df[combined_df['from'] <= "2023-01-07T15:00:00.000+01:00"]['from'],
                                y=combined_df[combined_df['from'] <= "2023-01-07T15:00:00.000+01:00"]['consumption'],
                                mode='lines',
                                name='Registrert',
                                line=dict(color='blue'))
    
    predicted_trace = go.Scatter(x=combined_df[combined_df['from'] > "2023-01-07T15:00:00.000+01:00"]['from'],
                                 y=combined_df[combined_df['from'] > "2023-01-07T15:00:00.000+01:00"]['consumption'],
                                 mode='lines',
                                 name='Forventet',
                                 line=dict(color='red'))
    
    fig = go.Figure(data=[original_trace, predicted_trace])
    
    fig.update_layout(xaxis_title='Dato', yaxis_title='Forbruk',
                      title='Strøm forbruk i desember og januar')
    
    image_path = '/app/static/comparison_consumption_plot.png'
    fig.write_image(image_path, format="png")
    
    return image_path

def main():
    # Load data
    consumption_data = load_json_data('/app/data/consumption.json')
    providers_data = load_json_data('/app/data/providers.json')

    # Preprocess consumption data
    consumption_df = preprocess_consumption_data(consumption_data)
    providers_data = pd.DataFrame(providers_data)

    # Split data into train and test sets
    train_size = int(0.8 * len(consumption_df))
    train = consumption_df[:train_size]
    test = consumption_df[train_size:]

    # SARIMAX hyperparameter tuning
    best_params, best_rmse = sarimax_hyperparameter_tuning(train, test)

    # SARIMA forecasting for January
    january_dataset = generate_january_dataset(consumption_df)
    forecasted_values_new, confidence_intervals_new = sarimax_forecast(train, january_dataset, best_params)
    forecasted_values_new = forecasted_values_new.reset_index()
    january_dataset["consumption"] = forecasted_values_new["predicted_mean"]
    january_dataset.rename(columns={'timestamp':'from'}, inplace=True)

    combined_df = pd.concat([consumption_df, january_dataset], axis=0)

    # Simulate spot-hourly and spot-monthly prices
    spot_hourly_prices = simulate_spot_hourly_prices(combined_df)
    spot_monthly_prices = simulate_spot_monthly_prices()

    # Calculate provider costs
    calculate_provider_costs(providers_data, combined_df, spot_hourly_prices, spot_monthly_prices)
    result = calculate_cheapest_provider(providers_data, combined_df)
    image_path = original_electricity_consumption(consumption_df)
    image_path_vs = original_vs_predicted_electricity_consumption(combined_df)
    
    return result

if __name__ == "__main__":
    main()