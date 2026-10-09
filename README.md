**all file is inside this folder**
# RailTrack

## AI-Powered Dynamic ETA Prediction for Coaching Trains

RailTrack is a real-time AI-powered railway ETA prediction system designed to predict when a train is expected to reach each upcoming station and its final destination.

Instead of only reporting a train's current location or current delay, RailTrack focuses on predicting what is likely to happen next. It combines live train information with historical delay patterns, route and station information, and available operational factors to dynamically forecast train delays and arrival times.

## Problem Statement

**Smart India Hackathon 2026 — Problem Statement 26028**

**Dynamic Forecast of Expected Time of Arrival (ETA) for Coaching Trains**

Conventional ETA estimation can become outdated when train conditions change during a journey. Delays may propagate because of speed restrictions, congestion, unscheduled stoppages, preceding train delays, weather and other operational conditions.

RailTrack addresses this challenge through machine-learning-based dynamic forecasting and continuous re-forecasting.

## Key Features

- Real-time train status and journey information
- Current delay and train movement information
- Historical delay and running-time patterns
- LSTM-based delay prediction
- Station-wise ETA prediction
- Final destination ETA prediction
- Continuous re-forecasting when new live information becomes available
- Interactive web dashboard
- API-driven backend for integration
- Fallback handling for live-data availability
- Designed for scalable expansion across routes and railway zones

## How It Works

```text
Live Train Data
       +
Historical Train Data
       +
Route / Station Information
       +
Weather / Operational Factors
       |
       v
Data Preprocessing
       |
       v
Feature Engineering
       |
       v
PyTorch LSTM Sequence Regressor
       |
       v
Dynamic Delay Prediction
       |
       v
ETA Calculation
       |
       +----------------------+
       |                      |
       v                      v
Station-wise ETA       Destination ETA
       |
       v
Live Dashboard / APIs
