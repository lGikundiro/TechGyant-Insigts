# TechGyant Insights - Investor Recommendation System

**Project**: AI-powered African startup recommendation system for tech investors

## Overview
TechGyant Insights uses linear regression to predict investor readiness scores for African tech startups based on **real data extracted from TechGyant articles**, founder backgrounds, and market alignment data. The system automatically scrapes and updates data from the TechGyant website to ensure predictions are based on the latest startup information.

## Mission
Amplifying African tech ecosystem growth by contextual storytelling and connecting promising startups with global investors through data-driven insights.

## Problem Statement
African tech startups often lack global attention with 3% global media coverage. This affects the growth of potential but small startups. TechGyants will be fostered to be the voice of every startup, particularly small but promising ones, and be a go-to site for every investor to learn about startups through data-driven insights.

## Features
- **Real TechGyant Data**: Automatically extracted from TechGyant website articles
- **Linear Regression Model**: Predicts startup investment readiness scores
- **Feature Engineering**: Extracts meaningful signals from startup data
- **FastAPI Endpoint**: RESTful API for real-time predictions
- **Data Visualization**: Comprehensive analysis and model performance plots
- **Model Comparison**: Linear Regression vs Decision Trees vs Random Forest
- **Automated Data Updates**: Continuous monitoring and extraction of new TechGyant articles

## Real TechGyant Data
This project uses **pure real data extracted from TechGyant articles** featuring:
- **15 verified African tech startups** (Flutterwave, M-KOPA, Moove, Lemfi, Academic Bridge, Strettch, etc.)
- **Automated extraction** from https://techgyant.com/articles/
- **Continuous dataset updates** when new articles are published
- **Rich features** including funding data, team info, market alignment, and more
- **No synthetic data** - only authentic startup information

### Sample Startups in Dataset:
- 🇳🇬 **Flutterwave** - FinTech (Digital Payment) - $500M funding
- 🇰🇪 **M-KOPA** - FinTech (Financial Inclusion) - $265M funding  
- 🇳🇬 **Moove** - FinTech (Supply Chain) - $600M funding
- 🇷🇼 **Ampersand** - CleanTech (Energy Access) - $21.5M funding
- 🇬🇭 **Kofa** - CleanTech (Energy Access) - $8M funding

### Limitation:
TechGyant Insights predictions do not offer a percentage investment-worthy score of a startup in its specific sector or country due to data limitation at the moment. However, it's the next near future enhancement.

### API Documentation:
https://techgyant-insigts.onrender.com/docs

## Project Structure
```
TechGyantInsights/
├── data/
│   ├── raw/                    # Real TechGyant startup data
│   │   └── techgyant_real_startups.csv     # Pure real dataset (16+ startups)
│   ├── processed/              # Cleaned and engineered features
│   ├── models/                 # Saved model files
│   └── monitoring/             # Data extraction configs and logs
├── src/
│   ├── techgyant_data_extractor.py  # Automated web scraping
│   ├── real_data_extractor.py       # Data extraction utilities
│   ├── data_preprocessing.py         # Data cleaning and feature engineering
│   ├── model_training.py             # Model training and evaluation
│   ├── model_comparison.py           # Compare different algorithms
│   └── prediction_service.py         # Prediction functions
├── api/
│   ├── main.py                # FastAPI application
│   ├── models.py              # Pydantic models
│   └── requirements.txt       # Dependencies
├── logs/                      # Extraction and system logs
├── notebooks/
│   └── exploratory_analysis.ipynb  # Data exploration
└── visualizations/            # Generated plots and charts
```
  
