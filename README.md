# TechGyant Insights - Investor Recommendation System

**Project**: AI-powered African startup recommendation system for tech investors

## Overview
TechGyant Insights uses linear regression to predict investor readiness scores for African tech startups based on **real data extracted from TechGyant articles**, founder backgrounds, and market alignment data. The system automatically scrapes and updates data from the TechGyant website to ensure predictions are based on the latest startup information.

## Features
- **Real TechGyant Data**: Automatically extracted from TechGyant website articles
- **Linear Regression Model**: Predicts startup investment readiness scores
- **Feature Engineering**: Extracts meaningful signals from startup data
- **FastAPI Endpoint**: RESTful API for real-time predictions
- **Data Visualization**: Comprehensive analysis and model performance plots
- **Model Comparison**: Linear Regression vs Decision Trees vs Random Forest
- **Automated Data Updates**: Continuous monitoring and extraction of new TechGyant articles

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

## Mission
Empower African tech ecosystem growth by connecting promising startups with global investors through data-driven insights.

## Real TechGyant Data
This project uses **pure real data extracted from TechGyant articles** featuring:
- **16+ verified African tech startups** (Flutterwave, M-KOPA, Moove, Lemfi, etc.)
- **Automated extraction** from https://techgyant.com/articles/
- **Continuous updates** when new articles are published
- **Rich features** including funding data, team info, market alignment, and more
- **No synthetic data** - only authentic startup information

### Sample Startups in Dataset:
- 🇳🇬 **Flutterwave** - FinTech (Digital Payment) - $500M funding
- 🇰🇪 **M-KOPA** - FinTech (Financial Inclusion) - $265M funding  
- 🇳🇬 **Moove** - FinTech (Supply Chain) - $600M funding
- 🇷🇼 **Ampersand** - CleanTech (Energy Access) - $21.5M funding
- 🇬🇭 **Kofa** - CleanTech (Energy Access) - $8M funding
