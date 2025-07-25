# TechGyant Insights - Render Deployment Guide

## 🚀 Deploy to Render

### Step 1: Prepare Repository
1. Push your code to GitHub repository
2. Make sure all these files are in the root directory:
   - `requirements.txt`
   - `Procfile`
   - `render.yaml`
   - `runtime.txt`

### Step 2: Deploy on Render
1. Go to https://render.com and sign up/login
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `techgyant-insights-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free` (for testing)

### Step 3: Environment Variables (Optional)
- `PYTHON_VERSION`: `3.11.0`

### Step 4: Deploy
- Click "Create Web Service"
- Wait for deployment (5-10 minutes)
- Your API will be available at: `https://your-service-name.onrender.com`

## 📝 API Endpoints for Testing

### Base URL: `https://your-service-name.onrender.com`

### 🏠 Homepage
- `GET /` - API homepage

### 🔍 Investment Recommendations
- `GET /recommendations/country/Nigeria` - Nigerian startups
- `GET /recommendations/sector/FinTech` - FinTech startups
- `GET /recommendations/country/Kenya/sector/AgriTech` - Kenyan AgriTech
- `GET /recommendations/top-countries` - Top investment countries
- `GET /recommendations/africa-overview` - Africa-wide overview

### 💡 Prediction (if model is loaded)
- `POST /predict` - Single startup prediction
- `POST /predict/batch` - Multiple predictions

### ❤️ Health Check
- `GET /health` - API status

## 📊 Sample API Responses

### Country Recommendations
```json
{
  "country": "Nigeria",
  "total_startups": 45,
  "recommended_startups": 23,
  "average_readiness_score": 67.5,
  "top_sectors": ["FinTech", "E-commerce", "HealthTech"],
  "startups": [...]
}
```

### Africa Overview
```json
{
  "total_startups": 515,
  "recommended_startups": 203,
  "countries_covered": 10,
  "sectors_covered": 10,
  "total_funding": 15000000000,
  "average_readiness_score": 64.2,
  "top_countries": [...],
  "startups": [...]
}
```

## 🧪 Testing with Postman

1. Import the API into Postman using: `https://your-api-url.onrender.com/openapi.json`
2. Test the endpoints listed above
3. Use query parameters:
   - `?min_score=70` for higher quality startups
   - `?limit=10` for top countries endpoint

## 🔧 Troubleshooting

- If deployment fails, check the build logs
- Ensure all dependencies are in `requirements.txt`
- The API works without ML models (recommendation endpoints only)
- Check `/health` endpoint to verify API status
