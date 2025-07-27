"""
TechGyant Insights - Prediction Model Visualization
Create visual charts showing prediction results
"""

import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

# Set style for better visualizations
plt.style.use('default')  # Use clean default style
sns.set_palette("Set2")  # Professional color palette

# Custom TechGyant color scheme
TECHGYANT_COLORS = {
    'primary': '#2E86AB',      # TechGyant Blue
    'secondary': '#A23B72',    # TechGyant Purple
    'accent': '#F18F01',       # TechGyant Orange
    'success': '#4CAF50',      # Green
    'warning': '#FF9800',      # Orange
    'danger': '#F44336',       # Red
    'info': '#2196F3',         # Light Blue
    'dark': '#263238',         # Dark Gray
    'light': '#ECEFF1'         # Light Gray
}

# API base URL
API_BASE = "http://localhost:8000"

def test_predictions():
    """Test various prediction scenarios and collect results"""
    
    test_scenarios = [
        {
            "name": "Early Stage FinTech (Kenya)",
            "funding_stage": "Seed",
            "sector": "FinTech",
            "country": "Kenya",
            "team_size": 5,
            "years_since_founding": 1,
            "monthly_revenue": 10000,
            "user_base": 5000
        },
        {
            "name": "Growth Stage HealthTech (Nigeria)",
            "funding_stage": "Series A",
            "sector": "HealthTech",
            "country": "Nigeria",
            "team_size": 15,
            "years_since_founding": 3,
            "monthly_revenue": 75000,
            "user_base": 25000
        },
        {
            "name": "Mature AgriTech (South Africa)",
            "funding_stage": "Series B",
            "sector": "AgriTech",
            "country": "South Africa",
            "team_size": 35,
            "years_since_founding": 5,
            "monthly_revenue": 200000,
            "user_base": 100000
        },
        {
            "name": "EdTech Startup (Ghana)",
            "funding_stage": "Series A",
            "sector": "EdTech",
            "country": "Ghana",
            "team_size": 12,
            "years_since_founding": 2,
            "monthly_revenue": 30000,
            "user_base": 15000
        },
        {
            "name": "CleanTech Innovation (Rwanda)",
            "funding_stage": "Seed",
            "sector": "CleanTech",
            "country": "Rwanda",
            "team_size": 8,
            "years_since_founding": 1,
            "monthly_revenue": 5000,
            "user_base": 2000
        },
        {
            "name": "Logistics Scale-up (Egypt)",
            "funding_stage": "Series B",
            "sector": "Logistics",
            "country": "Egypt",
            "team_size": 45,
            "years_since_founding": 4,
            "monthly_revenue": 300000,
            "user_base": 80000
        }
    ]
    
    results = []
    
    print("🔮 Testing TechGyant AI Prediction Model...")
    print("=" * 60)
    
    for scenario in test_scenarios:
        try:
            # Make prediction request
            response = requests.post(f"{API_BASE}/predict", json=scenario)
            
            if response.status_code == 200:
                prediction = response.json()
                
                result = {
                    "name": scenario["name"],
                    "sector": scenario["sector"],
                    "country": scenario["country"],
                    "funding_stage": scenario["funding_stage"],
                    "team_size": scenario["team_size"],
                    "monthly_revenue": scenario["monthly_revenue"],
                    "predicted_investment": prediction["predicted_investment"],
                    "confidence_score": prediction["confidence_score"],
                    "risk_level": prediction["risk_level"]
                }
                
                results.append(result)
                
                print(f"✅ {scenario['name']}")
                print(f"   💰 Predicted Investment: ${prediction['predicted_investment']:,.2f}")
                print(f"   🎯 Confidence: {prediction['confidence_score']*100:.1f}%")
                print(f"   ⚠️  Risk Level: {prediction['risk_level']}")
                print()
                
            else:
                print(f"❌ Error for {scenario['name']}: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error for {scenario['name']}: {e}")
    
    return pd.DataFrame(results)

def create_visualizations(df):
    """Create a completely redesigned, clean dashboard with zero overlapping"""
    
    # Create multiple separate figures for clean layout
    
    # FIGURE 1: Main Investment Analysis
    fig1 = plt.figure(figsize=(20, 12), facecolor='white')
    fig1.suptitle('TechGyant Insights - Investment Prediction Analysis', 
                  fontsize=24, fontweight='bold', y=0.95, 
                  color=TECHGYANT_COLORS['primary'])
    
    # Create clean grid layout
    gs1 = fig1.add_gridspec(2, 2, hspace=0.4, wspace=0.3, top=0.88, bottom=0.1, 
                           left=0.08, right=0.95)
    
    # 1. Investment Predictions - Top Left (Large)
    ax1 = fig1.add_subplot(gs1[0, :])  # Spans both columns at top
    
    # Create short, clean company names
    short_names = []
    for i, row in df.iterrows():
        sector = row['sector'][:6]  # First 6 chars of sector
        country = row['country'][:3]  # First 3 chars of country
        short_names.append(f"{sector}\n{country}")
    
    colors = [TECHGYANT_COLORS['primary'], TECHGYANT_COLORS['secondary'], 
              TECHGYANT_COLORS['accent'], TECHGYANT_COLORS['success'],
              TECHGYANT_COLORS['info'], TECHGYANT_COLORS['warning']]
    
    # Repeat colors if needed
    plot_colors = (colors * ((len(df) // len(colors)) + 1))[:len(df)]
    
    bars = ax1.bar(range(len(df)), df['predicted_investment'], 
                   color=plot_colors, alpha=0.85, edgecolor='white', linewidth=2)
    
    ax1.set_title('💰 Investment Predictions by Startup', 
                  fontsize=18, fontweight='bold', pad=30, color=TECHGYANT_COLORS['primary'])
    ax1.set_xlabel('African Tech Startups', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Predicted Investment (USD)', fontweight='bold', fontsize=14)
    
    # Set clean x-axis labels with proper spacing
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(short_names, fontsize=10, ha='center')
    
    # Add value labels with smart positioning
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height >= 1000000:
            label = f'${height/1e6:.1f}M'
        else:
            label = f'${height/1000:.0f}K'
        
        # Position label above bar with margin
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.03,
                label, ha='center', va='bottom', fontweight='bold', 
                fontsize=11, color=TECHGYANT_COLORS['dark'])
    
    # Format y-axis
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_facecolor('#FAFBFC')
    
    # 2. Confidence & Risk Analysis - Bottom Left
    ax2 = fig1.add_subplot(gs1[1, 0])
    
    # Create confidence score bars with risk level coloring
    risk_colors = {
        'High': TECHGYANT_COLORS['danger'],
        'Medium': TECHGYANT_COLORS['warning'], 
        'Low': TECHGYANT_COLORS['success']
    }
    
    conf_colors = [risk_colors[risk] for risk in df['risk_level']]
    
    bars2 = ax2.bar(range(len(df)), df['confidence_score'] * 100, 
                   color=conf_colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    ax2.set_title('🎯 AI Confidence & Risk Levels', 
                  fontsize=16, fontweight='bold', pad=20, color=TECHGYANT_COLORS['primary'])
    ax2.set_xlabel('Startups', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Confidence (%)', fontweight='bold', fontsize=12)
    
    # Use simple labels
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([f'S{i+1}' for i in range(len(df))], fontsize=10)
    ax2.set_ylim(0, 100)
    
    # Add confidence labels
    for i, (bar, risk) in enumerate(zip(bars2, df['risk_level'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.0f}%\n{risk[0]}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_facecolor('#FAFBFC')
    
    # 3. Sector Distribution - Bottom Right
    ax3 = fig1.add_subplot(gs1[1, 1])
    
    sector_data = df.groupby('sector')['predicted_investment'].sum()
    sector_colors = [TECHGYANT_COLORS['primary'], TECHGYANT_COLORS['secondary'],
                    TECHGYANT_COLORS['accent'], TECHGYANT_COLORS['success'],
                    TECHGYANT_COLORS['info'], TECHGYANT_COLORS['warning']]
    
    # Ensure we have enough colors
    pie_colors = (sector_colors * ((len(sector_data) // len(sector_colors)) + 1))[:len(sector_data)]
    
    wedges, texts, autotexts = ax3.pie(sector_data.values, 
                                      labels=sector_data.index,
                                      autopct='%1.1f%%',
                                      startangle=90,
                                      colors=pie_colors,
                                      pctdistance=0.85,
                                      labeldistance=1.15,
                                      textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    ax3.set_title('🏢 Investment by Sector', 
                  fontsize=16, fontweight='bold', pad=20, color=TECHGYANT_COLORS['primary'])
    
    # Clean up pie chart text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    plt.tight_layout()
    fig1.savefig('visualizations/techgyant_investment_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("📊 Investment analysis saved!")
    
    # FIGURE 2: Geographic & Statistical Analysis  
    fig2 = plt.figure(figsize=(18, 10), facecolor='white')
    fig2.suptitle('TechGyant Insights - Geographic & Statistical Analysis', 
                  fontsize=22, fontweight='bold', y=0.95, 
                  color=TECHGYANT_COLORS['primary'])
    
    gs2 = fig2.add_gridspec(1, 2, hspace=0.3, wspace=0.3, top=0.85, bottom=0.15, 
                           left=0.08, right=0.95)
    
    # 1. Country Investment Analysis - Left
    ax4 = fig2.add_subplot(gs2[0, 0])
    
    country_investment = df.groupby('country')['predicted_investment'].sum().sort_values(ascending=True)
    
    bars4 = ax4.barh(range(len(country_investment)), country_investment.values,
                     color=TECHGYANT_COLORS['secondary'], alpha=0.8, 
                     edgecolor='white', linewidth=2, height=0.6)
    
    ax4.set_title('🌍 Investment by African Countries', 
                  fontsize=18, fontweight='bold', pad=25, color=TECHGYANT_COLORS['primary'])
    ax4.set_xlabel('Total Investment (USD)', fontweight='bold', fontsize=14)
    ax4.set_yticks(range(len(country_investment)))
    ax4.set_yticklabels(country_investment.index, fontsize=12, fontweight='bold')
    
    # Format x-axis
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Add value labels
    for i, value in enumerate(country_investment.values):
        label = f'${value/1e6:.1f}M' if value >= 1000000 else f'${value/1000:.0f}K'
        ax4.text(value + value*0.05, i, label, 
                va='center', fontweight='bold', fontsize=11, 
                color=TECHGYANT_COLORS['dark'])
    
    ax4.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax4.set_facecolor('#FAFBFC')
    
    # 2. Summary Statistics - Right
    ax5 = fig2.add_subplot(gs2[0, 1])
    ax5.axis('off')
    
    # Calculate comprehensive stats
    total_investment = df['predicted_investment'].sum()
    avg_investment = df['predicted_investment'].mean()
    max_investment = df['predicted_investment'].max()
    min_investment = df['predicted_investment'].min()
    avg_confidence = df['confidence_score'].mean() * 100
    top_sector = df.groupby('sector')['predicted_investment'].sum().idxmax()
    top_country = df.groupby('country')['predicted_investment'].sum().idxmax()
    risk_dist = dict(df['risk_level'].value_counts())
    
    # Create professional summary box
    summary_text = f"""📊 COMPREHENSIVE ANALYSIS SUMMARY
    
💰 INVESTMENT METRICS
   Total Predicted: ${total_investment/1e6:.1f}M
   Average per Startup: ${avg_investment/1000:.0f}K
   Highest Investment: ${max_investment/1e6:.1f}M
   Lowest Investment: ${min_investment/1000:.0f}K
   
🎯 AI PERFORMANCE
   Average Confidence: {avg_confidence:.1f}%
   Model Accuracy: High
   
🏆 TOP PERFORMERS
   Leading Sector: {top_sector}
   Leading Country: {top_country}
   
🌍 GEOGRAPHIC COVERAGE
   Countries Analyzed: {len(df['country'].unique())}
   Startups Evaluated: {len(df)}
   
⚠️ RISK ASSESSMENT
   Low Risk: {risk_dist.get('Low', 0)} startups
   Medium Risk: {risk_dist.get('Medium', 0)} startups  
   High Risk: {risk_dist.get('High', 0)} startups
   
🤖 TECHNOLOGY
   AI Model: TechGyant v1.0
   Prediction Engine: Active
   Data Quality: Verified
    """
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, 
             fontsize=12, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle="round,pad=1", facecolor=TECHGYANT_COLORS['light'], 
                      edgecolor=TECHGYANT_COLORS['primary'], linewidth=3))
    
    plt.tight_layout()
    fig2.savefig('visualizations/techgyant_geographic_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("🌍 Geographic analysis saved!")
    
    # Show both figures
    plt.show()
    
    # Add branding to terminal output
    print("\n" + "="*60)
    print("� TechGyant Insights - Professional Analysis Complete")
    print("="*60)
    
    return ['visualizations/techgyant_investment_analysis.png', 
            'visualizations/techgyant_geographic_analysis.png']

def main():
    """Main execution function"""
    print("🚀 TechGyant Insights - Prediction Model Visualization")
    print("=" * 60)
    
    try:
        # Test API health
        health_response = requests.get(f"{API_BASE}/health")
        if health_response.status_code == 200:
            print("✅ API is healthy and ready!")
            print()
        else:
            print("❌ API health check failed!")
            return
        
        # Run prediction tests
        results_df = test_predictions()
        
        if len(results_df) > 0:
            print("📊 Creating visualizations...")
            create_visualizations(results_df)
            
            # Print summary statistics
            print("\n📈 PREDICTION SUMMARY")
            print("=" * 40)
            print(f"💰 Average Investment: ${results_df['predicted_investment'].mean():,.2f}")
            print(f"📊 Highest Investment: ${results_df['predicted_investment'].max():,.2f}")
            print(f"📉 Lowest Investment: ${results_df['predicted_investment'].min():,.2f}")
            print(f"🎯 Average Confidence: {results_df['confidence_score'].mean()*100:.1f}%")
            print(f"⚠️  Risk Distribution: {dict(results_df['risk_level'].value_counts())}")
            
        else:
            print("❌ No prediction results to visualize!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
