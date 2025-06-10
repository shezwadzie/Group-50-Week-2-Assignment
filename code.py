import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('water_pollution_disease.csv')

# Display basic information
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Set style for plots
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

# 1. Disease Cases by Country
plt.figure(figsize=(14, 8))
disease_cols = ['Diarrheal Cases per 100,000 people', 
                'Cholera Cases per 100,000 people', 
                'Typhoid Cases per 100,000 people']
country_disease = df.groupby('Country')[disease_cols].mean().sort_values(by='Diarrheal Cases per 100,000 people', ascending=False)
country_disease.plot(kind='bar', stacked=True)
plt.title('Average Waterborne Disease Cases per 100,000 People by Country')
plt.ylabel('Cases per 100,000 people')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Correlation between Water Quality and Diseases
water_quality = ['Contaminant Level (ppm)', 'pH Level', 'Turbidity (NTU)', 
                 'Dissolved Oxygen (mg/L)', 'Nitrate Level (mg/L)', 
                 'Lead Concentration (µg/L)', 'Bacteria Count (CFU/mL)']

corr_matrix = df[water_quality + disease_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix[disease_cols], annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Between Water Quality Parameters and Disease Cases')
plt.tight_layout()
plt.show()

# 3. Impact of Water Treatment Methods
plt.figure(figsize=(12, 6))
treatment_effect = df.groupby('Water Treatment Method')[disease_cols].mean()
treatment_effect.plot(kind='bar')
plt.title('Average Disease Cases by Water Treatment Method')
plt.ylabel('Cases per 100,000 people')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Disease Cases vs. Access to Clean Water
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Access to Clean Water (% of Population)', 
                y='Diarrheal Cases per 100,000 people', hue='Region')
plt.title('Diarrheal Cases vs. Access to Clean Water')
plt.tight_layout()
plt.show()

# 5. Infant Mortality vs. Water Contamination
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Lead Concentration (µg/L)', 
                y='Infant Mortality Rate (per 1,000 live births)', 
                hue='Country', size='Bacteria Count (CFU/mL)', sizes=(20, 200))
plt.title('Infant Mortality Rate vs. Lead Concentration')
plt.tight_layout()
plt.show()

# 6. Temporal Trends in Water Quality and Diseases
if 'Year' in df.columns:
    yearly_trend = df.groupby('Year')[water_quality + disease_cols].mean()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    yearly_trend[water_quality].plot(ax=axes[0])
    axes[0].set_title('Water Quality Trends Over Time')
    axes[0].set_ylabel('Measurement')
    
    yearly_trend[disease_cols].plot(ax=axes[1])
    axes[1].set_title('Disease Case Trends Over Time')
    axes[1].set_ylabel('Cases per 100,000 people')
    
    plt.tight_layout()
    plt.show()

# 7. Regional Analysis
plt.figure(figsize=(14, 8))
regional_analysis = df.groupby('Region')[disease_cols + ['Infant Mortality Rate (per 1,000 live births)']].mean()
regional_analysis.plot(kind='bar')
plt.title('Health Indicators by Region')
plt.ylabel('Rate/Cases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 8. Water Source Type Analysis
plt.figure(figsize=(12, 6))
source_analysis = df.groupby('Water Source Type')[disease_cols].mean()
source_analysis.plot(kind='bar')
plt.title('Disease Cases by Water Source Type')
plt.ylabel('Cases per 100,000 people')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9. GDP and Healthcare Access Impact
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='GDP per Capita (USD)', 
                y='Diarrheal Cases per 100,000 people', 
                hue='Healthcare Access Index (0-100)', size='Sanitation Coverage (% of Population)')
plt.title('Diarrheal Cases vs. GDP and Healthcare Access')
plt.tight_layout()
plt.show()