import pandas as pd
import random

# Sample phrases for generating synthetic stock market headlines
positive_headlines = [
    "Stock market surges as tech giants report strong earnings",
    "Investors confident as economy shows strong recovery",
    "Company X posts record-breaking revenue growth",
    "Federal Reserve policies boost investor sentiment",
    "Stock prices hit an all-time high amid strong GDP growth",
    "Tech sector booms as AI adoption increases",
    "Company Y secures major investment, stocks rise"
]

negative_headlines = [
    "Market crashes due to inflation concerns",
    "Economic downturn leads to massive sell-off",
    "Company Z faces bankruptcy, stock prices plummet",
    "Recession fears shake investor confidence",
    "Global trade war tensions lead to stock market slump",
    "Tech stocks plummet as regulatory crackdowns increase",
    "Oil price hike triggers stock market decline"
]

neutral_headlines = [
    "Analysts predict stable growth next quarter",
    "Market closes with mixed results today",
    "Stock trading volume remains steady this week",
    "Federal Reserve to announce interest rate decision",
    "Company X appoints new CEO, impact on stocks uncertain",
    "Quarterly earnings reports show expected performance",
    "No major market movements detected today"
]

# Function to generate a dataset
def generate_dataset(num_samples=1000):
    headlines = []
    labels = []
    
    for _ in range(num_samples // 3):
        headlines.append(random.choice(positive_headlines))
        labels.append(1)  # Positive sentiment
        
        headlines.append(random.choice(negative_headlines))
        labels.append(-1)  # Negative sentiment
        
        headlines.append(random.choice(neutral_headlines))
        labels.append(0)  # Neutral sentiment

    # Create DataFrame
    df = pd.DataFrame({"headline": headlines, "label": labels})

    # Save as CSV
    df.to_csv("news_headlines.csv", index=False)
    print("news_headlines.csv generated successfully!")

# Generate 1000+ headlines
generate_dataset(1200)
