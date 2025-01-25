"""
Tech Product Sentiment Analysis Generator
This script creates synthetic product reviews and generates a professional sentiment analysis visualization.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
from random import choice, randint

# Set random seed for reproducibility
np.random.seed(42)  # Ensures consistent random results across runs

# ========== CONFIGURATION SECTION ==========
NUM_REVIEWS = 1500       # Total number of reviews to generate
IMAGE_QUALITY = 300      # DPI for output image (300 = print quality)
IMAGE_FORMAT = 'png'     # Output format (options: png, pdf, jpeg, svg)
OUTPUT_FILENAME = 'tech_sentiment_analysis'  # Base name for output file

# Platform and product configuration
PLATFORMS = ['Amazon', 'BestBuy', 'Twitter', 'Reddit', 'TechRadar']
PRODUCT_CATEGORIES = ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Camera']

# Brand configuration with category mapping
BRANDS = {
    'Laptop': ['Dell', 'HP', 'Lenovo', 'Apple', 'Asus'],
    'Smartphone': ['Samsung', 'Apple', 'Google', 'OnePlus', 'Xiaomi'],
    'Headphones': ['Sony', 'Bose', 'JBL', 'Apple', 'Sennheiser'],
    'Tablet': ['Apple', 'Samsung', 'Microsoft', 'Amazon', 'Lenovo'],
    'Camera': ['Canon', 'Nikon', 'Sony', 'Fujifilm', 'GoPro']
}

# Technical specifications for different product categories
PRODUCT_SPECS = {
    'Laptop': {
        'models': ['XPS 13', 'Spectre x360', 'ThinkPad X1', 'MacBook Pro', 'ZenBook'],
        'specs': ['i7-1260P', 'Ryzen 7 6800U', '32GB DDR5', '1TB SSD', 'RTX 3050'],
        'features': ['4K touchscreen', 'backlit keyboard', 'fingerprint reader', 'Thunderbolt 4']
    },
    'Smartphone': {
        'models': ['Galaxy S23', 'iPhone 15', 'Pixel 8', 'Nord 3', 'Redmi Note 12'],
        'specs': ['Snapdragon 8 Gen2', 'A16 Bionic', '50MP camera', '120Hz AMOLED'],
        'features': ['5G', 'IP68 rating', 'wireless charging', 'under-display fingerprint']
    }
}

# Natural language review templates with placeholders
REVIEW_TEMPLATES = {
    "Laptop": {
        "positive": [
            "Absolutely loving my new {brand} {model}! The {feature} works flawlessly, getting {hours}h battery life. Perfect for {use_case}!",
            "Upgraded from my old laptop and the difference is incredible. {spec} handles {task} smoothly. Only wish {minor_issue}.",
            "Best tech purchase this year! The {feature} is revolutionary. {specific_praise}."
        ],
        "negative": [
            "Disappointed with {brand} {model}. {component} started {issue} within {days} days. {consequence}!",
            "Overheats during {task}. Can't even {basic_use} without fan noise. Avoid!",
            "{brand} quality declined. {feature} feels cheap and {specific_issue}."
        ]
    },
    "Smartphone": {
        "positive": [
            "Camera is amazing! {feature} takes stunning {photo_type} photos. Battery lasts {hours}h easily.",
            "Upgrade worth every penny. {specific_praise} makes {common_task} effortless. {casual_remark}",
            "Best {brand} phone yet. {feature} works perfectly. {specific_benefit}."
        ],
        "negative": [
            "Screen developed {issue} after {days}d. {brand} support was {support_exp}. Never again!",
            "Battery dies in {hours}h with light use. {frustration_phrase}",
            "{brand} {model} keeps {recurring_issue}. Should've bought {alternative}."
        ]
    }
}

# ========== HELPER FUNCTIONS ==========
def generate_context(category, brand, sentiment):
    """
    Generates dynamic context for review templates
    Parameters:
    - category: Product category (e.g., 'Laptop')
    - brand: Manufacturer name
    - sentiment: Review sentiment ('positive'/'negative')
    Returns dictionary of context variables for template filling
    """
    context = {'brand': brand}
    
    # Add technical specifications if available
    if category in PRODUCT_SPECS:
        specs = PRODUCT_SPECS[category]
        context.update({
            'model': choice(specs['models']),
            'spec': choice(specs['specs']),
            'feature': choice(specs['features'])
        })
    
    # Sentiment-specific context variables
    if sentiment == 'positive':
        context.update({
            'hours': randint(8, 14),  # Battery life/usage hours
            'days': randint(3, 7),    # Positive experience duration
            'use_case': choice(['gaming', 'work', 'content creation']),
            'task': choice(['4K rendering', 'multitasking', 'heavy workloads']),
            'photo_type': choice(['low-light', 'portrait', 'macro']),
            'common_task': choice(['photo editing', 'navigation', 'social media']),
            'specific_praise': choice(["The display is breathtaking", "Performance is blazing fast"]),
            'minor_issue': choice(["the webcam could be better", "it gets warm under load"])
        })
    else:
        context.update({
            'hours': randint(2, 5),   # Short battery life
            'days': randint(1, 14),   # Time until issue appeared
            'component': choice(['battery', 'keyboard', 'screen']),
            'issue': choice(['overheating', 'failing', 'malfunctioning']),
            'consequence': choice(["Lost important work", "Can't use for work"]),
            'support_exp': choice(["unhelpful", "rude", "slow to respond"]),
            'recurring_issue': choice(["crashing", "freezing", "overheating"]),
            'alternative': choice(["the competitor's model", "last year's flagship"])
        })
    
    return context

# ========== DATA GENERATION ==========
print("Generating synthetic review data...")
data = []
for _ in range(NUM_REVIEWS):
    # Random selection of product attributes
    category = choice(PRODUCT_CATEGORIES)
    brand = choice(BRANDS[category])
    platform = choice(PLATFORMS)
    
    # Platform-based sentiment distribution
    if platform in ['Twitter', 'Reddit']:
        sentiment = np.random.choice(['positive', 'neutral', 'negative'], p=[0.4, 0.3, 0.3])
    else:
        sentiment = np.random.choice(['positive', 'neutral', 'negative'], p=[0.5, 0.3, 0.2])
    
    # Generate review text using templates
    review = ""
    if category in REVIEW_TEMPLATES and sentiment in ['positive', 'negative']:
        template = choice(REVIEW_TEMPLATES[category][sentiment])
        context = generate_context(category, brand, sentiment)
        try:
            review = template.format(**context)
        except KeyError:
            review = "Great product overall!" if sentiment == 'positive' else "Disappointing experience"
    else:
        # Fallback generic reviews for categories without templates
        phrases = {
            'positive': [
                f"Excellent {category} from {brand}",
                f"Loving my new {brand} {category}",
                f"Best {category} I've ever used"
            ],
            'neutral': [
                f"Decent {category} but could improve",
                f"Average {category} experience",
                f"{brand} {category} meets basic needs"
            ],
            'negative': [
                f"Poor quality {category} from {brand}",
                f"Regret buying this {brand} {category}",
                f"Major issues with {brand} {category}"
            ]
        }
        review = choice(phrases[sentiment])
    
    # Generate sentiment score with normal distribution
    if sentiment == 'positive':
        score = np.random.normal(0.8, 0.1)  # Mean 0.8, Std Dev 0.1
    elif sentiment == 'neutral':
        score = np.random.normal(0.5, 0.1)  # Mean 0.5, Std Dev 0.1
    else:
        score = np.random.normal(0.3, 0.1)  # Mean 0.3, Std Dev 0.1
    score = max(0.0, min(1.0, score))      # Clamp score between 0-1
    
    data.append([category, brand, platform, review, score])

# Create DataFrame for analysis
df = pd.DataFrame(data, columns=['Category', 'Brand', 'Platform', 'Review', 'Sentiment Score'])

# ========== DATA ANALYSIS ==========
print("Calculating average sentiment scores...")
avg_sentiment = df.groupby(['Category', 'Platform'])['Sentiment Score'].mean().reset_index()

# ========== VISUALIZATION ==========
print("Creating visualization...")
plt.figure(figsize=(16, 10))  # HD aspect ratio (16:10)
sns.set_style("whitegrid")    # Modern visualization style
palette = sns.color_palette("husl", n_colors=len(PLATFORMS))  # Distinct platform colors

# Create main bar plot
ax = sns.barplot(
    x='Category',
    y='Sentiment Score',
    hue='Platform',
    data=avg_sentiment,
    palette=palette,
    ci=None                   # Remove confidence interval bars
)

# Plot customization
plt.title('Tech Product Sentiment Analysis by Category and Platform\n', 
         fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Product Categories', fontsize=14, labelpad=15)
plt.ylabel('Average Sentiment Score', fontsize=14, labelpad=15)
plt.ylim(0, 1)  # Fixed scale for comparison
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.3)  # Subtle grid lines

# Add value annotations
for p in ax.patches:
    ax.annotate(
        f'{p.get_height():.2f}',
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='center',
        xytext=(0, 10),
        textcoords='offset points',
        fontsize=10,
        color='black'
    )

# Legend customization
plt.legend(
    title='Platform',
    bbox_to_anchor=(1.05, 1),  # Position outside plot area
    loc='upper left',
    fontsize=12,
    title_fontsize=13
)

# Save high-quality image before showing
print(f"Saving visualization as {OUTPUT_FILENAME}.{IMAGE_FORMAT}...")
plt.savefig(
    f'{OUTPUT_FILENAME}.{IMAGE_FORMAT}',
    dpi=IMAGE_QUALITY,
    bbox_inches='tight',  # Prevent cropping
    facecolor='white'     # White background
)

# Display final plot
plt.tight_layout()
plt.show()

print("Analysis complete!")