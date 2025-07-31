#!/usr/bin/env python3
"""
Script to select top 50 most liquid/popular stocks for Render deployment
"""

import os
import shutil

# Top 50 most liquid/popular stocks (S&P 500 leaders)
TOP_50_STOCKS = [
    # Tech Giants
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'CRM', 'ADBE',
    
    # Financial
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'USB', 'COF',
    
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'GILD',
    
    # Consumer
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'DIS',
    
    # Industrial/Energy
    'JNJ', 'UNH', 'HD', 'PG', 'JPM', 'BAC', 'PFE', 'ABBV', 'TMO', 'ABT',
    
    # Additional popular stocks
    'V', 'MA', 'PYPL', 'INTC', 'AMD', 'ORCL', 'IBM', 'QCOM', 'TXN', 'AVGO'
]

def copy_top_50_models():
    """Copy only the top 50 stock models to a new directory"""
    
    # Create backup of current models
    if os.path.exists('models_backup'):
        shutil.rmtree('models_backup')
    shutil.copytree('models', 'models_backup')
    print(f"✅ Created backup of current models in 'models_backup'")
    
    # Create new models directory with only top 50
    if os.path.exists('models_reduced'):
        shutil.rmtree('models_reduced')
    os.makedirs('models_reduced')
    
    copied_count = 0
    missing_count = 0
    
    for stock in TOP_50_STOCKS:
        for risk_level in ['low', 'medium', 'high']:
            model_name = f"{stock}_ddpg_actor_{risk_level}.pth"
            source_path = os.path.join('models', model_name)
            dest_path = os.path.join('models_reduced', model_name)
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                copied_count += 1
                print(f"✅ Copied: {model_name}")
            else:
                missing_count += 1
                print(f"❌ Missing: {model_name}")
    
    print(f"\n📊 Summary:")
    print(f"✅ Copied: {copied_count} models")
    print(f"❌ Missing: {missing_count} models")
    print(f"📁 New models directory: 'models_reduced'")
    
    # Calculate size
    total_size = sum(os.path.getsize(os.path.join('models_reduced', f)) 
                    for f in os.listdir('models_reduced'))
    size_mb = total_size / (1024 * 1024)
    print(f"💾 Total size: {size_mb:.1f} MB")
    
    return copied_count

if __name__ == "__main__":
    print("🚀 Selecting top 50 most liquid stocks for Render deployment...")
    print("=" * 60)
    
    copied = copy_top_50_models()
    
    print("\n🎯 Next steps:")
    print("1. Replace 'models' directory with 'models_reduced'")
    print("2. Commit and push to git")
    print("3. Redeploy on Render")
    
    if copied > 0:
        print(f"\n✅ Success! Ready to deploy with {copied} models")
    else:
        print("\n❌ No models were copied. Check the stock list.") 