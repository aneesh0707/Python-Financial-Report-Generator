#!/usr/bin/env python3
"""
Quick Start Script for Python Report Generator
Run this to quickly test the tool with sample data or your own CSV file
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'reportlab', 'PIL']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install them using:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def run_sample_analysis():
    """Run analysis on the sample data"""
    print("\n🚀 Running sample data analysis...")
    
    if not os.path.exists('sample_data.csv'):
        print("❌ Sample data file not found. Please ensure 'sample_data.csv' exists.")
        return False
    
    # Import and run the main analysis
    try:
        from main import DataAnalyzer, DataVisualizer, ReportGenerator
        from datetime import datetime
        
        # Initialize analyzer
        analyzer = DataAnalyzer('sample_data.csv')
        
        # Load and inspect data
        if not analyzer.load_data():
            return False
        
        data_info = analyzer.inspect_data()
        
        # Clean data
        cleaned_df = analyzer.clean_data()
        
        # Create visualizations
        visualizer = DataVisualizer(cleaned_df, analyzer.numeric_columns, 
                                   analyzer.categorical_columns, analyzer.datetime_columns)
        figures = visualizer.create_visualizations()
        
        # Generate report
        output_path = f"Sample_Data_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        report_generator = ReportGenerator(data_info, {}, figures, output_path)
        report_generator.generate_report()
        
        print(f"\n🎉 Sample analysis complete! Report saved as: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False

def main():
    """Main quick start function"""
    print("="*60)
    print("PYTHON REPORT GENERATOR - QUICK START")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("\nOptions:")
    print("1. Run sample data analysis (recommended for first-time users)")
    print("2. Run with your own CSV file")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            if run_sample_analysis():
                print("\n✅ Sample analysis completed successfully!")
                print("You can now try with your own data using option 2 or run 'python main.py'")
            break
            
        elif choice == '2':
            print("\nTo analyze your own CSV file, run:")
            print("   python main.py")
            print("\nOr use the interactive mode:")
            print("   python -i main.py")
            break
            
        elif choice == '3':
            print("Goodbye! 👋")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
