#!/usr/bin/env python3
"""
Python Report Generator
A comprehensive tool for CSV data analysis and professional PDF report generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import io
from textwrap import dedent
from PIL import Image as PILImage
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PDF generation

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Handles data loading, cleaning, and analysis with enhanced financial capabilities"""
    
    def __init__(self, csv_path):
        """Initialize with CSV file path"""
        self.csv_path = csv_path
        self.df = None
        self.cleaned_df = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.financial_metrics = {}
        
    def load_data(self):
        """Load CSV data and perform initial inspection"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"OK Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
        except Exception as e:
            print(f"ERROR Error loading data: {e}")
            return False
    
    def inspect_data(self):
        """Inspect data structure and basic information"""
        print("\n" + "="*60)
        print("DATA INSPECTION")
        print("="*60)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nColumn Information:")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            null_count = self.df[col].isnull().sum()
            null_pct = (null_count / len(self.df)) * 100
            unique_count = self.df[col].nunique()
            
            print(f"  {col:20} | {str(dtype):15} | Null: {null_count:3} ({null_pct:5.1f}%) | Unique: {unique_count:3}")
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nData Types:")
        print(self.df.dtypes)
        
        return {
            'shape': self.df.shape,
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,
            'null_counts': self.df.isnull().sum().to_dict(),
            'dtypes': self.df.dtypes.to_dict()
        }
    
    def clean_data(self):
        """Clean the dataset by handling missing values and data types"""
        print("\n" + "="*60)
        print("DATA CLEANING")
        print("="*60)
        
        self.cleaned_df = self.df.copy()
        
        # Handle missing values
        print("\nHandling missing values...")
        for col in self.cleaned_df.columns:
            null_count = self.cleaned_df[col].isnull().sum()
            if null_count > 0:
                print(f"  {col}: {null_count} missing values")
                
                # For numeric columns, fill with median
                if pd.api.types.is_numeric_dtype(self.cleaned_df[col]):
                    median_val = self.cleaned_df[col].median()
                    self.cleaned_df[col].fillna(median_val, inplace=True)
                    print(f"    → Filled with median: {median_val}")
                
                # For categorical columns, fill with mode
                elif pd.api.types.is_string_dtype(self.cleaned_df[col]) or pd.api.types.is_object_dtype(self.cleaned_df[col]):
                    mode_val = self.cleaned_df[col].mode()[0] if len(self.cleaned_df[col].mode()) > 0 else "Unknown"
                    self.cleaned_df[col].fillna(mode_val, inplace=True)
                    print(f"    → Filled with mode: {mode_val}")
        
        # Remove duplicates
        initial_rows = len(self.cleaned_df)
        self.cleaned_df.drop_duplicates(inplace=True)
        final_rows = len(self.cleaned_df)
        duplicates_removed = initial_rows - final_rows
        
        if duplicates_removed > 0:
            print(f"\nRemoved {duplicates_removed} duplicate rows")
        
        # Infer and convert data types
        print("\nInferring and converting data types...")
        for col in self.cleaned_df.columns:
            # Try to convert to numeric if possible
            if pd.api.types.is_object_dtype(self.cleaned_df[col]):
                try:
                    pd.to_numeric(self.cleaned_df[col], errors='raise')
                    self.cleaned_df[col] = pd.to_numeric(self.cleaned_df[col], errors='coerce')
                    print(f"  {col}: Converted to numeric")
                except:
                    pass
            
            # Try to convert to datetime if possible
            if pd.api.types.is_object_dtype(self.cleaned_df[col]):
                try:
                    pd.to_datetime(self.cleaned_df[col], errors='raise')
                    self.cleaned_df[col] = pd.to_datetime(self.cleaned_df[col], errors='coerce')
                    print(f"  {col}: Converted to datetime")
                except:
                    pass
        
        # Categorize columns
        self._categorize_columns()
        
        # Calculate financial metrics
        self._calculate_financial_metrics()
        
        print(f"\nOK Data cleaning completed. Final shape: {self.cleaned_df.shape}")
        return self.cleaned_df
    
    def _categorize_columns(self):
        """Categorize columns by data type"""
        self.numeric_columns = self.cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.cleaned_df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_columns = self.cleaned_df.select_dtypes(include=['datetime64']).columns.tolist()
        
        print(f"\nColumn categorization:")
        print(f"  Numeric: {len(self.numeric_columns)} columns")
        print(f"  Categorical: {len(self.categorical_columns)} columns")
        print(f"  Datetime: {len(self.datetime_columns)} columns")
    
    def _calculate_financial_metrics(self):
        """Calculate key financial metrics and ratios"""
        print("\nCalculating financial metrics...")
        
        try:
            # Basic financial ratios
            if all(col in self.cleaned_df.columns for col in ['Assets', 'Liabilities']):
                self.cleaned_df['Debt_to_Equity'] = self.cleaned_df['Liabilities'] / (self.cleaned_df['Assets'] - self.cleaned_df['Liabilities'])
                self.cleaned_df['Current_Ratio'] = self.cleaned_df['Assets'] / self.cleaned_df['Liabilities']
                print("  OK Debt-to-Equity and Current Ratio calculated")
            
            if all(col in self.cleaned_df.columns for col in ['Revenue', 'Expenses']):
                self.cleaned_df['Expense_Ratio'] = self.cleaned_df['Expenses'] / self.cleaned_df['Revenue']
                self.cleaned_df['Revenue_Growth'] = self.cleaned_df['Revenue'].pct_change()
                print("  OK Expense Ratio and Revenue Growth calculated")
            
            if 'Profit' in self.cleaned_df.columns and 'Assets' in self.cleaned_df.columns:
                self.cleaned_df['ROA'] = self.cleaned_df['Profit'] / self.cleaned_df['Assets']
                print("  OK Return on Assets (ROA) calculated")
            
            if 'Cash_Flow' in self.cleaned_df.columns and 'Revenue' in self.cleaned_df.columns:
                self.cleaned_df['Cash_Flow_Margin'] = self.cleaned_df['Cash_Flow'] / self.cleaned_df['Revenue']
                print("  OK Cash Flow Margin calculated")
            
            # Monthly aggregations
            if 'Month' in self.cleaned_df.columns:
                monthly_metrics = self.cleaned_df.groupby('Month').agg({
                    'Revenue': 'sum',
                    'Expenses': 'sum',
                    'Profit': 'sum',
                    'Cash_Flow': 'sum',
                    'Investments': 'sum'
                }).round(2)
                
                monthly_metrics['Profit_Margin'] = (monthly_metrics['Profit'] / monthly_metrics['Revenue'] * 100).round(2)
                monthly_metrics['Cash_Flow_Margin'] = (monthly_metrics['Cash_Flow'] / monthly_metrics['Revenue'] * 100).round(2)
                
                self.financial_metrics['monthly_summary'] = monthly_metrics
                print("  OK Monthly financial summary calculated")
            
            # Overall financial health indicators
            if len(self.cleaned_df) > 0:
                self.financial_metrics['overall'] = {
                    'total_revenue': self.cleaned_df['Revenue'].sum(),
                    'total_expenses': self.cleaned_df['Expenses'].sum(),
                    'total_profit': self.cleaned_df['Profit'].sum(),
                    'avg_profit_margin': (self.cleaned_df['Profit'].sum() / self.cleaned_df['Revenue'].sum() * 100).round(2),
                    'avg_roi': self.cleaned_df['ROI'].mean() * 100 if 'ROI' in self.cleaned_df.columns else None
                }
                print("  OK Overall financial metrics calculated")
                
        except Exception as e:
            print(f"  ⚠ Warning: Some financial metrics could not be calculated: {e}")
    
    def get_financial_summary(self):
        """Get comprehensive financial summary"""
        if not self.financial_metrics:
            return "No financial metrics available"
        
        summary = "\n" + "="*60 + "\n"
        summary += "FINANCIAL PERFORMANCE SUMMARY\n"
        summary += "="*60 + "\n"
        
        if 'overall' in self.financial_metrics:
            overall = self.financial_metrics['overall']
            summary += f"\nOverall Performance:\n"
            summary += f"  Total Revenue: ${overall['total_revenue']:,.2f}\n"
            summary += f"  Total Expenses: ${overall['total_expenses']:,.2f}\n"
            summary += f"  Total Profit: ${overall['total_profit']:,.2f}\n"
            summary += f"  Average Profit Margin: {overall['avg_profit_margin']}%\n"
            if overall['avg_roi']:
                summary += f"  Average ROI: {overall['avg_roi']:.2f}%\n"
        
        if 'monthly_summary' in self.financial_metrics:
            summary += f"\nMonthly Breakdown:\n"
            summary += str(self.financial_metrics['monthly_summary'])
        
        return summary

class DataVisualizer:
    """Handles creation of data visualizations with enhanced financial charts"""
    
    def __init__(self, cleaned_df, numeric_columns, categorical_columns, datetime_columns):
        """Initialize with cleaned data and column information"""
        self.df = cleaned_df
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.datetime_columns = datetime_columns
        self.figures = []
        
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_visualizations(self):
        """Create comprehensive set of visualizations including financial charts"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # 1. Financial performance overview
        self._create_financial_overview()
        
        # 2. Distribution plots for numeric columns
        if self.numeric_columns:
            self._create_distribution_plots()
        
        # 3. Correlation heatmap for numeric columns
        if len(self.numeric_columns) > 1:
            self._create_correlation_heatmap()
        
        # 4. Categorical variable analysis
        if self.categorical_columns:
            self._create_categorical_plots()
        
        # 5. Time series plots if datetime columns exist
        if self.datetime_columns:
            self._create_time_series_plots()
        
        # 6. Scatter plots for numeric relationships
        if len(self.numeric_columns) > 1:
            self._create_scatter_plots()
        
        # 7. Financial ratio analysis
        self._create_financial_ratio_plots()
        
        print(f"\nOK Created {len(self.figures)} visualizations")
        return self.figures
    
    def _create_financial_overview(self):
        """Create comprehensive financial overview dashboard"""
        if not all(col in self.df.columns for col in ['Revenue', 'Expenses', 'Profit', 'Cash_Flow']):
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Financial Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Revenue vs Expenses trend
        if 'Date' in self.df.columns:
            dates = pd.to_datetime(self.df['Date'])
            ax1.plot(dates, self.df['Revenue'], label='Revenue', linewidth=2, color='green', marker='o')
            ax1.plot(dates, self.df['Expenses'], label='Expenses', linewidth=2, color='red', marker='s')
            ax1.fill_between(dates, self.df['Revenue'], self.df['Expenses'], 
                           where=(self.df['Revenue'] > self.df['Expenses']), 
                           alpha=0.3, color='green', label='Profit Zone')
            ax1.fill_between(dates, self.df['Revenue'], self.df['Expenses'], 
                           where=(self.df['Revenue'] <= self.df['Expenses']), 
                           alpha=0.3, color='red', label='Loss Zone')
            ax1.set_title('Revenue vs Expenses Trend')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Amount ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Profit margin over time
        if 'Profit_Margin' in self.df.columns:
            profit_margin = self.df['Profit_Margin'] * 100
            ax2.plot(dates, profit_margin, linewidth=2, color='blue', marker='o')
            ax2.axhline(y=profit_margin.mean(), color='red', linestyle='--', 
                       label=f'Average: {profit_margin.mean():.1f}%')
            ax2.set_title('Profit Margin Trend')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Profit Margin (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Cash flow analysis
        if 'Cash_Flow' in self.df.columns:
            cash_flow = self.df['Cash_Flow']
            colors = ['green' if x >= 0 else 'red' for x in cash_flow]
            ax3.bar(range(len(cash_flow)), cash_flow, color=colors, alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.set_title('Daily Cash Flow')
            ax3.set_xlabel('Day')
            ax3.set_ylabel('Cash Flow ($)')
            ax3.grid(True, alpha=0.3)
        
        # 4. Investment vs ROI
        if all(col in self.df.columns for col in ['Investments', 'ROI']):
            roi_pct = self.df['ROI'] * 100
            scatter = ax4.scatter(self.df['Investments'], roi_pct, 
                                c=roi_pct, cmap='viridis', s=50, alpha=0.7)
            ax4.set_title('Investment vs ROI')
            ax4.set_xlabel('Investment Amount ($)')
            ax4.set_ylabel('ROI (%)')
            ax4.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax4, label='ROI (%)')
        
        plt.tight_layout()
        self.figures.append(('financial_dashboard', fig, 'Financial Performance Dashboard'))
    
    def _create_financial_ratio_plots(self):
        """Create visualizations for financial ratios"""
        if not all(col in self.df.columns for col in ['Debt_to_Equity', 'Current_Ratio']):
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Financial Health Ratios', fontsize=14, fontweight='bold')
        
        # Debt-to-Equity ratio
        debt_equity = self.df['Debt_to_Equity'].dropna()
        ax1.plot(range(len(debt_equity)), debt_equity, linewidth=2, color='orange', marker='o')
        ax1.axhline(y=debt_equity.mean(), color='red', linestyle='--', 
                   label=f'Average: {debt_equity.mean():.2f}')
        ax1.set_title('Debt-to-Equity Ratio Trend')
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Debt-to-Equity Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Current Ratio
        current_ratio = self.df['Current_Ratio'].dropna()
        ax2.plot(range(len(current_ratio)), current_ratio, linewidth=2, color='purple', marker='s')
        ax2.axhline(y=current_ratio.mean(), color='red', linestyle='--', 
                   label=f'Average: {current_ratio.mean():.2f}')
        ax2.set_title('Current Ratio Trend')
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Current Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figures.append(('financial_ratios', fig, 'Financial Health Ratios'))
    
    def _create_distribution_plots(self):
        """Create distribution plots for numeric columns"""
        for col in self.numeric_columns[:6]:  # Limit to first 6 columns
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            ax1.hist(self.df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title(f'Distribution of {col}')
            ax1.set_xlabel(col)
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(self.df[col].dropna())
            ax2.set_title(f'Box Plot of {col}')
            ax2.set_ylabel(col)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.figures.append(('distribution', fig, f'Distribution Analysis: {col}'))
    
    def _create_correlation_heatmap(self):
        """Create correlation heatmap for numeric columns"""
        if len(self.numeric_columns) < 2:
            return
            
        corr_matrix = self.df[self.numeric_columns].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Correlation Heatmap of Financial Variables')
        plt.tight_layout()
        self.figures.append(('correlation', fig, 'Correlation Heatmap'))
    
    def _create_categorical_plots(self):
        """Create plots for categorical variables"""
        for col in self.categorical_columns[:4]:  # Limit to first 4 columns
            value_counts = self.df[col].value_counts().head(10)  # Top 10 values
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Bar chart
            value_counts.plot(kind='bar', ax=ax1, color='lightcoral')
            ax1.set_title(f'Top 10 Values in {col}')
            ax1.set_xlabel(col)
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Pie chart (top 5 values)
            top_5 = value_counts.head(5)
            ax2.pie(top_5.values, labels=top_5.index, autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Top 5 Values in {col}')
            
            plt.tight_layout()
            self.figures.append(('categorical', fig, f'Categorical Analysis: {col}'))
    
    def _create_time_series_plots(self):
        """Create time series plots if datetime columns exist"""
        for col in self.datetime_columns[:2]:  # Limit to first 2 datetime columns
            # Find a numeric column to plot against time
            for num_col in self.numeric_columns[:2]:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Group by date and calculate mean
                time_series = self.df.groupby(self.df[col].dt.date)[num_col].mean()
                
                ax.plot(time_series.index, time_series.values, marker='o', linewidth=2, markersize=4)
                ax.set_title(f'{num_col} Over Time ({col})')
                ax.set_xlabel('Date')
                ax.set_ylabel(num_col)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                self.figures.append(('timeseries', fig, f'Time Series: {num_col} vs {col}'))
                break
    
    def _create_scatter_plots(self):
        """Create scatter plots for numeric relationships"""
        if len(self.numeric_columns) < 2:
            return
            
        # Create scatter plots for first few numeric columns
        for i in range(min(3, len(self.numeric_columns))):
            for j in range(i+1, min(4, len(self.numeric_columns))):
                col1, col2 = self.numeric_columns[i], self.numeric_columns[j]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                ax.scatter(self.df[col1], self.df[col2], alpha=0.6, s=30)
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                ax.set_title(f'{col1} vs {col2}')
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(self.df[col1].dropna(), self.df[col2].dropna(), 1)
                p = np.poly1d(z)
                ax.plot(self.df[col1], p(self.df[col1]), "r--", alpha=0.8)
                
                plt.tight_layout()
                self.figures.append(('scatter', fig, f'Scatter Plot: {col1} vs {col2}'))
                
                if len(self.figures) >= 12:  # Increased limit for more financial charts
                    return

class ReportGenerator:
    """Generates professional PDF reports with enhanced financial analysis"""
    
    def __init__(self, data_info, analysis_results, figures, output_path, financial_metrics=None):
        """Initialize with analysis results, figures, and financial metrics"""
        self.data_info = data_info
        self.analysis_results = analysis_results
        self.figures = figures
        self.output_path = output_path
        self.financial_metrics = financial_metrics or {}
        
    def generate_report(self):
        """Generate comprehensive PDF report with financial focus"""
        print("\n" + "="*60)
        print("GENERATING FINANCIAL PDF REPORT")
        print("="*60)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            leftMargin=36,
            rightMargin=36,
            topMargin=48,
            bottomMargin=48,
        )
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=15,
            textColor=colors.darkgreen
        )
        
        # Custom style for bullet points with better spacing
        bullet_style = ParagraphStyle(
            'CustomBullet',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            spaceBefore=6,
            leftIndent=20,
            textColor=colors.black
        )
        
        # Custom style for numbered lists
        numbered_style = ParagraphStyle(
            'CustomNumbered',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            spaceBefore=8,
            leftIndent=25,
            textColor=colors.black
        )
        
        # Title page
        story.append(Paragraph("Financial Analysis Report", title_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
        story.append(Spacer(1, 40))
        story.append(Paragraph("Executive Summary", heading_style))
        total_rows = self.data_info['shape'][0]
        total_cols = self.data_info['shape'][1]
        
        story.append(Paragraph(f"This comprehensive financial analysis report presents a detailed examination of financial performance data spanning {total_rows:,} daily records across {total_cols} key financial metrics.", styles['Normal']))
        story.append(Spacer(1, 8))
        story.append(Paragraph("The analysis reveals critical insights into revenue generation, expense management, profitability trends, and overall financial health indicators.", styles['Normal']))
        story.append(Spacer(1, 8))
        story.append(Paragraph("Key financial highlights include revenue performance analysis, expense optimization opportunities, cash flow management insights, and investment return assessments.", styles['Normal']))
        story.append(Spacer(1, 8))
        story.append(Paragraph("The report provides actionable recommendations for improving financial performance and strategic decision-making based on data-driven insights.", styles['Normal']))
        story.append(PageBreak())
        
        # Table of Contents
        story.append(Paragraph("Table of Contents", heading_style))
        toc_items = [
            "1. Financial Performance Overview",
            "2. Data Quality Assessment",
            "3. Financial Metrics Analysis",
            "4. Key Financial Insights and Visualizations",
            "5. Risk Assessment",
            "6. Strategic Recommendations"
        ]
        
        for item in toc_items:
            story.append(Paragraph(f"• {item}", bullet_style))
        story.append(PageBreak())
        
        # Financial Performance Overview Section
        story.append(Paragraph("1. Financial Performance Overview", heading_style))
        story.append(Paragraph("Financial Performance Overview:", subheading_style))
        
        if self.financial_metrics:
            if 'overall' in self.financial_metrics:
                overall = self.financial_metrics['overall']
                story.append(Paragraph("Overall Performance:", numbered_style))
                story.append(Paragraph(f"• Total Revenue: ${overall['total_revenue']:,.2f}", bullet_style))
                story.append(Paragraph(f"• Total Expenses: ${overall['total_expenses']:,.2f}", bullet_style))
                story.append(Paragraph(f"• Total Profit: ${overall['total_profit']:,.2f}", bullet_style))
                story.append(Paragraph(f"• Average Profit Margin: {overall['avg_profit_margin']}%", bullet_style))
                if overall['avg_roi']:
                    story.append(Paragraph(f"• Average ROI: {overall['avg_roi']:.2f}%", bullet_style))
                story.append(Spacer(1, 8))
            
            if 'monthly_summary' in self.financial_metrics:
                story.append(Paragraph("Monthly Performance Summary:", numbered_style))
                monthly = self.financial_metrics['monthly_summary']
                for month, data in monthly.iterrows():
                    story.append(Paragraph(f"  {month}:", numbered_style))
                    story.append(Paragraph(f"    • Revenue: ${data['Revenue']:,.2f}", bullet_style))
                    story.append(Paragraph(f"    • Profit: ${data['Profit']:,.2f}", bullet_style))
                    story.append(Paragraph(f"    • Profit Margin: {data['Profit_Margin']}%", bullet_style))
                    story.append(Paragraph(f"    • Cash Flow: ${data['Cash_Flow']:,.2f}", bullet_style))
                    story.append(Spacer(1, 4))
        else:
            story.append(Paragraph("Financial metrics analysis not available.", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Data Quality Assessment
        story.append(Paragraph("2. Data Quality Assessment", heading_style))
        story.append(Paragraph("Data Quality Assessment:", subheading_style))
        
        null_info = self.data_info['null_counts']
        total_rows = self.data_info['shape'][0]
        
        for col, null_count in null_info.items():
            if null_count > 0:
                null_pct = (null_count / total_rows) * 100
                story.append(Paragraph(f"• {col}: {null_count:,} missing values ({null_pct:.1f}%)", bullet_style))
            else:
                story.append(Paragraph(f"• {col}: No missing values ✓", bullet_style))
        
        overall_completeness = ((total_rows * len(null_info) - sum(null_info.values())) / (total_rows * len(null_info)) * 100)
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"Overall data completeness: {overall_completeness:.1f}%", numbered_style))
        
        story.append(Spacer(1, 20))
        
        # Financial Metrics Analysis
        story.append(Paragraph("3. Financial Metrics Analysis", heading_style))
        story.append(Paragraph("Financial Metrics Analysis:", subheading_style))
        story.append(Paragraph("This section provides a comprehensive analysis of key financial ratios and metrics that are essential for understanding the organization's financial health and performance.", styles['Normal']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Key Financial Ratios Calculated:", numbered_style))
        story.append(Paragraph("• Debt-to-Equity Ratio: Measures financial leverage and risk", bullet_style))
        story.append(Paragraph("• Current Ratio: Indicates short-term liquidity and ability to meet obligations", bullet_style))
        story.append(Paragraph("• Return on Assets (ROA): Shows efficiency in using assets to generate profits", bullet_style))
        story.append(Paragraph("• Cash Flow Margin: Demonstrates ability to convert revenue to cash", bullet_style))
        story.append(Paragraph("• Expense Ratio: Shows cost structure efficiency relative to revenue", bullet_style))
        story.append(Spacer(1, 8))
        
        story.append(Paragraph("These metrics provide critical insights into:", numbered_style))
        story.append(Paragraph("• Financial stability and risk levels", bullet_style))
        story.append(Paragraph("• Operational efficiency and profitability", bullet_style))
        story.append(Paragraph("• Cash flow management effectiveness", bullet_style))
        story.append(Paragraph("• Investment performance and returns", bullet_style))
        
        story.append(Spacer(1, 20))
        
        # EDA Section
        story.append(Paragraph("4. Key Financial Insights and Visualizations", heading_style))
        story.append(Paragraph("The following visualizations provide comprehensive insights into financial performance, trends, and key metrics.", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Add all figures
        temp_img_paths = []
        for i, (plot_type, fig, title) in enumerate(self.figures):
            # Save figure to bytes
            img_bytes = io.BytesIO()
            fig.savefig(img_bytes, format='png', dpi=300, bbox_inches='tight')
            img_bytes.seek(0)
            
            # Convert to PIL Image and then to ReportLab Image
            pil_img = PILImage.open(img_bytes)
            img_path = f"temp_img_{i}.png"
            pil_img.save(img_path)
            temp_img_paths.append(img_path)
            
            # Add to report
            # Scale image to fit page width keeping aspect ratio
            max_width = doc.width
            max_height = 5.5 * inch
            iw, ih = pil_img.size
            aspect = ih / float(iw)
            target_w = min(max_width, iw)
            target_h = min(max_height, target_w * aspect)
            # If height exceeds max, scale down accordingly
            if target_h > max_height:
                target_h = max_height
                target_w = target_h / aspect

            fig_block = []
            fig_block.append(Paragraph(f"Figure {i+1}: {title}", subheading_style))
            fig_block.append(Image(img_path, width=target_w, height=target_h))
            fig_block.append(Spacer(1, 8))
            interpretation = self._get_financial_plot_interpretation(plot_type, title)
            fig_block.append(Paragraph(interpretation, styles['Normal']))
            fig_block.append(Spacer(1, 16))
            story.append(KeepTogether(fig_block))
        
        # Risk Assessment Section
        story.append(Paragraph("5. Risk Assessment", heading_style))
        story.append(Paragraph("Financial Risk Assessment:", subheading_style))
        story.append(Paragraph("Based on the comprehensive analysis of financial data, the following risk factors have been identified and assessed:", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Risk items with better formatting
        risk_items = [
            ("1. Liquidity Risk:", [
                "• Current ratio analysis indicates short-term financial strength",
                "• Cash flow patterns reveal operational cash generation capability", 
                "• Recommendations for maintaining adequate working capital"
            ]),
            ("2. Leverage Risk:", [
                "• Debt-to-equity ratios indicate financial leverage levels",
                "• Assessment of debt service capability",
                "• Recommendations for optimal capital structure"
            ]),
            ("3. Operational Risk:", [
                "• Expense ratio analysis reveals cost structure efficiency",
                "• Profit margin trends indicate operational performance",
                "• Recommendations for cost optimization and efficiency improvements"
            ]),
            ("4. Investment Risk:", [
                "• ROI analysis shows investment performance",
                "• Investment concentration and diversification assessment",
                "• Recommendations for portfolio optimization"
            ])
        ]
        
        for risk_title, risk_points in risk_items:
            story.append(Paragraph(risk_title, numbered_style))
            for point in risk_points:
                # Remove the bullet character from the point text since we're using custom styling
                clean_point = point.replace('• ', '')
                story.append(Paragraph(f"• {clean_point}", bullet_style))
            story.append(Spacer(1, 8))
        
        story.append(Spacer(1, 20))
        
        # Strategic Recommendations Section
        story.append(Paragraph("6. Strategic Recommendations", heading_style))
        story.append(Paragraph("Strategic Financial Recommendations:", subheading_style))
        story.append(Paragraph("Based on the comprehensive financial analysis conducted, the following strategic recommendations are provided:", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Recommendation items with better formatting
        rec_items = [
            ("1. Financial Performance Optimization:", [
                "• Implement cost control measures based on expense ratio analysis",
                "• Optimize pricing strategies to improve profit margins",
                "• Enhance cash flow management through improved working capital practices"
            ]),
            ("2. Risk Management Strategies:", [
                "• Maintain optimal debt-to-equity ratios for financial stability",
                "• Diversify investment portfolio to reduce concentration risk",
                "• Implement robust cash flow forecasting and monitoring systems"
            ]),
            ("3. Growth and Investment Opportunities:", [
                "• Identify high-ROI investment opportunities based on performance analysis",
                "• Develop strategic initiatives to improve return on assets",
                "• Consider expansion opportunities in high-margin business areas"
            ]),
            ("4. Operational Excellence:", [
                "• Streamline operational processes to improve efficiency ratios",
                "• Implement data-driven decision-making frameworks",
                "• Develop key performance indicators for continuous monitoring"
            ]),
            ("5. Long-term Financial Planning:", [
                "• Establish sustainable growth targets based on historical performance",
                "• Develop comprehensive financial forecasting models",
                "• Implement regular financial health assessments and reporting"
            ])
        ]
        
        for rec_title, rec_points in rec_items:
            story.append(Paragraph(rec_title, numbered_style))
            for point in rec_points:
                # Remove the bullet character from the point text since we're using custom styling
                clean_point = point.replace('• ', '')
                story.append(Paragraph(f"• {clean_point}", bullet_style))
            story.append(Spacer(1, 8))
        
        # Build PDF
        doc.build(story)
        
        # Clean up temporary image files after PDF build
        for p in temp_img_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        print(f"✓ Financial PDF report generated successfully: {self.output_path}")
    
    def _generate_executive_summary(self):
        """Generate executive summary with financial focus"""
        total_rows = self.data_info['shape'][0]
        total_cols = self.data_info['shape'][1]
        
        summary = f"""
        This comprehensive financial analysis report presents a detailed examination of financial 
        performance data spanning {total_rows:,} daily records across {total_cols} key financial metrics. 
        The analysis reveals critical insights into revenue generation, expense management, 
        profitability trends, and overall financial health indicators.
        
        Key financial highlights include revenue performance analysis, expense optimization 
        opportunities, cash flow management insights, and investment return assessments. 
        The report provides actionable recommendations for improving financial performance 
        and strategic decision-making based on data-driven insights.
        """
        return summary
    
    def _generate_financial_overview(self):
        """Generate financial performance overview section"""
        if not self.financial_metrics:
            return "Financial metrics analysis not available."
        
        overview = "Financial Performance Overview:\n\n"
        
        if 'overall' in self.financial_metrics:
            overall = self.financial_metrics['overall']
            overview += f"• Total Revenue: ${overall['total_revenue']:,.2f}\n"
            overview += f"• Total Expenses: ${overall['total_expenses']:,.2f}\n"
            overview += f"• Total Profit: ${overall['total_profit']:,.2f}\n"
            overview += f"• Average Profit Margin: {overall['avg_profit_margin']}%\n"
            if overall['avg_roi']:
                overview += f"• Average ROI: {overall['avg_roi']:.2f}%\n"
        
        if 'monthly_summary' in self.financial_metrics:
            overview += f"\nMonthly Performance Summary:\n"
            monthly = self.financial_metrics['monthly_summary']
            for month, data in monthly.iterrows():
                overview += f"  {month}:\n"
                overview += f"    Revenue: ${data['Revenue']:,.2f}\n"
                overview += f"    Profit: ${data['Profit']:,.2f}\n"
                overview += f"    Profit Margin: {data['Profit_Margin']}%\n"
                overview += f"    Cash Flow: ${data['Cash_Flow']:,.2f}\n"
        
        return overview
    
    def _generate_data_overview(self):
        """Generate data overview section"""
        overview = f"""
        Dataset Dimensions: {self.data_info['shape'][0]:,} rows × {self.data_info['shape'][1]} columns
        
        Memory Usage: {self.data_info['memory_usage']:.2f} MB
        
        Data Types Distribution:
        • Numeric columns: {len([col for col in self.data_info['dtypes'].values() 
                               if pd.api.types.is_numeric_dtype(pd.Series(dtype=col))])}
        • Categorical columns: {len([col for col in self.data_info['dtypes'].values() 
                                   if pd.api.types.is_string_dtype(pd.Series(dtype=col)) or 
                                   pd.api.types.is_object_dtype(pd.Series(dtype=col))])}
        • Datetime columns: {len([col for col in self.data_info['dtypes'].values() 
                                if 'datetime' in str(col)])}
        """
        return overview
    
    def _generate_financial_metrics_section(self):
        """Generate financial metrics analysis section"""
        metrics_text = """
        Financial Metrics Analysis:
        
        This section provides a comprehensive analysis of key financial ratios and metrics 
        that are essential for understanding the organization's financial health and performance.
        
        Key Financial Ratios Calculated:
        • Debt-to-Equity Ratio: Measures financial leverage and risk
        • Current Ratio: Indicates short-term liquidity and ability to meet obligations
        • Return on Assets (ROA): Shows efficiency in using assets to generate profits
        • Cash Flow Margin: Demonstrates ability to convert revenue to cash
        • Expense Ratio: Shows cost structure efficiency relative to revenue
        
        These metrics provide critical insights into:
        - Financial stability and risk levels
        - Operational efficiency and profitability
        - Cash flow management effectiveness
        - Investment performance and returns
        """
        return metrics_text
    
    def _generate_data_quality_section(self):
        """Generate data quality assessment section"""
        null_info = self.data_info['null_counts']
        total_rows = self.data_info['shape'][0]
        
        quality_text = "Data Quality Assessment:\n\n"
        
        for col, null_count in null_info.items():
            if null_count > 0:
                null_pct = (null_count / total_rows) * 100
                quality_text += f"• {col}: {null_count:,} missing values ({null_pct:.1f}%)\n"
            else:
                quality_text += f"• {col}: No missing values OK\n"
        
        quality_text += f"\nOverall data completeness: {((total_rows * len(null_info) - sum(null_info.values())) / (total_rows * len(null_info)) * 100):.1f}%"
        
        return quality_text
    
    def _generate_eda_section(self):
        """Generate EDA section"""
        eda_text = """
        Exploratory Data Analysis (EDA) was conducted to understand the underlying patterns, 
        distributions, and relationships in the financial dataset. The analysis included:
        
        • Distribution analysis of financial variables to identify patterns and outliers
        • Correlation analysis to understand relationships between financial metrics
        • Frequency analysis of categorical variables to identify dominant categories
        • Time series analysis to identify temporal trends in financial performance
        • Scatter plot analysis to visualize relationships between financial variables
        • Financial ratio analysis to assess overall financial health
        
        The visualizations in the following section provide detailed insights into these analyses.
        """
        return eda_text
    
    def _get_financial_plot_interpretation(self, plot_type, title):
        """Generate interpretation for each plot type with financial focus (safe evaluation)"""
        if plot_type == 'financial_dashboard':
            return ("This comprehensive financial dashboard provides a 360-degree view of financial "
                    "performance, including revenue vs. expenses trends, profit margin analysis, cash "
                    "flow patterns, and investment ROI relationships. The dashboard reveals key "
                    "performance indicators and trends that are critical for financial decision-making.")
        
        if plot_type == 'financial_ratios':
            return ("These financial health ratios demonstrate the organization's financial stability, "
                    "liquidity, and risk profile. The Debt-to-Equity ratio indicates leverage levels, "
                    "while the Current Ratio shows short-term financial strength and ability to meet obligations.")
        
        if plot_type == 'distribution':
            metric = title.split(': ', 1)[-1]
            return (f"This distribution plot shows the spread and shape of the financial data for {metric}. "
                    "The histogram reveals the frequency distribution, while the box plot highlights potential "
                    "outliers and the central tendency of the financial metrics.")
        
        if plot_type == 'correlation':
            return ("This correlation heatmap reveals the strength and direction of relationships between "
                    "financial variables. Understanding these correlations is crucial for portfolio management, "
                    "risk assessment, and strategic financial planning.")
        
        if plot_type == 'categorical':
            metric = title.split(': ', 1)[-1]
            return (f"This analysis shows the distribution of values in the {metric} variable. The bar chart "
                    "displays frequency counts, while the pie chart shows the proportional representation of the top categories.")
        
        if plot_type == 'timeseries':
            left = title.split(' vs ')[0]
            left = left.replace('Time Series: ', '')
            return (f"This time series plot shows how {left} changes over time. The trend line helps identify "
                    "overall patterns and direction of change in financial performance.")
        
        if plot_type == 'scatter':
            if ' vs ' in title:
                left, right = title.split(' vs ', 1)
                left = left.replace('Scatter Plot: ', '')
            else:
                left, right = title, 'the other variable'
            return (f"This scatter plot reveals the relationship between {left} and {right}. The trend line "
                    "indicates the overall direction of the relationship, while the scatter of points shows the "
                    "strength and consistency of this financial relationship.")
        
        return ("This financial visualization provides important insights into the data patterns and relationships "
                "that are essential for financial analysis and decision-making.")
    
    def _generate_risk_assessment(self):
        """Generate risk assessment section"""
        risk_text = """
        Financial Risk Assessment:
        
        Based on the comprehensive analysis of financial data, the following risk factors 
        have been identified and assessed:
        
        1. Liquidity Risk:
           • Current ratio analysis indicates short-term financial strength
           • Cash flow patterns reveal operational cash generation capability
           • Recommendations for maintaining adequate working capital
        
        2. Leverage Risk:
           • Debt-to-equity ratios indicate financial leverage levels
           • Assessment of debt service capability
           • Recommendations for optimal capital structure
        
        3. Operational Risk:
           • Expense ratio analysis reveals cost structure efficiency
           • Profit margin trends indicate operational performance
           • Recommendations for cost optimization and efficiency improvements
        
        4. Investment Risk:
           • ROI analysis shows investment performance
           • Investment concentration and diversification assessment
           • Recommendations for portfolio optimization
        """
        return risk_text
    
    def _generate_strategic_recommendations(self):
        """Generate strategic recommendations section with financial focus"""
        recommendations = """
        Strategic Financial Recommendations:
        
        Based on the comprehensive financial analysis conducted, the following strategic 
        recommendations are provided:
        
        1. Financial Performance Optimization:
           • Implement cost control measures based on expense ratio analysis
           • Optimize pricing strategies to improve profit margins
           • Enhance cash flow management through improved working capital practices
        
        2. Risk Management Strategies:
           • Maintain optimal debt-to-equity ratios for financial stability
           • Diversify investment portfolio to reduce concentration risk
           • Implement robust cash flow forecasting and monitoring systems
        
        3. Growth and Investment Opportunities:
           • Identify high-ROI investment opportunities based on performance analysis
           • Develop strategic initiatives to improve return on assets
           • Consider expansion opportunities in high-margin business areas
        
        4. Operational Excellence:
           • Streamline operational processes to improve efficiency ratios
           • Implement data-driven decision-making frameworks
           • Develop key performance indicators for continuous monitoring
        
        5. Long-term Financial Planning:
           • Establish sustainable growth targets based on historical performance
           • Develop comprehensive financial forecasting models
           • Implement regular financial health assessments and reporting
        """
        return recommendations

def main():
    """Main function to run the complete financial analysis pipeline"""
    print("="*60)
    print("PYTHON FINANCIAL REPORT GENERATOR")
    print("="*60)
    
    # Get CSV file path from user
    csv_path = input("\nEnter the path to your CSV file: ").strip()
    
    if not os.path.exists(csv_path):
        print("ERROR File not found. Please check the path and try again.")
        return
    
    # Initialize data analyzer
    analyzer = DataAnalyzer(csv_path)
    
    # Load and inspect data
    if not analyzer.load_data():
        return
    
    data_info = analyzer.inspect_data()
    
    # Clean data and calculate financial metrics
    cleaned_df = analyzer.clean_data()
    
    # Display financial summary
    print(analyzer.get_financial_summary())
    
    # Perform EDA
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Descriptive statistics for numeric columns
    if analyzer.numeric_columns:
        print("\nDescriptive Statistics for Financial Variables:")
        print(cleaned_df[analyzer.numeric_columns].describe())
        
        # Additional statistics
        print("\nAdditional Financial Statistics:")
        for col in analyzer.numeric_columns:
            print(f"\n{col}:")
            print(f"  Skewness: {cleaned_df[col].skew():.3f}")
            print(f"  Kurtosis: {cleaned_df[col].kurtosis():.3f}")
            print(f"  Range: {cleaned_df[col].max() - cleaned_df[col].min():.3f}")
            if col in ['Revenue', 'Expenses', 'Profit']:
                print(f"  Coefficient of Variation: {(cleaned_df[col].std() / cleaned_df[col].mean() * 100):.2f}%")
    
    # Frequency counts for categorical variables
    if analyzer.categorical_columns:
        print("\nFrequency Counts for Categorical Variables:")
        for col in analyzer.categorical_columns[:5]:  # Limit to first 5
            print(f"\n{col}:")
            print(cleaned_df[col].value_counts().head(10))
    
    # Create visualizations
    visualizer = DataVisualizer(cleaned_df, analyzer.numeric_columns, 
                               analyzer.categorical_columns, analyzer.datetime_columns)
    figures = visualizer.create_visualizations()
    
    # Generate report
    output_path = f"Financial_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    report_generator = ReportGenerator(data_info, {}, figures, output_path, analyzer.financial_metrics)
    report_generator.generate_report()
    
    print(f"\nDONE Financial analysis complete! Your report has been saved as: {output_path}")
    print("\nThe comprehensive financial report contains:")
    print("• Executive summary with key financial highlights")
    print("• Financial performance overview and metrics")
    print("• Data quality assessment")
    print("• Comprehensive financial analysis with visualizations")
    print("• Financial risk assessment")
    print("• Strategic financial recommendations")
    print("\nKey Financial Insights Generated:")
    if analyzer.financial_metrics and 'overall' in analyzer.financial_metrics:
        overall = analyzer.financial_metrics['overall']
        print(f"  • Total Revenue: ${overall['total_revenue']:,.2f}")
        print(f"  • Total Profit: ${overall['total_profit']:,.2f}")
        print(f"  • Average Profit Margin: {overall['avg_profit_margin']}%")
        if overall['avg_roi']:
            print(f"  • Average ROI: {overall['avg_roi']:.2f}%")

if __name__ == "__main__":
    main()
