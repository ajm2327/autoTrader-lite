import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd 
import numpy as np
from typing import Optional, List, Tuple
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_price_data(data: pd.DataFrame, ticker: str, title_suffix: str="")-> plt.Figure:
    """
    plots basic OHLC data
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    #plot price
    ax1.plot(data.index, data['Close'], label='Close', linewidth=1.5, color='#2E86AB')
    ax1.plot(data.index, data['High'], label='High', alpha=0.7, linewidth=0.8, color='#A23B72')
    ax1.plot(data.index, data['Low'], label='Low', alpha=0.7, linewidth=0.8, color='#F18F01')

    ax1.set_title(f'{ticker} Price Data {" -" + title_suffix if title_suffix else ""}',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Volume plot
    colors = ['green' if close >= open_price else 'red' 
              for close, open_price in zip(data['Close'], data['Open'])]
    ax2.bar(data.index, data['Volume'], color=colors, alpha=0.6, width=0.8)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Data', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Format x-axis 
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1,len(data)//10)))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
def plot_with_moving_averages(data:pd.DataFrame, ticker:str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(data.index, data['Close'], label='Close Price', linewidth=2, color='#2E86AB')

    # moving averages
    if 'SMA_20' in data.columns:
        ax.plot(data.index, data['SMA_20'], label='SMA 20', linewidth=1.5,
        color = '#F18F01', alpha=0.8)
    if 'SMA_50' in data.columns: 
        ax.plot(data.index, data['SMA_50'], label='SMA 50', linewidth = 1.5,
                color='#A23B72', alpha=0.8)
        
    if 'SMA_200' in data.columns:
        ax.plot(data.index, data['SMA_200'], label='SMA 200', linewidth=1.5,
                color='#C73E1D', alpha=0.8)
        
    ax.set_title(f'{ticker} Price with Moving Averages', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price $', fontsize = 12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data) // 10)))
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig

def plot_technical_indicators(data: pd.DataFrame, ticker: str) -> plt.Figure:
    indicators = []
    if 'RSI' in data.columns:
        indicators.append('RSI')
    if 'MACD' in data.columns and 'Signal_Line' in data.columns:
        indicators.append('MACD')
    if 'Upper_Band' in data.columns and 'Lower_Band' in data.columns:
        indicators.append('Bollinger')

    n_plots = len(indicators) + 1
    fig, axes = plt.subplots(n_plots, 1, figsize =(12, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # prie chart w bollinger bands
    ax = axes[plot_idx]
    ax.plot(data.index, data['Close'], label = 'Close Price', linewidth=2, color='#2E86AB')

    if 'Upper_Band' in data.columns and 'Lower_Band' in data.columns:
        ax.plot(data.index, data['Upper_Band'], label='Upper BB',
                linewidth=1, color='#F18F01', alpha=0.7)
        ax.plot(data.index, data['Lower_Band'], label='Lower BB',
                linewidth = 1, color='#F18F01', alpha=0.7)
        ax.fill_between(data.index, data['Upper_Band'], data['Lower_Band'],
                        alpha=0.1, color='#F18F01')
        
    ax.set_title(f'{ticker} Price with Technical Indicators', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # RSI
    if 'RSI' in data.columns:
        ax = axes[plot_idx]
        ax.plot(data.index, data['RSI'], label='RSI', linewidth=1.5, color = '#A23B72')
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
        ax.set_ylabel('RSI', fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # MACD
    if 'MACD' in data.columns and 'Signal_Line' in data.columns:
        ax = axes[plot_idx]
        ax.plot(data.index, data['MACD'], label='MACD', linewidth=1.5, color='#2E86AB')
        ax.plot(data.index, data['Signal_Line'], label='Signal Line',
                linewidth=1.5, color='#F18F01')
        ax.bar(data.index, data['MACD'] - data['Signal_Line'],
               label='Histogram', alpha=0.3, color='green', width=0.8)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_ylabel('MACD', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx +=1

    # format x axis for bottom plot
    axes[-1].set_xlabel('Date', fontsize=12)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data) // 10)))
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig

def plot_data_comparison(data_before: pd.DataFrame, data_after: pd.DataFrame,
                         ticker: str) -> plt.Figure:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15, 10))

    # before price data
    ax1.plot(data_before.index, data_before['Close'], label='Close', color='#2E86AB')
    ax1.set_title(f'{ticker} - Raw Price Data', fontweight = 'bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # After price with indicators
    ax2.plot(data_after.index, data_after['Close'], label='CLose', color='#2E68AB')
    if 'SMA_20' in data_after.columns:
        ax2.plot(data_after.index, data_after['SMA_20'], label='SMA 20',
                 color='#F18F01', alpha=0.8)
    ax2.set_title(f'{ticker} - Price with Moving Averages', fontweight = 'bold')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # data info comparison
    ax3.text(0.1, 0.8, f"Original Data SHape: {data_before.shape}",
             transform = ax3.transAxes, fontsize=12)
    ax3.text(0.1, 0.7, f"Original Columns: {len(data_before.columns)}",
             transform=ax3.transAxes, fontsize=12)
    ax3.text(0.1, 0.6, f"Columns: {', '.join(data_before.columns)}",
             transform=ax3.transAxes, fontsize=10, wrap=True)
    ax3.set_title('Original Data Info', fontweight='bold')
    ax3.axis('off')

    ax4.text(0.1, 0.8, f'Enhanced Data Shape: {data_after.shape}',
             transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.7, f"Enhanced Columns: {len(data_after.columns)}",
             transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.6, f"New Indicators: {', '.join([col for col in data_after.columns if col not in data_before.columns])}",
             transform=ax4.transAxes, fontsize=10, wrap=True)
    ax4.set_title('Enhanced Data Info', fontweight='bold')
    ax4.axis('off')

    plt.tight_layout()
    return fig

def plot_volume_analysis(data: pd.DataFrame, ticker: str) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,8), sharex=True)
    
    # VOlume with ma
    ax1.bar(data.index, data['Volume'], alpha=0.6, color='#2E86AB', width=0.8)

    # Add volume w ma
    vol_ma = data['Volume'].rolling(window=20).mean()
    ax1.plot(data.index, vol_ma, label='Volume MA (20)', color='red', linewidth=2)
    ax1.set_title(f'{ticker} Volume Analysis', fontsize=14, fontweight = 'bold')
    ax1.set_ylabel('Volume', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Relative Volume (current vs average)
    relative_vol = data['Volume'] / vol_ma
    colors = ['green' if rv > 1.5 else 'orange' if rv > 1 else 'red' for rv in relative_vol]

    ax2.bar(data.index, relative_vol, color=colors, alpha=0.7, width=0.8)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label= 'Average Volume')
    ax2.axhline(y=1.5, color='green', linestyle='--', alpha=0.7, label='High Volume (1.5x)')
    ax2.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Very High Volume (2x)')

    ax2.set_ylabel('Relative Volume', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data) // 10)))
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig

def save_plot(fig: plt.Figure, filename: str, dpi: int = 300) -> None:

    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Plot saved ad {filename}")


def show_data_summary(data: pd.DataFrame, ticker: str) -> None:
    print(f"\n{'='*50}")
    print(f"Data Summary For {ticker}")
    print(f"{'='*50}\n")
    print(f"Shape: {data.shape}")
    print(f"Date Range: {data.index.min()} to {data.index.max()}")
    print(f"Columns: {', '.join(data.columns)}")
    print(f"\nPrice range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print(f"Volume range: {data['Volume'].min():,} - {data['Volume'].max():,}")

    if 'RSI' in data.columns:
        print(f"RSI Range: {data['RSI'].min():.2f} - {data['RSI'].max():.2f}")

    print(f"\nMissing Values:\n{data.isnull().sum()}")
    print(f"{'='*50}\n")
    