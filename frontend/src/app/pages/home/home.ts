import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';

interface StockPrediction {
  symbol: string;
  name: string;
  prediction: 'RISE' | 'FALL';
  price: number;
  confidence: number;
  probability: number;
  signals: string;
}

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './home.html',
  styleUrl: './home.css'
})
export class HomeComponent {
  activeTab: string = 'top-stocks';

  topStocks: StockPrediction[] = [
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      prediction: 'RISE',
      price: 173.72,
      confidence: 85,
      probability: 78,
      signals: 'RSI: Neutral, MACD: Bullish'
    },
    {
      symbol: 'NVDA',
      name: 'NVIDIA Corporation',
      prediction: 'RISE',
      price: 485.09,
      confidence: 92,
      probability: 89,
      signals: 'RSI: Oversold, MACD: Strong Bullish'
    },
    {
      symbol: 'MSFT',
      name: 'Microsoft Corporation',
      prediction: 'RISE',
      price: 378.85,
      confidence: 88,
      probability: 82,
      signals: 'RSI: Neutral, MACD: Bullish'
    },
    {
      symbol: 'GOOGL',
      name: 'Alphabet Inc.',
      prediction: 'RISE',
      price: 142.56,
      confidence: 76,
      probability: 71,
      signals: 'RSI: Neutral, MACD: Neutral'
    },
    {
      symbol: 'AMZN',
      name: 'Amazon.com Inc.',
      prediction: 'RISE',
      price: 151.94,
      confidence: 81,
      probability: 75,
      signals: 'RSI: Oversold, MACD: Bullish'
    },
    {
      symbol: 'TSLA',
      name: 'Tesla Inc.',
      prediction: 'FALL',
      price: 237.49,
      confidence: 67,
      probability: 73,
      signals: 'RSI: Overbought, MACD: Bearish'
    },
    {
      symbol: 'META',
      name: 'Meta Platforms Inc.',
      prediction: 'RISE',
      price: 334.92,
      confidence: 79,
      probability: 74,
      signals: 'RSI: Neutral, MACD: Bullish'
    },
    {
      symbol: 'NFLX',
      name: 'Netflix Inc.',
      prediction: 'FALL',
      price: 492.19,
      confidence: 58,
      probability: 62,
      signals: 'RSI: Overbought, MACD: Neutral'
    },
    {
      symbol: 'ADBE',
      name: 'Adobe Inc.',
      prediction: 'RISE',
      price: 525.76,
      confidence: 83,
      probability: 77,
      signals: 'RSI: Neutral, MACD: Bullish'
    },
    {
      symbol: 'CRM',
      name: 'Salesforce Inc.',
      prediction: 'RISE',
      price: 251.94,
      confidence: 74,
      probability: 69,
      signals: 'RSI: Oversold, MACD: Neutral'
    }
  ];

  getConfidenceColor(confidence: number): string {
    if (confidence >= 80) return 'text-green-600';
    if (confidence >= 60) return 'text-yellow-600';
    return 'text-red-600';
  }

  switchTab(tab: string): void {
    this.activeTab = tab;
  }
}
