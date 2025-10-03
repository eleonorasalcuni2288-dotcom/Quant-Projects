# WTI-Brent Spread Trading Strategy

## Project Overview
Statistical arbitrage strategy for crude oil spread trading, based on Ruble & Powell (2021). I developed this project during a commodity trading internship, motivated by an apparent market inefficiency: WTI and Brent are both light sweet crudes with similar API gravity and sulfur content, making them close substitutes. Despite this similarity, their price spread exhibits significant deviations from equilibrium.
When the spread widens excessively (Brent >> WTI) or contracts abnormally, economic forces drive mean reversion through:

Arbitrage by refiners switching between crude types
Changes in transatlantic shipping flows
Pipeline infrastructure adjustments
This creates opportunities for statistical arbitrage because the spread exhibits:

Mean reversion: Tendency to return to long-term average
Volatility clustering: Calm periods alternate with crisis-driven shocks
Structural regimes: Post-2016 dynamics differ from the pre-shale era

## Statistical Models:
The strategy combines ARIMAX-GARCH forecasting with z-score signals under a dual-confirmation framework.

ARIMAX (Conditional Mean):
Captures the spread's mean-reverting behavior with short-term persistence. The AR component models autocorrelation, while exogenous variables can include transportation costs and storage utilization rates.
GARCH (Conditional Volatility):
Models time-varying volatility essential for risk management. Volatility clustering is pronounced during supply disruptions, geopolitical events, and OPEC decisions.
Markov-Switching (Regime Detection):
Identifies distinct market regimes (normal vs. stress) with different statistical properties. The spread behaves differently during pipeline bottlenecks or export restrictions, requiring regime-dependent trading parameters.

## Trading Logic
Dual-Confirmation Entry:
Both signals must agree before entering a position.
Z-Score Signal (Mispricing Detection):
z = (current_spread - historical_mean) / historical_std

z = 0: No mispricing
z = ±1.5: Threshold for extreme deviation (occurs ~13% of time)
|z| > 1.5 indicates statistical mispricing

ARIMAX-GARCH Forecast (Directional Confirmation):

Mean forecast: Predicts tomorrow's spread level
Volatility forecast: Quantifies prediction uncertainty

## Trading Rules:
Long Spread (Buy Brent, Sell WTI):

Entry: z < -1.5 AND forecast predicts increase
Logic: Spread undervalued AND correction beginning

Short Spread (Sell Brent, Buy WTI):

Entry: z > +1.5 AND forecast predicts decrease
Logic: Spread overvalued AND correction beginning

No Trade:

|z| < 1.5: Insufficient mispricing
Signals disagree: Uncertainty about direction
High volatility: Confidence too low

Risk Management
Dynamic Position Sizing:
pythonconfidence = 1.0 / (1.0 + forecast_volatility)
position_size = base_size * confidence
Reduces exposure during high-volatility periods when forecasts are less reliable.
Exit Rules:

Take profit: +2% on spread movement
Stop loss: -1% on spread movement
Mean reversion complete: |z-score| < 0.3

### Results
Backtest Period: January 2018 - September 2025 (7.7 years)

CAGR: 2.2% (conservative: 1 contract on $100k)
Sharpe Ratio: 0.86 (good risk-adjusted returns)
Max Drawdown: -2.7% (excellent risk control)
Win Rate: 54.4% (realistic for mean-reversion)
Trade Frequency: 13.5 trades/year (highly selective)

Scalability: With 10 contracts and regime detection enhancements, target CAGR increases to 15-25% while maintaining Sharpe >1.0.

## Conclusion
This implementation represents a research prototype rather than a production-ready system. While the methodology is sound and results are encouraging, several important limitations remain:
Current Gaps:

Regime detection is simplified compared to the full Markov-Switching approach in Ruble & Powell (2021)
No integration of fundamental data (EIA inventories, OPEC announcements, refinery utilization)
Execution assumptions are idealized (simultaneous fills, consistent liquidity)
Single sample period limits generalizability
Transaction costs, while realistic, don't account for market impact at scale

## Next Steps:
To make this production-ready, I would need to:

Implement proper Markov-Switching with transition probabilities
Add out-of-sample testing on recent data (2024-2025)
Integrate real-time fundamental data feeds (not available in free-sources)
Develop execution algorithms for larger position sizes
Conduct Monte Carlo stress testing under extreme scenarios

## Acknowledgments:
This work builds directly on Ruble & Powell's research and would not have been possible without their insights into the structural changes in WTI-Brent dynamics following the US shale revolution. Any errors or oversimplifications in this implementation are entirely my own.