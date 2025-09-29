"""
Options Greeks Calculator & Hedging Simulator
=============================================

A comprehensive Python implementation for options pricing, Greeks calculation,
and hedging strategy simulation with portfolio management capabilities.

Author: [Your Name]
Date: September 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("OPTIONS GREEKS CALCULATOR & HEDGING SIMULATOR")
print("=" * 60)

class BlackScholesModel:
    """
    Black-Scholes option pricing model with Greeks calculation.
    
    This class implements the complete Black-Scholes framework for
    European options pricing and risk management.
    """
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        """Calculate d1 parameter for Black-Scholes formula."""
        if T <= 0:
            return 0
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        """Calculate d2 parameter for Black-Scholes formula."""
        return BlackScholesModel.d1(S, K, T, r, sigma) - sigma*np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        """Calculate European call option price."""
        if T <= 0:
            return max(S - K, 0)
        
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        """Calculate European put option price."""
        if T <= 0:
            return max(K - S, 0)
        
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def delta(S, K, T, r, sigma, option_type='call'):
        """Calculate option delta (price sensitivity to underlying)."""
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S, K, T, r, sigma):
        """Calculate option gamma (delta sensitivity to underlying)."""
        if T <= 0:
            return 0
        
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def theta(S, K, T, r, sigma, option_type='call'):
        """Calculate option theta (time decay)."""
        if T <= 0:
            return 0
        
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if option_type == 'call':
            term2 = r * K * np.exp(-r*T) * norm.cdf(d2)
            return (term1 - term2) / 365  # Per day
        else:
            term2 = r * K * np.exp(-r*T) * norm.cdf(-d2)
            return (term1 + term2) / 365  # Per day
    
    @staticmethod
    def vega(S, K, T, r, sigma):
        """Calculate option vega (volatility sensitivity)."""
        if T <= 0:
            return 0
        
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change
    
    @staticmethod
    def rho(S, K, T, r, sigma, option_type='call'):
        """Calculate option rho (interest rate sensitivity)."""
        if T <= 0:
            return 0
        
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            return K * T * np.exp(-r*T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100


class Option:
    """
    Option contract class with pricing and Greeks calculation.
    """
    
    def __init__(self, S, K, T, r, sigma, option_type='call', quantity=1):
        self.S = S  # Spot price
        self.K = K  # Strike price
        self.T = T  # Time to expiry (years)
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility
        self.option_type = option_type.lower()
        self.quantity = quantity
        self.entry_price = self.price
        self.entry_spot = S
        
    @property
    def price(self):
        """Current option price."""
        if self.option_type == 'call':
            return BlackScholesModel.call_price(self.S, self.K, self.T, self.r, self.sigma)
        else:
            return BlackScholesModel.put_price(self.S, self.K, self.T, self.r, self.sigma)
    
    @property
    def delta(self):
        """Option delta."""
        return BlackScholesModel.delta(self.S, self.K, self.T, self.r, self.sigma, self.option_type)
    
    @property
    def gamma(self):
        """Option gamma."""
        return BlackScholesModel.gamma(self.S, self.K, self.T, self.r, self.sigma)
    
    @property
    def theta(self):
        """Option theta."""
        return BlackScholesModel.theta(self.S, self.K, self.T, self.r, self.sigma, self.option_type)
    
    @property
    def vega(self):
        """Option vega."""
        return BlackScholesModel.vega(self.S, self.K, self.T, self.r, self.sigma)
    
    @property
    def rho(self):
        """Option rho."""
        return BlackScholesModel.rho(self.S, self.K, self.T, self.r, self.sigma, self.option_type)
    
    def update_spot(self, new_spot):
        """Update underlying spot price."""
        self.S = new_spot
    
    def update_time(self, time_passed_days):
        """Update time to expiry."""
        self.T = max(0, self.T - time_passed_days/365)
    
    def pnl(self):
        """Calculate P&L relative to entry."""
        return (self.price - self.entry_price) * self.quantity
    
    def get_greeks_summary(self):
        """Return dictionary of all Greeks."""
        return {
            'Price': self.price,
            'Delta': self.delta,
            'Gamma': self.gamma,
            'Theta': self.theta,
            'Vega': self.vega,
            'Rho': self.rho,
            'P&L': self.pnl()
        }


class OptionsPortfolio:
    """
    Options portfolio management and risk analysis.
    """
    
    def __init__(self):
        self.positions = []
        self.hedge_position = 0  # Stock hedge position
        
    def add_position(self, option):
        """Add option position to portfolio."""
        self.positions.append(option)
        
    def remove_position(self, index):
        """Remove option position from portfolio."""
        if 0 <= index < len(self.positions):
            self.positions.pop(index)
    
    def clear_portfolio(self):
        """Clear all positions."""
        self.positions.clear()
        self.hedge_position = 0
    
    def update_market_data(self, new_spot, time_passed_days=0):
        """Update all positions with new market data."""
        for position in self.positions:
            position.update_spot(new_spot)
            if time_passed_days > 0:
                position.update_time(time_passed_days)
    
    def portfolio_greeks(self):
        """Calculate portfolio-level Greeks."""
        if not self.positions:
            return {greek: 0.0 for greek in ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']}
        
        portfolio_delta = sum(pos.delta * pos.quantity for pos in self.positions)
        portfolio_gamma = sum(pos.gamma * pos.quantity for pos in self.positions)
        portfolio_theta = sum(pos.theta * pos.quantity for pos in self.positions)
        portfolio_vega = sum(pos.vega * pos.quantity for pos in self.positions)
        portfolio_rho = sum(pos.rho * pos.quantity for pos in self.positions)
        
        return {
            'Delta': portfolio_delta,
            'Gamma': portfolio_gamma,
            'Theta': portfolio_theta,
            'Vega': portfolio_vega,
            'Rho': portfolio_rho
        }
    
    def total_pnl(self):
        """Calculate total portfolio P&L."""
        options_pnl = sum(pos.pnl() for pos in self.positions)
        
        # Add stock hedge P&L if any
        if hasattr(self, '_initial_spot') and self.hedge_position != 0:
            current_spot = self.positions[0].S if self.positions else self._initial_spot
            stock_pnl = self.hedge_position * (current_spot - self._initial_spot)
            return options_pnl + stock_pnl
        
        return options_pnl
    
    def portfolio_summary(self):
        """Generate comprehensive portfolio summary."""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for i, pos in enumerate(self.positions):
            greeks = pos.get_greeks_summary()
            data.append({
                'Position': i+1,
                'Type': pos.option_type.upper(),
                'Strike': pos.K,
                'Expiry': pos.T,
                'Quantity': pos.quantity,
                'Spot': pos.S,
                **greeks
            })
        
        df = pd.DataFrame(data)
        
        # Add portfolio totals
        portfolio_greeks = self.portfolio_greeks()
        totals = {
            'Position': 'TOTAL',
            'Type': '',
            'Strike': '',
            'Expiry': '',
            'Quantity': sum(pos.quantity for pos in self.positions),
            'Spot': '',
            'Price': sum(pos.price * pos.quantity for pos in self.positions),
            **portfolio_greeks,
            'P&L': self.total_pnl()
        }
        
        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
        return df


class HedgingSimulator:
    """
    Advanced hedging strategies simulation.
    """
    
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.hedging_log = []
        
    def delta_hedge(self, target_delta=0.0):
        """
        Implement delta hedging strategy.
        
        Args:
            target_delta: Target portfolio delta (default: 0 for delta neutral)
        """
        current_delta = self.portfolio.portfolio_greeks()['Delta']
        hedge_needed = target_delta - current_delta
        
        # Update hedge position
        self.portfolio.hedge_position += hedge_needed
        
        # Log hedging action
        current_spot = self.portfolio.positions[0].S if self.portfolio.positions else 100
        self.hedging_log.append({
            'strategy': 'Delta Hedge',
            'spot_price': current_spot,
            'portfolio_delta': current_delta,
            'hedge_adjustment': hedge_needed,
            'total_hedge_position': self.portfolio.hedge_position
        })
        
        if not hasattr(self.portfolio, '_initial_spot'):
            self.portfolio._initial_spot = current_spot
        
        return hedge_needed
    
    def gamma_scalping(self, price_move):
        """
        Implement gamma scalping strategy.
        
        Args:
            price_move: Recent price movement
        """
        portfolio_gamma = self.portfolio.portfolio_greeks()['Gamma']
        
        # Gamma scalping adjustment: buy/sell based on realized vs implied vol
        gamma_adjustment = 0.5 * portfolio_gamma * (price_move ** 2)
        
        # Update hedge position
        self.portfolio.hedge_position += gamma_adjustment
        
        current_spot = self.portfolio.positions[0].S if self.portfolio.positions else 100
        self.hedging_log.append({
            'strategy': 'Gamma Scalping',
            'spot_price': current_spot,
            'price_move': price_move,
            'gamma_adjustment': gamma_adjustment,
            'total_hedge_position': self.portfolio.hedge_position
        })
        
        return gamma_adjustment
    
    def get_hedging_summary(self):
        """Return hedging history as DataFrame."""
        return pd.DataFrame(self.hedging_log)


def monte_carlo_var(portfolio, confidence_level=0.95, time_horizon=1, num_simulations=10000):
    """
    Calculate Value at Risk using Monte Carlo simulation.
    
    Args:
        confidence_level: VaR confidence level (default: 95%)
        time_horizon: Time horizon in days
        num_simulations: Number of Monte Carlo simulations
    """
    if not portfolio.positions:
        return 0, 0
    
    # Current portfolio value
    current_value = sum(pos.price * pos.quantity for pos in portfolio.positions)
    
    # Market parameters
    current_spot = portfolio.positions[0].S
    vol = np.mean([pos.sigma for pos in portfolio.positions])
    
    # Simulate price paths
    dt = time_horizon / 365
    price_scenarios = np.random.lognormal(
        np.log(current_spot) + (-0.5 * vol**2) * dt,
        vol * np.sqrt(dt),
        num_simulations
    )
    
    # Calculate portfolio values under each scenario
    portfolio_values = []
    for price in price_scenarios:
        scenario_value = 0
        for pos in portfolio.positions:
            # Update position with scenario price
            if pos.option_type == 'call':
                scenario_price = BlackScholesModel.call_price(
                    price, pos.K, max(0, pos.T - dt), pos.r, pos.sigma
                )
            else:
                scenario_price = BlackScholesModel.put_price(
                    price, pos.K, max(0, pos.T - dt), pos.r, pos.sigma
                )
            scenario_value += scenario_price * pos.quantity
        
        portfolio_values.append(scenario_value)
    
    # Calculate VaR
    portfolio_values = np.array(portfolio_values)
    pnl_distribution = portfolio_values - current_value
    
    var_percentile = (1 - confidence_level) * 100
    var = np.percentile(pnl_distribution, var_percentile)
    
    # Expected Shortfall (Conditional VaR)
    expected_shortfall = np.mean(pnl_distribution[pnl_distribution <= var])
    
    return var, expected_shortfall


def stress_testing(portfolio):
    """
    Perform stress testing on the portfolio.
    """
    if not portfolio.positions:
        return pd.DataFrame()
    
    current_spot = portfolio.positions[0].S
    current_vol = np.mean([pos.sigma for pos in portfolio.positions])
    
    # Define stress scenarios
    scenarios = [
        {'name': 'Bull Market (+20%)', 'spot_shock': 1.2, 'vol_shock': 0.8},
        {'name': 'Bear Market (-20%)', 'spot_shock': 0.8, 'vol_shock': 1.3},
        {'name': 'Market Crash (-30%)', 'spot_shock': 0.7, 'vol_shock': 1.5},
        {'name': 'Vol Spike (+50%)', 'spot_shock': 1.0, 'vol_shock': 1.5},
        {'name': 'Vol Crush (-50%)', 'spot_shock': 1.0, 'vol_shock': 0.5},
        {'name': 'Combined Stress', 'spot_shock': 0.75, 'vol_shock': 1.4}
    ]
    
    results = []
    base_value = sum(pos.price * pos.quantity for pos in portfolio.positions)
    
    for scenario in scenarios:
        stressed_value = 0
        for pos in portfolio.positions:
            stressed_spot = current_spot * scenario['spot_shock']
            stressed_vol = current_vol * scenario['vol_shock']
            
            if pos.option_type == 'call':
                stressed_price = BlackScholesModel.call_price(
                    stressed_spot, pos.K, pos.T, pos.r, stressed_vol
                )
            else:
                stressed_price = BlackScholesModel.put_price(
                    stressed_spot, pos.K, pos.T, pos.r, stressed_vol
                )
            
            stressed_value += stressed_price * pos.quantity
        
        pnl = stressed_value - base_value
        pnl_pct = (pnl / abs(base_value)) * 100 if base_value != 0 else 0
        
        results.append({
            'Scenario': scenario['name'],
            'Spot_Change': f"{(scenario['spot_shock']-1)*100:+.0f}%",
            'Vol_Change': f"{(scenario['vol_shock']-1)*100:+.0f}%",
            'Portfolio_Value': stressed_value,
            'P&L': pnl,
            'P&L_%': pnl_pct
        })
    
    return pd.DataFrame(results)


def geometric_brownian_motion(S0, mu, sigma, T, dt, paths=1):
    """
    Simulate asset prices using Geometric Brownian Motion.
    
    Args:
        S0: Initial price
        mu: Drift (expected return)
        sigma: Volatility
        T: Time horizon
        dt: Time step
        paths: Number of simulation paths
    """
    n_steps = int(T / dt)
    times = np.linspace(0, T, n_steps + 1)
    
    # Generate random shocks
    dW = np.random.normal(0, np.sqrt(dt), (paths, n_steps))
    
    # Initialize price paths
    prices = np.zeros((paths, n_steps + 1))
    prices[:, 0] = S0
    
    # Simulate price evolution
    for i in range(n_steps):
        prices[:, i + 1] = prices[:, i] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * dW[:, i]
        )
    
    return times, prices


def main_demo():
    """
    Comprehensive demonstration of the options analytics system.
    """
    
    # Create sample portfolio
    portfolio = OptionsPortfolio()
    
    # Add some option positions
    call_option = Option(S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type='call', quantity=10)
    put_option = Option(S=100, K=95, T=0.25, r=0.05, sigma=0.20, option_type='put', quantity=5)
    
    portfolio.add_position(call_option)
    portfolio.add_position(put_option)
    
    print("\n1. PORTFOLIO SUMMARY")
    print("-" * 30)
    portfolio_df = portfolio.portfolio_summary()
    print(portfolio_df.round(4))
    
    print("\n2. PORTFOLIO GREEKS")
    print("-" * 30)
    greeks = portfolio.portfolio_greeks()
    for greek, value in greeks.items():
        print(f"{greek:>8}: {value:>8.4f}")
    
    print(f"\nTotal P&L: ${portfolio.total_pnl():>8.2f}")
    
    # Risk Analysis
    print("\n3. RISK ANALYSIS")
    print("-" * 30)
    var_95, expected_shortfall = monte_carlo_var(portfolio, confidence_level=0.95)
    print(f"95% VaR (1-day): ${var_95:.2f}")
    print(f"Expected Shortfall: ${expected_shortfall:.2f}")
    
    print("\n4. STRESS TESTING")
    print("-" * 30)
    stress_results = stress_testing(portfolio)
    print(stress_results.round(2))
    
    # Hedging Simulation
    hedging_sim = HedgingSimulator(portfolio)
    
    print("\n5. HEDGING SIMULATION")
    print("-" * 30)
    
    # Simulate some market moves and hedging
    price_moves = [2, -1.5, 3, -2, 1]
    
    for i, move in enumerate(price_moves):
        new_spot = call_option.S + move
        portfolio.update_market_data(new_spot)
        
        # Delta hedge
        hedge_adjustment = hedging_sim.delta_hedge()
        
        print(f"Day {i+1}: Spot=${new_spot:.2f}, "
              f"Portfolio Delta={portfolio.portfolio_greeks()['Delta']:.3f}, "
              f"Hedge Adjustment={hedge_adjustment:.3f}")
    
    print("\n6. MONTE CARLO ANALYSIS")
    print("-" * 30)
    
    # Run Monte Carlo simulation
    times, price_paths = geometric_brownian_motion(
        S0=100, mu=0.05, sigma=0.20, T=0.25, dt=1/252, paths=1000
    )
    
    final_prices = price_paths[:, -1]
    print(f"Simulated price statistics:")
    print(f"Mean final price: ${np.mean(final_prices):.2f}")
    print(f"Std final price: ${np.std(final_prices):.2f}")
    print(f"Min final price: ${np.min(final_prices):.2f}")
    print(f"Max final price: ${np.max(final_prices):.2f}")
    
    # Calculate portfolio value distribution
    portfolio_values = []
    for final_price in final_prices[:100]:  # Sample subset for speed
        call_value = max(final_price - 105, 0) * 10
        put_value = max(95 - final_price, 0) * 5
        portfolio_values.append(call_value + put_value)
    
    print(f"\nPortfolio value at expiry:")
    print(f"Mean: ${np.mean(portfolio_values):.2f}")
    print(f"Std: ${np.std(portfolio_values):.2f}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    # Run the demonstration
    main_demo()

# Imposta lo stile per i grafici
plt.style.use('default')
sns.set_palette("husl")

# Aggiungi queste nuove funzioni DOPO la classe HedgingSimulator:

def plot_greeks_surface(option_params):
    """Plot Greeks surface vs spot price and time."""
    S_range = np.linspace(80, 120, 50)
    T_range = np.linspace(0.01, 0.5, 30)
    S_grid, T_grid = np.meshgrid(S_range, T_range)
    
    # Calculate Greeks surface
    delta_surface = np.zeros_like(S_grid)
    gamma_surface = np.zeros_like(S_grid)
    
    for i in range(len(T_range)):
        for j in range(len(S_range)):
            delta_surface[i, j] = BlackScholesModel.delta(
                S_grid[i, j], option_params['K'], T_grid[i, j], 
                option_params['r'], option_params['sigma'], option_params['type']
            )
            gamma_surface[i, j] = BlackScholesModel.gamma(
                S_grid[i, j], option_params['K'], T_grid[i, j], 
                option_params['r'], option_params['sigma']
            )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Delta surface
    im1 = ax1.contourf(S_grid, T_grid, delta_surface, levels=20, cmap='RdYlBu')
    ax1.set_xlabel('Spot Price')
    ax1.set_ylabel('Time to Expiry')
    ax1.set_title('Delta Surface')
    plt.colorbar(im1, ax=ax1)
    
    # Gamma surface
    im2 = ax2.contourf(S_grid, T_grid, gamma_surface, levels=20, cmap='plasma')
    ax2.set_xlabel('Spot Price')
    ax2.set_ylabel('Time to Expiry')
    ax2.set_title('Gamma Surface')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('greeks_surface.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_payoff_diagram(portfolio, spot_range=None):
    """Plot portfolio payoff diagram."""
    if not portfolio.positions:
        print("No positions to plot")
        return
    
    current_spot = portfolio.positions[0].S
    if spot_range is None:
        spot_range = np.linspace(current_spot * 0.7, current_spot * 1.3, 100)
    
    # Calculate payoff at expiration (intrinsic value only)
    intrinsic_values = []
    current_values = []
    
    for spot in spot_range:
        intrinsic_pnl = 0
        current_pnl = 0
        
        for pos in portfolio.positions:
            # Intrinsic value at expiration
            if pos.option_type == 'call':
                intrinsic = max(spot - pos.K, 0)
            else:
                intrinsic = max(pos.K - spot, 0)
            intrinsic_pnl += (intrinsic - pos.entry_price) * pos.quantity
            
            # Current option value
            if pos.option_type == 'call':
                current_price = BlackScholesModel.call_price(spot, pos.K, pos.T, pos.r, pos.sigma)
            else:
                current_price = BlackScholesModel.put_price(spot, pos.K, pos.T, pos.r, pos.sigma)
            current_pnl += (current_price - pos.entry_price) * pos.quantity
        
        intrinsic_values.append(intrinsic_pnl)
        current_values.append(current_pnl)
    
    plt.figure(figsize=(12, 8))
    plt.plot(spot_range, intrinsic_values, 'b-', linewidth=2, label='Payoff at Expiration')
    plt.plot(spot_range, current_values, 'r--', linewidth=2, label='Current P&L')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.axvline(x=current_spot, color='green', linestyle='--', alpha=0.7, label=f'Current Spot (${current_spot})')
    
    plt.xlabel('Underlying Price at Expiry')
    plt.ylabel('Portfolio P&L ($)')
    plt.title('Portfolio Payoff Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('payoff_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_greeks_vs_spot(portfolio):
    """Plot Greeks sensitivity to spot price."""
    if not portfolio.positions:
        print("No positions to plot")
        return
    
    current_spot = portfolio.positions[0].S
    spot_range = np.linspace(current_spot * 0.8, current_spot * 1.2, 50)
    
    deltas = []
    gammas = []
    thetas = []
    vegas = []
    
    for spot in spot_range:
        # Update spot prices temporarily
        for pos in portfolio.positions:
            original_spot = pos.S
            pos.S = spot
        
        greeks = portfolio.portfolio_greeks()
        deltas.append(greeks['Delta'])
        gammas.append(greeks['Gamma'] * 100)  # Scale for visibility
        thetas.append(greeks['Theta'] * 365)  # Annual theta
        vegas.append(greeks['Vega'] * 10)     # Scale for visibility
        
        # Restore original spot prices
        for pos in portfolio.positions:
            pos.S = current_spot
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.plot(spot_range, deltas, 'b-', linewidth=2)
    ax1.set_title('Portfolio Delta vs Spot Price')
    ax1.set_xlabel('Spot Price')
    ax1.set_ylabel('Delta')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=current_spot, color='red', linestyle='--', alpha=0.7)
    
    ax2.plot(spot_range, gammas, 'g-', linewidth=2)
    ax2.set_title('Portfolio Gamma vs Spot Price (×100)')
    ax2.set_xlabel('Spot Price')
    ax2.set_ylabel('Gamma (×100)')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=current_spot, color='red', linestyle='--', alpha=0.7)
    
    ax3.plot(spot_range, thetas, 'r-', linewidth=2)
    ax3.set_title('Portfolio Theta vs Spot Price (Annual)')
    ax3.set_xlabel('Spot Price')
    ax3.set_ylabel('Theta (Annual)')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=current_spot, color='red', linestyle='--', alpha=0.7)
    
    ax4.plot(spot_range, vegas, 'm-', linewidth=2)
    ax4.set_title('Portfolio Vega vs Spot Price (×10)')
    ax4.set_xlabel('Spot Price')
    ax4.set_ylabel('Vega (×10)')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=current_spot, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('greeks_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_monte_carlo_simulation(portfolio, num_simulations=1000):
    """Visualize Monte Carlo price simulation and portfolio values."""
    if not portfolio.positions:
        return
    
    current_spot = portfolio.positions[0].S
    vol = np.mean([pos.sigma for pos in portfolio.positions])
    dt = 1/252  # Daily
    T = 30      # 30 days
    
    # Generate price paths
    times, price_paths = geometric_brownian_motion(
        S0=current_spot, mu=0.05, sigma=vol, T=T/365, dt=dt, paths=num_simulations
    )
    
    # Calculate portfolio values for each path
    portfolio_paths = np.zeros_like(price_paths)
    
    for i, path in enumerate(price_paths):
        for j, price in enumerate(path):
            portfolio_value = 0
            for pos in portfolio.positions:
                if pos.option_type == 'call':
                    option_value = BlackScholesModel.call_price(
                        price, pos.K, max(0.01, pos.T - j*dt), pos.r, pos.sigma
                    )
                else:
                    option_value = BlackScholesModel.put_price(
                        price, pos.K, max(0.01, pos.T - j*dt), pos.r, pos.sigma
                    )
                portfolio_value += option_value * pos.quantity
            portfolio_paths[i, j] = portfolio_value
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot sample price paths
    time_days = np.arange(len(times))
    for i in range(min(50, num_simulations)):  # Show only 50 paths for clarity
        ax1.plot(time_days, price_paths[i], alpha=0.3, color='blue', linewidth=0.5)
    
    ax1.plot(time_days, np.mean(price_paths, axis=0), 'r-', linewidth=2, label='Mean Path')
    ax1.set_title('Monte Carlo Price Simulation (Sample Paths)')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Stock Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot portfolio value distribution at end
    final_portfolio_values = portfolio_paths[:, -1]
    ax2.hist(final_portfolio_values, bins=50, alpha=0.7, color='green', density=True)
    ax2.axvline(x=np.mean(final_portfolio_values), color='red', linestyle='--', 
                label=f'Mean: ${np.mean(final_portfolio_values):.2f}')
    ax2.axvline(x=np.percentile(final_portfolio_values, 5), color='orange', linestyle='--',
                label=f'5th Percentile: ${np.percentile(final_portfolio_values, 5):.2f}')
    ax2.set_title('Portfolio Value Distribution (30 days)')
    ax2.set_xlabel('Portfolio Value ($)')
    ax2.set_ylabel('Probability Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('monte_carlo_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return final_portfolio_values


def plot_hedging_performance(hedging_sim):
    """Plot hedging performance over time."""
    hedging_df = hedging_sim.get_hedging_summary()
    
    if hedging_df.empty:
        print("No hedging data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot spot price and hedge position
    ax1.plot(range(len(hedging_df)), hedging_df['spot_price'], 'b-', linewidth=2, label='Spot Price')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(len(hedging_df)), hedging_df['total_hedge_position'], 'r--', linewidth=2, label='Hedge Position')
    
    ax1.set_ylabel('Spot Price', color='blue')
    ax1_twin.set_ylabel('Hedge Position', color='red')
    ax1.set_title('Hedging Strategy: Spot Price vs Hedge Position')
    ax1.grid(True, alpha=0.3)
    
    # Plot portfolio delta over time
    ax2.plot(range(len(hedging_df)), hedging_df['portfolio_delta'], 'g-', linewidth=2, label='Portfolio Delta')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Delta Neutral')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Portfolio Delta')
    ax2.set_title('Portfolio Delta Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hedging_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


def interactive_option_calculator():
    """Simple interactive option calculator."""
    print("\n" + "="*50)
    print("INTERACTIVE OPTION CALCULATOR")
    print("="*50)
    
    try:
        print("Enter option parameters:")
        S = float(input("Spot price: $"))
        K = float(input("Strike price: $"))
        T = float(input("Time to expiry (years): "))
        r = float(input("Risk-free rate (%): ")) / 100
        sigma = float(input("Volatility (%): ")) / 100
        option_type = input("Option type (call/put): ").lower()
        
        # Create option
        option = Option(S, K, T, r, sigma, option_type)
        
        print(f"\nOPTION VALUATION RESULTS:")
        print(f"Option Type: {option_type.upper()}")
        print(f"Price: ${option.price:.4f}")
        print(f"Delta: {option.delta:.4f}")
        print(f"Gamma: {option.gamma:.4f}")
        print(f"Theta: ${option.theta:.4f}/day")
        print(f"Vega: ${option.vega:.4f}")
        print(f"Rho: ${option.rho:.4f}")
        
        # Ask if user wants to see sensitivity analysis
        show_analysis = input("\nShow sensitivity analysis? (y/n): ").lower()
        if show_analysis == 'y':
            spot_range = np.linspace(S*0.8, S*1.2, 50)
            prices = []
            deltas = []
            
            for spot in spot_range:
                if option_type == 'call':
                    price = BlackScholesModel.call_price(spot, K, T, r, sigma)
                else:
                    price = BlackScholesModel.put_price(spot, K, T, r, sigma)
                delta = BlackScholesModel.delta(spot, K, T, r, sigma, option_type)
                
                prices.append(price)
                deltas.append(delta)
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(spot_range, prices, 'b-', linewidth=2)
            plt.axvline(x=S, color='red', linestyle='--', alpha=0.7, label=f'Current Spot (${S})')
            plt.axvline(x=K, color='green', linestyle='--', alpha=0.7, label=f'Strike (${K})')
            plt.xlabel('Spot Price')
            plt.ylabel('Option Price')
            plt.title(f'{option_type.upper()} Option Price vs Spot')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(spot_range, deltas, 'r-', linewidth=2)
            plt.axvline(x=S, color='red', linestyle='--', alpha=0.7, label=f'Current Spot (${S})')
            plt.axvline(x=K, color='green', linestyle='--', alpha=0.7, label=f'Strike (${K})')
            plt.xlabel('Spot Price')
            plt.ylabel('Delta')
            plt.title(f'{option_type.upper()} Option Delta vs Spot')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('interactive_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
    except ValueError:
        print("Invalid input. Please enter numerical values.")
    except KeyboardInterrupt:
        print("\nCalculation interrupted.")


# Modifica la funzione main_demo per includere le visualizzazioni:

def main_demo_with_plots():
    """
    Enhanced demonstration with visualizations.
    """
    
    # Create sample portfolio
    portfolio = OptionsPortfolio()
    
    # Add some option positions
    call_option = Option(S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type='call', quantity=10)
    put_option = Option(S=100, K=95, T=0.25, r=0.05, sigma=0.20, option_type='put', quantity=5)
    
    portfolio.add_position(call_option)
    portfolio.add_position(put_option)
    
    print("\n1. PORTFOLIO SUMMARY")
    print("-" * 30)
    portfolio_df = portfolio.portfolio_summary()
    print(portfolio_df.round(4))
    
    print("\n2. PORTFOLIO GREEKS")
    print("-" * 30)
    greeks = portfolio.portfolio_greeks()
    for greek, value in greeks.items():
        print(f"{greek:>8}: {value:>8.4f}")
    
    print(f"\nTotal P&L: ${portfolio.total_pnl():>8.2f}")
    
    # Risk Analysis
    print("\n3. RISK ANALYSIS")
    print("-" * 30)
    var_95, expected_shortfall = monte_carlo_var(portfolio, confidence_level=0.95)
    print(f"95% VaR (1-day): ${var_95:.2f}")
    print(f"Expected Shortfall: ${expected_shortfall:.2f}")
    
    print("\n4. STRESS TESTING")
    print("-" * 30)
    stress_results = stress_testing(portfolio)
    print(stress_results.round(2))
    
    # Hedging Simulation
    hedging_sim = HedgingSimulator(portfolio)
    
    print("\n5. HEDGING SIMULATION")
    print("-" * 30)
    
    # Simulate some market moves and hedging
    price_moves = [2, -1.5, 3, -2, 1]
    
    for i, move in enumerate(price_moves):
        new_spot = call_option.S + move
        portfolio.update_market_data(new_spot)
        
        # Delta hedge
        hedge_adjustment = hedging_sim.delta_hedge()
        
        print(f"Day {i+1}: Spot=${new_spot:.2f}, "
              f"Portfolio Delta={portfolio.portfolio_greeks()['Delta']:.3f}, "
              f"Hedge Adjustment={hedge_adjustment:.3f}")
    
    print("\n6. GENERATING VISUALIZATIONS...")
    print("-" * 30)
    
    # Create all plots
    print("Creating payoff diagram...")
    plot_payoff_diagram(portfolio)
    
    print("Creating Greeks sensitivity analysis...")
    plot_greeks_vs_spot(portfolio)
    
    print("Creating Monte Carlo simulation...")
    final_values = plot_monte_carlo_simulation(portfolio, num_simulations=500)
    
    print("Creating hedging performance chart...")
    plot_hedging_performance(hedging_sim)
    
    print("Creating Greeks surface plot...")
    option_params = {
        'K': 105, 'r': 0.05, 'sigma': 0.20, 'type': 'call'
    }
    plot_greeks_surface(option_params)
    
    print(f"\nAll plots saved as PNG files in current directory!")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Offer interactive calculator
    run_interactive = input("\nRun interactive option calculator? (y/n): ").lower()
    if run_interactive == 'y':
        interactive_option_calculator()


if __name__ == "__main__":
    # Run the enhanced demonstration with plots
    main_demo_with_plots()