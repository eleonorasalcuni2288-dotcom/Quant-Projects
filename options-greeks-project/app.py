import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

# Impostazioni pagina
st.set_page_config(
    page_title="Options Greeks Calculator & Hedging Simulator",
    page_icon="📈",
    layout="wide"
)

# Classe BlackScholes (copia dal main.py)
class BlackScholesModel:
    @staticmethod
    def d1(S, K, T, r, sigma):
        if T <= 0:
            return 0
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        return BlackScholesModel.d1(S, K, T, r, sigma) - sigma*np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        if T <= 0:
            return max(S - K, 0)
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        if T <= 0:
            return max(K - S, 0)
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def delta(S, K, T, r, sigma, option_type='call'):
        if T <= 0:
            return 1.0 if (S > K and option_type == 'call') else (0.0 if option_type == 'call' else -1.0 if S < K else 0.0)
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S, K, T, r, sigma):
        if T <= 0:
            return 0
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def theta(S, K, T, r, sigma, option_type='call'):
        if T <= 0:
            return 0
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type == 'call':
            term2 = r * K * np.exp(-r*T) * norm.cdf(d2)
            return (term1 - term2) / 365
        else:
            term2 = r * K * np.exp(-r*T) * norm.cdf(-d2)
            return (term1 + term2) / 365
    
    @staticmethod
    def vega(S, K, T, r, sigma):
        if T <= 0:
            return 0
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100
    
    @staticmethod
    def rho(S, K, T, r, sigma, option_type='call'):
        if T <= 0:
            return 0
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        if option_type == 'call':
            return K * T * np.exp(-r*T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100

# Titolo e descrizione
st.title("📈 Options Greeks Calculator & Hedging Simulator")
st.markdown("**Interactive web application for options pricing, Greeks calculation, and portfolio management**")

# Sidebar per i parametri
st.sidebar.header("Option Parameters")

# Input parametri
S = st.sidebar.number_input("Spot Price ($)", value=100.0, min_value=0.1, step=1.0)
K = st.sidebar.number_input("Strike Price ($)", value=105.0, min_value=0.1, step=1.0)
T = st.sidebar.number_input("Time to Expiry (years)", value=0.25, min_value=0.001, max_value=5.0, step=0.01)
r = st.sidebar.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0, max_value=20.0, step=0.1) / 100
sigma = st.sidebar.number_input("Volatility (%)", value=20.0, min_value=0.1, max_value=200.0, step=0.1) / 100
option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])

# Calcola option price e Greeks
option_type_lower = option_type.lower()

if option_type_lower == 'call':
    price = BlackScholesModel.call_price(S, K, T, r, sigma)
else:
    price = BlackScholesModel.put_price(S, K, T, r, sigma)

delta = BlackScholesModel.delta(S, K, T, r, sigma, option_type_lower)
gamma = BlackScholesModel.gamma(S, K, T, r, sigma)
theta = BlackScholesModel.theta(S, K, T, r, sigma, option_type_lower)
vega = BlackScholesModel.vega(S, K, T, r, sigma)
rho = BlackScholesModel.rho(S, K, T, r, sigma, option_type_lower)

# Layout principale con colonne
col1, col2 = st.columns(2)

with col1:
    st.subheader("Option Valuation")
    
    # Metrics cards
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("Option Price", f"${price:.4f}")
        st.metric("Delta", f"{delta:.4f}")
    
    with metrics_col2:
        st.metric("Gamma", f"{gamma:.4f}")
        st.metric("Theta", f"${theta:.4f}")
    
    with metrics_col3:
        st.metric("Vega", f"${vega:.4f}")
        st.metric("Rho", f"${rho:.4f}")

with col2:
    st.subheader("Moneyness & Time Value")
    
    intrinsic = max(S - K, 0) if option_type_lower == 'call' else max(K - S, 0)
    time_value = price - intrinsic
    moneyness = S / K
    
    st.metric("Intrinsic Value", f"${intrinsic:.4f}")
    st.metric("Time Value", f"${time_value:.4f}")
    st.metric("Moneyness (S/K)", f"{moneyness:.4f}")

# Tabs per diverse analisi
tab1, tab2, tab3, tab4 = st.tabs(["📊 Sensitivity Analysis", "💹 Payoff Diagram", "🎯 Monte Carlo", "⚖️ Portfolio"])

with tab1:
    st.subheader("Greeks Sensitivity Analysis")
    
    # Range di spot prices
    spot_range = np.linspace(S * 0.7, S * 1.3, 50)
    
    # Calcola Greeks per ogni spot price
    prices = []
    deltas = []
    gammas = []
    thetas = []
    vegas = []
    
    for spot in spot_range:
        if option_type_lower == 'call':
            p = BlackScholesModel.call_price(spot, K, T, r, sigma)
        else:
            p = BlackScholesModel.put_price(spot, K, T, r, sigma)
        
        prices.append(p)
        deltas.append(BlackScholesModel.delta(spot, K, T, r, sigma, option_type_lower))
        gammas.append(BlackScholesModel.gamma(spot, K, T, r, sigma))
        thetas.append(BlackScholesModel.theta(spot, K, T, r, sigma, option_type_lower))
        vegas.append(BlackScholesModel.vega(spot, K, T, r, sigma))
    
    # Plot con matplotlib
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    ax1.plot(spot_range, prices, 'b-', linewidth=2)
    ax1.axvline(x=S, color='red', linestyle='--', alpha=0.7, label=f'Current Spot')
    ax1.axvline(x=K, color='green', linestyle='--', alpha=0.7, label=f'Strike')
    ax1.set_title(f'{option_type} Price vs Spot')
    ax1.set_xlabel('Spot Price')
    ax1.set_ylabel('Option Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(spot_range, deltas, 'r-', linewidth=2)
    ax2.axvline(x=S, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(x=K, color='green', linestyle='--', alpha=0.7)
    ax2.set_title('Delta vs Spot')
    ax2.set_xlabel('Spot Price')
    ax2.set_ylabel('Delta')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(spot_range, gammas, 'g-', linewidth=2)
    ax3.axvline(x=S, color='red', linestyle='--', alpha=0.7)
    ax3.axvline(x=K, color='green', linestyle='--', alpha=0.7)
    ax3.set_title('Gamma vs Spot')
    ax3.set_xlabel('Spot Price')
    ax3.set_ylabel('Gamma')
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(spot_range, thetas, 'm-', linewidth=2)
    ax4.axvline(x=S, color='red', linestyle='--', alpha=0.7)
    ax4.axvline(x=K, color='green', linestyle='--', alpha=0.7)
    ax4.set_title('Theta vs Spot')
    ax4.set_xlabel('Spot Price')
    ax4.set_ylabel('Theta')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Payoff Diagram")
    
    # Payoff calculation
    payoff_range = np.linspace(S * 0.6, S * 1.4, 100)
    payoffs_expiry = []
    payoffs_current = []
    
    for spot in payoff_range:
        # Payoff at expiry
        if option_type_lower == 'call':
            payoff_exp = max(spot - K, 0) - price
            current_price = BlackScholesModel.call_price(spot, K, T, r, sigma)
        else:
            payoff_exp = max(K - spot, 0) - price
            current_price = BlackScholesModel.put_price(spot, K, T, r, sigma)
        
        payoffs_expiry.append(payoff_exp)
        payoffs_current.append(current_price - price)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(payoff_range, payoffs_expiry, 'b-', linewidth=2, label='Payoff at Expiry')
    ax.plot(payoff_range, payoffs_current, 'r--', linewidth=2, label='Current P&L')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.axvline(x=S, color='green', linestyle='--', alpha=0.7, label=f'Current Spot (${S})')
    ax.axvline(x=K, color='orange', linestyle='--', alpha=0.7, label=f'Strike (${K})')
    
    ax.set_xlabel('Underlying Price')
    ax.set_ylabel('Profit/Loss')
    ax.set_title(f'{option_type} Option Payoff Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Breakeven point
    if option_type_lower == 'call':
        breakeven = K + price
        st.info(f"**Breakeven Point:** ${breakeven:.2f}")
    else:
        breakeven = K - price
        st.info(f"**Breakeven Point:** ${breakeven:.2f}")

with tab3:
    st.subheader("Monte Carlo Simulation")
    
    num_sims = st.slider("Number of Simulations", 100, 5000, 1000)
    time_horizon = st.slider("Time Horizon (days)", 1, 90, 30)
    
    if st.button("Run Monte Carlo Simulation"):
        dt = time_horizon / 365
        
        # Generate random price scenarios
        price_scenarios = np.random.lognormal(
            np.log(S) + (r - 0.5 * sigma**2) * dt,
            sigma * np.sqrt(dt),
            num_sims
        )
        
        # Calculate option values
        final_values = []
        for price_scenario in price_scenarios:
            if option_type_lower == 'call':
                final_value = BlackScholesModel.call_price(
                    price_scenario, K, max(0.001, T - dt), r, sigma
                )
            else:
                final_value = BlackScholesModel.put_price(
                    price_scenario, K, max(0.001, T - dt), r, sigma
                )
            final_values.append(final_value)
        
        final_values = np.array(final_values)
        pnl = final_values - price
        
        # Results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean P&L", f"${np.mean(pnl):.4f}")
        with col2:
            st.metric("Std P&L", f"${np.std(pnl):.4f}")
        with col3:
            st.metric("95% VaR", f"${np.percentile(pnl, 5):.4f}")
        
        # Histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(price_scenarios, bins=50, alpha=0.7, color='blue')
        ax1.axvline(x=S, color='red', linestyle='--', label=f'Current: ${S}')
        ax1.set_title('Simulated Stock Prices')
        ax1.set_xlabel('Stock Price')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        ax2.hist(pnl, bins=50, alpha=0.7, color='green')
        ax2.axvline(x=0, color='red', linestyle='--', label='Breakeven')
        ax2.axvline(x=np.mean(pnl), color='orange', linestyle='--', label=f'Mean: ${np.mean(pnl):.2f}')
        ax2.set_title('P&L Distribution')
        ax2.set_xlabel('Profit/Loss')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)

with tab4:
    st.subheader("Portfolio Management")
    
    # Initialize portfolio in session state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []
    
    # Add position form
    st.write("**Add New Position:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pos_S = st.number_input("Spot", value=S, key="pos_S")
        pos_K = st.number_input("Strike", value=K, key="pos_K")
    with col2:
        pos_T = st.number_input("Expiry", value=T, key="pos_T")
        pos_r = st.number_input("Rate", value=r*100, key="pos_r") / 100
    with col3:
        pos_sigma = st.number_input("Vol", value=sigma*100, key="pos_sigma") / 100
        pos_type = st.selectbox("Type", ["Call", "Put"], key="pos_type")
    with col4:
        pos_qty = st.number_input("Quantity", value=1, step=1, key="pos_qty")
        if st.button("Add Position"):
            # Calculate position details
            pos_type_lower = pos_type.lower()
            if pos_type_lower == 'call':
                pos_price = BlackScholesModel.call_price(pos_S, pos_K, pos_T, pos_r, pos_sigma)
            else:
                pos_price = BlackScholesModel.put_price(pos_S, pos_K, pos_T, pos_r, pos_sigma)
            
            position = {
                'Type': pos_type,
                'Spot': pos_S,
                'Strike': pos_K,
                'Expiry': pos_T,
                'Rate': pos_r,
                'Vol': pos_sigma,
                'Quantity': pos_qty,
                'Price': pos_price,
                'Delta': BlackScholesModel.delta(pos_S, pos_K, pos_T, pos_r, pos_sigma, pos_type_lower),
                'Gamma': BlackScholesModel.gamma(pos_S, pos_K, pos_T, pos_r, pos_sigma),
                'Theta': BlackScholesModel.theta(pos_S, pos_K, pos_T, pos_r, pos_sigma, pos_type_lower),
                'Vega': BlackScholesModel.vega(pos_S, pos_K, pos_T, pos_r, pos_sigma),
                'Rho': BlackScholesModel.rho(pos_S, pos_K, pos_T, pos_r, pos_sigma, pos_type_lower)
            }
            st.session_state.portfolio.append(position)
            st.success(f"Added {pos_type} position!")
    
    # Display portfolio
    if st.session_state.portfolio:
        df = pd.DataFrame(st.session_state.portfolio)
        st.write("**Current Portfolio:**")
        st.dataframe(df.round(4))
        
        # Portfolio Greeks
        total_delta = sum(pos['Delta'] * pos['Quantity'] for pos in st.session_state.portfolio)
        total_gamma = sum(pos['Gamma'] * pos['Quantity'] for pos in st.session_state.portfolio)
        total_theta = sum(pos['Theta'] * pos['Quantity'] for pos in st.session_state.portfolio)
        total_vega = sum(pos['Vega'] * pos['Quantity'] for pos in st.session_state.portfolio)
        total_rho = sum(pos['Rho'] * pos['Quantity'] for pos in st.session_state.portfolio)
        
        st.write("**Portfolio Greeks:**")
        greek_col1, greek_col2, greek_col3, greek_col4, greek_col5 = st.columns(5)
        with greek_col1:
            st.metric("Delta", f"{total_delta:.4f}")
        with greek_col2:
            st.metric("Gamma", f"{total_gamma:.4f}")
        with greek_col3:
            st.metric("Theta", f"${total_theta:.4f}")
        with greek_col4:
            st.metric("Vega", f"${total_vega:.4f}")
        with greek_col5:
            st.metric("Rho", f"${total_rho:.4f}")
        
        # Clear portfolio button
        if st.button("Clear Portfolio"):
            st.session_state.portfolio = []
            st.success("Portfolio cleared!")
    else:
        st.info("No positions in portfolio. Add positions above.")

# Footer
st.markdown("---")
st.markdown("**Options Greeks Calculator & Hedging Simulator** - Built with Streamlit")
st.markdown("*For educational and demonstration purposes only*")