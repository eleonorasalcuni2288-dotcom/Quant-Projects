# Options Greeks Computation

This project emerged from a comprehensive notions I've learned in risk management, financial modeling and Derivatives courses, as an attempt to connect theoretical concepts with practical implementation. Working with the Black-Scholes model and options pricing theory, I developed a Python system that calculates option prices and their sensitivities, known as the Greeks.

The application handles basic portfolio management by aggregating multiple option positions and computing their combined risk metrics. It includes Monte Carlo simulation for Value-at-Risk analysis and implements some hedging strategies like delta hedging. Then, with AI support, I created both a command-line interface for comprehensive analysis and a web-based interface using Streamlit for interactive parameter adjustment.

## Usage

To run the analysis, install the required dependencies with `pip install -r requirements.txt`, then execute `python main.py` for the full analysis or `streamlit run app.py` for the interactive interface.

## Results

The system successfully computes portfolio Greeks, generates basic risk metrics, and produces visualizations of option strategies. While the implementation focuses on standard models and approaches, it demonstrates the practical application of financial mathematics concepts learned in academic study. The generated charts and risk analyses provide useful insights into option behavior and portfolio dynamics.

The technical pdf documentation in the docs folder contains detailed mathematical and structural explanation of the underlying methodology.
