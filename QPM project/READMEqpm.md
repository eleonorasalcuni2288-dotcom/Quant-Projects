### Quantitative Portfolio Management with Python
## Course Overview
This repository contains assignments for the Quantitative Portfolio Management with Python course, part of the MSc in Quantitative Finance program, I attended in my exchange semester.
The course bridges theoretical portfolio management concepts with practical implementation, covering:

Data Preparation: Working with high-dimensional stock market data and company characteristics
Modeling: Risk models, return estimation, objective functions, constraints, and transaction costs
Optimization: Mathematical programming approaches and regularization techniques
Backtesting & Simulation: Implementing realistic portfolio simulation frameworks
Machine Learning: Applying econometric and ML models for risk and return prediction

In teams we had the opportunity to develop hands-on programming skills using open-source solvers and Python to implement sophisticated portfolio strategies, emphasizing the subtle but crucial details necessary for successfully translating theoretical concepts into practice.

## First Assignment
In the first assignment, we developed a method to linearize turnover constraints within quadratic programming frameworks, addressing the fundamental trade-off between portfolio rebalancing and transaction costs. The implementation extends the QuadraticProgram class with techniques that transform non-linear turnover constraints into forms compatible with efficient solvers through auxiliary variable introduction and systematic constraint reformulation. This work demonstrates proficiency in mathematical optimization and numerical methods essential for practical portfolio management systems.

## Second Assignment
In the second assignment, we built a comprehensive backtesting framework that evaluates portfolio strategies under realistic market conditions, including both proportional transaction costs and fixed management fees. The implementation features a multi-period simulation engine that tracks portfolio evolution through time, computes turnover metrics, and calculates comprehensive performance statistics including Sharpe ratio, maximum drawdown, and volatility across multiple time horizons. This framework bridges the gap between theoretical optimization and realized performance, revealing how transaction costs impact real-world strategy returns.

## Third Assignment
We implemented an iterative algorithm that constructs portfolios maximizing risk-adjusted returns without requiring explicit risk aversion parameters, solving the classical problem of finding the optimal point on the efficient frontier. The MaxSharpe class employs sequential mean-variance optimization with adaptively adjusted parameters, while incorporating turnover penalties to control trading frequency and maintain practical implementability. Through calibration and backtesting, we demonstrated the real-world trade-offs between theoretical optimality and transaction costs, targeting approximately 100% annual turnover.

## Forth Assignment
In the fourth assignment, We tackled the problem of constructing portfolios that maximize risk-adjusted returns without requiring explicit specification of risk aversion parameters. Unlike standard mean-variance optimization where the investor must choose a target return or risk level, maximum Sharpe ratio optimization seeks the portfolio on the efficient frontier with the steepest capital allocation line. The implementation develops an iterative algorithm that approximates the maximum Sharpe ratio portfolio through sequential solution of mean-variance problems, with the MaxSharpe class encapsulating this iterative logic and adaptively adjusting the risk aversion parameter based on current portfolio characteristics until convergence. Additionally, this assignment addresses the critical practical consideration of controlling portfolio turnover through a penalty approach, calibrating the penalty parameter through backtesting to achieve approximately 100% annual turnover—a level that balances adaptation to changing market conditions against transaction costs.

## Fifth Assignment
In the final assignment, we developed a complete end-to-end quantitative investment system for the Swiss equity market, integrating data processing, machine learning predictions, multi-constraint optimization, and realistic backtesting into a cohesive production-ready framework. The system features dynamic stock selection with quality and liquidity filters, machine learning models for return and risk forecasting trained on fundamental data, and a flexible optimization architecture incorporating sector limits, position bounds, and turnover controls. The comprehensive backtesting engine simulates portfolio performance with realistic costs (1% annual fixed, 0.2% transaction costs) and generates detailed HTML reports with performance visualizations, statistical analysis across multiple periods, and benchmark comparison against the Swiss Performance Index.