"""
Forecast: the model core of AI Scout v3.

Predicts a player's per-90 output next season at a given club, from his past
seasons, the league and team he moves to, his expected share of the team's
play, and his age. Every term is fitted on past seasons and kept only if it
beats a simpler model in the backtest (see forecast/backtest.py).
"""
