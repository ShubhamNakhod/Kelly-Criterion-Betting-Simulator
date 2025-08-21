# Kelly Criterion Betting Simulator

Simulates bankroll growth under uncertainty using the Kelly criterion.  
Computes optimal bet sizing, expected log-growth, drawdowns, and risk of ruin.  
Supports both coin-toss style bets and sports betting moneylines.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install numpy matplotlib


How to Run
1. Binary bet (coin toss / fixed odds)

- p: win probability

- decimal-odds: payout including stake (2.00 = even money)

'''bash
python kelly_sim.py binary --p 0.55 --decimal-odds 2.0 --bets 200 --sims 10000 --grid 0.25,0.5,1.0

python kelly_sim.py binary --wins 57 --losses 43 --american-odds -110 --bets 300 --sims 20000


2. Sports moneyline

Provide subjective win probability and American odds:
'''bash
python kelly_sim.py sports --p 0.55 --american-odds -110 --bets 300 --sims 20000 --grid 0.25,0.5,1.0

Example Output

Kelly Criterion Betting Simulator
Inputs: p=0.5500, decimal_odds=2.0000 (b=1.0000), full_kelly=0.1000
Simulating 10000 paths x 200 bets, bankroll0=1000.0, ruin_threshold=0.0

mult    f       E[ln(1+R)]   ruin_prob   median_TW   p10_TW  p1_TW   med_maxDD   p90_maxDD
0.25    0.0250  0.002188     0.0000      1548.96     987.57  695.88  22.80%      35.08%
0.50    0.0500  0.003753     0.0000      2118.10     860.51  427.07  41.68%      59.98%
1.00    0.1000  0.005008     0.0000      2722.83     447.37  109.80  69.87%      87.35%


Extra Features

Export to CSV
Add --export results.csv to save the summary table.

Plotting
Add --plot --samples 3 to visualize sample bankroll paths and growth vs fraction.

'''bash
python kelly_sim.py binary --p 0.55 --decimal-odds 2.0 --bets 200 --sims 10000 --grid 0.25,0.5,1.0 --samples 3 --plot

## Theory

### Odds conversion

- Decimal odds: $b = d - 1$ where $d$ is decimal odds  
- American odds $A$:  
  - if $A > 0 \;\Rightarrow\; d = 1 + \tfrac{A}{100}$  
  - if $A < 0 \;\Rightarrow\; d = 1 + \tfrac{100}{|A|}$  

---

### Full Kelly fraction

$$
f^* = \frac{b p - (1 - p)}{b}
$$

---

### Expected log-growth

$$
g(f) = p \ln(1 + f b) + (1 - p) \ln(1 - f)
$$

---

### Interpretation

- Betting more than Kelly ($f > f^*$) reduces long-term growth and increases drawdowns.  
- Fractional Kelly (e.g., half-Kelly) is often used to balance growth and risk.


Example Plots (with --plot)

- Sample bankroll trajectories from Monte Carlo simulations

- Expected log-growth vs bet fraction, showing the optimal Kelly point
​
