import numpy as np
import matplotlib.pyplot as plt

# Define pesticide concentration based on decision variables
def compute_C(t, x, C0, k, T):
    return sum(x[i] * C0 * np.exp(-k * (t - i * T)) for i in range(len(x)) if t >= i * T)

# Define the agricultural ecosystem ODE
def agricultural_ecosystem_step(P, H, B, t, params, x):
    C_t = compute_C(t, x, params['C0'], params['k'], params['T'])

    dPdt = (params['r1'] / (1 + (params['C_w1'] * C_t)**2)) * (
        P * (1 - P / params['K_P']) - params['w1'] * H * P
    )
    dHdt = (params['r2'] * H * (1 - H / params['K_H'])) - params['beta'] * B * H - params['C_w2'] * C_t * H
    dBdt = (params['r3'] * B * (1 - B / params['K_B']))

    if np.isnan(dPdt) or np.isnan(dHdt) or np.isnan(dBdt):
        print(f"NaN detected at time {t:.2f}: dPdt={dPdt}, dHdt={dHdt}, dBdt={dBdt}")

    return dPdt, dHdt, dBdt

# Numerical integration using fixed time steps
def solve_ecosystem_with_harvest(y0, t_span, dt, params, x, harvest_times, harvest_rate):
    P, H, B = y0
    t_values = np.arange(t_span[0], t_span[1] + dt, dt)
    results = {'t': [], 'P': [], 'H': [], 'B': []}

    for t in t_values:
        # Record values at this step
        results['t'].append(t)
        results['P'].append(P)
        results['H'].append(H)
        results['B'].append(B)

        # Check if harvest event occurs
        if any(abs(t - ht) < dt / 2 for ht in harvest_times):
            print(f"Harvest event at time {t:.2f}")
            print(f"Before harvest: P={P:.2f}, H={H:.2f}, B={B:.2f}")
            P *= (1 - harvest_rate)  # Apply harvest
            print(f"After harvest: P={P:.2f}, H={H:.2f}, B={B:.2f}")

        # Compute derivatives
        dPdt, dHdt, dBdt = agricultural_ecosystem_step(P, H, B, t, params, x)

        # Update variables using Euler's method
        P += dPdt * dt
        H += dHdt * dt
        B += dBdt * dt

        # Prevent negative values
        P = max(P, 0)
        H = max(H, 0)
        B = max(B, 0)

    return results

# Parameters
params = {
    'r1': 10,       # Growth rate of crops
    'r2': 0.5,       # Growth rate of pests
    'r3': 0.5,       # Growth rate of beneficial insects
    'w1': 0.05,      # Interaction coefficient between pests and crops
    'w2': 1,         # Interaction coefficient between crops and pests
    'w3': 2,         # Interaction coefficient between pests and beneficial insects
    'beta': 0.003,    # Impact of beneficial insects on pests
    'G': 0.01,       # Seasonal harvesting rate
    'C0': 1.0,       # Initial pesticide concentration
    'C_w1': 1,       # Chemical impact on nutrients
    'C_w2': 0.05,     # Chemical impact on pests
    'k': 0.03,       # Decay rate for pesticide
    'T': 13,         # Pesticide application frequency (quarterly in weeks)
    'K_P': 2000,      # Carrying capacity for crops
    'K_H': 15,
    'K_B': 20
}

# Initial conditions and settings
y0 = [100, 30, 10]  # Initial populations [P, H, B]
t_span = (0, 260)   # 5 years in weeks
dt = 1              # Time step (1 week)
x = [1.5] * 20      # Constant pesticide input for 20 quarters
harvest_times = [39 + k * 52 for k in range(5)]  # Harvest at 3rd quarter each year
harvest_rate = 0.85  # 85% of crops are harvested

# Solve the system
results = solve_ecosystem_with_harvest(y0, t_span, dt, params, x, harvest_times, harvest_rate)

# Plot the results
plt.figure(figsize=(14, 8))
plt.plot(results['t'], results['P'], label='Crops (P)', color='green', linewidth=2)
plt.plot(results['t'], results['H'], label='Pests (H)', color='red', linewidth=2)
plt.plot(results['t'], results['B'], label='Beneficial Insects (B)', color='blue', linewidth=2)
plt.xlabel('Time (weeks)', fontsize=14)
plt.ylabel('Population', fontsize=14)
plt.title('Population Dynamics with Harvest Events Over 5 Years', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.5)
plt.xlim(0, 250)  # Set x-axis limit to 0-250
plt.tight_layout()
plt.show()
