# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
# from pymoo.algorithms.moo.nsga3 import NSGA3
# from pymoo.core.problem import ElementwiseProblem
# from pymoo.optimize import minimize
# from pymoo.util.ref_dirs import get_reference_directions

# # Define constants and parameters
# r1, r2, r3 = 0.5, 0.3, 0.2  # Growth rates
# K_P = 100  # Carrying capacity of crops
# w1, w2, w3 = 0.1, 0.2, 0.3  # Interaction coefficients
# beta, C_w1, C_w2 = 0.1, 0.05, 0.1  # Pesticide effects
# G, C0, k = 0.85, 1.0, 0.1  # Harvest disturbance rate, pesticide concentration, decay
# T = 1.0  # Time interval per quarter
# crop_price, pesticide_unit_cost = 10.0, 2.0  # Costs
# w1_env, w2_env = 0.5, 0.5  # Environmental weights

# # Define system dynamics
# def system_dynamics(t, y, x):
#     P, H, B = y
#     C = sum(x_i * C0 * np.exp(-k * (t - i * T)) for i, x_i in enumerate(x) if t >= i * T)
#     dP_dt = (r1 / (1 + (C_w1 * C)**2)) * (P * (1 - P / K_P) - w1 * H * P)
#     dH_dt = r2 * H * (1 - H / (w2 * P)) - beta * B * H - C_w2 * C * H
#     dB_dt = r3 * B * (1 - B / (w3 * H))
#     return [dP_dt, dH_dt, dB_dt]

# # Define the multi-objective problem
# class PesticideOptimizationProblem(ElementwiseProblem):
#     def __init__(self):
#         super().__init__(n_var=16, n_obj=2, n_constr=0, xl=0, xu=4)

#     def _evaluate(self, x, out, *args, **kwargs):
#         sol = solve_ivp(system_dynamics, [0, 16], [50, 10, 5], args=(x,), t_eval=np.arange(0, 17))
#         P, H, B = sol.y

#         # Economic objective
#         harvest_revenue = 0.85 * np.sum(P[::4]) * crop_price
#         total_pesticide_cost = np.sum(x) * pesticide_unit_cost
#         F1 = harvest_revenue - total_pesticide_cost

#         # Normalize F1
#         max_harvest_revenue = crop_price * 0.85 * np.sum(P)
#         F1_normalized = F1 / max_harvest_revenue

#         # Environmental objective
#         S_n = 1 / (1 + 0.1 * np.sum(x))  # Soil fertility index
#         p_P, p_H, p_B = P / (P + H + B), H / (P + H + B), B / (P + H + B)

#         # Avoid log of zero or negative values
#         p_P = np.clip(p_P, 1e-10, 1)
#         p_H = np.clip(p_H, 1e-10, 1)
#         p_B = np.clip(p_B, 1e-10, 1)

#         H_n = -np.nansum([p_P * np.log(p_P), p_H * np.log(p_H), p_B * np.log(p_B)])  # Handle NaNs
#         max_H_n = np.log(3)
#         F2_normalized = w1_env * (S_n / 1) + w2_env * (H_n / max_H_n)

#         out["F"] = [-F1_normalized, -F2_normalized]

# # Initialize the problem
# problem = PesticideOptimizationProblem()
# ref_dirs = get_reference_directions("das-dennis", n_dim=2, n_points=100)

# # Define the NSGA-III algorithm
# algorithm = NSGA3(ref_dirs=ref_dirs)

# # Solve the problem using NSGA-III
# res = minimize(
#     problem,
#     algorithm,
#     ('n_gen', 200),
#     verbose=True
# )

# # Extract the results
# F = res.F  # Objective values
# X = res.X  # Decision variables

# # Plot all solutions
# plt.figure(figsize=(10, 8))
# plt.scatter(-F[:, 0], -F[:, 1], c='gray', alpha=0.5, label="All Solutions")

# # Identify non-Pareto solutions (dominated solutions)
# def is_dominated(f, front, eps=1e-6):
#     for pf in front:
#         if all(pf <= f + eps) and any(pf < f - eps):
#             return True
#     return False

# pareto_front = []
# non_pareto = []

# for i, f in enumerate(F):
#     if not is_dominated(f, F):
#         pareto_front.append((f, X[i], i))
#     else:
#         non_pareto.append((f, X[i], i))

# # Debugging: Print the count of Pareto and Non-Pareto solutions
# print(f"Number of Pareto solutions: {len(pareto_front)}")
# print(f"Number of Non-Pareto solutions: {len(non_pareto)}")

# # Split objectives and decisions for plotting
# pareto_f = np.array([pf[0] for pf in pareto_front])
# non_pareto_f = np.array([npf[0] for npf in non_pareto]) if non_pareto else np.array([])

# # Plot Pareto front and non-Pareto points
# plt.scatter(-pareto_f[:, 0], -pareto_f[:, 1], c='blue', label="Pareto Front", edgecolors='black', s=100)
# if non_pareto_f.size > 0:  # Only plot if non_pareto_f is not empty
#     plt.scatter(-non_pareto_f[:, 0], -non_pareto_f[:, 1], c='red', label="Non-Pareto Points", alpha=0.8)

# # Annotate each point with its index
# for idx, f in enumerate(F):
#     plt.annotate(idx, (-f[0], -f[1]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

# plt.title("All Solutions with Pareto and Non-Pareto Points")
# plt.xlabel("Economic Objective (F1)")
# plt.ylabel("Environmental Objective (F2)")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

# # Plot some decision variables corresponding to Pareto front
# plt.figure(figsize=(8, 6))
# for i, (_, x, idx) in enumerate(pareto_front[:10]):  # Plot for a subset of Pareto solutions
#     plt.plot(x, label=f'Pareto Solution {idx}')

# plt.title("Decision Variables for Pareto Front Solutions")
# plt.xlabel("Time (quarters)")
# plt.ylabel("Pesticide Input")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

# # Plot some decision variables corresponding to non-Pareto points
# if non_pareto:  # Only plot if non-Pareto points exist
#     plt.figure(figsize=(8, 6))
#     for i, (_, x, idx) in enumerate(non_pareto[:10]):  # Plot for a subset of Non-Pareto solutions
#         plt.plot(x, label=f'Non-Pareto Solution {idx}')

#     plt.title("Decision Variables for Non-Pareto Points")
#     plt.xlabel("Time (quarters)")
#     plt.ylabel("Pesticide Input")
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()
#     plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

# Define constants and parameters
r1, r2, r3 = 0.5, 0.3, 0.2  # Growth rates
K_P = 100  # Carrying capacity of crops
w1, w2, w3 = 0.1, 0.2, 0.3  # Interaction coefficients
beta, C_w1, C_w2 = 0.1, 0.05, 0.1  # Pesticide effects
G, C0, k = 0.85, 1.0, 0.1  # Harvest disturbance rate, pesticide concentration, decay
T = 1.0  # Time interval per quarter
crop_price, pesticide_unit_cost = 10.0, 2.0  # Costs
w1_env, w2_env = 0.5, 0.5  # Environmental weights

# Define system dynamics
def system_dynamics(t, y, x):
    P, H, B = y
    C = sum(x_i * C0 * np.exp(-k * (t - i * T)) for i, x_i in enumerate(x) if t >= i * T)
    dP_dt = (r1 / (1 + (C_w1 * C)**2)) * (P * (1 - P / K_P) - w1 * H * P)
    dH_dt = r2 * H * (1 - H / (w2 * P)) - beta * B * H - C_w2 * C * H
    dB_dt = r3 * B * (1 - B / (w3 * H))
    return [dP_dt, dH_dt, dB_dt]

# Define the multi-objective problem
class PesticideOptimizationProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=16, n_obj=2, n_constr=0, xl=0, xu=4)

    def _evaluate(self, x, out, *args, **kwargs):
        sol = solve_ivp(system_dynamics, [0, 16], [50, 10, 5], args=(x,), t_eval=np.arange(0, 17))
        P, H, B = sol.y

        # Economic objective
        harvest_revenue = 0.85 * np.sum(P[::4]) * crop_price
        total_pesticide_cost = np.sum(x) * pesticide_unit_cost
        F1 = harvest_revenue - total_pesticide_cost

        # Normalize F1
        max_harvest_revenue = crop_price * 0.85 * np.sum(P)
        F1_normalized = F1 / max_harvest_revenue

        # Environmental objective
        S_n = 1 / (1 + 0.1 * np.sum(x))  # Soil fertility index
        p_P, p_H, p_B = P / (P + H + B), H / (P + H + B), B / (P + H + B)

        # Avoid log of zero or negative values
        p_P = np.clip(p_P, 1e-10, 1)
        p_H = np.clip(p_H, 1e-10, 1)
        p_B = np.clip(p_B, 1e-10, 1)

        H_n = -np.nansum([p_P * np.log(p_P), p_H * np.log(p_H), p_B * np.log(p_B)])  # Handle NaNs
        max_H_n = np.log(3)
        F2_normalized = w1_env * (S_n / 1) + w2_env * (H_n / max_H_n)

        out["F"] = [-F1_normalized, -F2_normalized]

# Initialize the problem
problem = PesticideOptimizationProblem()
ref_dirs = get_reference_directions("das-dennis", n_dim=2, n_points=100)

# Define the NSGA-III algorithm
algorithm = NSGA3(ref_dirs=ref_dirs)

n_gen = 200
# Solve the problem using NSGA-III
res = minimize(
    problem,
    algorithm,
    ('n_gen', n_gen),
    verbose=True
)

# Extract the results
F = res.F  # Objective values
X = res.X  # Decision variables

# Plot all solutions
plt.figure(figsize=(10, 8))
plt.scatter(-F[:, 0], -F[:, 1], c='gray', alpha=0.5, label="All Solutions")

# Identify non-Pareto solutions (dominated solutions)
def is_dominated(f, front, eps=1e-6):
    for pf in front:
        if all(pf <= f + eps) and any(pf < f - eps):
            return True
    return False

pareto_front = []
non_pareto = []

for i, f in enumerate(F):
    if not is_dominated(f, F):
        pareto_front.append((f, X[i], i))
    else:
        non_pareto.append((f, X[i], i))

# Debugging: Print the count of Pareto and Non-Pareto solutions
print(f"Number of Pareto solutions: {len(pareto_front)}")
print(f"Number of Non-Pareto solutions: {len(non_pareto)}")

# Split objectives and decisions for plotting
pareto_f = np.array([pf[0] for pf in pareto_front])
non_pareto_f = np.array([npf[0] for npf in non_pareto]) if non_pareto else np.array([])

# Plot Pareto front and non-Pareto points
plt.scatter(-pareto_f[:, 0], -pareto_f[:, 1], c='blue', label="Pareto Front", alpha=0.7, edgecolors='none', s=100)
if non_pareto_f.size > 0:  # Only plot if non_pareto_f is not empty
    plt.scatter(-non_pareto_f[:, 0], -non_pareto_f[:, 1], c='red', label="Non-Pareto Points", alpha=0.5, edgecolors='none')

plt.title("All Solutions with Pareto and Non-Pareto Points")
plt.xlabel("Economic Objective (F1)")
plt.ylabel("Environmental Objective (F2)")
plt.legend(loc='upper left', fontsize=8, ncol=2)
plt.grid()
plt.tight_layout()
plt.savefig(f"pareto_solutions_n_gen_{res.algorithm.n_gen}.png")
plt.show()

# Plot some decision variables corresponding to Pareto front
plt.figure(figsize=(8, 6))
for i, (_, x, idx) in enumerate(pareto_front):  # Plot for a subset of Pareto solutions
    plt.plot(x, label=f'Pareto Solution {idx}')

plt.title("Decision Variables for Pareto Front Solutions")
plt.xlabel("Time (quarters")
plt.ylabel("Pesticide Input")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"pareto_decision_variables_n_gen_{res.algorithm.n_gen}.png")
plt.show()

# Plot some decision variables corresponding to non-Pareto points
if non_pareto:  # Only plot if non-Pareto points exist
    plt.figure(figsize=(8, 6))
    for i, (_, x, idx) in enumerate(non_pareto[:10]):  # Plot for a subset of Non-Pareto solutions
        plt.plot(x, label=f'Non-Pareto Solution {idx}')

    plt.title("Decision Variables for Non-Pareto Points")
    plt.xlabel("Time (quarters")
    plt.ylabel("Pesticide Input")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
