## Finalized version for Pareto front

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
import random

# Define constants and parameters
r1, r2, r3 = 0.5, 0.3, 0.2  # Growth rates
K_P = 100  # Carrying capacity of crops
w1, w2, w3 = 0.1, 0.2, 0.3  # Interaction coefficients
beta, C_w1, C_w2 = 0.1, 0.05, 0.1  # Pesticide effects
G, C0, k = 0.85, 1.0, 0.1  # Harvest disturbance rate, pesticide concentration, decay
T = 1.0  # Time interval per quarter
crop_price, pesticide_unit_cost = 10.0, 2.0  # Costs
w1_env, w2_env = 0.5, 0.5  # Environmental weights

xu = 3

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
        super().__init__(n_var=4, n_obj=2, n_constr=0, xl=0, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # sol = solve_ivp(system_dynamics, [0, 16], [50, 10, 5], args=(x,), t_eval=np.arange(0, 17))
        sol = solve_ivp(system_dynamics,[0, 4], [50, 10, 5],  args=(x,), t_eval=np.arange(0, 5))

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
ref_dirs = get_reference_directions("das-dennis", n_dim=2, n_points=200)

# Define the NSGA-III algorithm
algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=200)
# algorithm = NSGA2(pop_size=10)

n_gen = 200

# Solve the problem using NSGA-III
res = minimize(
    problem,
    algorithm,
    ('n_gen', n_gen),
    verbose=True,
    seed=42,
    save_history=True,
    eliminate_duplicates=True,
    n_jobs=94
)


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
# plt.scatter(-pareto_f[:, 0], -pareto_f[:, 1], c='blue', label="Pareto Front", alpha=0.7, edgecolors='none', s=100)
# if non_pareto_f.size > 0:  # Only plot if non_pareto_f is not empty
#     plt.scatter(-non_pareto_f[:, 0], -non_pareto_f[:, 1], c='red', label="Non-Pareto Points", alpha=0.5, edgecolors='none')

# plt.title("All Solutions with Pareto and Non-Pareto Points")
# plt.xlabel("Economic Objective (F1)")
# plt.ylabel("Environmental Objective (F2)")
# plt.legend(loc='upper left', fontsize=8, ncol=2)
# plt.grid()
# plt.tight_layout()
# plt.savefig(f"pareto_solutions_n_gen_{n_gen}_xu_{xu}.png")
# plt.show()

# # Plot some decision variables corresponding to Pareto front
# plt.figure(figsize=(8, 6))
# for i, (_, x, idx) in enumerate(pareto_front):  # Plot for a subset of Pareto solutions
#     plt.plot(x, label=f'Pareto Solution {idx}')

# plt.title("Decision Variables for Pareto Front Solutions")
# plt.xlabel("Time (quarters")
# plt.ylabel("Pesticide Input")
# # plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(f"pareto_decision_variables_n_gen_{n_gen}_xu_{xu}.png")
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
plt.savefig(f"pareto_solutions_n_gen_{n_gen}_xu_{xu}_with_non_pareto.png")
plt.show()

# # # Plot some decision variables corresponding to Pareto front
# # plt.figure(figsize=(8, 6))
# # for i, (_, x, idx) in enumerate(pareto_front):  # Plot for a subset of Pareto solutions
# #     plt.plot(x, label=f'Pareto Solution {idx}')

# # plt.title("Decision Variables for Pareto Front Solutions")
# # plt.xlabel("Time (quarters)")
# # plt.ylabel("Pesticide Input")
# # # plt.legend()
# # plt.grid()
# # plt.tight_layout()
# # plt.savefig(f"pareto_decision_variables_n_gen_{n_gen}_xu_{xu}.png")
# # plt.show()

# # 使用糖果色的色彩映射
# cmap = plt.cm.rainbow  # 色彩映射表
# norm = plt.Normalize(vmin=0, vmax=len(pareto_front) - 1)  # 将 Pareto 解索引映射到颜色范围

# # 绘制 Pareto 前沿解的决策变量条形图
# plt.figure(figsize=(10, 8))

# for i, (_, x, idx) in enumerate(pareto_front):  # 遍历 Pareto 解
#     color = cmap(norm(i))  # 为每个 Pareto 解分配颜色
#     plt.bar(np.arange(len(x)) + i * 0.1, x, width=0.1, color=color)

# 使用糖果色的色彩映射
cmap = plt.cm.rainbow  # 彩虹色映射表

# 设置随机选择 k 个解
k = 10  # 你可以调整 k 的值
random_pareto_solutions = random.sample(pareto_front, k)  # 随机选择 k 个解

# 获取随机解在原始 pareto_front 中的索引范围
random_indices = [idx for _, _, idx in random_pareto_solutions]
norm = plt.Normalize(vmin=min(random_indices), vmax=max(random_indices))  # 动态调整索引范围

# 第一个图：Pareto 前沿解的决策变量折线图
plt.figure(figsize=(10, 8))
for i, (_, x, idx) in enumerate(random_pareto_solutions):  # 遍历随机选择的 Pareto 解
    color = cmap(norm(idx))  # 为每个随机解分配彩虹色
    plt.plot(x, color=color, label=f'Pareto Solution {idx}', alpha=0.8)

# 图例和美化
plt.title(f"Decision Variables for {k} Random Pareto Front Solutions (Line Plot)")
plt.xlabel("Time (year)")
plt.ylabel("Pesticide Input")
plt.legend(loc='upper left', fontsize=8, ncol=2, title="Solutions")
plt.grid()
plt.tight_layout()

# 保存折线图
plt.savefig(f"pareto_decision_variables_plot_random_{k}_n_gen_{n_gen}_xu_{xu}.png")
plt.show()

# 第二个图：Pareto 前沿解的决策变量条形图
plt.figure(figsize=(10, 8))
for i, (_, x, idx) in enumerate(random_pareto_solutions):  # 遍历随机选择的 Pareto 解
    color = cmap(norm(idx))  # 为每个随机解分配彩虹色
    plt.bar(np.arange(len(x)) + i * 0.1, x, width=0.1, color=color, label=f'Pareto Solution {idx}')

# 图例和美化
plt.title(f"Decision Variables for {k} Random Pareto Front Solutions (Bar Chart)")
plt.xlabel("Time (quarters)")
plt.ylabel("Pesticide Input")
plt.legend(loc='upper left', fontsize=8, ncol=2, title="Solutions")
plt.grid(axis='y')
plt.tight_layout()

# 保存条形图
plt.savefig(f"pareto_decision_variables_bar_random_{k}_n_gen_{n_gen}_xu_{xu}.png")
plt.show()
# # 第一个图：Pareto 前沿解的决策变量折线图
# plt.figure(figsize=(10, 8))
# for i, (_, x, idx) in enumerate(pareto_front):  # 遍历 Pareto 解
#     color = cmap(norm(i))  # 为每个 Pareto 解分配颜色
#     plt.plot(x, color=color, label=f'Pareto Solution {idx}', alpha=0.8)

# # 图例和美化
# plt.title("Decision Variables for Pareto Front Solutions (Line Plot)")
# plt.xlabel("Time (quarters)")
# plt.ylabel("Pesticide Input")
# # plt.legend(loc='upper left', fontsize=8, ncol=2, title="Solutions")
# plt.grid()
# plt.tight_layout()

# # 保存折线图
# plt.savefig(f"pareto_decision_variables_plot_n_gen_{n_gen}_xu_{xu}.png")
# plt.show()

# # 第二个图：Pareto 前沿解的决策变量条形图
# plt.figure(figsize=(10, 8))
# for i, (_, x, idx) in enumerate(pareto_front):  # 遍历 Pareto 解
#     color = cmap(norm(i))  # 为每个 Pareto 解分配颜色
#     plt.bar(np.arange(len(x)) + i * 0.1, x, width=0.1, color=color, label=f'Pareto Solution {idx}')

# # 图例和美化
# plt.title("Decision Variables for Pareto Front Solutions (Bar Chart)")
# plt.xlabel("Time (quarters)")
# plt.ylabel("Pesticide Input")
# # plt.legend(loc='upper left', fontsize=8, ncol=2, title="Solutions")
# plt.grid(axis='y')
# plt.tight_layout()

# # 保存条形图
# plt.savefig(f"pareto_decision_variables_bar_n_gen_{n_gen}_xu_{xu}.png")
# plt.show()


# # 图例和美化
# plt.title("Decision Variables for Pareto Front Solutions")
# plt.xlabel("Time (quarters)")
# plt.ylabel("Pesticide Input")
# # plt.legend(loc='upper left', fontsize=8, ncol=2, title="Solutions")
# plt.grid(axis='y')
# plt.tight_layout()

# # 保存图像
# plt.savefig(f"pareto_decision_variables_rainbow_n_gen_{n_gen}_xu_{xu}.png")
# plt.show()


# Plot some decision variables corresponding to non-Pareto points
if non_pareto:  # Only plot if non-Pareto points exist
    plt.figure(figsize=(8, 6))
    for i, (_, x, idx) in enumerate(non_pareto[:10]):  # Plot for a subset of Non-Pareto solutions
        plt.plot(x, label=f'Non-Pareto Solution {idx}')

    plt.title("Decision Variables for Non-Pareto Points")
    plt.xlabel("Time (quarters)")
    plt.ylabel("Pesticide Input")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
# 绘制逐代解的演化过程
plt.figure(figsize=(10, 8))

# 使用糖果色的色彩映射
cmap = plt.cm.rainbow  # 可以换成其他映射表，如 viridis, plasma, coolwarm 等
norm = plt.Normalize(vmin=0, vmax=len(res.history) - 1)  # 将代数映射到颜色范围

# 遍历每一代的历史解
for gen_idx, gen in enumerate(res.history):
    gen_F = gen.pop.get("F")  # 获取当前代的目标函数值
    color = cmap(norm(gen_idx))  # 根据代数为当前代生成颜色
    scatter = plt.scatter(-gen_F[:, 0], -gen_F[:, 1], alpha=0.6, c=[color], label=f"Generation {gen_idx}" if gen_idx % 10 == 0 else "")

# 最终 Pareto 前沿
pareto_f = np.array([pf[0] for pf in pareto_front])
plt.scatter(-pareto_f[:, 0], -pareto_f[:, 1], c='blue', label="Final Pareto Front", s=100, edgecolors='none', alpha=0.8)

# 添加颜色条以表示代数
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # 创建 mappable 对象
sm.set_array([])  # 设置空数组，表示只显示颜色映射
cbar = plt.colorbar(sm, ax=plt.gca())  # 绑定当前的 Axes
cbar.set_label("Generation Index")  # 添加标签

# 图像美化和保存
plt.title("Objective Space Evolution Across Generations (Candy Colors)")
plt.xlabel("Economic Objective (F1)")
plt.ylabel("Environmental Objective (F2)")
plt.grid()
plt.tight_layout()
plt.savefig(f"evolution_generations_candy_colors_n_gen_{n_gen}_xu_{xu}.png")
plt.show()
