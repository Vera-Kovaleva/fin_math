"""
=============================================================
КУРСОВАЯ РАБОТА по финансовой математике
ПАО «Лукойл» (LKOH) — Биномиальная модель + СДП
=============================================================

Структура кода:
  1. Параметры модели
  2. Биномиальный алгоритм (оба сценария)
  3. График 1 — биномиальные деревья (оптим. + пессим.)
  4. Стохастическое динамическое программирование
  5. График 2 — оптимальные стратегии
  6. График 3 — сравнение распределений
  7. График 4 — динамика цены LKOH (иллюстративный)
=============================================================
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm, t as sp_t

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# 1. ПАРАМЕТРЫ МОДЕЛИ
# ─────────────────────────────────────────────────────────────

S0 = 5600.0   # Цена акции LKOH на 7 апреля 2026 г., руб.
T  = 3        # Горизонт моделирования, лет

# ── Оптимистический сценарий ──────────────────────────────────
# Ключевая ставка снижается; нефтяной рынок восстанавливается
R_opt = [0.105, 0.085, 0.070]   # Ставки по вкладам (R_i)
u_opt = [1.2200, 1.2600, 1.3050] # Множители роста
d_opt = [0.8050, 0.8250, 0.8480] # Множители падения
pu_opt = [0.73, 0.76, 0.72]      # Объективные вероятности роста
pd_opt = [0.28, 0.24, 0.28]      # Объективные вероятности падения
K_opt = 10900.0                   # Страйк колл-опциона, руб.

# ── Пессимистический сценарий ─────────────────────────────────
# Высокие ставки сохраняются; нефть дешевеет; геополитика давит
R_pes = [0.140, 0.130, 0.120]
u_pes = [1.1600, 1.1400, 1.1300]
d_pes = [0.6900, 0.6800, 0.6750]
pu_pes = [0.35, 0.30, 0.25]
pd_pes = [0.65, 0.70, 0.75]
K_pes = 8300.0


# ─────────────────────────────────────────────────────────────
# 2. БИНОМИАЛЬНЫЙ АЛГОРИТМ
# ─────────────────────────────────────────────────────────────

def binomial_model(S0, T, R, u, d, pu, pd, K):
    """
    Строит биномиальное дерево, считает цену опциона и
    хеджирующий портфель.

    Возвращает словарь с деревьями S, V и портфелями h.
    Нотация узлов: (downs, time) — сколько раз цена падала.
    """
    r = [1 + ri for ri in R]

    # Мартингальные вероятности на каждом шаге
    qu = [(r[t] - d[t]) / (u[t] - d[t]) for t in range(T)]
    qd = [1 - q for q in qu]

    # Ожидаемый рост и стандартное отклонение случайной компоненты
    k     = [pu[t]*u[t] + pd[t]*d[t] for t in range(T)]
    sigma = [float(np.sqrt(pu[t]*u[t]**2 + pd[t]*d[t]**2 - k[t]**2))
             for t in range(T)]

    # ── Дерево цен акции ─────────────────────────────────────
    # Узел (i, t): i = число падений, t = момент времени
    S = {}
    S[(0,0)] = S0
    S[(0,1)] = S0*u[0];        S[(1,0)] = S0*d[0]
    S[(0,2)] = S[(0,1)]*u[1];  S[(1,1)] = S[(0,1)]*d[1];  S[(2,0)] = S[(1,0)]*d[1]
    S[(0,3)] = S[(0,2)]*u[2];  S[(1,2)] = S[(0,2)]*d[2]
    S[(2,1)] = S[(2,0)]*u[2];  S[(3,0)] = S[(2,0)]*d[2]

    # ── Обратный ход: цена опциона ───────────────────────────
    V = {}
    # Терминальные выплаты (t = T = 3)
    for node in [(0,3), (1,2), (2,1), (3,0)]:
        V[node] = max(S[node] - K, 0)

    # t = 2
    for node, up_node, dn_node, ti in [
        ((0,2), (0,3), (1,2), 2),
        ((1,1), (1,2), (2,1), 2),
        ((2,0), (2,1), (3,0), 2),
    ]:
        V[node] = (qu[ti]*V[up_node] + qd[ti]*V[dn_node]) / r[ti]

    # t = 1
    for node, up_node, dn_node, ti in [
        ((0,1), (0,2), (1,1), 1),
        ((1,0), (1,1), (2,0), 1),
    ]:
        V[node] = (qu[ti]*V[up_node] + qd[ti]*V[dn_node]) / r[ti]

    # t = 0
    V[(0,0)] = (qu[0]*V[(0,1)] + qd[0]*V[(1,0)]) / r[0]

    # ── Хеджирующий портфель ─────────────────────────────────
    # h(node) = (x, y): x — облигации, y — акции
    def hedge(Vu, Vd, S_cur, u_t, d_t, r_t):
        x = (u_t*Vd - d_t*Vu) / ((u_t - d_t) * r_t)
        y = (Vu - Vd) / (S_cur * (u_t - d_t))
        return x, y

    H = {}
    for node, un, dn, ti, sn in [
        ((0,0), (0,1), (1,0), 0, (0,0)),
        ((0,1), (0,2), (1,1), 1, (0,1)),
        ((1,0), (1,1), (2,0), 1, (1,0)),
        ((0,2), (0,3), (1,2), 2, (0,2)),
        ((1,1), (1,2), (2,1), 2, (1,1)),
        ((2,0), (2,1), (3,0), 2, (2,0)),
    ]:
        H[node] = hedge(V[un], V[dn], S[sn], u[ti], d[ti], r[ti])

    return {
        'S': S, 'V': V, 'H': H,
        'r': r, 'qu': qu, 'qd': qd,
        'k': k, 'sigma': sigma,
        'option_price': V[(0,0)]
    }


# Считаем оба сценария
res_opt = binomial_model(S0, T, R_opt, u_opt, d_opt, pu_opt, pd_opt, K_opt)
res_pes = binomial_model(S0, T, R_pes, u_pes, d_pes, pu_pes, pd_pes, K_pes)

# Вывод результатов в консоль
print("=" * 60)
print("РЕЗУЛЬТАТЫ БИНОМИАЛЬНОЙ МОДЕЛИ")
print("=" * 60)
for name, res, K in [("ОПТИМИСТИЧЕСКИЙ", res_opt, K_opt),
                      ("ПЕССИМИСТИЧЕСКИЙ", res_pes, K_pes)]:
    print(f"\n── {name} СЦЕНАРИЙ (K = {K:.0f} руб.) ──")
    print(f"  Безарбитражность (d ≤ 1+R ≤ u):")
    R_cur = R_opt if name == "ОПТИМИСТИЧЕСКИЙ" else R_pes
    u_cur = u_opt if name == "ОПТИМИСТИЧЕСКИЙ" else u_pes
    d_cur = d_opt if name == "ОПТИМИСТИЧЕСКИЙ" else d_pes
    for t in range(T):
        r_t = 1 + R_cur[t]
        ok = d_cur[t] <= r_t <= u_cur[t]
        print(f"    {2026+t}: {d_cur[t]} ≤ {r_t} ≤ {u_cur[t]}  →  {'✓ OK' if ok else '✗ НАРУШЕНО'}")
    print(f"  Дерево цен (в руб.):")
    for key in sorted(res['S'].keys(), key=lambda x: (x[1], x[0])):
        print(f"    S{key} = {res['S'][key]:.2f}")
    print(f"  Цена опциона: П(0;X) = {res['option_price']:.2f} руб.")
    print(f"  Хеджирующий портфель в (0,0): x={res['H'][(0,0)][0]:.4f}, y={res['H'][(0,0)][1]:.6f}")


# ─────────────────────────────────────────────────────────────
# 3. ГРАФИК 1 — БИНОМИАЛЬНЫЕ ДЕРЕВЬЯ
# ─────────────────────────────────────────────────────────────

def draw_tree(ax, res, title, color_up, color_mid, color_dn):
    """Рисует биномиальное дерево на оси ax."""

    S, V, H = res['S'], res['V'], res['H']

    # Координаты узлов: ключ (downs, time) → (x, y)
    pos = {
        (0,0): (0.0, 4.0),
        (0,1): (2.0, 6.0),  (1,0): (2.0, 2.0),
        (0,2): (4.0, 7.5),  (1,1): (4.0, 4.0),  (2,0): (4.0, 0.5),
        (0,3): (6.0, 8.5),  (1,2): (6.0, 6.0),  (2,1): (6.0, 3.0),  (3,0): (6.0, 0.5),
    }

    # Рёбра дерева
    edges = [
        ((0,0),(0,1)), ((0,0),(1,0)),
        ((0,1),(0,2)), ((0,1),(1,1)),
        ((1,0),(1,1)), ((1,0),(2,0)),
        ((0,2),(0,3)), ((0,2),(1,2)),
        ((1,1),(1,2)), ((1,1),(2,1)),
        ((2,0),(2,1)), ((2,0),(3,0)),
    ]
    for a, b in edges:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                'k-', lw=0.9, alpha=0.4, zorder=1)

    # Узлы
    for node, (xv, yv) in pos.items():
        downs, time = node
        # Цвет: верхний путь — синий/красный, средний — оранжевый, нижний — серый
        if downs == 0:
            color = color_up
        elif downs == time:
            color = color_dn
        else:
            color = color_mid

        # Прямоугольник узла
        ax.add_patch(mpatches.FancyBboxPatch(
            (xv - 0.42, yv - 0.62), 0.84, 1.24,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor='white', linewidth=1.5, zorder=2, alpha=0.92
        ))

        # Текст: цена и значение портфеля
        s_val = S.get(node, 0)
        v_val = V.get(node, 0)
        ax.text(xv, yv + 0.38, f'S={s_val:.0f}', ha='center', va='center',
                fontsize=6.8, color='white', fontweight='bold', zorder=3)
        ax.text(xv, yv - 0.10, f'V={v_val:.2f}', ha='center', va='center',
                fontsize=6.2, color='white', zorder=3)

        # Для нетерминальных узлов — состав портфеля
        if node in H:
            x_h, y_h = H[node]
            ax.text(xv, yv - 0.46, f'x={x_h:.1f}, y={y_h:.3f}',
                    ha='center', va='center', fontsize=5.0, color='white',
                    alpha=0.85, zorder=3)

    # Метки осей времени
    for t, year in enumerate([2026, 2027, 2028, 'Исп.']):
        ax.text(t * 2, -0.5, f't={t}\n({year})', ha='center', va='top',
                fontsize=8, color='#333333')
        ax.axvline(t * 2, color='#CCCCCC', lw=0.5, alpha=0.5)

    ax.set_xlim(-0.7, 7.2)
    ax.set_ylim(-1.2, 10.0)
    ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold', color='#1A3A5C', pad=10)


fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig1.patch.set_facecolor('#F8F9FA')
for ax in [ax1, ax2]:
    ax.set_facecolor('#F8F9FA')

draw_tree(ax1, res_opt,
          f'Оптимистический сценарий\nK = {K_opt:.0f} руб.   П(0;X) = {res_opt["option_price"]:.2f} руб.',
          color_up='#1565C0', color_mid='#1976D2', color_dn='#546E7A')

draw_tree(ax2, res_pes,
          f'Пессимистический сценарий\nK = {K_pes:.0f} руб.   П(0;X) = {res_pes["option_price"]:.2f} руб.',
          color_up='#B71C1C', color_mid='#E53935', color_dn='#455A64')

fig1.suptitle('Биномиальные деревья цен акции ПАО «Лукойл» (LKOH)',
              fontsize=13, fontweight='bold', color='#1A3A5C', y=0.98)
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('fig1_trees.png', dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print("\nГрафик 1 сохранён: fig1_trees.png")


# ─────────────────────────────────────────────────────────────
# 4. СТОХАСТИЧЕСКОЕ ДИНАМИЧЕСКОЕ ПРОГРАММИРОВАНИЕ
# ─────────────────────────────────────────────────────────────

# Параметры СДП
Z0     = res_opt['option_price']   # Начальный капитал = цена опциона
r_sdp  = res_opt['r']              # Коэффициенты роста облигации
k_sdp  = res_opt['k']              # Ожидаемый рост акции
sig    = res_opt['sigma']          # СКО случайной компоненты

Nz      = 150    # Число узлов сетки по капиталу
Nu      = 30     # Число значений управлений
M       = 2000   # Сценарии Монте-Карло
N_runs  = 10     # Испытания для усреднения
U_MAX   = 3.3    # Верхняя граница доли рискового актива

z_grid = np.linspace(Z0 * 0.05, Z0 * 8.0, Nz)
u_grid = np.linspace(0.0, U_MAX, Nu)


def sample_xi(t, size, dist):
    """
    Генерирует size реализаций случайной компоненты ξ_t
    для выбранного распределения.
    Все три распределения имеют E[ξ] = 0 и Var[ξ] = sigma[t]^2.
    """
    s = sig[t]
    if dist == 'normal':
        # Нормальное N(0, σ²)
        return np.random.normal(0.0, s, size)
    elif dist == 'uniform':
        # Равномерное U(-√3σ, √3σ) → дисперсия = σ²
        a = float(np.sqrt(3.0) * s)
        return np.random.uniform(-a, a, size)
    else:
        # Стьюдента t(5), масштабированное: Var[t_5] = 5/3 → scale = σ·√(3/5)
        n = 5
        scale = float(s * np.sqrt((n - 2.0) / n))
        return np.random.standard_t(n, size) * scale


def run_sdp(dist):
    """
    Один запуск алгоритма обратной индукции.
    Возвращает оптимальные управления u* и ожидаемый конечный капитал.
    """
    # Терминальное условие: V_T(z) = z
    V_next = z_grid.copy()
    opt_u_table = np.zeros((T, Nz))

    for t in range(T - 1, -1, -1):
        V_curr     = np.zeros(Nz)
        best_u_arr = np.zeros(Nz)

        # Генерируем M реализаций ξ для шага t
        xi = sample_xi(t, M, dist)

        for iz in range(Nz):
            z        = z_grid[iz]
            best_val = -1e18
            best_uj  = 0.0

            for uj in u_grid:
                # Уравнение динамики капитала:
                # z_{t+1} = z * (r_t + (k_t - r_t)*u + ξ*u)
                z_next = z * (r_sdp[t] + (k_sdp[t] - r_sdp[t]) * uj + xi * uj)

                # Линейная интерполяция V_{t+1}
                Vi = np.interp(z_next, z_grid, V_next)
                ev = float(np.mean(Vi))

                if ev > best_val:
                    best_val = ev
                    best_uj  = uj

            V_curr[iz]     = best_val
            best_u_arr[iz] = best_uj

        V_next         = V_curr
        opt_u_table[t] = best_u_arr

    # Интерполируем результат для начального капитала Z0
    u_opt = [float(np.interp(Z0, z_grid, opt_u_table[t])) for t in range(T)]
    V0    = float(np.interp(Z0, z_grid, V_next))
    return u_opt, V0


# Запуск для трёх распределений
distributions = ['normal', 'uniform', 'student']
dist_labels   = {
    'normal':  'Нормальное N(0, σ²)',
    'uniform': 'Равномерное U(−√3σ, √3σ)',
    'student': 'Стьюдента t(5)'
}

results = {}
print("\n" + "=" * 60)
print("СТОХАСТИЧЕСКОЕ ДИНАМИЧЕСКОЕ ПРОГРАММИРОВАНИЕ")
print(f"V₀ = {Z0:.2f} руб.,  T = {T},  U_max = {U_MAX},  M = {M},  N = {N_runs}")
print("=" * 60)

for dist in distributions:
    all_u = [[], [], []]
    all_V = []
    for _ in range(N_runs):
        u_opt, V0 = run_sdp(dist)
        for t in range(T):
            all_u[t].append(u_opt[t])
        all_V.append(V0)

    results[dist] = {
        'u0': float(np.mean(all_u[0])),
        'u1': float(np.mean(all_u[1])),
        'u2': float(np.mean(all_u[2])),
        'V0': float(np.mean(all_V))
    }
    rs = results[dist]
    print(f"\n  {dist_labels[dist]}:")
    print(f"    u₀ = {rs['u0']:.4f},  u₁ = {rs['u1']:.4f},  u₂ = {rs['u2']:.4f}")
    print(f"    Ожидаемый конечный капитал = {rs['V0']:.2f} руб.")

# Математическое ожидание выплаты по опциону (по объективным вероятностям)
V = res_opt['V']
E_pay = (
    pu_opt[0]*pu_opt[1]*pu_opt[2] * V[(0,3)]
    + (pu_opt[0]*pu_opt[1]*pd_opt[2] + pu_opt[0]*pd_opt[1]*pu_opt[2] + pd_opt[0]*pu_opt[1]*pu_opt[2]) * V[(1,2)]
    + (pu_opt[0]*pd_opt[1]*pd_opt[2] + pd_opt[0]*pu_opt[1]*pd_opt[2] + pd_opt[0]*pd_opt[1]*pu_opt[2]) * V[(2,1)]
    + pd_opt[0]*pd_opt[1]*pd_opt[2] * V[(3,0)]
)
print(f"\n  E[выплата по опциону] (объект. вер.) = {E_pay:.2f} руб.")


# ─────────────────────────────────────────────────────────────
# 5. ГРАФИК 2 — ОПТИМАЛЬНЫЕ СТРАТЕГИИ (СДП)
# ─────────────────────────────────────────────────────────────

fig2, ax = plt.subplots(figsize=(8, 5))
fig2.patch.set_facecolor('#F8F9FA')
ax.set_facecolor('#F8F9FA')

colors  = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['s-', 'o--', '^-.']

for i, dist in enumerate(distributions):
    uv = [results[dist]['u0'], results[dist]['u1'], results[dist]['u2']]
    ax.plot([0, 1, 2], uv, markers[i], color=colors[i],
            label=f"{dist_labels[dist]}\n"
                  f"u₀={results[dist]['u0']:.4f}, EV={results[dist]['V0']:.2f} руб.",
            lw=2.2, ms=9, markeredgecolor='white', markeredgewidth=1)

ax.set_xlabel('Период, t', fontsize=12)
ax.set_ylabel('Оптимальная доля рискового актива, $u_t$', fontsize=12)
ax.set_title(f'Средняя оптимальная стратегия (V₀ = {Z0:.2f} руб.)',
             fontsize=12, color='#1A3A5C', fontweight='bold')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['t = 0\n(2026)', 't = 1\n(2027)', 't = 2\n(2028)'])
ax.axhline(U_MAX, color='gray', ls=':', lw=1.2, alpha=0.6, label=f'U_max = {U_MAX}')
ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5))
ax.grid(alpha=0.3)
ax.set_ylim(0.5, U_MAX + 0.4)

plt.tight_layout()
plt.savefig('fig2_strategy.png', dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print("\nГрафик 2 сохранён: fig2_strategy.png")


# ─────────────────────────────────────────────────────────────
# 6. ГРАФИК 3 — СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ
# ─────────────────────────────────────────────────────────────

fig3, ax = plt.subplots(figsize=(8, 5))
fig3.patch.set_facecolor('#F8F9FA')
ax.set_facecolor('#F8F9FA')

s_p = sig[0]   # СКО на первом шаге
x_ax = np.linspace(-3.8 * s_p, 3.8 * s_p, 600)

# Нормальное
ax.plot(x_ax, norm.pdf(x_ax, 0, s_p), color=colors[0], lw=2.2,
        label=f'Нормальное $N(0,\\sigma^2)$,  $\\sigma$ = {s_p:.4f}')

# Равномерное
a_u = float(np.sqrt(3.0) * s_p)
ax.plot(x_ax, np.where(np.abs(x_ax) <= a_u, 1.0 / (2 * a_u), 0.0),
        color=colors[1], lw=2.2, ls='--',
        label=f'Равномерное $U(-\\sqrt{{3}}\\sigma,\\,\\sqrt{{3}}\\sigma)$')

# Стьюдента
n_df = 5
sc_t = float(s_p * np.sqrt((n_df - 2.0) / n_df))
ax.plot(x_ax, sp_t.pdf(x_ax / sc_t, n_df) / sc_t,
        color=colors[2], lw=2.2, ls='-.',
        label=f'Стьюдента $t({n_df})$, масштаб. до $\\sigma^2$')

# Подписи хвостов
for side in [-1, 1]:
    ax.annotate('тяжёлый\nхвост (t)', xy=(side*2.8*s_p, sp_t.pdf(2.8*s_p/sc_t, 5)/sc_t),
                xytext=(side*3.2*s_p, 0.8), fontsize=7.5, color=colors[2],
                ha='center', arrowprops=dict(arrowstyle='->', color=colors[2], lw=0.8))

ax.set_xlabel('Значение случайной компоненты $\\xi$', fontsize=12)
ax.set_ylabel('Плотность распределения $p(\\xi)$', fontsize=12)
ax.set_title('Сравнение законов распределения случайной компоненты',
             fontsize=12, color='#1A3A5C', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(-3.5*s_p, 3.5*s_p)

plt.tight_layout()
plt.savefig('fig3_distributions.png', dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print("График 3 сохранён: fig3_distributions.png")


# ─────────────────────────────────────────────────────────────
# 7. ГРАФИК 4 — ДИНАМИКА ЦЕНЫ LKOH (иллюстративный)
# ─────────────────────────────────────────────────────────────
# Реальные годовые цены LKOH (начало года, руб.)
years_hist = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
price_hist = [5190, 5750, 6170, 6430, 4650, 7200, 6300, 5600]

# Прогнозные пути биномиальной модели (t=0,1,2,3 → 2026,2027,2028,2029)
t_proj = [2026, 2027, 2028, 2029]
S = res_opt['S']
paths_opt = {
    'uuu': [S0, S[(0,1)], S[(0,2)], S[(0,3)]],
    'uud': [S0, S[(0,1)], S[(0,2)], S[(1,2)]],
    'udd': [S0, S[(1,0)], S[(1,1)], S[(2,1)]],
    'ddd': [S0, S[(1,0)], S[(2,0)], S[(3,0)]],
}
path_labels = {
    'uuu': '↑↑↑ (3 роста)',
    'uud': '↑↑↓ (2 роста)',
    'udd': '↓↑↓ (1 рост)',
    'ddd': '↓↓↓ (3 падения)',
}
path_colors = ['#1565C0', '#42A5F5', '#FFA726', '#B71C1C']
path_alpha  = [0.9, 0.7, 0.7, 0.9]

fig4, ax = plt.subplots(figsize=(11, 5))
fig4.patch.set_facecolor('#F8F9FA')
ax.set_facecolor('#F8F9FA')

# Исторические данные
ax.plot(years_hist, price_hist, 'ko-', lw=2, ms=6, label='Исторические цены LKOH', zorder=5)
ax.axvline(2026, color='gray', ls='--', lw=1.2, alpha=0.7, label='Дата начала (апр. 2026)')

# Прогнозные пути
for (key, path), color, alpha in zip(paths_opt.items(), path_colors, path_alpha):
    ax.plot(t_proj, path, 'o--', color=color, lw=1.8, ms=6, alpha=alpha,
            label=f'{path_labels[key]}: {path[-1]:.0f} руб.')

# Страйк
ax.axhline(K_opt, color='purple', ls=':', lw=1.5, alpha=0.8,
           label=f'Страйк K = {K_opt:.0f} руб.')

ax.fill_betweenx([min(price_hist)*0.8, K_opt*1.1],
                  2025.8, 2029.2, alpha=0.04, color='blue')

ax.set_xlabel('Год', fontsize=12)
ax.set_ylabel('Цена акции, руб.', fontsize=12)
ax.set_title('Динамика цены акции ПАО «Лукойл» (LKOH): история и прогноз',
             fontsize=12, color='#1A3A5C', fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xlim(2018.5, 2029.5)

plt.tight_layout()
plt.savefig('fig4_price_dynamics.png', dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print("График 4 сохранён: fig4_price_dynamics.png")


# ─────────────────────────────────────────────────────────────
# ИТОГОВАЯ СВОДКА
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ИТОГОВАЯ СВОДКА")
print("=" * 60)
print(f"\nАктив: ПАО «Лукойл» (LKOH),  S₀ = {S0:.0f} руб.,  T = {T} года")
print(f"\nОПТИМИСТИЧЕСКИЙ СЦЕНАРИЙ (K = {K_opt:.0f} руб.):")
print(f"  Цена опциона П(0;X) = {res_opt['option_price']:.2f} руб.")
print(f"  h(0,0): x = {res_opt['H'][(0,0)][0]:.4f},  y = {res_opt['H'][(0,0)][1]:.6f}")
print(f"\nПЕССИМИСТИЧЕСКИЙ СЦЕНАРИЙ (K = {K_pes:.0f} руб.):")
print(f"  Цена опциона П(0;X) = {res_pes['option_price']:.2f} руб.")
print(f"  h(0,0): x = {res_pes['H'][(0,0)][0]:.4f},  y = {res_pes['H'][(0,0)][1]:.6f}")
print(f"\nСДП (V₀ = {Z0:.2f} руб.):")
print(f"{'Распределение':<35} {'u₀':>7} {'u₁':>7} {'u₂':>7} {'EV, руб.':>10}")
print("-" * 65)
for dist in distributions:
    rs = results[dist]
    print(f"  {dist_labels[dist]:<33} {rs['u0']:>7.4f} {rs['u1']:>7.4f} {rs['u2']:>7.4f} {rs['V0']:>10.2f}")
print(f"\n  E[выплата по опциону] (объект. вер.) = {E_pay:.2f} руб.")
print("\nВсе графики сохранены:")
print("  fig1_trees.png          — биномиальные деревья")
print("  fig2_strategy.png       — оптимальные стратегии СДП")
print("  fig3_distributions.png  — сравнение распределений")
print("  fig4_price_dynamics.png — динамика цены LKOH")
