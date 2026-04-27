import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# ==============================================================================
# ДАННЫЕ ВАРИАНТА 26
# ==============================================================================
data = np.array([
    [1.01, 1.15, 0.32],
    [2.07, 2.12, 2.48],
    [2.97, 2.99, 3.85],
    [4.04, 3.86, 1.11],
    [4.98, 5.09, 0.21]
])

X_raw = data[:, 0]
Y_raw = data[:, 1]
Z = data[:, 2]

print("=" * 60)
print("ВАРИАНТ 26")
print("=" * 60)
print("Исходные данные:")
for i in range(len(X_raw)):
    print(f"  {i}: x={X_raw[i]:.3f}, y={Y_raw[i]:.3f}, z={Z[i]:.3f}")

# ==============================================================================
# ЗАДАНИЕ 3: Константные модели
# ==============================================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 3: Константные модели")
print("=" * 60)

c_mse = np.mean(Z)
c_mae = np.median(Z)

print(f"Константа по MSE (среднее):     c = {c_mse:.4f}")
print(f"Константа по MAE (медиана):     c = {c_mae:.4f}")
print(f"MSE константной модели:         {np.mean((Z - c_mse) ** 2):.6f}")
print(f"MAE константной модели:         {np.mean(np.abs(Z - c_mae)):.6f}")

# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ВИЗУАЛИЗАЦИИ
# ==============================================================================
x_grid_raw = np.linspace(0.5, 5.5, 50)
y_grid_raw = np.linspace(0.5, 5.5, 50)
X_grid_raw, Y_grid_raw = np.meshgrid(x_grid_raw, y_grid_raw)


def save_3d_and_contour(Z_grid, title_base, filename_base, residuals=None, centers=None):
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X_grid_raw, Y_grid_raw, Z_grid, cmap='plasma', alpha=0.7)
    ax1.scatter(X_raw, Y_raw, Z, c='blue', s=80, edgecolors='black')
    ax1.set_title(f'{title_base} + данные')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X_grid_raw, Y_grid_raw, Z_grid, levels=20, cmap='plasma')
    ax2.contour(X_grid_raw, Y_grid_raw, Z_grid, levels=10, colors='white', alpha=0.5)

    if residuals is not None:
        sc = ax2.scatter(X_raw, Y_raw, c=residuals, s=100, cmap='coolwarm', edgecolors='black')
        plt.colorbar(sc, ax=ax2, label='невязка')

    if centers is not None:
        ax2.scatter(centers[:, 0], centers[:, 1], marker='*', s=200,
                    c='gold', edgecolors='black', linewidth=1.5, label='Центры RBF')
        ax2.legend()

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'Линии уровня {title_base.lower()}')
    plt.colorbar(contour, ax=ax2, label='z')
    plt.tight_layout()
    plt.savefig(f'{filename_base}.png', dpi=150)
    plt.show()


# ==============================================================================
# ЗАДАНИЕ 1: Двумерная гауссиана
# ==============================================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 1: Двумерная гауссиана")
print("=" * 60)


def gaussian_2d(x, y, A, x0, y0, sx, sy, offset):
    return A * np.exp(-((x - x0) ** 2) / (2 * sx ** 2) - ((y - y0) ** 2) / (2 * sy ** 2)) + offset


def gaussian_loss(params):
    A, x0, y0, sx, sy, offset = params
    if sx <= 0 or sy <= 0 or A <= 0:
        return 1e10
    pred = gaussian_2d(X_raw, Y_raw, A, x0, y0, sx, sy, offset)
    return np.mean((Z - pred) ** 2)


max_idx = np.argmax(Z)
A_start = Z[max_idx] - np.min(Z) + 0.1
x0_start = X_raw[max_idx]
y0_start = Y_raw[max_idx]
sx_start = (np.max(X_raw) - np.min(X_raw)) / 4
sy_start = (np.max(Y_raw) - np.min(Y_raw)) / 4
offset_start = np.min(Z)

params_start_g = [A_start, x0_start, y0_start, sx_start, sy_start, offset_start]

print("Оптимизация гауссианы (L-BFGS-B)...")
result_g = minimize(gaussian_loss, params_start_g, method='L-BFGS-B',
                    bounds=[(0.1, 15), (0.5, 5.5), (0.5, 5.5), (0.1, 3), (0.1, 3), (-1, 2)])
A_opt, x0_opt, y0_opt, sx_opt, sy_opt, offset_opt = result_g.x

print(f"\nПараметры гауссианы: A={A_opt:.4f}, x0={x0_opt:.4f}, y0={y0_opt:.4f}")
print(f"sigma_x={sx_opt:.4f}, sigma_y={sy_opt:.4f}, offset={offset_opt:.4f}, MSE={result_g.fun:.8f}")

Z_pred_g = gaussian_2d(X_raw, Y_raw, A_opt, x0_opt, y0_opt, sx_opt, sy_opt, offset_opt)
residuals_g = Z - Z_pred_g

print("\nНевязки гауссианы:")
for i in range(len(X_raw)):
    print(f"  {i}: факт={Z[i]:.3f}, предск={Z_pred_g[i]:.3f}, невязка={residuals_g[i]:.6f}")


# РЕАЛЬНАЯ кривая обучения гауссианы (градиентный спуск)
def gaussian_grad(params):
    A, x0, y0, sx, sy, offset = params
    pred = gaussian_2d(X_raw, Y_raw, A, x0, y0, sx, sy, offset)
    err = pred - Z
    phi = np.exp(-((X_raw - x0) ** 2) / (2 * sx ** 2) - ((Y_raw - y0) ** 2) / (2 * sy ** 2))
    grad_A = 2 * np.mean(err * phi)
    grad_x0 = 2 * np.mean(err * A * phi * (X_raw - x0) / sx ** 2)
    grad_y0 = 2 * np.mean(err * A * phi * (Y_raw - y0) / sy ** 2)
    grad_sx = 2 * np.mean(err * A * phi * (X_raw - x0) ** 2 / sx ** 3)
    grad_sy = 2 * np.mean(err * A * phi * (Y_raw - y0) ** 2 / sy ** 3)
    grad_offset = 2 * np.mean(err)
    return np.array([grad_A, grad_x0, grad_y0, grad_sx, grad_sy, grad_offset])


params_gd_g = np.array(params_start_g, dtype=float)
lr_g = 0.01
losses_g = []
n_iter_g = 300

for step in range(n_iter_g):
    pred = gaussian_2d(X_raw, Y_raw, *params_gd_g)
    loss = np.mean((Z - pred) ** 2)
    losses_g.append(loss)
    grad = gaussian_grad(params_gd_g)
    grad = np.clip(grad, -5, 5)
    params_gd_g -= lr_g * grad
    if step > 20 and losses_g[-1] > losses_g[-5]:
        lr_g *= 0.99

plt.figure(figsize=(10, 4))
plt.plot(range(len(losses_g)), losses_g, 'orange', linewidth=2)
plt.yscale('log')
plt.xlabel('Итерация')
plt.ylabel('MSE')
plt.title('Кривая обучения (двумерная гауссиана)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_gaussian.png', dpi=150)
plt.show()

Z_grid_g_raw = gaussian_2d(X_grid_raw, Y_grid_raw, A_opt, x0_opt, y0_opt, sx_opt, sy_opt, offset_opt)
save_3d_and_contour(Z_grid_g_raw, 'Гауссиана', 'surface_gaussian', residuals_g)

# ==============================================================================
# ЗАДАНИЕ 2: Эллиптический параболоид (с нормальной кривой обучения)
# ==============================================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 2: Эллиптический параболоид")
print("=" * 60)


def paraboloid_predict(x, y, params):
    a, b, c, d, e, f = params
    return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f


def paraboloid_loss(params):
    pred = paraboloid_predict(X_raw, Y_raw, params)
    return np.mean((Z - pred) ** 2)


# Нормализуем данные для устойчивости градиентного спуска
X_mean, X_std = np.mean(X_raw), np.std(X_raw)
Y_mean, Y_std = np.mean(Y_raw), np.std(Y_raw)
X_norm = (X_raw - X_mean) / X_std
Y_norm = (Y_raw - Y_mean) / Y_std


def paraboloid_predict_norm(x, y, params):
    a, b, c, d, e, f = params
    return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f


def paraboloid_loss_norm(params):
    pred = paraboloid_predict_norm(X_norm, Y_norm, params)
    return np.mean((Z - pred) ** 2)


# Начальное приближение в нормализованных координатах
params_start = [0.0, 0.0, 0.0, 0.0, 0.0, np.mean(Z)]

# Сначала найдём хорошее решение через L-BFGS-B (для точности)
print("  Точная оптимизация (L-BFGS-B)...")
result_p = minimize(paraboloid_loss_norm, params_start, method='L-BFGS-B')
params_opt_norm = result_p.x


# Теперь построим РЕАЛЬНУЮ кривую обучения через градиентный спуск
def paraboloid_grad_norm(params):
    a, b, c, d, e, f = params
    pred = a * X_norm ** 2 + b * Y_norm ** 2 + c * X_norm * Y_norm + d * X_norm + e * Y_norm + f
    err = pred - Z
    grad_a = 2 * np.mean(err * X_norm ** 2)
    grad_b = 2 * np.mean(err * Y_norm ** 2)
    grad_c = 2 * np.mean(err * X_norm * Y_norm)
    grad_d = 2 * np.mean(err * X_norm)
    grad_e = 2 * np.mean(err * Y_norm)
    grad_f = 2 * np.mean(err)
    return np.array([grad_a, grad_b, grad_c, grad_d, grad_e, grad_f])


# Градиентный спуск с логированием
params_gd = np.array(params_start, dtype=float)
lr = 0.05  # подходящий learning rate для нормализованных данных
losses_p = []
n_iter = 200

for step in range(n_iter):
    pred = paraboloid_predict_norm(X_norm, Y_norm, params_gd)
    loss = np.mean((Z - pred) ** 2)
    losses_p.append(loss)

    grad = paraboloid_grad_norm(params_gd)
    # Ограничиваем градиенты для устойчивости
    grad = np.clip(grad, -1, 1)
    params_gd -= lr * grad

    # Адаптивное уменьшение learning rate
    if step > 10 and losses_p[-1] < losses_p[-2]:
        pass  # продолжаем
    elif step > 10 and losses_p[-1] > losses_p[-2] * 1.01:
        lr *= 0.95

# Переводим параметры обратно в исходные координаты для вывода
a_opt = params_gd[0] / X_std ** 2
b_opt = params_gd[1] / Y_std ** 2
c_opt = params_gd[2] / (X_std * Y_std)
d_opt = -2 * params_gd[0] * X_mean / X_std ** 2 - params_gd[2] * Y_mean / (X_std * Y_std) + params_gd[3] / X_std
e_opt = -2 * params_gd[1] * Y_mean / Y_std ** 2 - params_gd[2] * X_mean / (X_std * Y_std) + params_gd[4] / Y_std
f_opt = (params_gd[0] * X_mean ** 2 / X_std ** 2 + params_gd[1] * Y_mean ** 2 / Y_std ** 2 +
         params_gd[2] * X_mean * Y_mean / (X_std * Y_std) - params_gd[3] * X_mean / X_std -
         params_gd[4] * Y_mean / Y_std + params_gd[5])

print(f"\nПараметры параболоида (в исходных координатах):")
print(f"  a = {a_opt:.6f}, b = {b_opt:.6f}, c = {c_opt:.6f}")
print(f"  d = {d_opt:.6f}, e = {e_opt:.6f}, f = {f_opt:.6f}")
print(f"  MSE = {losses_p[-1]:.8f}")

# Предсказания и невязки
Z_pred_p = paraboloid_predict(X_raw, Y_raw, [a_opt, b_opt, c_opt, d_opt, e_opt, f_opt])
residuals_p = Z - Z_pred_p

print("\nНевязки параболоида:")
for i in range(len(X_raw)):
    print(f"  {i}: факт={Z[i]:.3f}, предск={Z_pred_p[i]:.3f}, невязка={residuals_p[i]:.4f}")

# Кривая обучения
plt.figure(figsize=(10, 4))
plt.plot(range(len(losses_p)), losses_p, linewidth=2, color='blue')
plt.xlabel('Итерация')
plt.ylabel('MSE')
plt.title('Кривая обучения (эллиптический параболоид)')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_paraboloid.png', dpi=150)
plt.show()


# Визуализация
def paraboloid_predict_raw(x, y):
    return (a_opt * x ** 2 + b_opt * y ** 2 + c_opt * x * y + d_opt * x + e_opt * y + f_opt)


Z_grid_p_raw = paraboloid_predict_raw(X_grid_raw, Y_grid_raw)
save_3d_and_contour(Z_grid_p_raw, 'Эллиптический параболоид', 'surface_paraboloid', residuals_p)

# ==============================================================================
# ЗАДАНИЕ 5: RBF-сеть с РЕАЛЬНОЙ кривой обучения
# ==============================================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 5: RBF-сеть")
print("=" * 60)

# Простая RBF-сеть с фиксированными центрами (все точки данных)
centers = np.column_stack([X_raw, Y_raw])
n_centers = len(centers)

# Вычисляем sigma как среднее расстояние до ближайшего центра
sigma = 0.0
for i in range(n_centers):
    dists = [np.linalg.norm(centers[i] - centers[j]) for j in range(n_centers) if j != i]
    sigma += min(dists) if dists else 1.0
sigma = sigma / n_centers

print(f"Центров: {n_centers}, sigma = {sigma:.4f}")


def rbf_predict(X, Y, weights, offset):
    pred = np.full(len(X), offset)
    for j, (cx, cy) in enumerate(centers):
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        pred += weights[j] * np.exp(-r2 / (2 * sigma ** 2))
    return pred


def rbf_loss(params):
    weights = params[:-1]
    offset = params[-1]
    pred = rbf_predict(X_raw, Y_raw, weights, offset)
    return np.mean((Z - pred) ** 2)


# Инициализация
weights_start = np.random.randn(n_centers) * 0.1
offset_start = np.mean(Z)
params_start_rbf = np.concatenate([weights_start, [offset_start]])

# Оптимизация
print("Оптимизация RBF-сети (L-BFGS-B)...")
result_rbf = minimize(rbf_loss, params_start_rbf, method='L-BFGS-B')
weights_opt = result_rbf.x[:-1]
offset_opt = result_rbf.x[-1]

Z_pred_rbf = rbf_predict(X_raw, Y_raw, weights_opt, offset_opt)
residuals_rbf = Z - Z_pred_rbf
mse_rbf = np.mean(residuals_rbf ** 2)

print(f"\nMSE = {mse_rbf:.8f}, offset = {offset_opt:.4f}")
print("\nВеса нейронов:")
for j, w in enumerate(weights_opt):
    print(f"  Нейрон {j}: w={w:.6f}, центр=({centers[j, 0]:.4f}, {centers[j, 1]:.4f})")

significant = np.abs(weights_opt) > 1e-4
print(f"\nЗначимые нейроны (|w| > 1e-4): {np.sum(significant)} из {n_centers}")

print("\nНевязки RBF-сети:")
for i in range(len(X_raw)):
    print(f"  {i}: факт={Z[i]:.3f}, предск={Z_pred_rbf[i]:.3f}, невязка={residuals_rbf[i]:.6f}")


# РЕАЛЬНАЯ кривая обучения RBF (градиентный спуск)
def rbf_grad(params):
    weights = params[:-1]
    offset = params[-1]
    pred = rbf_predict(X_raw, Y_raw, weights, offset)
    err = pred - Z

    grad_weights = np.zeros(len(weights))
    for j, (cx, cy) in enumerate(centers):
        r2 = (X_raw - cx) ** 2 + (Y_raw - cy) ** 2
        phi = np.exp(-r2 / (2 * sigma ** 2))
        grad_weights[j] = 2 * np.mean(err * phi)

    grad_offset = 2 * np.mean(err)
    return np.concatenate([grad_weights, [grad_offset]])


params_gd_rbf = np.array(params_start_rbf, dtype=float)
lr_rbf = 0.01
losses_rbf = []
n_iter_rbf = 500

for step in range(n_iter_rbf):
    pred = rbf_predict(X_raw, Y_raw, params_gd_rbf[:-1], params_gd_rbf[-1])
    loss = np.mean((Z - pred) ** 2)
    losses_rbf.append(loss)
    grad = rbf_grad(params_gd_rbf)
    grad = np.clip(grad, -5, 5)
    params_gd_rbf -= lr_rbf * grad
    if step > 50 and losses_rbf[-1] > losses_rbf[-10]:
        lr_rbf *= 0.99

plt.figure(figsize=(10, 4))
plt.plot(range(len(losses_rbf)), losses_rbf, 'green', linewidth=2)
plt.yscale('log')
plt.xlabel('Итерация')
plt.ylabel('MSE')
plt.title('Кривая обучения (RBF-сеть)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_rbf.png', dpi=150)
plt.show()


# Визуализация RBF
def rbf_predict_grid(x, y):
    return rbf_predict(np.array([x]), np.array([y]), weights_opt, offset_opt)[0]


Z_grid_rbf_raw = np.zeros_like(X_grid_raw)
for i in range(X_grid_raw.shape[0]):
    for j in range(X_grid_raw.shape[1]):
        Z_grid_rbf_raw[i, j] = rbf_predict_grid(X_grid_raw[i, j], Y_grid_raw[i, j])

save_3d_and_contour(Z_grid_rbf_raw, 'RBF-сеть', 'surface_rbf', residuals_rbf, centers)

# ==============================================================================
# ВЫВОД АНАЛИТИЧЕСКОГО ВИДА
# ==============================================================================
print("\n" + "=" * 60)
print("ИТОГОВЫЕ МОДЕЛИ")
print("=" * 60)

print("\n📌 Константные модели:")
print(f"   MSE: z = {c_mse:.4f}")
print(f"   MAE: z = {c_mae:.4f}")

print("\n📌 Эллиптический параболоид:")
print(f"   z(x,y) = {a_opt:.6f}·x² + {b_opt:.6f}·y² + {c_opt:.6f}·x·y + {d_opt:.6f}·x + {e_opt:.6f}·y + {f_opt:.6f}")

print("\n📌 Двумерная гауссиана:")
print(
    f"   z(x,y) = {A_opt:.4f}·exp(-(x-{x0_opt:.4f})²/(2·{sx_opt:.4f}²) - (y-{y0_opt:.4f})²/(2·{sy_opt:.4f}²)) + {offset_opt:.4f}")

print("\n📌 RBF-сеть:")
print(f"   z(x,y) = {offset_opt:.4f}")
significant = np.abs(weights_opt) > 1e-4
for j, w in enumerate(weights_opt):
    if np.abs(w) > 1e-4:
        cx, cy = centers[j]
        print(f"           + {w:.4f}·exp(-((x-{cx:.4f})² + (y-{cy:.4f})²)/(2·{sigma:.4f}²))")
if np.sum(significant) < n_centers:
    print(f"   (остальные {n_centers - np.sum(significant)} нейронов имеют пренебрежимо малые веса)")

print("\n" + "=" * 60)
print("✅ ГОТОВО! Сохранены файлы:")
print("   loss_gaussian.png, surface_gaussian.png")
print("   loss_paraboloid.png, surface_paraboloid.png")
print("   loss_rbf.png, surface_rbf.png")
print("=" * 60)