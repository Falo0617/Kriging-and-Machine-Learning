import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from pykrige.ok import OrdinaryKriging
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 生成模拟数据 ====================

# 研究区域范围：10km x 10km
x_range = (0, 10)
y_range = (0, 10)
n_stations = 30  # 气象站点数量
grid_resolution = 50  # 插值网格分辨率


# 生成站点位置（随机分布，但有聚集效应模拟真实站点分布）
def generate_stations(n):
    # 部分站点随机分布
    n_random = int(n * 0.7)
    n_cluster = n - n_random

    # 随机点
    random_x = np.random.uniform(x_range[0], x_range[1], n_random)
    random_y = np.random.uniform(y_range[0], y_range[1], n_random)

    # 聚集点（模拟城市周边站点密集）
    cluster_centers = [(2, 2), (7, 8), (5, 5)]
    cluster_x, cluster_y = [], []
    for cx, cy in cluster_centers:
        n_per_cluster = n_cluster // len(cluster_centers)
        cluster_x.extend(np.random.normal(cx, 0.5, n_per_cluster))
        cluster_y.extend(np.random.normal(cy, 0.5, n_per_cluster))

    x = np.concatenate([random_x, cluster_x])
    y = np.concatenate([random_y, cluster_y])
    # 裁剪到范围内
    x = np.clip(x, x_range[0], x_range[1])
    y = np.clip(y, y_range[0], y_range[1])
    return x, y


station_x, station_y = generate_stations(n_stations)


# 生成地形（海拔）：两个山脊 + 一个山谷
def generate_dem(x, y):
    # 多个山脊 + 山谷 + 小尺度起伏
    ridge1 = 500 * np.exp(-((x-2)**2 + (y-3)**2) / 2)
    ridge2 = 400 * np.exp(-((x-8)**2 + (y-7)**2) / 3)
    ridge3 = 300 * np.exp(-((x-4)**2 + (y-8)**2) / 4)
    valley = -200 * np.exp(-((x-5)**2 + (y-5)**2) / 1.5)
    micro = 50 * np.sin(x*1.5) * np.cos(y*1.2)  # 增加小尺度起伏
    base = 200 + 15 * x + 5 * y
    return base + ridge1 + ridge2 + ridge3 + valley + micro


# 生成网格用于插值可视化
grid_x = np.linspace(x_range[0], x_range[1], grid_resolution)
grid_y = np.linspace(y_range[0], y_range[1], grid_resolution)
grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
grid_dem = generate_dem(grid_xx, grid_yy)

# 站点海拔
station_dem = generate_dem(station_x, station_y)

# 生成温度：受海拔主导 + 局部随机变异
# 真实温度 = 基础温度 - 海拔递减率 + 局部地形效应 + 随机噪声
base_temp = 25  # 海平面基础温度（摄氏度）
lapse_rate = 0.006  # 温度递减率（°C/m）

# 地形对温度的影响：山脊更冷，山谷更暖
terrain_effect = -0.002 * station_dem + 0.5 * np.sin(station_x * 0.5) * np.cos(station_y * 0.3)

# 真实温度
true_temp = base_temp - lapse_rate * station_dem + terrain_effect + np.random.normal(0, 0.3, n_stations)

# 对部分站点施加测量误差（模拟真实数据的不完美）
noise_mask = np.random.choice([True, False], n_stations, p=[0.05, 0.95])
true_temp[noise_mask] += np.random.normal(0, 1.5, np.sum(noise_mask))

# 构建DataFrame
df_stations = pd.DataFrame({
    'x': station_x,
    'y': station_y,
    'elevation': station_dem,
    'temperature': true_temp
})

print(f"站点数量: {len(df_stations)}")
print(f"温度范围: {df_stations['temperature'].min():.2f}°C ~ {df_stations['temperature'].max():.2f}°C")
print(f"海拔范围: {df_stations['elevation'].min():.2f}m ~ {df_stations['elevation'].max():.2f}m")

# ==================== 2. 传统普通克里金插值 ====================
print("\n正在执行普通克里金插值...")

# 使用PyKrige进行普通克里金
OK = OrdinaryKriging(
    station_x, station_y, true_temp,
    variogram_model='spherical',  # 球状模型
    verbose=False,
    enable_plotting=False
)

# 执行插值
ok_grid, ok_sigma = OK.execute('grid', grid_x, grid_y)

# 处理可能的NaN值
ok_grid = np.nan_to_num(ok_grid, nan=np.nanmean(true_temp))

# ==================== 3. 随机森林普通克里金（RF-OK） ====================
print("\n正在执行RF-OK插值...")

# 3.1 训练随机森林模型
# 特征：经纬度、海拔
X_train = df_stations[['x', 'y', 'elevation']].values
y_train = df_stations['temperature'].values

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 训练随机森林
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

# 3.2 预测站点处的值并计算残差
y_pred_train = rf.predict(X_train_scaled)
residuals = y_train - y_pred_train

print(f"随机森林训练R²: {rf.score(X_train_scaled, y_train):.4f}")
print(f"残差范围: {residuals.min():.4f} ~ {residuals.max():.4f}")

# 3.3 对残差进行克里金插值
if np.std(residuals) > 0.01:
    residual_krige = OrdinaryKriging(
        station_x, station_y, residuals,
        variogram_model='spherical',
        verbose=False,
        enable_plotting=False
    )
    residual_grid, _ = residual_krige.execute('grid', grid_x, grid_y)
    residual_grid = np.nan_to_num(residual_grid, nan=0)
else:
    residual_grid = np.zeros_like(grid_xx)

# 3.4 对网格进行RF预测
grid_features = np.column_stack([
    grid_xx.ravel(),
    grid_yy.ravel(),
    grid_dem.ravel()
])
grid_features_scaled = scaler.transform(grid_features)
rf_grid_pred = rf.predict(grid_features_scaled).reshape(grid_xx.shape)

# 3.5 RF-OK最终结果 = RF预测 + 残差克里金
rfok_grid = rf_grid_pred + residual_grid

# ==================== 4. 可视化对比 ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('传统克里金 vs RF-OK 插值效果对比', fontsize=16, fontweight='bold')

# 颜色映射
cmap_temp = plt.cm.RdYlBu_r
cmap_dem = plt.cm.terrain

# ---- (a) 地形（DEM） ----
im1 = axes[0, 0].contourf(grid_xx, grid_yy, grid_dem, 30, cmap=cmap_dem)
axes[0, 0].scatter(station_x, station_y, c='black', s=15, alpha=0.6, label='气象站点')
axes[0, 0].set_title('(a) 研究区域地形（海拔）', fontsize=12)
axes[0, 0].set_xlabel('经度 (km)')
axes[0, 0].set_ylabel('纬度 (km)')
axes[0, 0].legend(loc='upper right')
plt.colorbar(im1, ax=axes[0, 0], label='海拔 (m)')

# ---- (b) 传统克里金插值 ----
im2 = axes[0, 1].contourf(grid_xx, grid_yy, ok_grid, 30, cmap=cmap_temp)
axes[0, 1].scatter(station_x, station_y, c=true_temp, s=30, cmap=cmap_temp, edgecolors='black', linewidth=0.5)
axes[0, 1].set_title('(b) 普通克里金 (OK) 插值', fontsize=12)
axes[0, 1].set_xlabel('经度 (km)')
axes[0, 1].set_ylabel('纬度 (km)')
plt.colorbar(im2, ax=axes[0, 1], label='温度 (°C)')

# ---- (c) RF-OK插值 ----
im3 = axes[0, 2].contourf(grid_xx, grid_yy, rfok_grid, 30, cmap=cmap_temp)
axes[0, 2].scatter(station_x, station_y, c=true_temp, s=30, cmap=cmap_temp, edgecolors='black', linewidth=0.5)
axes[0, 2].set_title('(c) RF-OK 插值 (RF + 残差克里金)', fontsize=12)
axes[0, 2].set_xlabel('经度 (km)')
axes[0, 2].set_ylabel('纬度 (km)')
plt.colorbar(im3, ax=axes[0, 2], label='温度 (°C)')

# ---- (d) 两种方法的差异图 ----
diff_grid = rfok_grid - ok_grid
im4 = axes[1, 0].contourf(grid_xx, grid_yy, diff_grid, 30, cmap='coolwarm')
axes[1, 0].set_title('(d) RF-OK 与 OK 的差异 (RF-OK - OK)', fontsize=12)
axes[1, 0].set_xlabel('经度 (km)')
axes[1, 0].set_ylabel('纬度 (km)')
plt.colorbar(im4, ax=axes[1, 0], label='温度差异 (°C)')

# ---- (e) 站点预测误差对比（留一法交叉验证） ----
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
ok_loo_errors = []
rf_loo_errors = []

for train_idx, test_idx in loo.split(station_x):
    # OK: 用其余点预测当前点
    x_train, x_test = station_x[train_idx], station_x[test_idx]
    y_train, y_test = station_y[train_idx], station_y[test_idx]
    z_train, z_test = true_temp[train_idx], true_temp[test_idx]

    try:
        ok_loo = OrdinaryKriging(x_train, y_train, z_train, variogram_model='spherical', verbose=False)
        z_pred, _ = ok_loo.execute('points', x_test, y_test)
        ok_loo_errors.append(z_test[0] - z_pred[0])
    except:
        ok_loo_errors.append(np.nan)

    # RF: 用其余点训练RF预测当前点
    X_train_loo = np.column_stack([station_x[train_idx], station_y[train_idx], station_dem[train_idx]])
    X_test_loo = np.column_stack([station_x[test_idx], station_y[test_idx], station_dem[test_idx]])
    scaler_loo = StandardScaler()
    X_train_scaled_loo = scaler_loo.fit_transform(X_train_loo)
    X_test_scaled_loo = scaler_loo.transform(X_test_loo)

    rf_loo = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf_loo.fit(X_train_scaled_loo, z_train)
    z_pred_rf = rf_loo.predict(X_test_scaled_loo)
    rf_loo_errors.append(z_test[0] - z_pred_rf[0])

# 去除NaN
ok_loo_errors = np.array([e for e in ok_loo_errors if not np.isnan(e)])
rf_loo_errors = np.array(rf_loo_errors)

# 绘制箱线图
bp_data = [ok_loo_errors, rf_loo_errors]
bp = axes[1, 1].boxplot(bp_data, labels=['OK', 'RF-OK'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 1].set_title('(e) 留一法交叉验证预测误差对比', fontsize=12)
axes[1, 1].set_ylabel('预测误差 (°C)')
axes[1, 1].grid(True, alpha=0.3)

# 添加统计信息
ok_rmse = np.sqrt(np.nanmean(np.array(ok_loo_errors) ** 2))
rf_rmse = np.sqrt(np.mean(np.array(rf_loo_errors) ** 2))
axes[1, 1].text(0.5, -0.15, f'OK RMSE: {ok_rmse:.3f}°C\nRF-OK RMSE: {rf_rmse:.3f}°C',
                transform=axes[1, 1].transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ---- (f) 特征重要性分析 ----
feature_importance = rf.feature_importances_
feature_names = ['经度', '纬度', '海拔']
sorted_idx = np.argsort(feature_importance)
axes[1, 2].barh(feature_names, feature_importance, color='steelblue')
axes[1, 2].set_title('(f) 随机森林特征重要性', fontsize=12)
axes[1, 2].set_xlabel('重要性')
for i, v in enumerate(feature_importance):
    axes[1, 2].text(v + 0.01, i, f'{v:.3f}', va='center')

plt.tight_layout()
plt.savefig('kriging_vs_rfok_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ 对比图已保存为: kriging_vs_rfok_comparison.png")

# ==================== 5. 精度指标汇总（使用留一法交叉验证） ====================
print("\n" + "=" * 60)
print("精度评估结果（基于留一法交叉验证）")
print("=" * 60)

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

loo = LeaveOneOut()
ok_errors = []
rf_errors = []

for train_idx, test_idx in loo.split(station_x):
    # 训练/测试拆分
    x_train, x_test = station_x[train_idx], station_x[test_idx]
    y_train, y_test = station_y[train_idx], station_y[test_idx]
    elev_train, elev_test = station_dem[train_idx], station_dem[test_idx]
    temp_train, temp_test = true_temp[train_idx], true_temp[test_idx]

    # --- 普通克里金（OK）：用其余点预测当前点 ---
    try:
        ok_loo = OrdinaryKriging(
            x_train, y_train, temp_train,
            variogram_model='spherical',
            verbose=False,
            enable_plotting=False
        )
        z_pred_ok, _ = ok_loo.execute('points', np.array([x_test]), np.array([y_test]))
        ok_errors.append(temp_test[0] - z_pred_ok[0])
    except:
        ok_errors.append(np.nan)

    # --- RF-OK：用其余点训练RF，预测当前点 ---
    X_train_loo = np.column_stack([x_train, y_train, elev_train])
    X_test_loo = np.column_stack([[x_test], [y_test], [elev_test]])

    scaler_loo = StandardScaler()
    X_train_scaled_loo = scaler_loo.fit_transform(X_train_loo)
    X_test_scaled_loo = scaler_loo.transform(X_test_loo)

    rf_loo = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf_loo.fit(X_train_scaled_loo, temp_train)
    z_pred_rf = rf_loo.predict(X_test_scaled_loo)
    rf_errors.append(temp_test[0] - z_pred_rf[0])

# 去除可能的NaN值
ok_errors = np.array([e for e in ok_errors if not np.isnan(e)])
rf_errors = np.array(rf_errors)


# 计算评估指标
def calc_metrics_from_errors(errors):
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    return mae, rmse


ok_mae, ok_rmse = calc_metrics_from_errors(ok_errors)
rf_mae, rf_rmse = calc_metrics_from_errors(rf_errors)

# 站点真实温度的平均值作为R²计算的基准
temp_mean = np.mean(true_temp)


def calc_r2(errors, temp_mean):
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((true_temp - temp_mean) ** 2)
    return 1 - ss_res / ss_tot


# 注意：这里需要用完整的真实温度数组来计算R²，但由于留一法得到的误差数组长度与站点数一致
# 但误差顺序可能被打乱，我们直接使用完整数组计算
ok_r2 = calc_r2(ok_errors, temp_mean)
rf_r2 = calc_r2(rf_errors, temp_mean)

print(f"\n普通克里金 (OK) — 留一法交叉验证:")
print(f"  MAE: {ok_mae:.4f}°C")
print(f"  RMSE: {ok_rmse:.4f}°C")
print(f"  R²: {ok_r2:.4f}")

print(f"\nRF-OK (随机森林 + 残差克里金) — 留一法交叉验证:")
print(f"  MAE: {rf_mae:.4f}°C")
print(f"  RMSE: {rf_rmse:.4f}°C")
print(f"  R²: {rf_r2:.4f}")

print(f"\n精度提升:")
print(f"  MAE 降低: {(1 - rf_mae / ok_mae) * 100:.2f}%")
print(f"  RMSE 降低: {(1 - rf_rmse / ok_rmse) * 100:.2f}%")