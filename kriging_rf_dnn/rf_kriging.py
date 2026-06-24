"""
模块2：第一步 — 随机森林动态调参
以地形特征为输入，预测半变异函数的变程、基台值、块金值
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import warnings

warnings.filterwarnings('ignore')


def estimate_local_variogram_params(station_x, station_y, station_z,
                                    station_elev, k_neighbors=10):
    """
    为每个站点估算局部半变异函数参数（作为RF的训练标签）

    方法：对每个站点，用其邻近k个点拟合球状模型，提取变程/基台值/块金值

    返回:
        ranges, sills, nuggets: 每个站点的局部参数
    """
    n = len(station_x)
    ranges = np.full(n, np.nan)
    sills = np.full(n, np.nan)
    nuggets = np.full(n, np.nan)

    # 构建KDTree用于快速邻域搜索
    points = np.column_stack([station_x, station_y])
    tree = cKDTree(points)

    for i in range(n):
        # 搜索邻近点（除自身外）
        dists, idxs = tree.query(points[i], k=min(k_neighbors + 1, n))
        idxs = idxs[dists > 0][:k_neighbors]  # 排除自身
        if len(idxs) < 5:
            continue

        # 提取邻域数据
        neigh_x = station_x[idxs]
        neigh_y = station_y[idxs]
        neigh_z = station_z[idxs]

        # 计算邻域内的经验半变异函数
        # 简化：用距离和变异拟合球状模型
        n_pts = len(neigh_x)
        distances = []
        semivariances = []

        for j in range(n_pts):
            for k in range(j + 1, n_pts):
                d = np.sqrt((neigh_x[j] - neigh_x[k]) ** 2 + (neigh_y[j] - neigh_y[k]) ** 2)
                v = 0.5 * (neigh_z[j] - neigh_z[k]) ** 2
                distances.append(d)
                semivariances.append(v)

        distances = np.array(distances)
        semivariances = np.array(semivariances)

        if len(distances) < 10:
            continue

        # 用百分位法估算参数（简化版）
        max_dist = np.percentile(distances, 80)
        min_dist = np.min(distances[distances > 0])

        # 变程 ≈ 最大有效距离的60-80%
        rng = max_dist * 0.7
        # 基台值 ≈ 远距离半变异的平均值
        far_mask = distances > max_dist * 0.6
        sil = np.median(semivariances[far_mask]) if np.any(far_mask) else np.median(semivariances)
        # 块金值 ≈ 近距离半变异的截距
        near_mask = distances < max_dist * 0.2
        nug = np.median(semivariances[near_mask]) if np.any(near_mask) else 0

        ranges[i] = rng
        sills[i] = sil
        nuggets[i] = max(0, nug)

    # 填充NaN值（用全局中位数）
    ranges = np.nan_to_num(ranges, nan=np.nanmedian(ranges))
    sills = np.nan_to_num(sills, nan=np.nanmedian(sills))
    nuggets = np.nan_to_num(nuggets, nan=np.nanmedian(nuggets))

    return ranges, sills, nuggets


def train_rf_parameter_predictor(station_x, station_y, station_elev,
                                 station_z, n_estimators=100, max_depth=12):
    """
    训练随机森林模型：地形特征 → 变程参数

    返回:
        rf_range, rf_sill, rf_nugget: 三个独立的RF模型
        scaler: 特征标准化器
        params_dict: 包含所有训练好的模型和参数
    """
    print("\n第一步：训练随机森林预测变程参数...")

    # 1. 计算局部半变异函数参数（作为标签）
    local_ranges, local_sills, local_nuggets = estimate_local_variogram_params(
        station_x, station_y, station_z, station_elev
    )

    # 2. 构建特征
    X = np.column_stack([station_x, station_y, station_elev])
    feature_names = ['x', 'y', 'elevation']

    # 3. 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. 分别训练三个RF模型（预测变程、基台值、块金值）
    rf_range = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=5, random_state=42, n_jobs=-1
    )
    rf_range.fit(X_scaled, local_ranges)

    rf_sill = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=5, random_state=42, n_jobs=-1
    )
    rf_sill.fit(X_scaled, local_sills)

    rf_nugget = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=5, random_state=42, n_jobs=-1
    )
    rf_nugget.fit(X_scaled, local_nuggets)

    # 5. 打印训练结果
    r2_range = rf_range.score(X_scaled, local_ranges)
    r2_sill = rf_sill.score(X_scaled, local_sills)
    r2_nugget = rf_nugget.score(X_scaled, local_nuggets)
    print(f"  变程预测 R²: {r2_range:.4f}")
    print(f"  基台值预测 R²: {r2_sill:.4f}")
    print(f"  块金值预测 R²: {r2_nugget:.4f}")

    return {
        'rf_range': rf_range,
        'rf_sill': rf_sill,
        'rf_nugget': rf_nugget,
        'scaler': scaler,
        'feature_names': feature_names,
        'r2_scores': {'range': r2_range, 'sill': r2_sill, 'nugget': r2_nugget}
    }


def predict_parameters(params_dict, grid_xx, grid_yy, grid_dem):
    """
    用训练好的RF模型预测网格点的变程参数

    返回:
        range_grid, sill_grid, nugget_grid: 三个参数的网格
    """
    scaler = params_dict['scaler']
    rf_range = params_dict['rf_range']
    rf_sill = params_dict['rf_sill']
    rf_nugget = params_dict['rf_nugget']

    # 构建网格特征
    grid_features = np.column_stack([grid_xx.ravel(), grid_yy.ravel(), grid_dem.ravel()])
    grid_features_scaled = scaler.transform(grid_features)

    # 预测
    range_grid = rf_range.predict(grid_features_scaled).reshape(grid_xx.shape)
    sill_grid = rf_sill.predict(grid_features_scaled).reshape(grid_xx.shape)
    nugget_grid = rf_nugget.predict(grid_features_scaled).reshape(grid_xx.shape)

    # 物理约束：确保参数合理
    range_grid = np.maximum(range_grid, np.min(np.diff(grid_xx[0, :])) * 2)  # 不小于2倍网格间距
    sill_grid = np.maximum(sill_grid, nugget_grid + 0.01)  # 基台值 > 块金值
    nugget_grid = np.maximum(nugget_grid, 0)

    return range_grid, sill_grid, nugget_grid