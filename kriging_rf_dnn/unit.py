"""
模块0：公共工具函数
供所有模块调用的通用函数
"""

import numpy as np
from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


def interpolate_dem_to_grid(station_x, station_y, station_elev, grid_xx, grid_yy):
    """
    基于站点海拔，对网格点进行DEM插值（线性 + 最近邻补缺）

    参数:
        station_x, station_y: 站点坐标（米）
        station_elev: 站点海拔（米）
        grid_xx, grid_yy: 网格坐标矩阵

    返回:
        grid_dem: 网格海拔矩阵
    """
    points = np.column_stack([station_x, station_y])
    grid_points = np.column_stack([grid_xx.ravel(), grid_yy.ravel()])

    # 先尝试线性插值
    grid_dem = griddata(points, station_elev, grid_points, method='linear')
    grid_dem = grid_dem.reshape(grid_xx.shape)

    # 如果存在NaN（边界外），用最近邻填充
    if np.any(np.isnan(grid_dem)):
        grid_dem_nn = griddata(points, station_elev, grid_points, method='nearest')
        grid_dem = np.where(np.isnan(grid_dem), grid_dem_nn.reshape(grid_xx.shape), grid_dem)

    return grid_dem


def loo_cross_validation(station_x, station_y, station_z, station_elev,
                          variogram_model='spherical', n_estimators=50, max_depth=10):
    """
    留一法交叉验证，同时评估 OK 和 RF-OK

    参数:
        station_x, station_y: 站点坐标（米）
        station_z: 站点温度（°C）
        station_elev: 站点海拔（米）
        variogram_model: 半变异函数模型
        n_estimators, max_depth: RF参数

    返回:
        ok_errors, rf_errors: 两种方法的预测误差数组
    """
    n = len(station_x)
    ok_errors = []
    rf_errors = []

    print("执行留一法交叉验证...")

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]

        # --- 普通克里金 ---
        try:
            ok = OrdinaryKriging(
                station_x[train_idx], station_y[train_idx], station_z[train_idx],
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False
            )
            pred, _ = ok.execute('points', np.array([station_x[i]]), np.array([station_y[i]]))
            ok_errors.append(station_z[i] - pred[0])
        except:
            ok_errors.append(np.nan)

        # --- RF-OK（简化版）---
        try:
            X_tr = np.column_stack([
                station_x[train_idx], station_y[train_idx], station_elev[train_idx]
            ])
            X_te = np.column_stack([
                [station_x[i]], [station_y[i]], [station_elev[i]]
            ])

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)

            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            rf.fit(X_tr_scaled, station_z[train_idx])
            pred = rf.predict(X_te_scaled)
            rf_errors.append(station_z[i] - pred[0])
        except:
            rf_errors.append(np.nan)

    ok_errors = np.array([e for e in ok_errors if not np.isnan(e)])
    rf_errors = np.array([e for e in rf_errors if not np.isnan(e)])

    return ok_errors, rf_errors


def calc_metrics(errors):
    """计算误差指标"""
    if len(errors) == 0:
        return np.nan, np.nan
    mae = np.nanmean(np.abs(errors))
    rmse = np.sqrt(np.nanmean(errors**2))
    return mae, rmse


def get_grid(station_x, station_y, grid_res=50, pad_ratio=0.1):
    """
    根据站点范围创建插值网格

    返回:
        grid_xx, grid_yy: 网格坐标矩阵
        grid_x, grid_y: 网格坐标向量
    """
    x_range = (station_x.min(), station_x.max())
    y_range = (station_y.min(), station_y.max())

    pad_x = (x_range[1] - x_range[0]) * pad_ratio
    pad_y = (y_range[1] - y_range[0]) * pad_ratio

    grid_x = np.linspace(x_range[0] - pad_x, x_range[1] + pad_x, grid_res)
    grid_y = np.linspace(y_range[0] - pad_y, y_range[1] + pad_y, grid_res)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    return grid_xx, grid_yy, grid_x, grid_y


def stations_to_xy(df_stations, lon_center=None, lat_center=None):
    """
    将经纬度转换为平面坐标（单位：米）

    参数:
        df_stations: 包含 latitude, longitude 的 DataFrame
        lon_center, lat_center: 中心点经纬度（默认取均值）

    返回:
        df: 添加了 x, y 列的 DataFrame
        lon_center, lat_center: 使用的中心点坐标
    """
    df = df_stations.copy()
    if lon_center is None:
        lon_center = df['longitude'].mean()
    if lat_center is None:
        lat_center = df['latitude'].mean()

    # 1° ≈ 111.32 km
    df['x'] = (df['longitude'] - lon_center) * 111.32 * 1000
    df['y'] = (df['latitude'] - lat_center) * 111.32 * 1000

    return df, lon_center, lat_center