"""
从本地GSOD数据文件夹读取气象数据，对比 OK 与 RF-OK 插值效果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import griddata
import warnings
import os
from datetime import datetime
import glob

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 从本地GSOD文件夹读取数据
# ============================================================

def load_gsod_from_local(folder_path, target_lat=None, target_lon=None, radius_km=None, max_stations=50, year=2024):
    """
    从本地GSOD文件夹读取气象数据

    参数:
        folder_path: 本地GSOD CSV文件所在文件夹路径
        target_lat, target_lon: 目标中心点坐标（用于筛选站点）
        radius_km: 搜索半径（公里），如果为None则读取所有站点
        max_stations: 最大站点数
        year: 目标年份（仅读取该年份数据）

    返回:
        DataFrame: 包含站点坐标、海拔、温度等信息的表格
    """
    print(f"\n正在从本地文件夹读取GSOD数据: {folder_path}")

    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print(f"❌ 未找到CSV文件")
        return None

    print(f"找到 {len(csv_files)} 个CSV文件")

    all_stations = []
    file_count = 0

    for file_path in csv_files:
        try:
            # 读取CSV
            df = pd.read_csv(file_path)

            # 检查必要列
            if 'LATITUDE' not in df.columns or 'LONGITUDE' not in df.columns or 'TEMP' not in df.columns:
                continue

            # 筛选年份
            if 'DATE' in df.columns:
                df['DATE'] = pd.to_datetime(df['DATE'])
                df = df[df['DATE'].dt.year == year]
                if df.empty:
                    continue

            # 计算年均温（GSOD温度需要除以10）
            temp_mean = df['TEMP'].mean() / 10.0

            # 检查温度是否合理
            if np.isnan(temp_mean) or temp_mean < -50 or temp_mean > 50:
                continue

            # 提取站点信息
            lat = df['LATITUDE'].iloc[0]
            lon = df['LONGITUDE'].iloc[0]
            elev = df['ELEVATION'].iloc[0] if 'ELEVATION' in df.columns else 0

            # 检查经纬度是否有效
            if pd.isna(lat) or pd.isna(lon):
                continue
            if lat < -90 or lat > 90 or lon < -180 or lon > 180:
                continue

            all_stations.append({
                'station_id': os.path.basename(file_path).replace('.csv', ''),
                'latitude': lat,
                'longitude': lon,
                'elevation': elev,
                'temperature': temp_mean,
                'data_count': len(df)
            })
            file_count += 1

            # 进度提示
            if file_count % 1000 == 0:
                print(f"  已处理 {file_count} 个文件...")

        except Exception as e:
            continue

    if not all_stations:
        print("❌ 未读取到有效数据")
        return None

    df_stations = pd.DataFrame(all_stations)
    print(f"✅ 成功读取 {len(df_stations)} 个站点的数据")

    # 如果指定了中心点和半径，进行空间筛选
    if target_lat is not None and target_lon is not None and radius_km is not None:
        print(f"\n正在筛选 {target_lat}°N, {target_lon}°E 附近 {radius_km}km 范围内的站点...")

        # 计算与目标点的距离
        lat_diff = (df_stations['latitude'] - target_lat) * 111.32
        lon_diff = (df_stations['longitude'] - target_lon) * 111.32 * np.cos(np.radians(target_lat))
        df_stations['distance'] = np.sqrt(lat_diff**2 + lon_diff**2)

        # 筛选
        df_stations = df_stations[df_stations['distance'] <= radius_km]
        df_stations = df_stations.sort_values('distance').head(max_stations)
        print(f"筛选后剩余 {len(df_stations)} 个站点")

    # 删除距离列（如果存在）
    if 'distance' in df_stations.columns:
        df_stations = df_stations.drop(columns=['distance'])

    print(f"温度范围: {df_stations['temperature'].min():.2f}°C ~ {df_stations['temperature'].max():.2f}°C")
    print(f"海拔范围: {df_stations['elevation'].min():.1f}m ~ {df_stations['elevation'].max():.1f}m")

    return df_stations


def stations_to_xy(df_stations, lon_center=None, lat_center=None):
    """将经纬度转换为平面坐标"""
    df = df_stations.copy()
    if lon_center is None:
        lon_center = df['longitude'].mean()
    if lat_center is None:
        lat_center = df['latitude'].mean()

    df['x'] = (df['longitude'] - lon_center) * 111.32 * 1000
    df['y'] = (df['latitude'] - lat_center) * 111.32 * 1000
    return df, lon_center, lat_center


# ============================================================
# 2. 插值函数
# ============================================================

def interpolate_dem_to_grid(station_x, station_y, station_elev, grid_xx, grid_yy):
    """网格DEM插值"""
    points = np.column_stack([station_x, station_y])
    grid_points = np.column_stack([grid_xx.ravel(), grid_yy.ravel()])
    grid_dem = griddata(points, station_elev, grid_points, method='linear').reshape(grid_xx.shape)
    if np.any(np.isnan(grid_dem)):
        grid_dem_nn = griddata(points, station_elev, grid_points, method='nearest').reshape(grid_xx.shape)
        grid_dem = np.where(np.isnan(grid_dem), grid_dem_nn, grid_dem)
    return grid_dem


def perform_ok_interpolation(station_x, station_y, station_z, grid_res=50):
    x_range, y_range = (station_x.min(), station_x.max()), (station_y.min(), station_y.max())
    pad_x, pad_y = (x_range[1]-x_range[0])*0.1, (y_range[1]-y_range[0])*0.1
    grid_x = np.linspace(x_range[0]-pad_x, x_range[1]+pad_x, grid_res)
    grid_y = np.linspace(y_range[0]-pad_y, y_range[1]+pad_y, grid_res)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    try:
        ok = OrdinaryKriging(station_x, station_y, station_z, variogram_model='spherical', verbose=False)
        ok_grid, _ = ok.execute('grid', grid_x, grid_y)
        ok_grid = np.nan_to_num(ok_grid, nan=np.nanmean(station_z))
    except:
        ok_grid = np.full_like(grid_xx, np.nanmean(station_z))
    return grid_xx, grid_yy, ok_grid


def perform_rfok_interpolation(df_stations, grid_xx, grid_yy):
    x, y, z = df_stations['x'].values, df_stations['y'].values, df_stations['temperature'].values
    elev = df_stations['elevation'].values

    grid_dem = interpolate_dem_to_grid(x, y, elev, grid_xx, grid_yy)

    X_train = np.column_stack([x, y, elev])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, z)

    residuals = z - rf.predict(X_train_scaled)
    print(f"  RF R²: {rf.score(X_train_scaled, z):.4f}")

    if np.std(residuals) > 0.01:
        try:
            rk = OrdinaryKriging(x, y, residuals, variogram_model='spherical', verbose=False)
            res_grid, _ = rk.execute('grid', np.unique(grid_xx[:,0]), np.unique(grid_yy[0,:]))
            res_grid = np.nan_to_num(res_grid, nan=0)
        except:
            res_grid = np.zeros_like(grid_xx)
    else:
        res_grid = np.zeros_like(grid_xx)

    grid_feat = np.column_stack([grid_xx.ravel(), grid_yy.ravel(), grid_dem.ravel()])
    rf_pred = rf.predict(scaler.transform(grid_feat)).reshape(grid_xx.shape)

    return rf_pred + res_grid, rf, scaler


def loo_cv(df_stations):
    n = len(df_stations)
    x, y, z, e = df_stations['x'].values, df_stations['y'].values, df_stations['temperature'].values, df_stations['elevation'].values
    ok_err, rf_err = [], []
    print("执行留一法交叉验证...")

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        try:
            ok = OrdinaryKriging(x[train_idx], y[train_idx], z[train_idx], variogram_model='spherical', verbose=False)
            pred, _ = ok.execute('points', np.array([x[i]]), np.array([y[i]]))
            ok_err.append(z[i] - pred[0])
        except:
            ok_err.append(np.nan)

        try:
            X_tr = np.column_stack([x[train_idx], y[train_idx], e[train_idx]])
            X_te = np.column_stack([[x[i]], [y[i]], [e[i]]])
            scaler = StandardScaler()
            rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            rf.fit(scaler.fit_transform(X_tr), z[train_idx])
            pred = rf.predict(scaler.transform(X_te))
            rf_err.append(z[i] - pred[0])
        except:
            rf_err.append(np.nan)

    return np.array([e for e in ok_err if not np.isnan(e)]), np.array([e for e in rf_err if not np.isnan(e)])


# ============================================================
# 3. 主程序
# ============================================================

def main():
    # ========== 请修改以下配置 ==========
    # 本地GSOD数据文件夹路径
    DATA_FOLDER = r"D:\pycharm\Kriging and Machine Learning\2024_gsod_data"

    # 研究区域中心点（以广州为例）
    TARGET_LAT = 23.13
    TARGET_LON = 113.26

    # 搜索半径（公里）
    SEARCH_RADIUS = 1000

    # 最大站点数
    MAX_STATIONS = 100

    # 数据年份
    YEAR = 2024
    # ===================================

    # 读取本地数据
    df = load_gsod_from_local(DATA_FOLDER, TARGET_LAT, TARGET_LON, SEARCH_RADIUS, MAX_STATIONS, YEAR)

    if df is None or len(df) < 10:
        print("数据不足，程序退出")
        return

    # 坐标转换
    df, lon_center, lat_center = stations_to_xy(df)

    # OK插值
    print("\n执行 OK 插值...")
    gx, gy, ok_grid = perform_ok_interpolation(df['x'].values, df['y'].values, df['temperature'].values)

    # RF-OK插值
    print("执行 RF-OK 插值...")
    rfok_grid, rf, scaler = perform_rfok_interpolation(df, gx, gy)

    # 交叉验证
    ok_err, rf_err = loo_cv(df)

    # 精度指标
    ok_rmse = np.sqrt(np.nanmean(ok_err**2))
    rf_rmse = np.sqrt(np.nanmean(rf_err**2))
    ok_mae = np.nanmean(np.abs(ok_err))
    rf_mae = np.nanmean(np.abs(rf_err))

    print("\n" + "="*60)
    print("精度评估结果")
    print("="*60)
    print(f"OK   -> MAE: {ok_mae:.4f}°C, RMSE: {ok_rmse:.4f}°C")
    print(f"RFOK -> MAE: {rf_mae:.4f}°C, RMSE: {rf_rmse:.4f}°C")
    print(f"提升: RMSE {(1-rf_rmse/ok_rmse)*100:.2f}%, MAE {(1-rf_mae/ok_mae)*100:.2f}%")

    # 绘图
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'普通克里金 vs RF-OK 插值对比 (GSOD本地数据, {YEAR}年)', fontsize=16, fontweight='bold')

    axs[0,0].scatter(df['x']/1000, df['y']/1000, c=df['temperature'], s=60, cmap='RdYlBu_r', edgecolors='k')
    axs[0,0].set_title('(a) 站点分布与温度')
    axs[0,0].set_xlabel('km'); axs[0,0].set_ylabel('km')
    plt.colorbar(axs[0,0].collections[0], ax=axs[0,0])

    im = axs[0,1].contourf(gx/1000, gy/1000, ok_grid, 30, cmap='RdYlBu_r')
    axs[0,1].scatter(df['x']/1000, df['y']/1000, c='k', s=10, alpha=0.5)
    axs[0,1].set_title('(b) 普通克里金 (OK)')
    plt.colorbar(im, ax=axs[0,1])

    im2 = axs[0,2].contourf(gx/1000, gy/1000, rfok_grid, 30, cmap='RdYlBu_r')
    axs[0,2].scatter(df['x']/1000, df['y']/1000, c='k', s=10, alpha=0.5)
    axs[0,2].set_title('(c) RF-OK 插值')
    plt.colorbar(im2, ax=axs[0,2])

    im3 = axs[1,0].contourf(gx/1000, gy/1000, rfok_grid - ok_grid, 30, cmap='coolwarm')
    axs[1,0].set_title('(d) RF-OK 与 OK 差异')
    plt.colorbar(im3, ax=axs[1,0])

    bp = axs[1,1].boxplot([ok_err, rf_err], labels=['OK', 'RF-OK'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    axs[1,1].axhline(0, c='k', ls='--')
    axs[1,1].set_title('(e) 交叉验证误差')
    axs[1,1].grid(True, alpha=0.3)
    axs[1,1].set_ylabel('误差 (°C)')
    axs[1,1].text(0.5, -0.15, f'OK RMSE: {ok_rmse:.3f}°C\nRF RMSE: {rf_rmse:.3f}°C',
                  transform=axs[1,1].transAxes, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

    axs[1,2].text(0.5, 0.5, f'站点数: {len(df)}\nRMSE 降低: {(1-rf_rmse/ok_rmse)*100:.1f}%\nMAE 降低: {(1-rf_mae/ok_mae)*100:.1f}%',
                  transform=axs[1,2].transAxes, ha='center', va='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axs[1,2].set_title('(f) 精度汇总')
    axs[1,2].axis('off')

    plt.tight_layout()
    plt.savefig('gsod_local_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✅ 对比图已保存为: gsod_local_comparison.png")


if __name__ == "__main__":
    main()