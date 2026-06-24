"""
模块1：数据加载
从本地GSOD文件夹读取气象数据，或生成模拟数据
"""

import numpy as np
import pandas as pd
import os
import glob


def load_gsod_from_local(folder_path, target_lat, target_lon, radius_km,
                         max_stations=100, year=2024):
    """
    从本地GSOD文件夹读取气象数据

    参数:
        folder_path: GSOD CSV文件所在文件夹
        target_lat, target_lon: 中心点坐标
        radius_km: 搜索半径
        max_stations: 最大站点数
        year: 数据年份

    返回:
        DataFrame: 包含 station_id, latitude, longitude, elevation, temperature
    """
    print(f"\n正在从本地读取GSOD数据: {folder_path}")

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print("未找到CSV文件，使用模拟数据")
        return generate_simulated_data()

    print(f"找到 {len(csv_files)} 个CSV文件")

    all_stations = []
    for i, file_path in enumerate(csv_files):
        if i % 1000 == 0 and i > 0:
            print(f"  已处理 {i} 个文件...")
        try:
            df = pd.read_csv(file_path)
            if 'LATITUDE' not in df.columns or 'LONGITUDE' not in df.columns or 'TEMP' not in df.columns:
                continue

            if 'DATE' in df.columns:
                df['DATE'] = pd.to_datetime(df['DATE'])
                df = df[df['DATE'].dt.year == year]
                if df.empty:
                    continue

            temp_mean = df['TEMP'].mean() / 10.0
            if np.isnan(temp_mean) or temp_mean < -50 or temp_mean > 50:
                continue

            lat, lon = df['LATITUDE'].iloc[0], df['LONGITUDE'].iloc[0]
            if pd.isna(lat) or pd.isna(lon):
                continue
            if lat < -90 or lat > 90 or lon < -180 or lon > 180:
                continue

            elev = df['ELEVATION'].iloc[0] if 'ELEVATION' in df.columns else 0

            all_stations.append({
                'station_id': os.path.basename(file_path).replace('.csv', ''),
                'latitude': lat,
                'longitude': lon,
                'elevation': elev,
                'temperature': temp_mean
            })
        except:
            continue

    if not all_stations:
        print("未读取到有效数据，使用模拟数据")
        return generate_simulated_data()

    df_stations = pd.DataFrame(all_stations)
    print(f"成功读取 {len(df_stations)} 个站点")

    # 空间筛选
    lat_diff = (df_stations['latitude'] - target_lat) * 111.32
    lon_diff = (df_stations['longitude'] - target_lon) * 111.32 * np.cos(np.radians(target_lat))
    df_stations['distance'] = np.sqrt(lat_diff ** 2 + lon_diff ** 2)
    df_stations = df_stations[df_stations['distance'] <= radius_km]
    df_stations = df_stations.sort_values('distance').head(max_stations)

    print(f"筛选后剩余 {len(df_stations)} 个站点")
    print(f"温度范围: {df_stations['temperature'].min():.2f} ~ {df_stations['temperature'].max():.2f}°C")
    print(f"海拔范围: {df_stations['elevation'].min():.1f} ~ {df_stations['elevation'].max():.1f}m")

    return df_stations


def generate_simulated_data(n_stations=80, seed=42):
    """
    生成模拟数据（含复杂地形），用于算法验证
    """
    print("\n生成模拟数据...")
    np.random.seed(seed)

    x = np.random.uniform(0, 10, n_stations)
    y = np.random.uniform(0, 10, n_stations)

    # 复杂地形：多个山脊 + 山谷 + 微地形
    dem = (200 + 15 * x + 5 * y +
           300 * np.exp(-((x - 2) ** 2 + (y - 3) ** 2) / 2) +
           400 * np.exp(-((x - 8) ** 2 + (y - 7) ** 2) / 3) +
           -200 * np.exp(-((x - 5) ** 2 + (y - 5) ** 2) / 1.5) +
           50 * np.sin(x * 1.5) * np.cos(y * 1.2))

    # 温度 = 25 - 0.006*海拔 + 噪声
    temp = 25 - 0.006 * dem + np.random.normal(0, 0.5, n_stations)

    df = pd.DataFrame({
        'station_id': [f'S{i:03d}' for i in range(n_stations)],
        'latitude': y * 0.01 + 23.0,
        'longitude': x * 0.01 + 113.0,
        'elevation': dem,
        'temperature': temp
    })

    print(f"生成了 {len(df)} 个站点")
    print(f"温度范围: {temp.min():.2f} ~ {temp.max():.2f}°C")
    print(f"海拔范围: {dem.min():.1f} ~ {dem.max():.1f}m")
    return df


def stations_to_xy(df_stations):
    """将经纬度转换为平面坐标（单位：米）"""
    df = df_stations.copy()
    lon_center = df['longitude'].mean()
    lat_center = df['latitude'].mean()
    df['x'] = (df['longitude'] - lon_center) * 111.32 * 1000
    df['y'] = (df['latitude'] - lat_center) * 111.32 * 1000
    return df, lon_center, lat_center