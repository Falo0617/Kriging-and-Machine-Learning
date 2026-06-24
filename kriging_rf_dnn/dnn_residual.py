"""
模块3：第三步 — 深度学习残差修正
用深度神经网络学习并修正克里金插值的残差
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def train_dnn_residual_corrector(station_x, station_y, station_elev,
                                 station_z, kriged_at_stations,
                                 hidden_layer_sizes=(64, 32, 16)):
    """
    训练DNN模型：地形特征 → 残差

    输入: 站点的地形特征（x, y, elevation）
    输出: 残差 = 真实温度 - 克里金预测温度

    返回:
        dnn_model: 训练好的MLP模型
        scaler: 特征标准化器
        residual_rmse: 训练集上的残差RMSE
    """
    print("\n第三步：训练深度学习残差修正模型...")

    # 计算残差
    residuals = station_z - kriged_at_stations
    print(f"  原始残差范围: {residuals.min():.4f} ~ {residuals.max():.4f}°C")
    print(f"  原始残差标准差: {np.std(residuals):.4f}°C")

    # 构建特征
    X = np.column_stack([station_x, station_y, station_elev])

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练MLP（作为DNN的轻量替代，也可用PyTorch实现更复杂的网络）
    dnn = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    dnn.fit(X_scaled, residuals)

    # 训练集预测
    pred_residuals = dnn.predict(X_scaled)
    residual_rmse = np.sqrt(np.mean((residuals - pred_residuals) ** 2))
    residual_r2 = 1 - np.sum((residuals - pred_residuals) ** 2) / np.sum((residuals - np.mean(residuals)) ** 2)

    print(f"  残差修正 R²: {residual_r2:.4f}")
    print(f"  残差修正 RMSE: {residual_rmse:.4f}°C")

    return {
        'dnn': dnn,
        'scaler': scaler,
        'residual_rmse': residual_rmse,
        'residual_r2': residual_r2
    }


def apply_dnn_correction(dnn_params, grid_xx, grid_yy, grid_dem, kriged_grid):
    """
    用DNN对克里金插值结果进行残差修正

    返回:
        corrected_grid: 修正后的预测网格
        residual_grid: DNN预测的残差网格
    """
    print("\n第三步：应用深度学习残差修正...")

    dnn = dnn_params['dnn']
    scaler = dnn_params['scaler']

    # 构建网格特征
    grid_features = np.column_stack([grid_xx.ravel(), grid_yy.ravel(), grid_dem.ravel()])
    grid_features_scaled = scaler.transform(grid_features)

    # 预测残差
    residual_grid = dnn.predict(grid_features_scaled).reshape(grid_xx.shape)

    # 修正结果 = 克里金插值 + 预测残差
    corrected_grid = kriged_grid + residual_grid

    print(f"  修正后范围: {corrected_grid.min():.2f} ~ {corrected_grid.max():.2f}°C")
    print(f"  残差修正量范围: {residual_grid.min():.4f} ~ {residual_grid.max():.4f}°C")

    return corrected_grid, residual_grid


def complete_rf_dnn_workflow(params_dict, dnn_params, station_x, station_y,
                             station_elev, station_z, grid_xx, grid_yy, grid_dem):
    """
    完整三步流程：RF定参 → 克里金插值 → DNN残差修正
    """
    # 第一步+第二步：RF定参 + 克里金插值
    from rf_kriging import predict_parameters
    from kriging_interp import interpolate_with_dynamic_params

    range_grid, sill_grid, nugget_grid = predict_parameters(params_dict, grid_xx, grid_yy, grid_dem)

    # 用RF预测的参数做克里金插值
    kriged_grid = interpolate_with_dynamic_params(
        station_x, station_y, station_z,
        grid_xx, grid_yy,
        range_grid, sill_grid, nugget_grid
    )

    # 第三步：DNN修正残差
    # 需要先获取克里金在站点位置的预测值
    # 这里简化：用网格插值结果在站点位置的采样值
    from scipy.interpolate import griddata
    kriged_at_stations = griddata(
        (grid_xx.ravel(), grid_yy.ravel()),
        kriged_grid.ravel(),
        (station_x, station_y),
        method='linear'
    )
    # 处理NaN
    nan_mask = np.isnan(kriged_at_stations)
    if np.any(nan_mask):
        kriged_at_stations[nan_mask] = np.nanmean(kriged_at_stations)

    # 直接调用本模块内的训练函数（避免自导入导致循环）
    dnn_params_temp = train_dnn_residual_corrector(
        station_x, station_y, station_elev, station_z, kriged_at_stations
    )

    corrected_grid, residual_grid = apply_dnn_correction(
        dnn_params_temp, grid_xx, grid_yy, grid_dem, kriged_grid
    )

    return {
        'kriged_grid': kriged_grid,
        'corrected_grid': corrected_grid,
        'residual_grid': residual_grid,
        'range_grid': range_grid,
        'sill_grid': sill_grid,
        'nugget_grid': nugget_grid
    }