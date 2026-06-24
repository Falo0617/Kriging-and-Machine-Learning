"""
主程序：完整的三步流程（包含基于新增地形/局部统计特征的残差建模与CV诊断打印）
1. 数据加载
2. 随机森林动态调参
3. 克里金初步插值（RF-OK）
4. 用特征工程 + GBDT/MLP 拟合残差并在网格上修正
5. 留一法交叉验证与可视化对比
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import warnings

warnings.filterwarnings('ignore')

# 导入各模块
from data_loader import load_gsod_from_local, generate_simulated_data, stations_to_xy
from rf_kriging import train_rf_parameter_predictor, predict_parameters
from kriging_interp import interpolate_with_dynamic_params
from dnn_residual import train_dnn_residual_corrector, apply_dnn_correction
from visualize import plot_full_comparison

# 特征工程模块（你新建的）
from feature_engineering import build_station_feature_matrix, build_grid_feature_matrix

# 导入克里金用于传统OK对比
from pykrige.ok import OrdinaryKriging

# ML 库（LightGBM 优先，否则使用 sklearn 的 RandomForest）
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    LGB_AVAILABLE = False

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def perform_traditional_ok(station_x, station_y, station_z, grid_xx, grid_yy):
    """传统普通克里金（作为基准对比）"""
    print("\n执行传统普通克里金（基准）...")
    try:
        ok = OrdinaryKriging(station_x, station_y, station_z,
                             variogram_model='spherical', verbose=False, enable_plotting=False)
        # 正确构造 grid_x（横向）和 grid_y（纵向）
        grid_x = np.unique(grid_xx[0, :])
        grid_y = np.unique(grid_yy[:, 0])
        ok_grid, _ = ok.execute('grid', grid_x, grid_y)
        ok_grid = np.asarray(ok_grid)
        # 确保返回的网格与 grid_xx 同形
        if ok_grid.shape != grid_xx.shape:
            try:
                ok_grid = ok_grid.reshape(grid_yy.shape)
            except:
                ok_grid = np.full_like(grid_xx, np.nanmean(station_z))
        ok_grid = np.nan_to_num(ok_grid, nan=np.nanmean(station_z))
    except Exception as e:
        print(f"  传统OK失败: {e}")
        ok_grid = np.full_like(grid_xx, np.nanmean(station_z))
    return ok_grid


def interpolate_dem_to_grid(station_x, station_y, station_elev, grid_xx, grid_yy):
    """网格DEM插值（线性+最近邻补缺）"""
    points = np.column_stack([station_x, station_y])
    grid_points = np.column_stack([grid_xx.ravel(), grid_yy.ravel()])
    grid_dem = griddata(points, station_elev, grid_points, method='linear')
    grid_dem = grid_dem.reshape(grid_xx.shape)
    if np.any(np.isnan(grid_dem)):
        grid_dem_nn = griddata(points, station_elev, grid_points, method='nearest')
        grid_dem = np.where(np.isnan(grid_dem), grid_dem_nn.reshape(grid_xx.shape), grid_dem)
    return grid_dem


def loo_cross_validation(df_stations, params_dict=None, dnn_params=None):
    """留一法交叉验证（稳健版）"""
    n = len(df_stations)
    x, y, z = df_stations['x'].values, df_stations['y'].values, df_stations['temperature'].values
    elev = df_stations['elevation'].values

    ok_errors = []
    rf_errors = []

    print("\n执行留一法交叉验证...")

    # 全局量用于约束
    domain_diag = np.hypot(x.max() - x.min(), y.max() - y.min())
    unique_x = np.unique(np.sort(x))
    diffs = np.diff(unique_x)
    diffs = diffs[diffs > 0]
    min_dx = np.min(diffs) if len(diffs) > 0 else domain_diag * 0.01
    min_range_default = max(min_dx * 2.0, 1.0)
    max_range_default = max(domain_diag * 2.0, min_range_default)
    global_var = np.var(z)
    default_sill = max(global_var, 0.1)
    default_nugget = max(global_var * 0.1, 0.01)

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]

        # 传统OK
        try:
            ok = OrdinaryKriging(
                x[train_idx], y[train_idx], z[train_idx],
                variogram_model='spherical', verbose=False, enable_plotting=False
            )
            pred, _ = ok.execute('points', np.array([x[i]]), np.array([y[i]]))
            ok_errors.append(z[i] - pred[0])
        except Exception:
            ok_errors.append(np.nan)

        # RF-OK（若提供 params_dict）
        if params_dict is not None:
            try:
                scaler = params_dict['scaler']
                rf_range = params_dict['rf_range']
                rf_sill = params_dict['rf_sill']
                rf_nugget = params_dict['rf_nugget']

                X_te = np.column_stack([[x[i]], [y[i]], [elev[i]]])
                X_te_scaled = scaler.transform(X_te)

                rng = float(rf_range.predict(X_te_scaled)[0])
                sil = float(rf_sill.predict(X_te_scaled)[0])
                nug = float(rf_nugget.predict(X_te_scaled)[0])

                # 物理约束
                nug = max(nug, 0.0)
                sil = max(sil, nug + 0.01, default_sill * 0.1)
                rng = np.clip(rng, min_range_default, max_range_default)

                if not np.isfinite(rng) or not np.isfinite(sil) or not np.isfinite(nug):
                    rng = min_range_default * 3
                    sil = default_sill
                    nug = default_nugget

                try:
                    ok_loo = OrdinaryKriging(
                        x[train_idx], y[train_idx], z[train_idx],
                        variogram_model='spherical',
                        variogram_parameters={'sill': sil, 'range': rng, 'nugget': nug},
                        verbose=False,
                        enable_plotting=False
                    )
                    pred, _ = ok_loo.execute('points', np.array([x[i]]), np.array([y[i]]))
                    rf_errors.append(z[i] - pred[0])
                except Exception:
                    ok2 = OrdinaryKriging(
                        x[train_idx], y[train_idx], z[train_idx],
                        variogram_model='spherical', verbose=False, enable_plotting=False
                    )
                    pred2, _ = ok2.execute('points', np.array([x[i]]), np.array([y[i]]))
                    rf_errors.append(z[i] - pred2[0])
            except Exception:
                rf_errors.append(np.nan)

    ok_errors = np.array([e for e in ok_errors if not np.isnan(e)])
    rf_errors = np.array([e for e in rf_errors if not np.isnan(e)])

    return ok_errors, rf_errors, np.array([])


def evaluate_residual_model_and_apply(station_x, station_y, station_elev, station_z,
                                      grid_xx, grid_yy, grid_dem,
                                      rfok_grid, n_splits=5, radius_m=800):
    """
    使用 feature_engineering 构建特征，在站点层面做 KFold CV 比较残差模型（增加诊断打印），
    然后在全域网格上预测残差并返回修正后的网格与统计信息。
    返回 dict 包含: rmse_rfok_mean, rmse_corrected_mean, final_model, scaler, corrected_grid, pred_resid_grid
    """
    # 采样 rfok 在站点处（线性插值 + 最近邻补充）
    rfok_at_stations = griddata((grid_xx.ravel(), grid_yy.ravel()), rfok_grid.ravel(),
                                (station_x, station_y), method='linear')
    nans = np.isnan(rfok_at_stations)
    if np.any(nans):
        rfok_at_stations[nans] = griddata((grid_xx.ravel(), grid_yy.ravel()), rfok_grid.ravel(),
                                          (station_x[nans], station_y[nans]), method='nearest')

    # 构建站点特征矩阵
    X_stat, y_resid, feat_names, terrain = build_station_feature_matrix(
        station_x, station_y, station_elev, station_z,
        grid_xx, grid_yy, grid_dem, rfok_at_stations, radius_m=radius_m
    )

    # KFold CV 比较（加入诊断打印）
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_rfok = []
    rmse_corrected = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_stat), start=1):
        Xtr, Xte = X_stat[train_idx], X_stat[test_idx]
        ytr, yte = y_resid[train_idx], y_resid[test_idx]

        # 诊断打印：训练目标方差与前几列特征方差
        ytr_std = np.std(ytr)
        feat_var = np.var(Xtr, axis=0)
        print(f"\n[CV] Fold {fold}: ytr.std = {ytr_std:.6f}")
        print(f"[CV] Fold {fold}: first 6 feature variances = {feat_var[:6]}")

        # 若目标方差极小，提示可能没有可学信号
        if ytr_std < 1e-6:
            print(f"[CV][WARN] Fold {fold}: target residual std is very small ({ytr_std}), model may not learn useful splits.")

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        if LGB_AVAILABLE:
            model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)

        model.fit(Xtr_s, ytr)
        pred_resid = model.predict(Xte_s)

        # 原始 RFOK 误差
        rfok_pred_at_test = rfok_at_stations[test_idx]
        obs_at_test = station_z[test_idx]
        err_rfok = obs_at_test - rfok_pred_at_test
        rmse_rfok.append(np.sqrt(np.mean(err_rfok**2)))

        # 修正后误差
        corrected_pred = rfok_pred_at_test + pred_resid
        err_corr = obs_at_test - corrected_pred
        rmse_corrected.append(np.sqrt(np.mean(err_corr**2)))

    print("KFold CV RFOK RMSE (per-fold):", np.array(rmse_rfok))
    print("KFold CV Corrected RMSE (per-fold):", np.array(rmse_corrected))
    print("Mean RFOK RMSE:", np.mean(rmse_rfok), "Mean Corrected RMSE:", np.mean(rmse_corrected))

    # 在全数据上训练最终模型并在网格上预测残差
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X_stat)
    if LGB_AVAILABLE:
        final_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42)
    else:
        final_model = RandomForestRegressor(n_estimators=400, max_depth=14, random_state=42, n_jobs=-1)
    final_model.fit(X_full_s, y_resid)

    # 构建网格特征并预测（传入站点以确保列一致）
    X_grid, grid_feat_names = build_grid_feature_matrix(
        grid_xx, grid_yy, grid_dem, terrain, rfok_grid,
        station_x=station_x, station_y=station_y, station_z=station_z, radius_m=radius_m
    )
    X_grid_s = scaler_full.transform(X_grid)
    pred_resid_grid = final_model.predict(X_grid_s).reshape(grid_xx.shape)

    corrected_grid = rfok_grid + pred_resid_grid

    return {
        'rmse_rfok_mean': float(np.mean(rmse_rfok)),
        'rmse_corrected_mean': float(np.mean(rmse_corrected)),
        'final_model': final_model,
        'scaler': scaler_full,
        'corrected_grid': corrected_grid,
        'pred_resid_grid': pred_resid_grid,
        'feat_names': feat_names,
        'grid_feat_names': grid_feat_names
    }


def main():
    print("=" * 70)
    print("基于随机森林+深度学习的克里金插值优化方案（含残差特征工程与CV诊断）")
    print("=" * 70)

    # ========== 配置 ==========
    DATA_SOURCE = 'gsod'  # 'gsod' 或 'simulated'
    GSOD_FOLDER = r"D:\pycharm\Kriging and Machine Learning\2024_gsod_data"
    TARGET_LAT, TARGET_LON = 23.13, 113.26
    SEARCH_RADIUS = 2000
    MAX_STATIONS = 100
    YEAR = 2024
    # 网格分辨率
    grid_res = 50
    # ===========================

    # 1. 加载数据
    if DATA_SOURCE == 'gsod':
        df = load_gsod_from_local(GSOD_FOLDER, TARGET_LAT, TARGET_LON,
                                  SEARCH_RADIUS, MAX_STATIONS, YEAR)
        if df is None or len(df) < 10:
            print("GSOD 数据不足或未找到，回退到模拟数据")
            df = generate_simulated_data(n_stations=100)
    else:
        df = generate_simulated_data(n_stations=100)

    if df is None or len(df) < 10:
        print("数据不足，程序退出")
        return

    # 坐标转换
    df, lon_center, lat_center = stations_to_xy(df)

    # 提取站点数据
    station_x = df['x'].values
    station_y = df['y'].values
    station_elev = df['elevation'].values
    station_z = df['temperature'].values

    # 创建插值网格
    x_range = (station_x.min(), station_x.max())
    y_range = (station_y.min(), station_y.max())
    pad_x, pad_y = (x_range[1] - x_range[0]) * 0.1, (y_range[1] - y_range[0]) * 0.1
    grid_x = np.linspace(x_range[0] - pad_x, x_range[1] + pad_x, grid_res)
    grid_y = np.linspace(y_range[0] - pad_y, y_range[1] + pad_y, grid_res)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    # 网格DEM插值
    grid_dem = interpolate_dem_to_grid(station_x, station_y, station_elev, grid_xx, grid_yy)

    # 2. 传统OK（基准）
    ok_grid = perform_traditional_ok(station_x, station_y, station_z, grid_xx, grid_yy)

    # 3. 第一步：训练RF预测变程参数
    params_dict = train_rf_parameter_predictor(
        station_x, station_y, station_elev, station_z
    )

    # 4. 第二步：用RF预测的参数做克里金插值（RF-OK）
    range_grid, sill_grid, nugget_grid = predict_parameters(
        params_dict, grid_xx, grid_yy, grid_dem
    )

    rfok_grid = interpolate_with_dynamic_params(
        station_x, station_y, station_z,
        grid_xx, grid_yy,
        range_grid, sill_grid, nugget_grid
    )

    # 5. 残差建模：用特征工程训练残差模型并在网格上修正
    res_result = evaluate_residual_model_and_apply(
        station_x, station_y, station_elev, station_z,
        grid_xx, grid_yy, grid_dem,
        rfok_grid, n_splits=5, radius_m=800
    )

    corrected_grid = res_result['corrected_grid']
    pred_resid_grid = res_result['pred_resid_grid']

    # 6. 交叉验证（LOO）用于对比 OK 与 RF-OK（保留 DNN 逐点 LOO 可选）
    ok_errors, rf_errors, dnn_errors = loo_cross_validation(df, params_dict, None)

    # 打印精度汇总
    print("\n" + "=" * 70)
    print("精度评估结果（留一法交叉验证）")
    print("=" * 70)

    ok_rmse = np.sqrt(np.nanmean(ok_errors ** 2)) if len(ok_errors) > 0 else np.nan
    ok_mae = np.nanmean(np.abs(ok_errors)) if len(ok_errors) > 0 else np.nan
    print(f"传统OK     -> MAE: {ok_mae:.4f}°C, RMSE: {ok_rmse:.4f}°C")

    if len(rf_errors) > 0:
        rf_rmse = np.sqrt(np.nanmean(rf_errors ** 2))
        rf_mae = np.nanmean(np.abs(rf_errors))
        print(f"RF-OK      -> MAE: {rf_mae:.4f}°C, RMSE: {rf_rmse:.4f}°C")
        if np.isfinite(ok_rmse) and ok_rmse > 0:
            print(f"  RMSE 提升: {(1 - rf_rmse / ok_rmse) * 100:.2f}%")

    # 打印残差模型的 CV 指标
    print("\n残差模型（站点层 KFold）结果：")
    print(f"  RFOK 平均 RMSE: {res_result['rmse_rfok_mean']:.4f}")
    print(f"  RFOK+Residual 平均 RMSE: {res_result['rmse_corrected_mean']:.4f}")
    if res_result['rmse_rfok_mean'] > 0:
        print(f"  改进: {(1 - res_result['rmse_corrected_mean'] / res_result['rmse_rfok_mean']) * 100:.2f}%")

    # 7. 可视化
    plot_full_comparison(
        df, grid_xx, grid_yy, grid_dem,
        ok_grid, rfok_grid, corrected_grid,
        range_grid, sill_grid,
        ok_errors, rf_errors, dnn_errors,
        save_path='kriging_rf_dnn_result.png'
    )

    print("\n✅ 方案完整流程执行完毕！")


if __name__ == "__main__":
    main()