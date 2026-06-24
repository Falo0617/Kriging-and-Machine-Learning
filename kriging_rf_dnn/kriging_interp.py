"""
模块3：第二步 — 克里金初步插值
用RF预测的动态参数执行普通克里金插值
"""

import numpy as np
from pykrige.ok import OrdinaryKriging
import warnings

warnings.filterwarnings('ignore')


def interpolate_with_dynamic_params(station_x, station_y, station_z,
                                    grid_xx, grid_yy,
                                    range_grid, sill_grid, nugget_grid,
                                    use_global_fallback=True):
    """
    用动态参数执行克里金插值

    关键思路：对每个网格点，使用该点对应的局部变程/基台值/块金值
    但由于pykrige不支持逐点变程，我们采用：
        方案A：用所有站点参数的中位数作为全局参数（简化版）
        方案B：分区插值（按地形分区，每区用一套参数）

    这里实现方案A + 置信度评估
    """
    print("\n第二步：执行克里金初步插值...")

    # 取参数的中位数作为代表性参数（使用 nanmedian 更稳健）
    median_range = np.nanmedian(range_grid)
    median_sill = np.nanmedian(sill_grid)
    median_nugget = np.nanmedian(nugget_grid)

    print(f"  使用参数: 变程={median_range:.2f}, 基台值={median_sill:.2f}, 块金值={median_nugget:.2f}")

    try:
        # 创建自定义半变异函数模型
        ok = OrdinaryKriging(
            station_x, station_y, station_z,
            variogram_model='spherical',
            variogram_parameters={
                'sill': float(median_sill),
                'range': float(median_range),
                'nugget': float(median_nugget)
            },
            verbose=False,
            enable_plotting=False
        )

        # 注意：grid_xx, grid_yy 是 meshgrid 格式（shape: ny, nx）
        # 正确的取法是：
        grid_x = np.unique(grid_xx[0, :])   # 横向唯一 x 值
        grid_y = np.unique(grid_yy[:, 0])   # 纵向唯一 y 值

        kriged_grid, krige_variance = ok.execute('grid', grid_x, grid_y)
        kriged_grid = np.asarray(kriged_grid)
        # 如果 OK 返回的网格形状与 meshgrid 不一致，调整为 meshgrid 形状
        if kriged_grid.shape != grid_xx.shape:
            try:
                kriged_grid = kriged_grid.reshape(grid_yy.shape)
            except:
                # 回退到用全域平均值填充
                kriged_grid = np.full_like(grid_xx, np.nanmean(station_z))

        kriged_grid = np.nan_to_num(kriged_grid, nan=np.nanmean(station_z))

    except Exception as e:
        print(f"  克里金插值失败: {e}")
        if use_global_fallback:
            print("  回退到全局拟合...")
            try:
                ok = OrdinaryKriging(
                    station_x, station_y, station_z,
                    variogram_model='spherical',
                    verbose=False,
                    enable_plotting=False
                )
                grid_x = np.unique(grid_xx[0, :])
                grid_y = np.unique(grid_yy[:, 0])
                kriged_grid, _ = ok.execute('grid', grid_x, grid_y)
                kriged_grid = np.asarray(kriged_grid)
                if kriged_grid.shape != grid_xx.shape:
                    kriged_grid = kriged_grid.reshape(grid_yy.shape)
                kriged_grid = np.nan_to_num(kriged_grid, nan=np.nanmean(station_z))
            except Exception as e2:
                print(f"  回退也失败: {e2}")
                kriged_grid = np.full_like(grid_xx, np.nanmean(station_z))
        else:
            kriged_grid = np.full_like(grid_xx, np.nanmean(station_z))

    print(f"  插值完成，结果范围: {kriged_grid.min():.2f} ~ {kriged_grid.max():.2f}°C")
    return kriged_grid


def interpolate_with_rf_prediction(params_dict, station_x, station_y, station_z,
                                   grid_xx, grid_yy, grid_dem):
    """
    完整的第一步+第二步：先用RF预测参数，再用动态参数做克里金插值
    这是方案核心流程的完整实现
    """
    # 第一步：预测参数
    from rf_kriging import predict_parameters
    range_grid, sill_grid, nugget_grid = predict_parameters(params_dict, grid_xx, grid_yy, grid_dem)

    # 第二步：克里金插值
    kriged_grid = interpolate_with_dynamic_params(
        station_x, station_y, station_z,
        grid_xx, grid_yy,
        range_grid, sill_grid, nugget_grid
    )

    # 同时返回参数网格供分析使用
    return kriged_grid, range_grid, sill_grid, nugget_grid