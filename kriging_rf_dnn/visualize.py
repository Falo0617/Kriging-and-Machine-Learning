"""
模块5：可视化对比
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_full_comparison(df_stations, grid_xx, grid_yy, grid_dem,
                         ok_grid, rfok_grid, corrected_grid,
                         range_grid, sill_grid,
                         ok_errors=None, rf_errors=None, dnn_errors=None,
                         save_path='kriging_rf_dnn_comparison.png'):
    """
    绘制完整的三步对比图
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('技术方案三步流程插值效果对比', fontsize=16, fontweight='bold')

    x_km = grid_xx / 1000
    y_km = grid_yy / 1000
    cmap_temp = plt.cm.RdYlBu_r

    # (1) 站点分布
    ax = axes[0, 0]
    sc = ax.scatter(df_stations['x'] / 1000, df_stations['y'] / 1000,
                    c=df_stations['temperature'], s=50, cmap=cmap_temp,
                    edgecolors='black', linewidth=0.5)
    ax.set_title('(a) 站点分布与温度')
    ax.set_xlabel('km');
    ax.set_ylabel('km')
    plt.colorbar(sc, ax=ax, label='°C')

    # (2) 传统OK
    ax = axes[0, 1]
    im = ax.contourf(x_km, y_km, ok_grid, 30, cmap=cmap_temp)
    ax.scatter(df_stations['x'] / 1000, df_stations['y'] / 1000,
               c='black', s=10, alpha=0.3)
    ax.set_title('(b) 第二步：克里金插值 (OK)')
    ax.set_xlabel('km')
    plt.colorbar(im, ax=ax, label='°C')

    # (3) RF-OK（第二步）
    ax = axes[0, 2]
    im2 = ax.contourf(x_km, y_km, rfok_grid, 30, cmap=cmap_temp)
    ax.scatter(df_stations['x'] / 1000, df_stations['y'] / 1000,
               c='black', s=10, alpha=0.3)
    ax.set_title('(c) 第一步+第二步：RF定参+克里金')
    ax.set_xlabel('km')
    plt.colorbar(im2, ax=ax, label='°C')

    # (4) 最终结果（第三步）
    ax = axes[0, 3]
    im3 = ax.contourf(x_km, y_km, corrected_grid, 30, cmap=cmap_temp)
    ax.scatter(df_stations['x'] / 1000, df_stations['y'] / 1000,
               c='black', s=10, alpha=0.3)
    ax.set_title('(d) 第三步：DNN残差修正')
    ax.set_xlabel('km')
    plt.colorbar(im3, ax=ax, label='°C')

    # (5) 变程分布
    ax = axes[1, 0]
    im4 = ax.contourf(x_km, y_km, range_grid, 30, cmap='viridis')
    ax.set_title('(e) RF预测的变程分布')
    ax.set_xlabel('km');
    ax.set_ylabel('km')
    plt.colorbar(im4, ax=ax, label='变程 (m)')

    # (6) 基台值分布
    ax = axes[1, 1]
    im5 = ax.contourf(x_km, y_km, sill_grid, 30, cmap='plasma')
    ax.set_title('(f) RF预测的基台值分布')
    ax.set_xlabel('km')
    plt.colorbar(im5, ax=ax, label='基台值')

    # (7) 残差修正量
    ax = axes[1, 2]
    residual_grid = corrected_grid - rfok_grid
    im6 = ax.contourf(x_km, y_km, residual_grid, 30, cmap='coolwarm')
    ax.set_title('(g) DNN残差修正量')
    ax.set_xlabel('km')
    plt.colorbar(im6, ax=ax, label='°C')

    # (8) 精度汇总
    ax = axes[1, 3]
    ax.axis('off')

    # 计算误差（如果有交叉验证结果）
    info = f"站点数: {len(df_stations)}\n\n"
    if ok_errors is not None:
        ok_rmse = np.sqrt(np.nanmean(np.array(ok_errors) ** 2))
        info += f"OK RMSE: {ok_rmse:.4f}°C\n"
    if rf_errors is not None:
        rf_rmse = np.sqrt(np.nanmean(np.array(rf_errors) ** 2))
        info += f"RF-OK RMSE: {rf_rmse:.4f}°C\n"
    if dnn_errors is not None:
        dnn_rmse = np.sqrt(np.nanmean(np.array(dnn_errors) ** 2))
        info += f"DNN修正 RMSE: {dnn_rmse:.4f}°C\n"

    ax.text(0.5, 0.5, info, transform=ax.transAxes, ha='center', va='center',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('(h) 精度汇总')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✅ 对比图已保存为: {save_path}")