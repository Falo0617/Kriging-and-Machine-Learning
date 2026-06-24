import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

def dem_terrain_features(grid_xx, grid_yy, grid_dem):
    """
    从网格 DEM 计算坡度、坡向、曲率（简单差分近似）。
    返回字典：{'slope':, 'aspect':, 'curvature':}
    grid_xx, grid_yy: meshgrid arrays
    grid_dem: ny x nx
    """
    # 计算坐标轴间隔
    x_unique = np.unique(grid_xx[0, :])
    y_unique = np.unique(grid_yy[:, 0])
    dx = x_unique[1] - x_unique[0] if len(x_unique) > 1 else 1.0
    dy = y_unique[1] - y_unique[0] if len(y_unique) > 1 else 1.0

    # 梯度（注意 numpy.gradient 参数顺序）
    dgy, dgx = np.gradient(grid_dem, dy, dx, edge_order=2)
    slope = np.hypot(dgx, dgy)            # magnitude of gradient
    aspect = np.arctan2(dgy, dgx)         # angle in radians
    # 曲率近似（Laplacian）
    dxx = np.gradient(np.gradient(grid_dem, axis=1), axis=1) / (dx**2)
    dyy = np.gradient(np.gradient(grid_dem, axis=0), axis=0) / (dy**2)
    curvature = dxx + dyy
    return {'slope': slope, 'aspect': aspect, 'curvature': curvature}

def sample_grid_features_at_stations(grid_xx, grid_yy, feat_grid, station_x, station_y):
    """
    将网格特征插值到站点位置（linear，若 NaN 用 nearest 补全）
    feat_grid: ny x nx array
    station_x, station_y: arrays (n,)
    """
    pts = (grid_xx.ravel(), grid_yy.ravel())
    vals = feat_grid.ravel()
    sampled = griddata(pts, vals, (station_x, station_y), method='linear')
    if np.any(np.isnan(sampled)):
        nn = griddata(pts, vals, (station_x, station_y), method='nearest')
        sampled = np.where(np.isnan(sampled), nn, sampled)
    return sampled

def local_spatial_stats(station_x, station_y, station_z, radius_m=500):
    """
    计算每个站点邻域统计：局部均值、局部标准差、邻居数量、最近邻距离
    返回字典
    """
    pts = np.column_stack([station_x, station_y])
    tree = cKDTree(pts)
    n = len(pts)
    local_mean = np.zeros(n)
    local_std = np.zeros(n)
    n_neighbors = np.zeros(n, dtype=int)
    dist_to_nearest = np.zeros(n)
    # 使用 query_ball_point 获取邻域
    for i in range(n):
        idxs = tree.query_ball_point(pts[i], r=radius_m)
        idxs = [j for j in idxs if j != i]
        if len(idxs) == 0:
            local_mean[i] = station_z[i]
            local_std[i] = 0.0
            n_neighbors[i] = 0
        else:
            vals = station_z[idxs]
            local_mean[i] = np.mean(vals)
            local_std[i] = np.std(vals)
            n_neighbors[i] = len(idxs)
        # 最近邻距离
        dists, ids = tree.query(pts[i], k=2)  # 包含自身
        dist_to_nearest[i] = dists[1] if len(dists) > 1 else 0.0
    return {'local_mean': local_mean, 'local_std': local_std,
            'n_neighbors': n_neighbors, 'dist_to_nearest': dist_to_nearest}

def build_station_feature_matrix(station_x, station_y, station_elev, station_z,
                                 grid_xx, grid_yy, grid_dem, kriged_at_stations,
                                 radius_m=500, include_interactions=True):
    """
    构建站点级特征矩阵 X 和目标 y（残差）
    返回 X (n, p), y (n,), feature_names list
    """
    terrain = dem_terrain_features(grid_xx, grid_yy, grid_dem)
    slope_s = sample_grid_features_at_stations(grid_xx, grid_yy, terrain['slope'], station_x, station_y)
    aspect_s = sample_grid_features_at_stations(grid_xx, grid_yy, terrain['aspect'], station_x, station_y)
    curv_s = sample_grid_features_at_stations(grid_xx, grid_yy, terrain['curvature'], station_x, station_y)

    sp_stats = local_spatial_stats(station_x, station_y, station_z, radius_m=radius_m)

    # 构造特征
    feats = [
        station_x, station_y, station_elev,
        slope_s, np.sin(aspect_s), np.cos(aspect_s),
        curv_s,
        sp_stats['local_mean'], sp_stats['local_std'],
        sp_stats['n_neighbors'], sp_stats['dist_to_nearest'],
        kriged_at_stations
    ]
    names = ['x','y','elev','slope','sin_aspect','cos_aspect','curvature',
             'local_mean_temp','local_std_temp','n_neighbors','dist_to_nearest','kriged_at_station']

    X = np.column_stack(feats)
    if include_interactions:
        X = np.hstack([X, (station_x*station_y)[:,None], (station_x**2)[:,None], (station_y**2)[:,None]])
        names += ['x*y','x2','y2']
    y = station_z - kriged_at_stations
    return X, y, names, terrain

def build_grid_feature_matrix(grid_xx, grid_yy, grid_dem, terrain, kriged_grid,
                             station_x=None, station_y=None, station_z=None,
                             radius_m=800, include_interactions=True):
    """
    构建网格级特征矩阵（用于在网格上预测残差）
    使得网格特征与站点特征的列名和顺序一致，返回 X_grid (ny*nx, p) 以及 feature names.

    新增参数:
      station_x, station_y, station_z: optional, 用于计算每个网格点到最近站点的距离以及基于站点的邻域计数（若传入）
      radius_m: 用于站点邻域计数（若 station_x/station_y 提供）
    """
    import numpy as np
    from scipy.ndimage import uniform_filter
    from scipy.spatial import cKDTree

    gx = grid_xx.ravel()
    gy = grid_yy.ravel()
    gelev = grid_dem.ravel()
    gslope = terrain['slope'].ravel()
    gaspect = terrain['aspect'].ravel()
    gcurv = terrain['curvature'].ravel()
    gkrig = kriged_grid.ravel()

    # 局部平均/标准差：在网格上用滑动窗口（uniform_filter）近似
    # uniform_filter 要求输入为 2D，所以直接对 grid 计算再展平
    # 选择窗口大小：以 radius_m 对应的像素数估算（粗略）
    ny, nx = grid_xx.shape
    # 估计像素分辨率（假设等间距）
    x_unique = np.unique(grid_xx[0, :])
    y_unique = np.unique(grid_yy[:, 0])
    dx = x_unique[1] - x_unique[0] if len(x_unique) > 1 else 1.0
    dy = y_unique[1] - y_unique[0] if len(y_unique) > 1 else 1.0
    # 半径对应的像素数（向上取整）
    rx = max(1, int(np.ceil(radius_m / max(dx, 1e-6))))
    ry = max(1, int(np.ceil(radius_m / max(dy, 1e-6))))
    # 使用 uniform_filter 计算滑动窗口均值与均方，然后得 std
    try:
        from scipy.ndimage import uniform_filter
        gkrig_2d = kriged_grid.reshape((ny, nx))
        mean_local = uniform_filter(gkrig_2d, size=(ry, rx), mode='nearest')
        mean_sq = uniform_filter(gkrig_2d**2, size=(ry, rx), mode='nearest')
        var_local = np.maximum(0.0, mean_sq - mean_local**2)
        std_local = np.sqrt(var_local)
        g_local_mean = mean_local.ravel()
        g_local_std = std_local.ravel()
    except Exception:
        # 若 uniform_filter 不可用或失败，降级为用整个网格的全局值
        g_local_mean = np.full_like(gkrig, np.nanmean(gkrig))
        g_local_std = np.full_like(gkrig, np.nanstd(gkrig))

    # n_neighbors 与 dist_to_nearest：若提供站点坐标则基于 KDTree 计算，否则用网格点间近似或常数
    if (station_x is not None) and (station_y is not None):
        pts_grid = np.column_stack([gx, gy])
        tree = cKDTree(np.column_stack([station_x, station_y]))
        # 对每个网格点统计邻居数与最近邻距离
        # query_ball_point 会为每个点返回邻居索引列表
        # 为节省时间，先使用 query to nearest k=1 获取距离（最近邻）
        dists, ids = tree.query(pts_grid, k=1)
        g_dist_to_nearest = dists
        # 计算邻居数量（在 radius_m 内）
        # 使用 query_ball_point for all grid points (could be heavy but acceptable for moderate grids)
        idxs_list = tree.query_ball_point(pts_grid, r=radius_m)
        g_n_neighbors = np.array([len([j for j in l]) for l in idxs_list], dtype=float)
    else:
        # 未提供站点，则以 0 填充邻居数量，dist 设置为一个较大值
        g_n_neighbors = np.zeros_like(gkrig, dtype=float)
        g_dist_to_nearest = np.full_like(gkrig, np.max([np.ptp(gx), np.ptp(gy)]))

    # 现在按和站点特征相同的顺序构造特征
    feats = [
        gx, gy, gelev,
        gslope, np.sin(gaspect), np.cos(gaspect),
        gcurv,
        g_local_mean, g_local_std,
        g_n_neighbors, g_dist_to_nearest,
        gkrig
    ]
    names = ['x','y','elev','slope','sin_aspect','cos_aspect','curvature',
             'local_mean_temp','local_std_temp','n_neighbors','dist_to_nearest','kriged_grid']

    Xg = np.column_stack(feats)
    if include_interactions:
        Xg = np.hstack([Xg, (gx*gy)[:,None], (gx**2)[:,None], (gy**2)[:,None]])
        names += ['x*y','x2','y2']

    return Xg, names