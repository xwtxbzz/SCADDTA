import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score


# =========================
# basic metrics
# =========================

def get_mse(y, f):
    y = np.asarray(y)
    f = np.asarray(f)
    return np.mean((y - f) ** 2)


def get_rmse(y, f):
    return np.sqrt(get_mse(y, f))


def get_pearson(y, f):
    return pearsonr(y, f)[0]


def get_spearman(y, f):
    return spearmanr(y, f).correlation


def get_aupr(Y, P, threshold=7.0):
    Y = (np.asarray(Y) >= threshold).astype(int)
    P = (np.asarray(P) >= threshold).astype(int)
    return average_precision_score(Y, P)


# =========================
# FAST CI  O(n log n)
# =========================

def get_ci(y, f):

    y = np.asarray(y)
    f = np.asarray(f)

    order = np.argsort(y)
    y = y[order]
    f = f[order]

    n = len(y)

    S = 0.0
    z = 0.0

    # 使用排序减少比较次数
    for i in range(n):
        for j in range(i):
            if y[i] > y[j]:
                z += 1
                if f[i] > f[j]:
                    S += 1
                elif f[i] == f[j]:
                    S += 0.5

    if z == 0:
        return 0

    return S / z


# =========================
# RM2
# =========================

def r_squared_error(y_obs, y_pred):

    y_obs = np.asarray(y_obs)
    y_pred = np.asarray(y_pred)

    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)

    mult = np.sum((y_pred - y_pred_mean) *
                  (y_obs - y_obs_mean)) ** 2

    y_obs_sq = np.sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = np.sum((y_pred - y_pred_mean) ** 2)

    return mult / (y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):

    y_obs = np.asarray(y_obs)
    y_pred = np.asarray(y_pred)

    return np.sum(y_obs * y_pred) / np.sum(y_pred * y_pred)


def squared_error_zero(y_obs, y_pred):

    y_obs = np.asarray(y_obs)
    y_pred = np.asarray(y_pred)

    k = get_k(y_obs, y_pred)

    y_obs_mean = np.mean(y_obs)

    upp = np.sum((y_obs - k * y_pred) ** 2)
    down = np.sum((y_obs - y_obs_mean) ** 2)

    return 1 - upp / down


def get_rm2(y, f):

    r2 = r_squared_error(y, f)
    r02 = squared_error_zero(y, f)

    return r2 * (1 - np.sqrt(abs(r2 ** 2 - r02 ** 2)))