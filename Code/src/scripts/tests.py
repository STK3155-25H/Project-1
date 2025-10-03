# tests/test_vs_sklearn.py
import os
import importlib
import numpy as np
import pytest

# sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="module")
def seeded_src():
    """
    Imposta un seed fisso PRIMA dell'import di src (il tuo modulo lo legge a import-time).
    Restituisce direttamente i simboli che ci servono dal pacchetto src.
    """
    os.environ["SEED"] = "314"
    # Import lazy per essere sicuri che prenda il seed dall'ambiente
    import src  # noqa: F401
    importlib.reload(src)
    from src import (
        runge_function, R2_score, MSE,
        polynomial_features, polynomial_features_scaled, split_scale,
        OLS_parameters, Ridge_parameters, Gradient_descent_advanced,
        MSE_Bias_Variance
    )
    return {
        "runge_function": runge_function,
        "R2_score": R2_score,
        "MSE": MSE,
        "polynomial_features": polynomial_features,
        "polynomial_features_scaled": polynomial_features_scaled,
        "split_scale": split_scale,
        "OLS_parameters": OLS_parameters,
        "Ridge_parameters": Ridge_parameters,
        "Gradient_descent_advanced": Gradient_descent_advanced,
        "MSE_Bias_Variance": MSE_Bias_Variance,
    }


# ----------------------------
# Polynomial features
# ----------------------------
@pytest.mark.parametrize("degree,intercept", [(1, True), (3, True), (3, False), (7, True)])
def test_polynomial_features_matches_sklearn(seeded_src, degree, intercept):
    pf = seeded_src["polynomial_features"]
    rng = np.linspace(-1, 1, 31)
    X_my = pf(rng, degree, intercept=intercept)

    pf_skl = PolynomialFeatures(degree=degree, include_bias=intercept)
    X_skl = pf_skl.fit_transform(rng.reshape(-1, 1))
    np.testing.assert_allclose(X_my, X_skl, rtol=1e-12, atol=1e-12)


def test_polynomial_features_scaled_matches_standard_scaler(seeded_src):
    pfs = seeded_src["polynomial_features_scaled"]
    # train/test split deterministico per confrontare scaling colonne (tranne l'intercetta)
    x = np.linspace(-2, 2, 50)
    X_train, X_test = x[:35], x[35:]
    deg = 4

    Xtr_my, means, stds = pfs(X_train, deg, intercept=True, return_stats=True)
    Xte_my = pfs(X_test, deg, intercept=True, col_means=means, col_stds=stds)

    # Costruisco le stesse feature grezze e applico StandardScaler SOLO alle colonne non-intercetta
    pf_skl = PolynomialFeatures(degree=deg, include_bias=True)
    Xtr_raw = pf_skl.fit_transform(X_train.reshape(-1, 1))
    Xte_raw = pf_skl.transform(X_test.reshape(-1, 1))

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_scaled = Xtr_raw.copy()
    Xte_scaled = Xte_raw.copy()
    Xtr_scaled[:, 1:] = scaler.fit_transform(Xtr_raw[:, 1:])
    Xte_scaled[:, 1:] = scaler.transform(Xte_raw[:, 1:])

    np.testing.assert_allclose(Xtr_my, Xtr_scaled, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Xte_my, Xte_scaled, rtol=1e-12, atol=1e-12)


# ----------------------------
# Metriche
# ----------------------------
def test_mse_and_r2_match_sklearn(seeded_src):
    MSE = seeded_src["MSE"]
    R2 = seeded_src["R2_score"]

    rng = np.linspace(-1, 1, 50)
    y_true = np.sin(3 * rng)
    y_pred = y_true + 0.1 * np.cos(5 * rng)

    assert np.isclose(MSE(y_true, y_pred), mean_squared_error(y_true, y_pred))
    assert np.isclose(R2(y_true, y_pred), r2_score(y_true, y_pred))


# ----------------------------
# OLS
# ----------------------------
@pytest.mark.parametrize("degree", [1, 3, 7])
def test_ols_parameters_match_sklearn_linear_regression(seeded_src, degree):
    pf_scaled = seeded_src["polynomial_features_scaled"]
    OLS = seeded_src["OLS_parameters"]

    # dati "Runge" senza rumore per confronto pulito
    x = np.linspace(-1, 1, 80)
    y = seeded_src["runge_function"](x, noise=False)

    # split e scaling come nel tuo progetto
    X_train, X_test, y_train, y_test = seeded_src["split_scale"](x, y)

    # feature polinomiali + scaling colonne (intercetta inclusa)
    Xtr, mu, sig = pf_scaled(X_train.ravel(), degree, return_stats=True)
    Xte = pf_scaled(X_test.ravel(), degree, col_means=mu, col_stds=sig)

    theta = OLS(Xtr, y_train)
    y_hat = Xte @ theta

    # sklearn: stesso design matrix con bias già dentro -> fit_intercept=False
    lr = LinearRegression(fit_intercept=False)
    lr.fit(Xtr, y_train)
    y_hat_skl = lr.predict(Xte)

    # Confronto coefficienti e predizioni
    np.testing.assert_allclose(theta, lr.coef_, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(y_hat, y_hat_skl, rtol=1e-10, atol=1e-10)


# ----------------------------
# Ridge
# ----------------------------
@pytest.mark.parametrize("degree,alpha", [(3, 1e-3), (5, 1e-2), (7, 0.1)])
def test_ridge_parameters_match_sklearn_no_intercept_penalty(seeded_src, degree, alpha):
    pf_scaled = seeded_src["polynomial_features_scaled"]
    Ridge_parameters = seeded_src["Ridge_parameters"]

    x = np.linspace(-1, 1, 100)
    y = seeded_src["runge_function"](x, noise=False)

    X_train, X_test, y_train, y_test = seeded_src["split_scale"](x, y)

    # Per confronto diretto col Ridge sklearn: NON includo la colonna d'intercetta e penalizzo tutte le colonne.
    Xtr_no1, mu, sig = pf_scaled(X_train.ravel(), degree, intercept=False, return_stats=True)
    Xte_no1 = pf_scaled(X_test.ravel(), degree, intercept=False, col_means=mu, col_stds=sig)

    theta = Ridge_parameters(Xtr_no1, y_train, lam=alpha, intercept=False)
    y_hat = Xte_no1 @ theta

    rr = Ridge(alpha=alpha, fit_intercept=False, solver="auto", random_state=None)
    rr.fit(Xtr_no1, y_train)
    y_hat_skl = rr.predict(Xte_no1)

    np.testing.assert_allclose(theta, rr.coef_, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(y_hat, y_hat_skl, rtol=1e-9, atol=1e-9)


# ----------------------------
# GD vs soluzione analitica (OLS)
# ----------------------------
@pytest.mark.parametrize("degree,lr", [(3, 0.05), (5, 0.02)])
def test_gradient_descent_converges_to_ols_ols_case(seeded_src, degree, lr):
    pf_scaled = seeded_src["polynomial_features_scaled"]
    OLS = seeded_src["OLS_parameters"]
    GD = seeded_src["Gradient_descent_advanced"]

    x = np.linspace(-1, 1, 120)
    y = seeded_src["runge_function"](x, noise=False)

    X_train, X_test, y_train, y_test = seeded_src["split_scale"](x, y)
    Xtr, mu, sig = pf_scaled(X_train.ravel(), degree, return_stats=True)

    theta_star = OLS(Xtr, y_train)

    theta_gd = GD(
        Xtr, y_train, Type=0, lam=0.0, lr=lr,
        n_iter=50000, tol=1e-10, method="vanilla", theta_history=False
    )
    # Devono coincidere (entro tolleranza numerica)
    np.testing.assert_allclose(theta_gd, theta_star, rtol=1e-5, atol=1e-6)


# ----------------------------
# Bias-Variance decomposition coerenza interna
# ----------------------------
def test_mse_bias_variance_identity(seeded_src):
    # Creiamo più predizioni bootstrappate attorno a un target fisso
    rng = np.random.default_rng(42)
    y_true = np.linspace(-1, 1, 40)
    preds = []
    for _ in range(100):
        preds.append(y_true + rng.normal(0, 0.2, size=y_true.size))
    preds = np.vstack(preds)

    mse, bias2, var = seeded_src["MSE_Bias_Variance"](y_true, preds)
    # Identità: mse ≈ bias^2 + var (qui y_true è deterministico)
    assert np.isclose(mse, bias2 + var, rtol=1e-6, atol=1e-8)
