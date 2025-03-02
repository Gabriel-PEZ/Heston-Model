import numpy as np

def generate_heston_paths(S_0, T, r, kappa, theta, v0, rho, xi,
                          steps, Npaths, seed=None):
    """
    Génère des trajectoires de prix (S) et de variance (v) selon le modèle de Heston.
    Si seed est renseignée, on fixe le générateur aléatoire pour une reproductibilité.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    prices = np.zeros((Npaths, steps))
    sigs   = np.zeros((Npaths, steps))
    
    # Initialisation des trajectoires
    S_t = np.full(Npaths, S_0)
    v_t = np.full(Npaths, v0)
    
    for t in range(steps):
        WT = np.random.multivariate_normal(mean=[0, 0],cov=[[1, rho], 
                                                            [rho, 1]],
                                                            size=Npaths) * np.sqrt(dt)
        
        S_t = S_t * np.exp((r - 0.5 * np.maximum(v_t, 0)) * dt + np.sqrt(np.maximum(v_t, 0)) * WT[:, 0])
        v_t = v_t + kappa * (theta - np.maximum(v_t, 0)) * dt + xi * np.sqrt(np.maximum(v_t, 0)) * WT[:, 1]
        v_t = np.maximum(v_t, 0)  # Forcer la positivité
        
        prices[:, t] = S_t
        sigs[:, t] = v_t
    
    return prices, sigs

def mc_price_call_put_heston(S_0, r, T, kappa, theta, v0, rho, xi,
                             steps, Npaths, K, seed=None):
    """
    Calcule le prix d'un Call et d'un Put européen via simulation Monte Carlo sous le modèle de Heston.
    """
    prices, _ = generate_heston_paths(S_0, T, r, kappa, theta, v0, rho, xi,
                                      steps, Npaths, seed=seed)
    S_T = prices[:, -1]  # Valeur du sous-jacent à maturité
    
    payoff_call = np.maximum(S_T - K, 0)
    payoff_put  = np.maximum(K - S_T, 0)
    
    discount_factor = np.exp(-r * T)
    call_price = discount_factor * np.mean(payoff_call)
    put_price  = discount_factor * np.mean(payoff_put)
    
    return call_price, put_price


# Calcul des Grecs par Différences Finies
def compute_all_greeks(S_0, r, T, kappa, theta, v0, rho, xi,
                       steps, Npaths, K,
                       seed=None,
                       eps_ratio_S=0.01,   # Pour Delta et Gamma
                       eps_ratio_r=0.01,   # Pour Rho
                       eps_ratio_T=0.01,   # Pour Theta
                       eps_ratio_v0=0.01   # Pour Vega
                       ):
    """
    Calcule les principaux grecs (Delta, Gamma, Vega, Theta, Rho) pour le Call et le Put
    en utilisant des différences finies centrées.
    """
    # Prix de base pour les calculs du gamma avec les différences finies de second ordre
    call_0, put_0 = mc_price_call_put_heston(S_0, r, T, kappa, theta, v0, rho, xi, steps, Npaths, K, seed=seed)
    

    # Delta & Gamma 
    eps_S = eps_ratio_S * S_0
    call_pS, put_pS = mc_price_call_put_heston(S_0 + eps_S, r, T, kappa, theta, v0, rho, xi, steps, Npaths, K, seed=seed) # Partie epsilon positive
    call_mS, put_mS = mc_price_call_put_heston(S_0 - eps_S, r, T, kappa, theta, v0, rho, xi, steps, Npaths, K, seed=seed) # Partie epsilon négative
    
    delta_call = (call_pS - call_mS) / (2 * eps_S)
    delta_put  = (put_pS - put_mS) / (2 * eps_S)
    
    gamma_call = (call_pS - 2 * call_0 + call_mS) / (eps_S ** 2) # On utilise la formule d'ordre 2 pour les gamma des put et call
    gamma_put  = (put_pS - 2 * put_0 + put_mS) / (eps_S ** 2)
    
    # Rho 
    eps_r = eps_ratio_r * r if r != 0 else 0.0001
    call_pr, put_pr = mc_price_call_put_heston(S_0, r + eps_r, T, kappa, theta, v0, rho, xi, steps, Npaths, K, seed=seed) # Partie epsilon positive
    call_mr, put_mr = mc_price_call_put_heston(S_0, r - eps_r, T, kappa, theta, v0, rho, xi, steps, Npaths, K, seed=seed) #  Partie epsilon négative
    
    rho_call = (call_pr - call_mr) / (2 * eps_r)
    rho_put  = (put_pr - put_mr) / (2 * eps_r)
   
    # Theta 
    eps_T = eps_ratio_T * T if T != 0 else 0.0001
    call_pT, put_pT = mc_price_call_put_heston(S_0, r, T + eps_T, kappa, theta, v0, rho, xi,steps, Npaths, K, seed=seed)
    T_minus = max(T - eps_T, 0.00001) # On s'assure que T reste positif sinon incohérent on peut pas avoir de maturité négative

    call_mT, put_mT = mc_price_call_put_heston(S_0, r, T_minus, kappa, theta, v0, rho, xi, steps, Npaths, K, seed=seed)
    theta_call = (call_pT - call_mT) / (2 * eps_T)
    theta_put  = (put_pT - put_mT) / (2 * eps_T)
    
    # Vega
    eps_v0 = eps_ratio_v0 * v0 if v0 != 0 else 0.0001
    call_pv, put_pv = mc_price_call_put_heston(S_0, r, T, kappa, theta, v0 + eps_v0, rho, xi, steps, Npaths, K, seed=seed)
    v0_minus = max(v0 - eps_v0, 0.0000001) # Idem que pour T, on peut avoir une volatilité négative
    call_mv, put_mv = mc_price_call_put_heston(S_0, r, T, kappa, theta, v0_minus, rho, xi, steps, Npaths, K, seed=seed)
    vega_call = (call_pv - call_mv) / (2 * eps_v0)
    vega_put  = (put_pv - put_mv) / (2 * eps_v0)
    
    greeks = {
        "delta_call": delta_call,
        "delta_put":  delta_put,
        "gamma_call": gamma_call,
        "gamma_put":  gamma_put,
        "vega_call":  vega_call,
        "vega_put":   vega_put,
        "theta_call": theta_call,
        "theta_put":  theta_put,
        "rho_call":   rho_call,
        "rho_put":    rho_put
    }
    return greeks