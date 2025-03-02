import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pricing_utils.pricing_functions import generate_heston_paths, mc_price_call_put_heston, compute_all_greeks # type: ignore


def main():

    st.markdown(
        """
        <style>
        div.block-container {
            max-width: 90%;
            margin: auto;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("Pricer d’options Heston + Calcul des Grecs")
    st.markdown("""
    Cette application simule le **modèle de Heston** en Monte Carlo et calcule le prix d'un **Call & Put** ainsi que leurs principaux **grecs** (Delta, Gamma, Vega, Theta, Rho) par différences finies.
    """)
    
    # Paramètres Heston dans la sidebar
    st.sidebar.header("Paramètres Heston")
    S_0   = st.sidebar.number_input("Prix initial S₀", value=100.0, step=1.0)
    r     = st.sidebar.number_input("Taux sans risque r", value=0.05, step=0.01)
    T     = st.sidebar.number_input("Maturité (années) T", value=1.0, step=0.5)
    kappa = st.sidebar.number_input("kappa", value=3.0, step=0.1)
    theta = st.sidebar.number_input("theta", value=0.04, step=0.01)
    v0    = st.sidebar.number_input("v₀ (variance initiale)", value=0.04, step=0.01)
    rho   = st.sidebar.slider("Corrélation rho", min_value=-0.99, max_value=0.99, value=-0.7, step=0.01)
    xi    = st.sidebar.number_input("Vol of vol xi", value=0.3, step=0.05)
    
    steps  = st.sidebar.number_input("Nombre de pas (time steps)", value=200, step=50)
    Npaths = st.sidebar.number_input("Nombre de trajectoires", value=500, step=50)
    
    st.sidebar.header("Option & Simulation")
    K = st.sidebar.number_input("Strike K", value=100.0, step=1.0)
    
    st.sidebar.header("Seed (optionnel)")
    seed_input = st.sidebar.text_input("Saisir une seed (laisser vide pour non fixe)", value="")
    seed = int(seed_input) if seed_input != "" else None
    
    st.sidebar.header("Paramètres de différences finies")
    eps_ratio_S   = st.sidebar.number_input("Epsilon ratio pour S (Delta, Gamma)", value=0.01, step=0.001, format="%.4f")
    eps_ratio_r   = st.sidebar.number_input("Epsilon ratio pour r (Rho)", value=0.01, step=0.001, format="%.4f")
    eps_ratio_T   = st.sidebar.number_input("Epsilon ratio pour T (Theta)", value=0.01, step=0.001, format="%.4f")
    eps_ratio_v0  = st.sidebar.number_input("Epsilon ratio pour v₀ (Vega)", value=0.01, step=0.001, format="%.4f")
    
    if st.sidebar.button("Calculer prix & grecs"):
        with st.spinner("Simulation en cours..."):
            # Calcul du prix
            call_price, put_price = mc_price_call_put_heston(
                S_0, r, T, kappa, theta, v0, rho, xi,
                steps, Npaths, K, seed=seed
            )
            # Calcul des grecs
            greeks = compute_all_greeks(
                S_0, r, T, kappa, theta, v0, rho, xi,
                steps, Npaths, K,
                seed=seed,
                eps_ratio_S=eps_ratio_S,
                eps_ratio_r=eps_ratio_r,
                eps_ratio_T=eps_ratio_T,
                eps_ratio_v0=eps_ratio_v0
            )
        st.success("Simulation terminée !")
        
        # Affichage des résultats de prix
        st.subheader("Prix estimés")
        col1, col2 = st.columns(2)
        col1.metric("Prix du Call", f"{call_price:.4f}")
        col2.metric("Prix du Put", f"{put_price:.4f}")
        
        st.subheader("Grecs estimés (Call)")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Delta", f"{greeks['delta_call']:.4f}")
        col2.metric("Gamma", f"{greeks['gamma_call']:.4f}")
        col3.metric("Vega",  f"{greeks['vega_call']:.4f}")
        col4.metric("Theta", f"{greeks['theta_call']:.4f}")
        col5.metric("Rho",   f"{greeks['rho_call']:.4f}")

        st.subheader("Grecs estimés (Put)")
        col6, col7, col8, col9, col10 = st.columns(5)
        col6.metric("Delta", f"{greeks['delta_put']:.4f}")
        col7.metric("Gamma", f"{greeks['gamma_put']:.4f}")
        col8.metric("Vega",  f"{greeks['vega_put']:.4f}")
        col9.metric("Theta", f"{greeks['theta_put']:.4f}")
        col10.metric("Rho",  f"{greeks['rho_put']:.4f}")
        
        # Optionnel : Affichage de quelques trajectoires simulées
        prices, sigs = generate_heston_paths(S_0, T, r, kappa, theta, v0, rho, xi,
                                             steps, Npaths, seed=seed)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for i in range(min(100, Npaths)):
            ax[0].plot(prices[i], alpha=0.7)
            ax[1].plot(np.sqrt(sigs[i]), alpha=0.7)
        ax[0].set_title(f"Simulation des {min(Npaths,100)} premières trajectoires de prix S(t)")
        ax[0].set_xlabel("Temps (pas)")
        ax[0].set_ylabel("Prix")
        ax[1].set_title(f"Simulation des {min(Npaths,100)} premières trajectoires volatilité (√v(t))")
        ax[1].set_xlabel("Temps (pas)")
        ax[1].set_ylabel("Volatilité")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Sélectionnez vos paramètres puis cliquez sur 'Calculer prix & grecs'.")

if __name__ == "__main__":
    main()

