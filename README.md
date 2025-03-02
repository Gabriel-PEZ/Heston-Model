# Heston Model Pricer

## Description

Ce projet implémente un pricer d'options basé sur le **modèle de Heston**, utilisant une simulation de Monte Carlo. Il permet de :

- Simuler des trajectoires du modèle de Heston
- Calculer les prix des options **Call** et **Put**
- Évaluer les principaux **Grecs** (Delta, Gamma, Vega, Theta, Rho) par différences finies
- Visualiser les trajectoires des prix et de la volatilité

L'application est développée avec **Streamlit** pour offrir une interface interactive.

## Installation

### Prérequis

Assurez-vous d'avoir **Python 3** installé sur votre machine.

### Installation des dépendances

Cloner le dépôt et installez les dépendances avec :

```bash
pip install -r requirements.txt
