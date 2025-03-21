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
```

## Utilisation

Lancer l'application Streamlit avec la commande suivante :

```bash
streamlit run main.py
```
Ensuite, configurez les paramètres du modèle Heston et de l'option via l'interface graphique, puis cliquez sur "Calculer prix & grecs".

## Structure du Projet

```plaintext
Heston-Model/
│-- pricing_utils/              # Contient les fonctions de pricing et de simulation
│-- main.py                     # Fichier principal Streamlit
│-- requirements.txt             # Liste des dépendances
│-- README.md                   # Documentation du projet
```

## Auteur

**Gabriel PEZENNEC**
