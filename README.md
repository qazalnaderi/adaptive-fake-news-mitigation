# Adaptive Fake News Mitigation in Social Networks

This repository contains a course project for **Game Theory**, based on the implementation and extension of a published evolutionary game-theoretic model of fake news spread on social networks.

The project is inspired by:
> *Containing misinformation: Modeling spatial games of fake news*  
> Jones et al., PNAS Nexus (2024)

---

## Project Overview

The original paper models fake news propagation as a **spatial evolutionary coordination game** on social networks, where individuals may:
- Share true news (A)
- Share fake news (B)
- Act as sanctioners / fact-checkers (C)

In the original model, both the **number** and **placement** of sanctioners are fixed throughout the simulation.

This project implements the original model and extends it by introducing **adaptive and dynamic intervention mechanisms**.

---

## Key Extensions

### 1. Adaptive Sanctioner Density

Instead of keeping the fraction of sanctioners fixed, the model dynamically adjusts the number of sanctioners based on the current prevalence of fake news.

- Higher fake news levels lead to increased sanctioning effort.
- Lower fake news levels reduce the need for sanctioners.

This adaptive mechanism reflects realistic moderation strategies where resources respond to problem severity.

---

### 2. Boundary-Based Sanctioner Placement

Rather than placing sanctioners randomly or solely based on network centrality, this project identifies **boundary nodes** located between fake-news and non–fake-news regions.

Sanctioners are placed at these boundary locations to:
- Maximize exposure to fake-news spreaders
- Disrupt the formation and persistence of echo chambers

---

### 3. Dynamic Repositioning of Sanctioners

Sanctioners are periodically reselected and repositioned based on the evolving state of the network.

This models mobile or redeployable fact-checking agents, such as platform-level interventions or automated moderation systems.

---

## Model Details

- Network topology: Watts–Strogatz small-world network
- Update rule: Asynchronous imitation dynamics
- Game type: Spatial evolutionary coordination game
- Payoff matrix: Identical to Jones et al. (2024)
- Selection strength: β = 0.5
- Sanctioner behavior: Fixed strategy (C nodes do not imitate others)

---

## Results

Compared to a baseline model with fixed, targeted sanctioners, the adaptive approach:
- Suppresses fake news more rapidly
- Weakens and destabilizes echo chambers
- Achieves better outcomes with fewer sanctioners on average

---

## Motivation

This project demonstrates how **adaptive control** and **local, boundary-aware interventions** can significantly improve misinformation mitigation in networked systems.

It connects theoretical evolutionary game models with practical strategies relevant to:
- Social media moderation
- Fact-checking systems
- Automated policy enforcement

---

## Requirements

- Python 3.8+
- networkx
- numpy
- matplotlib

---

## Usage

Run the main script to compare:
- Fixed targeted sanctioning
- Adaptive boundary-based sanctioning

The output includes:
- Time-series plots of true vs. fake news spreaders
- Dynamic changes in the sanctioner fraction

---

## Course Context

This project was completed as part of a **Game Theory** course and focuses on evolutionary and networked games.

---

## Author

Independent course project implementing and extending a published evolutionary game-theoretic model for academic and portfolio purposes.
