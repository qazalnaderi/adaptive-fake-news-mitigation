# Adaptive Boundary-Aware Fact-Checker Placement for Misinformation Suppression in Social Networks

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

A paper-aligned Python implementation of an adaptive, boundary-aware
fact-checker allocation strategy for misinformation suppression in
complex social networks.

## Associated Publication

This repository is associated with the following peer-reviewed article:

> M. T. Firouzjaee, **G. Naderi**, R. Gore, and N. Moghim,  
> “Adaptive Boundary-Aware Fact-Checker Placement for Misinformation
> Suppression in Social Networks,”  
> *Applied Sciences*, vol. 16, no. 10, Article 4740, 2026.

- **Published article:**  
  https://www.mdpi.com/2076-3417/16/10/4740

- **DOI:**  
  https://doi.org/10.3390/app16104740

- **BibTeX citation:** See the [Citation](#citation) section.

## About This Repository

This repository originally began as a Game Theory course project based
on a spatial evolutionary game model of misinformation diffusion.

Following the development and publication of the associated article,
the original implementation was refactored and aligned with the core
methodology described in the paper.

The current implementation includes:

- Static random fact-checker placement as the baseline
- Adaptive regulation of fact-checker density
- Normalized boundary-aware placement
- Periodic fact-checker reallocation
- Probabilistic lasting correction
- Payoff-biased asynchronous imitation dynamics
- Small-world, scale-free, and random network topologies
- Single-seed visualization
- Multi-seed experiment execution and CSV export

## Repository Status

The repository provides a **paper-aligned implementation of the core
simulation framework**.

It does not yet claim to be a complete reproduction package for every
figure, table, ablation experiment, sensitivity analysis, and
statistical test reported in the published article.