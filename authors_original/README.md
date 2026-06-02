# Emergent Fast-Slow Dynamics in Multi-Agent Q-Learning for Networked Stochastic Games

Code and supplementary material for the AAAI 2026 paper
**"Emergent Fast-Slow Dynamics in Multi-Agent Q-Learning for Networked
Stochastic Games"**.

This repository contains:

- The Supplementary Material.
- The code to reproduce all figures, in which we validate the theoretical model against agent-based simulations.

## Repository structure

| File | Description |
|------|-------------|
| `sim.py` | Agent-based simulation. |
| `theory.py` | The theoretical model. |
| `utils.py` | Graph construction helpers. |
| `experiments_sim.py` | Runs the simulation experiments and saves results to `data/`. |
| `experiments_theory.py` | Runs the theory experiments and saves results to `data/`. |
| `plot_main.py` | Generates the main-text figures into `figures/`. |
| `plot_SI.py` | Generates the supplementary figures into `figures/`. |
| `SI.pdf` | Supplementary Material PDF. |

## Requirements

The code was developed and tested with **Python 3.12.8** and the following package

| Package | Version |
|---------|---------|
| jax | 0.7.2 |
| jaxlib | 0.7.2 |
| numpy | 2.2.4 |
| networkx | 3.4.2 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |
| tqdm | 4.67.1 |

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{geng2026emergent,
  title={Emergent Fast-Slow Dynamics in Multi-Agent Q-Learning for Networked Stochastic Games},
  author={Geng, Yuxin and Barfuss, Wolfram and Chen, Xingru},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={35},
  pages={29450--29458},
  year={2026}
}
```
