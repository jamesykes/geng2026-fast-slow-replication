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
| `run_high_alpha.py` | Runs a higher-learning-rate `alpha` sweep based on the Figure 3 setup. |
| `SI.pdf` | Supplementary Material PDF. |

## High learning-rate experiments

The authors' Figure 3 alpha sweep uses `alpha = 0.0002, 0.0005, 0.001`.
To probe larger learning rates without overwriting the reproduction data, run:

```bash
cd authors_original
python run_high_alpha.py
```

By default this writes:

- `data/high_alpha/high_alpha_sweep.npz`
- `data/high_alpha/high_alpha_summary.csv`
- `figures/high_alpha/high_alpha_sweep.png`

For a quick smoke run:

```bash
python run_high_alpha.py --alphas 0.001,0.01,0.1 --time-steps 200 --num-reps 2
```

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
