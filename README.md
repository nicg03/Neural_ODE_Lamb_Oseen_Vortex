# Neural ODEs for Multi-Vortex Lamb–Oseen Dynamics

This repository investigates how **Neural ODEs (NODE)** and **Augmented Neural ODEs (ANODE)** learn and reproduce the dynamics of particles advected by a **multi-vortex Lamb–Oseen flow**.  
The goal is to compare classic feedforward models with continuous-time neural formulations and evaluate them using physically meaningful metrics.

---

## Overview

We aim to learn the continuous law

$\dot{\mathbf{r}}(t) = \mathbf{v}(\mathbf{r}(t), t)$

where $\(\mathbf{v}\)$ is the multi-vortex Lamb–Oseen velocity field.  
Three models are implemented:

- **Feedforward Network (FF)** — discrete mapping $(x,y,t) \to (u_x, u_y)$
- **Neural ODE (NODE)** — learns the differential law directly via `torchdiffeq`  
- **Augmented Neural ODE (ANODE)** — extends the ODE state with latent variables for greater expressiveness  

The evaluation focuses on **physical coherence**, including vorticity, divergence, circulation, step-invariance, and trajectory correctness.


