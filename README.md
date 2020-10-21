# Contour Stochastic Gradient Langevin Dynamics

Simulations of multi-modal distributions can be very **costly** and often lead to **unreliable** predictions. To accelerate the computations, we propose to sample from a **flattened** distribution to **accelerate the computations** and estimate the **importance weights** between the original distribution and the flattened distribution to **ensure the correctness** of the distribution. 



<p float="left">
  <img src="figures/SGLD.gif" width="200" title="SGLD"/>
  <img src="figures/cycSGLD.gif" width="200" alt="Made with Angular" title="Angular" /> 
  <img src="figures/reSGLD.gif" width="200" alt="hello!" title="adam solomon's hello"/>
  <img src="figures/CSGLD.gif" width="200" />
</p>



| Methods   |      Speed      | Special features  | Cost |
|----------|:-------------:|:-------------:|:-------------:|
| SGLD (ICML'11) |  Extremely slow | None | None |
| Cycic SGLD (ICLR'20) |    Medium   | Cyclic learning rates  | More cycles |
| Replica exchange SGLD (ICML'20) | Fast | Swaps/Jumps | Parallel chains |
| Contour SGLD (NeurIPS'20) | Fast | Bouncy moves | Estimate latent vector |






## References:

1. Max Welling, Yee Whye Teh. [Bayesian Learning via Stochastic Gradient Langevin Dynamics](https://pdfs.semanticscholar.org/aeed/631d6a84100b5e9a021ec1914095c66de415.pdf). ICML'11

2. R. Zhang, C. Li, J. Zhang, C. Chen, A. Wilson. [Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning](https://arxiv.org/pdf/1902.03932.pdf). ICLR'20

3. W. Deng, Q. Feng, L. Gao, F. Liang, G. Lin. [Non-convex Learning via Replica Exchange Stochastic Gradient MCMC](https://arxiv.org/pdf/2008.05367.pdf). ICML'20.

4. W. Deng, G. Lin, F. Liang. [A Contour Stochastic Gradient Langevin Dynamics Algorithm for Simulations of Multi-modal Distributions](https://arxiv.org/pdf/2010.09800.pdf). NeurIPS'20.
