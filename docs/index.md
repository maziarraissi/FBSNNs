---
layout: default
---
### Author
[Maziar Raissi](http://www.dam.brown.edu/people/mraissi/)

### Abstract

Classical numerical methods for solving [partial differential equations](https://en.wikipedia.org/wiki/Partial_differential_equation) suffer from the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) mainly due to their reliance on meticulously generated spatio-temporal grids. Inspired by [modern deep learning based techniques](https://maziarraissi.github.io/DeepHPMs/) for solving forward and inverse problems associated with partial differential equations, we circumvent the tyranny of numerical discretization by devising an algorithm that is scalable to high-dimensions. In particular, we approximate the unknown solution by a [deep neural network](https://en.wikipedia.org/wiki/Deep_learning) which essentially enables us to benefit from the merits of [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation). To train the aforementioned neural network we leverage the [well-known connection](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.20168) between high-dimensional partial differential equations and [forward-backward stochastic differential equations](https://www.mathematik.hu-berlin.de/~perkowsk/files/bsde.pdf). In fact, independent realizations of a standard [Brownian motion](https://en.wikipedia.org/wiki/Brownian_motion) will act as training data. We test the effectiveness of our approach for a couple of benchmark problems spanning a number of scientific domains including [Black-Scholes-Barenblatt](https://en.wikipedia.org/wiki/Black–Scholes_model) and [Hamilton-Jacobi-Bellman](https://en.wikipedia.org/wiki/Hamilton–Jacobi–Bellman_equation) equations, both in 100-dimensions.

* * * * * *
#### Problem setup and solution methodology

In [this work](https://arxiv.org/abs/1804.07010), we consider coupled forward-backward stochastic differential equations of the general form

$$
\begin{array}{l}
dX_t = \mu(t, X_t, Y_t, Z_t)dt + \sigma(t, X_t, Y_t)dW_t, ~~~ t \in [0,T],\\
X_0 = \xi,\\
dY_t = \varphi(t, X_t, Y_t, Z_t)dt + Z_t'\sigma(t,X_t,Y_t)dW_t, ~~~ t \in [0,T),\\
Y_T = g(X_T),
\end{array}
$$

where $$W_t$$ is a vector-valued Brownian motion. A solution to these equations consists of the stochastic processes $$X_t$$, $$Y_t$$, and $$Z_t$$. It is [well-known](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.20168) that coupled forward-backward stochastic differential equations are related to quasi-linear partial differential equations of the form

$$
u_t = f(t,x,u,Du,D^2u),
$$

with terminal condition $$u(T,x) = g(x)$$, where $$u(t,x)$$ is the unknown solution and

$$
f(t,x,y,z,\gamma) = \varphi(t,x,y,z) - \mu(t,x,y,z)'z - \frac{1}{2}\text{Tr}[\sigma(t,x,y)\sigma(t,x,y)'\gamma].
$$

Here, $$Du$$ and $$D^2u$$ denote the gradient vector and the Hessian matrix of $$u$$, respectively. In particular, it follows directly from [Ito's formula](https://en.wikipedia.org/wiki/Itô%27s_lemma) that solutions of forward-backward stochastic differential equations and quasi-linear partial differential equations are related according to

$$
Y_t = u(t, X_t), ~ \text{and} ~ Z_t = D u(t, X_t).
$$

Inspired by recent developments in [physics-informed deep learning](https://maziarraissi.github.io/PINNs/) and [deep hidden physics models](https://maziarraissi.github.io/DeepHPMs/), we proceed by approximating the unknown solution $$u(t,x)$$ by a deep neural network. We obtain the required gradient vector $$Du(t,x)$$ by applying the chain rule for differentiating compositions of functions using automatic differentiation. In particular, to compute the derivatives we rely on [Tensorflow](https://www.tensorflow.org) which is a popular and relatively well documented open source software library for automatic differentiation and deep learning computations.

Parameters of the neural network representing $$u(t,x)$$ can be learned by minimizing the following loss function obtained from discretizing the forward-backward stochastic differential equation using the standard [Euler-Maruyama scheme](https://en.wikipedia.org/wiki/Euler–Maruyama_method). To be more specific, let us employ the Euler-Maruyama scheme and obtain

$$
\begin{array}{l}
X^{n+1} \approx X^n + \mu(t^n,X^n,Y^n,Z^n)\Delta t^n + \sigma(t^n,X^n,Y^n)\Delta W^n,\\
Y^{n+1} \approx Y^n + \varphi(t^n,X^n,Y^n,Z^n)\Delta t^n + (Z^n)'\sigma(t^n,X^n,Y^n)\Delta W^n,
\end{array}
$$

for $$n = 0, 1, \ldots, N-1$$, where $$\Delta t^n := t^{n+1} - t^n = T/N$$ and $$\Delta W^n \sim \mathcal{N}(0, \Delta t^n)$$ is a random variable with mean $$0$$ and standard deviation $$\sqrt{\Delta t^n}$$. The loss function is then given by

$$
\sum_{m=1}^M \sum_{n=0}^{N-1} |Y^{n+1}_m - Y^n_m - \Phi^n_m \Delta t^n - (Z^n_m)' \Sigma^n_m \Delta W_m^n|^2 + \sum_{m=1}^M |Y^N_m - g(X^N_m)|^2,
$$

which corresponds to $$M$$ different realizations of the underlying Brownian motion. Here, $$\Phi^n_m := \varphi(t^n,X^n_m,Y^n_m,Z^n_m)$$ and $$\Sigma^n_m := \sigma(t^n,X^n_m,Y^n_m)$$. The subscript $$m$$ corresponds to the $$m$$-th realization of the underlying Brownian motion while the superscript $$n$$ corresponds to time $$t^n$$. It is worth recalling that $$Y_m^n = u(t^n, X_m^n)$$ and $$Z_m^n = D u(t^n, X_m^n)$$, and consequently the loss is a function of the parameters of the neural network $$u(t,x)$$. Furthermore, we have

$$
X^{n+1}_m = X^n_m + \mu(t^n,X^n_m,Y^n_m,Z^n_m)\Delta t^n + \sigma(t^n_m,X^n_m,Y^n_m)\Delta W^n_m,
$$

and $$X^0_m = \xi$$ for every $$m$$.

* * * * * *
#### Results
**Two-dimensional damped oscillator**

As a first illustrative example, let us consider the two-dimensional damped harmonic oscillator with cubic dynamics; i.e.,

$$
\begin{array}{l}
\dot{x} = -0.1\ x^3 + 2.0\ y^3,\\
\dot{y} = -2.0\ x^3 - 0.1\ y^3.
\end{array}
$$

We use $$[x_0\ y_0]^T = [2\ 0]^T$$ as initial condition and collect data from $$t = 0$$ to $$t = 25$$ with a time-step size of $$\Delta t = 0.01$$. The data are plotted in the following figure. We employ a neural network with one hidden layer and 256 neurons to represent the nonlinear dynamics. As for the multistep scheme, we use Adams-Moulton with $$M=1$$ steps (i.e., the trapezoidal rule). Upon training the neural network, we solve the identified system using the same initial condition as the one above. The following figure provides a qualitative assessment of the accuracy in identifying the correct nonlinear dynamics. Specifically, by comparing the exact and predicted trajectories of the system, as well as the resulting phase portraits, we observe that the algorithm can correctly capture the dynamic evolution of the system.

![](http://www.dam.brown.edu/people/mraissi/assets/img/Cubic2D.png)
> _Harmonic Oscillator:_ Trajectories of the two-dimensional damped harmonic oscillator with cubic dynamics are depicted in the left panel while the corresponding phase portrait is plotted in the right panel. Solid colored lines represent the exact dynamics while the dashed black lines demonstrate the learned dynamics. The identified system correctly captures the form of the dynamics and accurately reproduces the phase portrait.

**Lorenz system**

To explore the identification of chaotic dynamics evolving on a finite dimensional attractor, we consider the nonlinear [Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system)

$$
\begin{array}{l}
\dot{x} = 10 (y - x),\\
\dot{y} = x (28 - z) - y,\\
\dot{z} = x y - (8/3) z.
\end{array}
$$

We use $$[x_0\ y_0\ z_0]^T = [-8\ 7\ 27]^T$$ as initial condition and collect data from $$t = 0$$ to $$t = 25$$ with a time-step size of $$\Delta t = 0.01$$. The data are plotted in the following figure. We employ a neural network with one hidden layer and 256 neurons to represent the nonlinear dynamics. As for the multistep scheme, we use Adams-Moulton with $$M=1$$ steps (i.e., the trapezoidal rule). Upon training the neural network, we solve the identified system using the same initial condition as the one above. As depicted in the following figure, the learned system correctly captures the form of the attractor. The Lorenz system has a positive Lyapunov exponent, and small differences between the exact and learned models grow exponentially, even though the attractor remains intact. 

![](http://www.dam.brown.edu/people/mraissi/assets/img/Lorenz.png)
> _Lorenz System:_ The exact phase portrait of the Lorenz system (left panel) is compared to the corresponding phase portrait of the learned dynamics (right panel).

**Fluid flow behind a cylinder**

In this example we collect data for the fluid flow past a cylinder at Reynolds number 100 using direct numerical simulations of the two dimensional Navier-Stokes equations (see the following figure).

![](http://www.dam.brown.edu/people/mraissi/assets/img/Cylinder_vorticity.png)
> _Flow past a cylinder:_ A snapshot of the vorticity field of a solution to the Navier-Stokes equations for the fluid flow past a cylinder.

In particular, we simulate the Navier-Stokes equations describing the two-dimensional fluid flow past a circular cylinder at Reynolds number 100 using the [Immersed Boundary Projection Method](https://github.com/cwrowley/ibpm). This approach utilizes a multi-domain scheme with four nested domains, each successive grid being twice as large as the previous one. Length and time are non-dimensionalized so that the cylinder has unit diameter and the flow has unit velocity. Data is collected on the finest domain with dimensions $$9 \times 4$$ at a grid resolution of $$449 \times 199$$. The flow solver uses a 3rd-order [Runge-Kutta](https://en.wikipedia.org/wiki/Runge–Kutta_methods) integration scheme with a time step of $$t = 0.02$$, which has been verified to yield well-resolved and converged flow fields. After simulations converge to steady periodic [vortex shedding](https://en.wikipedia.org/wiki/Vortex_shedding), flow snapshots are saved every $$\Delta t = 0.02$$. We then reduce the dimension of the system by [proper orthogonal decomposition](https://en.wikipedia.org/wiki/Principal_component_analysis) (POD). The POD results in a hierarchy of orthonormal modes that, when truncated, capture most of the energy of the original system for the given rank truncation. The first two most energetic POD modes capture a significant portion of the energy; the steady-state vortex shedding is a [limit cycle](https://en.wikipedia.org/wiki/Limit_cycle) in these coordinates. An additional mode, called the shift mode, is included to capture the transient dynamics connecting the unstable steady state with the mean of the limit cycle. The resulting POD coefficients are depicted in the following figure.

![](http://www.dam.brown.edu/people/mraissi/assets/img/Cylinder.png)
> _Flow past a cylinder:_ The exact phase portrait of the cylinder wake trajectory in reduced coordinates (left panel) is compared to the corresponding phase portrait of the learned dynamics (right panel).

We employ a neural network with one hidden layer and $$256$$ neurons to represent the nonlinear dynamics shown in the above figure. As for the linear multistep scheme, we use Adams-Moulton with $$M=1$$ steps (i.e., the trapezoidal rule). Upon training the neural network, we solve the identified system. As depicted in the above figure, the learned system correctly captures the form of the dynamics and accurately reproduces the phase portrait, including both the transient regime as well as the limit cycle attained once the flow dynamics converge to the well known Karman vortex street.

**Hopf bifurcation**

Many real-world systems depend on parameters and, when the parameters are varied, they may go through [bifurcations](https://en.wikipedia.org/wiki/Bifurcation_theory). To illustrate the ability of our method to identify parameterized dynamics, let us consider the Hopf normal form

$$
\begin{array}{l}
\dot{x} = \mu x + y - x(x^2 + y^2),\\
\dot{y} = -x + \mu y - y(x^2 + y^2).
\end{array}
$$

Our algorithm can be readily extended to encompass parameterized systems. In particular, the above system can be equivalently written as

$$
\begin{array}{l}
\dot{\mu} = 0,\\
\dot{x} = \mu x + y - x(x^2 + y^2),\\
\dot{y} = -x + \mu y - y(x^2 + y^2).
\end{array}
$$

We collect data from the Hopf system for various initial conditions corresponding to different parameter values for $$\mu$$. The data is depicted in the following figure. The identified parameterized dynamics is shown in the following figure for a set of parameter values different from the ones used during model training. The learned system correctly captures the transition from the fixed point for $$\mu < 0$$ to the limit cycle for $$\mu>0$$.

![](http://www.dam.brown.edu/people/mraissi/assets/img/Hopf.png)
> _Hopf bifurcation:_ Training data from the Hopf system for various initial conditions corresponding to different parameter values (left panel) is compared to the corresponding phase portrait of the learned dynamics (right panel). It is worth highlighting that the algorithm is tested on initial conditions different from the ones used during training.

**Glycolytic oscillator**

As an example of complicated nonlinear dynamics typical of biological systems, we simulate the [glycolytic oscillator model](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0119821). The model consists of ordinary differential equations for the concentrations of 7 biochemical species; i.e.,

$$
\begin{array}{l}
\frac{dS_1}{dt} = J_0 - \frac{k_1 S_1 S_6}{1 + (S_6/K_1)^q},\\
\frac{dS_2}{dt} = 2\frac{k_1 S_1 S_6}{1 + (S_6/K_1)^q} - k_2 S_2 (N - S_5) - k_6 S_2 S_5,\\
\frac{dS_3}{dt} = k_2 S_2 (N - S_5) - k_3 S_3 (N - S_6),\\
\frac{dS_4}{dt} = k_3 S_3 (A - S_6) - k_4 S_4 S_5 - \kappa (S_4 - S_7),\\
\frac{dS_5}{dt} = k_2 S_2 (N - S_5) - k_4 S_4 S_5 - k_6 S_2 S_5,\\
\frac{dS_6}{dt} = -2\frac{k_1 S_1 S_6}{1 + (S_6/K_1)^q} + 2 k_3 S_3 (A - S_6) - k_5 S_6,\\
\frac{dS_7}{dt} = \psi \kappa (S_4 - S_7) - k S_7.
\end{array}
$$

As shown in the following figure, data from a simulation of this equation are collected from $$t = 0$$ to $$t = 10$$ with a time-step size of $$\Delta t = 0.01$$. We employ a neural network with one hidden layer and $$256$$ neurons to represent the nonlinear dynamics. As for the multi-step scheme, we use Adams-Moulton with $$M=1$$ steps (i.e., the trapezoidal rule). Upon training the neural network, we solve the identified system using the same initial condition as the ones used for the exact system. As depicted in the following figure, the learned system correctly captures the form of the dynamics.

![](http://www.dam.brown.edu/people/mraissi/assets/img/Glycolytic.png)
> _Glycolytic oscillator:_ Exact versus learned dynamics for random initial conditions.

* * * * *

**Conclusion**

We have presented a [machine learning approach](https://arxiv.org/abs/1801.01236) for extracting nonlinear dynamical systems from time-series data. The proposed algorithm leverages the structure of well studied multi-step time-stepping schemes such as Adams-Bashforth, Adams Moulton, and BDF families, to construct efficient algorithms for learning dynamical systems using deep neural networks. Although state-of-the-art results are presented for a diverse collection of benchmark problems, there exist a series of open questions mandating further investigation. How could one handle a variable temporal gap $$\Delta{t}$$, i.e., irregularly sampled data in time? How would common techniques such as batch normalization, drop out, and $$\mathcal{L}_1$$/$$\mathcal{L}_2$$ regularization enhance the robustness of the proposed algorithm and mitigate the effects of over-fitting? How could one incorporate partial knowledge of the dynamical system in cases where certain interaction terms are already known? In terms of future work, interesting directions include the application of [convolutional architectures](https://en.wikipedia.org/wiki/Convolutional_neural_network) for mitigating the complexity associated with very high-dimensional inputs.

* * * * *

**Acknowledgements**

This work received support by the DARPA EQUiPS grant N66001-15-2-4055 and the AFOSR grant FA9550-17-1-0013. All data and codes are publicly available on [GitHub](https://github.com/maziarraissi/MultistepNNs).

* * * * *
## Citation

	@article{raissi2018multistep,
	  title={Multistep Neural Networks for Data-driven Discovery of Nonlinear Dynamical Systems},
	  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
	  journal={arXiv preprint arXiv:1801.01236},
	  year={2018}
	}

