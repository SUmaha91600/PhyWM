# The Noether-Discovery Paradox Connection: A Mathematical Framework

**Author:** Mahadevan Sutharsan
**Date:** February 3, 2026
**Status:** Theoretical Foundation for Publication

---

## Abstract

We establish a deep mathematical connection between Noether's theorem from classical mechanics and the Discovery Paradox in neural network training. We show that the same mathematical structure that *creates* conservation laws (symmetry invariance) also *hides* them from gradient-based learning. This connection is formalized using differential geometry, symplectic topology, and information theory. We prove that conservation laws lie in the null space of the prediction loss gradient at optimum, and propose Physics-First Pretraining as a principled solution grounded in Noether's insight.

---

## Table of Contents

1. [Noether's Theorem: Complete Derivation](#1-noethers-theorem-complete-derivation)
2. [The Discovery Paradox: Formal Proof](#2-the-discovery-paradox-formal-proof)
3. [The Mathematical Connection](#3-the-mathematical-connection)
4. [Topological Perspective](#4-topological-perspective)
5. [Information-Theoretic Analysis](#5-information-theoretic-analysis)
6. [Resolution: Physics-First Pretraining](#6-resolution-physics-first-pretraining)
7. [Emergent Mathematical Structures](#7-emergent-mathematical-structures)

---

## 1. Noether's Theorem: Complete Derivation

### 1.1 The Lagrangian Framework

**Definition 1.1 (Configuration Space):** Let Q be an n-dimensional smooth manifold representing all possible configurations of a physical system. A point q ∈ Q specifies the system's configuration.

**Definition 1.2 (Tangent Bundle):** The tangent bundle TQ consists of pairs (q, q̇) where q ∈ Q and q̇ ∈ T_qQ is a tangent vector (velocity) at q.

**Definition 1.3 (Lagrangian):** A Lagrangian is a smooth function:
$$L: TQ \times \mathbb{R} \to \mathbb{R}$$
$$L = L(q, \dot{q}, t)$$

For mechanical systems:
$$L = T - V = \frac{1}{2}\sum_{i,j} g_{ij}(q)\dot{q}^i\dot{q}^j - V(q)$$

where $g_{ij}$ is the metric tensor (kinetic energy metric) and V is the potential energy.

### 1.2 The Action Functional

**Definition 1.4 (Action):** The action is a functional on the space of paths:
$$S[\gamma] = \int_{t_1}^{t_2} L(q(t), \dot{q}(t), t) \, dt$$

where γ: [t₁, t₂] → Q is a path in configuration space.

**Theorem 1.1 (Hamilton's Principle):** Physical trajectories are stationary points of the action functional:
$$\delta S = 0$$

### 1.3 Derivation of Euler-Lagrange Equations

Consider a variation of the path: q(t) → q(t) + εη(t), where η(t₁) = η(t₂) = 0.

$$\delta S = \frac{d}{d\varepsilon}\bigg|_{\varepsilon=0} S[q + \varepsilon\eta]$$

$$= \int_{t_1}^{t_2} \left( \frac{\partial L}{\partial q^i}\eta^i + \frac{\partial L}{\partial \dot{q}^i}\dot{\eta}^i \right) dt$$

Integration by parts on the second term:
$$\int_{t_1}^{t_2} \frac{\partial L}{\partial \dot{q}^i}\dot{\eta}^i \, dt = \left[\frac{\partial L}{\partial \dot{q}^i}\eta^i\right]_{t_1}^{t_2} - \int_{t_1}^{t_2} \frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}^i}\right)\eta^i \, dt$$

Since η vanishes at endpoints:
$$\delta S = \int_{t_1}^{t_2} \left( \frac{\partial L}{\partial q^i} - \frac{d}{dt}\frac{\partial L}{\partial \dot{q}^i} \right)\eta^i \, dt = 0$$

For arbitrary η, we obtain:

**Theorem 1.2 (Euler-Lagrange Equations):**
$$\boxed{\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}^i}\right) - \frac{\partial L}{\partial q^i} = 0}$$

### 1.4 Noether's Theorem: Precise Statement and Proof

**Definition 1.5 (One-Parameter Group of Transformations):** A one-parameter group of transformations is a smooth map:
$$\Phi: \mathbb{R} \times Q \to Q$$
$$\Phi_\varepsilon(q) = \Phi(\varepsilon, q)$$

satisfying:
- Φ₀ = id (identity)
- Φ_ε ∘ Φ_δ = Φ_{ε+δ} (group property)

The infinitesimal generator is:
$$X^i(q) = \frac{\partial \Phi^i_\varepsilon(q)}{\partial \varepsilon}\bigg|_{\varepsilon=0}$$

**Definition 1.6 (Symmetry):** A transformation Φ_ε is a symmetry of the Lagrangian if:
$$L(\Phi_\varepsilon(q), D\Phi_\varepsilon \cdot \dot{q}, t) = L(q, \dot{q}, t) + \frac{d}{dt}F_\varepsilon(q, t)$$

for some function F (allowing for gauge transformations).

For strict invariance: L(Φ_ε(q), DΦ_ε · q̇, t) = L(q, q̇, t)

**Theorem 1.3 (Noether's Theorem):** If Φ_ε is a symmetry of the Lagrangian with infinitesimal generator X, then the Noether charge:
$$\boxed{Q = \sum_i \frac{\partial L}{\partial \dot{q}^i} X^i = p_i X^i}$$

is conserved along solutions of the Euler-Lagrange equations: dQ/dt = 0.

**Proof:**

Define the canonical momentum:
$$p_i = \frac{\partial L}{\partial \dot{q}^i}$$

The Noether charge is:
$$Q = p_i X^i(q)$$

Compute the time derivative:
$$\frac{dQ}{dt} = \frac{dp_i}{dt}X^i + p_i\frac{dX^i}{dt}$$

From Euler-Lagrange: $\frac{dp_i}{dt} = \frac{\partial L}{\partial q^i}$

And: $\frac{dX^i}{dt} = \frac{\partial X^i}{\partial q^j}\dot{q}^j$

So:
$$\frac{dQ}{dt} = \frac{\partial L}{\partial q^i}X^i + p_i\frac{\partial X^i}{\partial q^j}\dot{q}^j$$

Under the symmetry transformation, to first order in ε:
$$\delta L = \frac{\partial L}{\partial q^i}\varepsilon X^i + \frac{\partial L}{\partial \dot{q}^i}\varepsilon \frac{dX^i}{dt} = 0$$

(for strict invariance)

This gives:
$$\frac{\partial L}{\partial q^i}X^i + p_i\frac{\partial X^i}{\partial q^j}\dot{q}^j = 0$$

Therefore:
$$\boxed{\frac{dQ}{dt} = 0}$$

∎

### 1.5 Fundamental Examples

**Example 1.5.1 (Time Translation → Energy):**

Symmetry: t → t + ε (time translation)
Generator: ∂/∂t acting on trajectories

If ∂L/∂t = 0, define the Hamiltonian:
$$H = p_i\dot{q}^i - L$$

Compute:
$$\frac{dH}{dt} = \frac{dp_i}{dt}\dot{q}^i + p_i\ddot{q}^i - \frac{dL}{dt}$$

$$= \frac{\partial L}{\partial q^i}\dot{q}^i + p_i\ddot{q}^i - \frac{\partial L}{\partial q^i}\dot{q}^i - \frac{\partial L}{\partial \dot{q}^i}\ddot{q}^i - \frac{\partial L}{\partial t}$$

$$= -\frac{\partial L}{\partial t}$$

If ∂L/∂t = 0: **dH/dt = 0 (Energy Conservation)**

**Example 1.5.2 (Space Translation → Momentum):**

Symmetry: q^i → q^i + εn^i (translation in direction n)
Generator: X^i = n^i

If ∂L/∂q^i = 0 for some i:
$$Q = p_i n^i$$

is conserved. For full translation invariance: **p = (p₁, p₂, p₃) is conserved (Momentum Conservation)**

**Example 1.5.3 (Rotation → Angular Momentum):**

Symmetry: SO(3) rotations
Generator: X = ω × r for rotation with angular velocity ω

The Noether charge:
$$L = r \times p$$

is the **Angular Momentum**, conserved when V = V(|r|).

---

## 2. The Discovery Paradox: Formal Proof

### 2.1 Setup: World Models

**Definition 2.1 (World Model):** A world model is a parameterized function:
$$f_\theta: \mathcal{X} \to \mathcal{X}$$

that predicts the next state x_{t+1} from current state x_t.

**Definition 2.2 (Prediction Loss):** The prediction loss is:
$$\mathcal{L}_{pred}(\theta) = \mathbb{E}_{(x_t, x_{t+1}) \sim \mathcal{D}} \left[ \|f_\theta(x_t) - x_{t+1}\|^2 \right]$$

### 2.2 Conserved Quantities in State Space

**Definition 2.3 (Conserved Quantity):** A function Q: X → ℝ is conserved under the true dynamics if:
$$Q(x_t) = Q(x_{t+1}) = c \quad \forall t$$

for some constant c depending only on the trajectory.

### 2.3 The Discovery Paradox Theorem

**Theorem 2.1 (Discovery Paradox):** Let f_θ* be an optimal world model (achieving minimum prediction loss). Then for any conserved quantity Q:

$$\nabla_\theta \mathcal{L}_{pred}\big|_{\theta=\theta^*} \perp \nabla_\theta Q_{encoded}$$

where Q_encoded represents how Q is encoded in the model's representation.

More precisely: conserved quantities lie in the **null space** of the Hessian of the prediction loss at optimum.

**Proof:**

**Step 1:** Decompose the state space.

Let x = (q, Q) where q represents changing quantities and Q represents conserved quantities.

At any time t: x_t = (q_t, Q) and x_{t+1} = (q_{t+1}, Q)

Note: Q is the same at both times (conserved).

**Step 2:** Analyze the prediction loss gradient.

$$\mathcal{L}_{pred} = \mathbb{E}\left[ \|f_\theta(x_t) - x_{t+1}\|^2 \right]$$

$$= \mathbb{E}\left[ \|f_\theta^{(q)}(x_t) - q_{t+1}\|^2 + \|f_\theta^{(Q)}(x_t) - Q\|^2 \right]$$

**Step 3:** At optimum, consider the Q component.

At the optimum θ*, the model predicts correctly:
$$f_{\theta^*}^{(Q)}(x_t) = Q$$

The gradient of the loss with respect to parameters affecting Q:
$$\frac{\partial \mathcal{L}}{\partial \theta} \bigg|_Q = 2\mathbb{E}\left[ (f_\theta^{(Q)}(x_t) - Q) \cdot \frac{\partial f_\theta^{(Q)}}{\partial \theta} \right]$$

At optimum: $f_{\theta^*}^{(Q)}(x_t) = Q$, so:
$$\frac{\partial \mathcal{L}}{\partial \theta} \bigg|_Q = 2\mathbb{E}\left[ 0 \cdot \frac{\partial f_\theta^{(Q)}}{\partial \theta} \right] = 0$$

**Step 4:** The null space argument.

The Hessian of the loss at optimum:
$$H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial \theta_i \partial \theta_j}\bigg|_{\theta^*}$$

For directions v that only affect how Q is encoded:
$$v^T H v = 0$$

because Q is already correctly predicted (Q_pred = Q_true), and there's no gradient signal to improve the *representation* of Q.

**Step 5:** Conclusion.

The conserved quantity Q contributes zero to the loss gradient at optimum. Any reparameterization of how Q is encoded leaves the loss unchanged.

Therefore: **Q is in the null space of learning dynamics.**

∎

### 2.4 Corollary: No Incentive for Explicit Representation

**Corollary 2.1:** A world model trained only with prediction loss has no incentive to explicitly represent conserved quantities in an interpretable way.

**Proof:** Since any encoding of Q that satisfies Q_pred ≈ Q_true achieves the same loss, the model may encode Q implicitly, distribute it across many dimensions, or not represent it distinctly at all. The loss landscape is flat in the Q-encoding directions.

∎

---

## 3. The Mathematical Connection

### 3.1 The Parallel Structure

We now reveal the deep parallel between Noether's theorem and the Discovery Paradox.

**Noether's Structure:**
$$\frac{\partial L}{\partial q^i} = 0 \quad \text{(symmetry)} \implies \frac{dp_i}{dt} = 0 \quad \text{(conservation)}$$

The Lagrangian being **independent** of q^i implies the conjugate momentum p_i is **constant**.

**Discovery Paradox Structure:**
$$\frac{\partial \mathcal{L}_{pred}}{\partial Q_{encoding}} = 0 \quad \text{(at optimum)} \implies \frac{d(\text{Q representation})}{d(\text{training})} = 0$$

The prediction loss being **independent** of Q encoding (at optimum) implies the representation of Q **doesn't evolve** during training.

### 3.2 The Fundamental Duality

**Theorem 3.1 (Noether-Discovery Duality):** There is a mathematical duality:

| Noether's Theorem | Discovery Paradox |
|-------------------|-------------------|
| Configuration space Q | Parameter space Θ |
| Lagrangian L(q, q̇) | Loss L_pred(θ) |
| Symmetry: ∂L/∂q = 0 | Optimality: ∂L_pred/∂θ_Q = 0 |
| Conserved momentum p | Frozen representation |
| Time evolution | Training dynamics |

**Interpretation:**

In physics, symmetry (independence) creates conservation (constancy).
In learning, optimality (independence) creates stagnation (no learning signal).

### 3.3 The Deeper Unity: Invariance and Null Spaces

Both theorems are fundamentally about **invariance implying degeneracy**.

**In Noether:**
- The Lagrangian is invariant under a transformation
- This creates a degenerate direction in the equations of motion
- Motion along this direction is "free" → conservation

**In Discovery:**
- The loss is invariant under re-encoding of Q (at optimum)
- This creates a degenerate direction in parameter space
- Parameters in this direction don't change → no learning of Q

**Unified Statement:** Invariance of an objective function implies the existence of null directions in its derivative, leading to conserved quantities (physics) or frozen representations (learning).

---

## 4. Topological Perspective

### 4.1 Phase Space Topology

**Definition 4.1 (Phase Space):** The phase space M = T*Q is the cotangent bundle of configuration space, with coordinates (q, p).

**Definition 4.2 (Symplectic Form):** Phase space carries a natural symplectic form:
$$\omega = \sum_i dq^i \wedge dp_i$$

This 2-form is:
- Closed: dω = 0
- Non-degenerate: ω^n ≠ 0

### 4.2 Energy Surfaces as Submanifolds

**Definition 4.3 (Energy Surface):** For a Hamiltonian H, the energy surface at energy E is:
$$\Sigma_E = \{(q, p) \in M : H(q, p) = E\}$$

This is a (2n-1)-dimensional submanifold of the 2n-dimensional phase space.

**Theorem 4.1 (Liouville):** Hamiltonian flow preserves:
1. The symplectic form ω (canonical structure)
2. The energy surface Σ_E (trajectories lie on constant-energy surfaces)
3. Phase space volume (Liouville's theorem)

### 4.3 Topology of Conservation

**Theorem 4.2 (Topology of Conserved Quantities):** If Q₁, ..., Q_k are independent conserved quantities, trajectories are confined to the intersection:
$$\mathcal{M} = \{x : Q_1(x) = c_1, ..., Q_k(x) = c_k\}$$

This is a (2n - k)-dimensional submanifold.

**For a completely integrable system** (k = n independent conserved quantities in involution), the motion is confined to n-dimensional tori (Liouville-Arnold theorem).

### 4.4 Topological View of the Discovery Paradox

**Definition 4.4 (Representation Manifold):** Consider the manifold of possible representations:
$$\mathcal{R} = \{(z_{inv}, z_{dyn}) : z_{inv} \text{ encodes conserved quantities}\}$$

**Theorem 4.3 (Topological Discovery Paradox):** The prediction loss L_pred defines a function on R. At optimum:

1. The level sets {L_pred = L*} form a submanifold
2. Re-encodings of Q that preserve prediction accuracy form a **gauge orbit**
3. This gauge orbit is the **fiber** over each point in the "physical" representation space

**Interpretation:** There's a fiber bundle structure:
```
Gauge orbits (Q encodings) → Representation space → Physical predictions
```

The prediction loss only sees the base space, not the fibers. Conservation encoding lives in the fibers → invisible to gradient descent.

### 4.5 The Null Space as a Tangent Space

**Proposition 4.1:** The null space of ∇²L_pred at optimum is isomorphic to the tangent space of the gauge orbit.

**Proof Sketch:**
- Directions that don't change the loss correspond to moving along the fiber
- The fiber is the set of equivalent Q-encodings
- Tangent vectors to the fiber are null vectors of the Hessian

∎

### 4.6 Visualization: The Cylinder Analogy

Consider a particle on a cylinder (S¹ × ℝ):

```
        ↑ z (height = conserved)
        |
    ════╪════  Energy surface (circle at fixed height)
        |
    ════╪════
        |
      ──┴──→ θ (angle = changes)
```

- **Dynamics**: Particle moves around the circle (θ changes)
- **Conservation**: Height z stays constant
- **Prediction**: To predict θ_{t+1}, you don't need to know z!

This is the topological essence of the Discovery Paradox:
- z lives in a direction **orthogonal** to the dynamics
- Prediction loss has no gradient in the z direction

---

## 5. Information-Theoretic Analysis

### 5.1 Mutual Information Framework

**Definition 5.1 (Mutual Information):** For random variables X and Y:
$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

where H is Shannon entropy.

### 5.2 Information Content of Conservation

**Proposition 5.1:** For a conserved quantity Q:
$$H(Q_t | Q_{t-1}, Q_{t-2}, ...) = 0$$

**Proof:** Since Q_t = Q_{t-1} = c (conservation), Q_t is completely determined by any past value. Conditional entropy is zero.

∎

**Corollary 5.1:** Conservation implies **zero conditional entropy**, meaning Q carries no *new* information at each time step.

### 5.3 Prediction and Information

**Theorem 5.1 (Information Bottleneck of Prediction):** Optimal prediction requires minimizing:
$$I(Z; X_t) - \beta I(Z; X_{t+1})$$

where Z is the learned representation.

For conserved Q:
- I(Z; Q) can be arbitrarily low (Q doesn't help predict *changes*)
- The information bottleneck principle pushes Q out of the representation

### 5.4 The Information Geometry View

**Definition 5.2 (Fisher Information):** The Fisher information matrix for a model p_θ(x):
$$F_{ij} = \mathbb{E}\left[ \frac{\partial \log p_\theta}{\partial \theta_i} \frac{\partial \log p_\theta}{\partial \theta_j} \right]$$

**Theorem 5.2 (Fisher Information and Conservation):** For parameters θ_Q that only affect the encoding of conserved quantities:
$$F_{θ_Q, θ_Q} → 0 \quad \text{as prediction accuracy → 1}$$

**Interpretation:** The Fisher information (which measures how much data tells us about parameters) vanishes for Q-encoding parameters. The data provides no information about how to encode Q.

### 5.5 Information-Theoretic Statement of Discovery Paradox

**Theorem 5.3 (Information-Theoretic Discovery Paradox):**
$$\lim_{L_{pred} \to L^*} I(\theta_Q; \mathcal{D}) = 0$$

At optimal prediction, the dataset provides zero information about how conserved quantities should be encoded.

**Proof Sketch:**
1. At optimum, any encoding of Q that satisfies Q_pred ≈ Q achieves same likelihood
2. Different θ_Q values are indistinguishable from data
3. Mutual information between θ_Q and data vanishes

∎

---

## 6. Resolution: Physics-First Pretraining

### 6.1 Breaking the Null Space

The Discovery Paradox arises because prediction loss has a null space containing Q-encodings. To learn Q explicitly, we must **break this degeneracy**.

**Definition 6.1 (Conservation Loss):** Define:
$$\mathcal{L}_{cons} = \text{Var}_t[Q_{encoded}(z_t)]$$

This loss is **non-zero** precisely when Q is not constant over time → provides gradient signal.

### 6.2 The Noether-Inspired Loss

**Definition 6.2 (Noether Loss):** Inspired by Noether's theorem:
$$\mathcal{L}_{Noether} = \mathcal{L}_{pred} + \lambda_{inv} \mathcal{L}_{cons} + \lambda_{sym} \mathcal{L}_{symmetry}$$

where:
- L_pred: prediction accuracy
- L_cons: conservation enforcement (z_inv should be constant over time)
- L_symmetry: symmetry detection (learn which symmetries are present)

### 6.3 Mathematical Justification

**Theorem 6.1 (Null Space Breaking):** The combined loss L_Noether has no null space in the Q-encoding directions.

**Proof:**

At any point θ:
$$\nabla_\theta \mathcal{L}_{Noether} = \nabla_\theta \mathcal{L}_{pred} + \lambda_{inv} \nabla_\theta \mathcal{L}_{cons}$$

Even when ∇L_pred = 0 (at prediction optimum), if Q is not perfectly constant:
$$\nabla_\theta \mathcal{L}_{cons} \neq 0$$

The conservation loss provides gradient signal where prediction loss fails.

∎

### 6.4 Two-Phase Training: Formal Description

**Phase 1: Physics Pretraining**

Learn the Noether mapping: Symmetry → Conservation

$$f_{physics}: \text{Sym} \to \text{Cons}$$

Training objective:
$$\mathcal{L}_{Phase1} = \mathcal{L}_{Noether\_mapping} + \mathcal{L}_{conservation\_verification}$$

**Phase 2: Visual Grounding**

Map visual features to physics-aware space:
$$g_{ground}: \mathcal{X}_{visual} \to \mathcal{Z}_{physics}$$

where Z_physics = (z_inv, z_dyn) with z_inv corresponding to conserved quantities.

Training objective:
$$\mathcal{L}_{Phase2} = \mathcal{L}_{recon} + \lambda_1 \mathcal{L}_{inv\_constancy} + \lambda_2 \mathcal{L}_{energy\_correlation}$$

### 6.5 Why This Works: A Topological Argument

**Theorem 6.2 (Fiber Fixing):** Physics-First pretraining selects a **canonical section** of the representation fiber bundle.

**Interpretation:**
- Without physics pretraining: any point in the fiber (any Q-encoding) is equally good
- With physics pretraining: we specify "Q should be encoded as learned Noether charge"
- This **fixes the gauge** and selects a unique representation

---

## 7. Emergent Mathematical Structures

### 7.1 The Symmetry-Learning Correspondence

We observe a correspondence:

| Physical Concept | Learning Concept |
|------------------|------------------|
| Symmetry group G | Invariance class of models |
| Noether charge Q | Latent dimension z_inv |
| Conservation dQ/dt = 0 | Temporal constancy Var_t[z_inv] = 0 |
| Phase space (q, p) | Latent space (z_dyn, z_inv) |
| Symplectic structure | Loss geometry |

### 7.2 Conjecture: Learning as Symplectic Reduction

**Conjecture 7.1:** The process of learning conserved quantities from data can be understood as **symplectic reduction**.

In symplectic geometry, given a symmetry group G acting on phase space M:
$$M \xrightarrow{\text{moment map}} \mathfrak{g}^* \xrightarrow{\text{reduction}} M // G$$

The reduced space M // G captures the "essential" degrees of freedom.

**Analogy in Learning:**
- Full representation space = unreduced phase space
- Conserved quantities = moment map values
- Proper representation = symplectically reduced space

### 7.3 The Representation Fiber Bundle

**Structure:** There exists a fiber bundle:
$$F \hookrightarrow \mathcal{R} \xrightarrow{\pi} \mathcal{P}$$

where:
- R = full representation space
- P = "physical" prediction space (what matters for prediction)
- F = fiber of equivalent Q-encodings

**Prediction loss** is a function on P (base space).
**Conservation loss** is a function on the total space R.

The combined loss L_Noether defines a function on R that:
1. Restricts to L_pred on P
2. Has gradients along fibers F (from L_cons)

### 7.4 Potential New Mathematics

**Open Question 7.1:** Is there a natural connection (in the differential geometric sense) on this fiber bundle that corresponds to optimal Q-encoding?

**Open Question 7.2:** Can we define a "conservation metric" on representation space:
$$d_{cons}(z, z') = |Q(z) - Q(z')|$$
and study its geometry?

**Open Question 7.3:** Is there an analogue of the Liouville-Arnold theorem for learned representations? Specifically, if we learn k conserved quantities, does the representation naturally decompose into a k-dimensional invariant part and a (n-k)-dimensional dynamic part?

### 7.5 Towards a General Theory

We propose that there exists a general theory of **Learning Conservation** with the following structure:

1. **Axioms:**
   - A1: Conservation is characterized by temporal invariance
   - A2: Prediction loss has null space containing conserved quantities
   - A3: Additional conservation loss breaks the degeneracy

2. **Theorems:**
   - T1: (Discovery Paradox) Prediction alone cannot learn conservation
   - T2: (Noether Principle) Teaching symmetry enables conservation discovery
   - T3: (Representation Theorem) Physics-aware representations have canonical structure

3. **Constructions:**
   - C1: The Noether loss functional
   - C2: The physics-first training protocol
   - C3: The invariant-dynamic latent space decomposition

---

## 8. Conclusion

We have established a deep mathematical connection between Noether's theorem and the Discovery Paradox:

1. **Noether** shows that symmetry (Lagrangian independence) implies conservation
2. **Discovery Paradox** shows that optimality (loss independence) implies learning stagnation
3. **Both** are manifestations of invariance creating null spaces

This connection suggests that **physics-informed machine learning is not merely a heuristic but a mathematically principled approach** grounded in the same structures that govern physical law.

The resolution—Physics-First Pretraining—can be understood as:
- Breaking the null space degeneracy
- Fixing a gauge in the representation fiber bundle
- Teaching Noether's insight before grounding in data

---

## References

1. Noether, E. (1918). "Invariante Variationsprobleme." Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen.

2. Arnold, V.I. (1989). "Mathematical Methods of Classical Mechanics." Springer.

3. Marsden, J.E. & Ratiu, T.S. (1999). "Introduction to Mechanics and Symmetry." Springer.

4. Our previous work on the Discovery Paradox (see `Noether_Exp/theory/discovery_paradox.py`)

---

## Appendix A: Notation Summary

| Symbol | Meaning |
|--------|---------|
| Q (config) | Configuration manifold |
| TQ | Tangent bundle (positions + velocities) |
| T*Q | Cotangent bundle (phase space) |
| L | Lagrangian |
| H | Hamiltonian |
| S | Action functional |
| Q (conserved) | Conserved Noether charge |
| ω | Symplectic form |
| L_pred | Prediction loss |
| L_cons | Conservation loss |
| z_inv | Invariant latent dimensions |
| z_dyn | Dynamic latent dimensions |
| θ | Neural network parameters |

---

## Appendix B: Key Equations

**Euler-Lagrange:**
$$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}^i}\right) - \frac{\partial L}{\partial q^i} = 0$$

**Noether Charge:**
$$Q = p_i X^i = \frac{\partial L}{\partial \dot{q}^i} X^i$$

**Discovery Paradox:**
$$\nabla_\theta \mathcal{L}_{pred}\big|_{optimum} \perp \nabla_\theta Q_{encoding}$$

**Noether Loss:**
$$\mathcal{L}_{Noether} = \mathcal{L}_{pred} + \lambda \text{Var}_t[z_{inv}]$$

---

*Document prepared for Physics World Model Publication*
*Last updated: February 3, 2026*
