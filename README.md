# Species-Area Relationships in Changing Environments
## Integrating Habitat Loss and Resource Competition into SAR

**Complex System Simulation Course Project**  
*MSc Computational Science | University of Amsterdam & VU Amsterdam*  
*January 2026*

---

## Overview

Species-Area Relationships (SAR) describe how biodiversity scales with habitat size, a fundamental pattern in ecology often exhibiting power-law behavior: $S(A) \sim A^z$. This project investigates when and why this power-law emerges, and critically examines its robustness under environmental perturbations and ecological dynamics.

Building on the baseline model by García Martín & Goldenfeld (2006) [1], we extend classical SAR theory by integrating:
- **Habitat loss** via Extinction-Area Relationships (EAR)
- **Resource competition** through spatial resource allocation dynamics

Our simulations reveal that while power-law SAR emerges elegantly from spatial clustering and lognormal abundance, it is fragile under realistic ecological pressures with important implications for conservation biology.

---

## Research Questions

### 1. When and why does the SAR follow a power-law?
**Hypothesis:** Power-law emerges from the combination of spatial clustering (fractal branching patterns) and lognormal species abundance distribution.

**Approach:** Implement fractal tree generation with clustering parameter α, compute proximity-based species detection, validate against theoretical predictions.

### 2. How do population thresholds alter extinction predictions from SAR and EAR models?
**Hypothesis:** Classical SAR overestimates extinctions by ignoring spatial distribution and rarity. Extinction-Area Relationship (EAR) models, which account for critical abundance thresholds, provide better predictions (Kitzes & Hart, 2014) [2].

**Approach:** Compare SAR vs. EAR extinction predictions under varying habitat loss scenarios, using minimum viable population sizes ($n_c$).

### 3. Does the inclusion of resource competition affect the SAR power law?
**Hypothesis:** Power-law SAR persists but with modified parameters as interspecific competition reshapes spatial distributions and abundance patterns.

**Approach:** Implement grid-based resource competition model with gamma-distributed resources, competition radii, and survival thresholds.

---

## The Model

### Baseline: Power-Law SAR from Fractal Clustering

Following García Martín & Goldenfeld (2006) [1], we generate species distributions using **self-similar fractal trees**:

1. **Spatial clustering:** Individuals of each species cluster according to a branching process controlled by parameter $\alpha$ $(0.5 \leq  \alpha$ \leq 1$)
   - Each tree starts at a random position within diameter d
   - Branches bifurcate with probability determined by $\alpha$
   - Branch length decreases geometrically: $l_n = l_0 · 1.5^{-(n-1)}$
   - Branching angle varies: $\theta ± \delta_{max}$, where $\delta_{max}$ depends on iteration depth

2. **Species-Area Relationship:** Computed via proximity function approach $S_C(R) = \sum F_s(R) = ∫ h(R,c) p(c) dc$
   Where:
   - **$F_s(R)$:** Probability that species s is within distance R of a random sample point
   - **$h(R,c)$:** Detection probability for species with cover c
   - **$p(c)$:** Lognormal abundance distribution

3. **Implementation:** Sample N random points in region $\Omega$, compute minimum distance to each species, count species within radius $R$, average over samples.

**Expected result:** $S(A) \sim A^z$ with $z \in [0.2, 0.4]$.

### Extension 1: Extinction-Area Relationship (EAR)

Classical SAR assumes species go extinct when all individuals are removed. EAR improves this by modeling **extinction probability** based on critical abundance:

1. **Critical abundance ($n_c$):** Minimum viable population size below which extinction occurs
2. **Extinction probability:** For species with n individuals: **ADD**
3. **Parameter estimation:** Solve for q using numerical root-finding to match observed extinction patterns

**Expected result:** EAR predicts fewer extinctions than classical SAR, especially for spatially aggregated species.

### Extension 2: Resource Competition

Grid-based spatial competition model:

1. **Resource distribution:** Resources spread on grid according to gamma distribution $\Gamma(k, \theta)$
2. **Species competition:** Each species has competition coefficient $\beta_s$
3. **Competition radius:** Species within distance $r_{comp}$ compete for resources
4. **Survival rule:** Individuals with resources below threshold T die
5. **SAR computation:** Calculate S(A) for surviving populations across different grid sizes

**Expected result:** Power-law persists but with altered exponent z and modified abundance distributions.

---

## Repository Structure

`plots.py` contains all configuration for plotting the final data.

`habitat.py` contains any code related to extinction.

`main.py` is the main entry point of the project, and handles the species generation alongside the needed analysis for the core SAR analysis and resource competition.

## Installation & Usage

### Prerequisites

The following packages are required to run the modelling:
* numpy
* matplotlib
* scikit-learn
* scipy

## Conclusions

### Baseline Model Validated 
- Power-law SAR successfully reproduced: $S(A) \sim A^{0.36} (R^2 = 1.00)$
- Clustering (fractal branching) is necessary condition for scale-free behavior
- Confirms García Martín & Goldenfeld (2006) theoretical framework

### Extensions Reveal Limitations
**Habitat loss disrupts power-law**
  - ...
  - EAR models provide more accurate predictions under fragmentation
  
**Resource competition modifies equilibria**
  - Species interactions reshape spatial patterns beyond pure geometry
  - Power-law persists but with altered parameters

---

## References
[1] García Martín, H., & Goldenfeld, N. (2006). On the origin and robustness of power-law species–area relationships in ecology. Proceedings of the National Academy of Sciences, 103(27), 10310–10315. https://doi.org/10.1073/pnas.0510605103

[2‌] Kitzes, J., & Harte, J. (2013). Beyond the species-area relationship: improving macroecological extinction estimates. Methods in Ecology and Evolution, 5(1), 1–8. https://doi.org/10.1111/2041-210x.12130

[3] Traill, L. W., Bradshaw, C. J. A., & Brook, B. W. (2007). Minimum viable population size: A meta-analysis of 30 years of published estimates. Biological Conservation, 139(1-2), 159–166. https://doi.org/10.1016/j.biocon.2007.06.011


---

## AI Usage Declaration

This project utilized AI as a coding assistant, mainly for documentation structure and debugging. AI strategy is available at `AI-usage.md`.

---

## Authors
- [Tadhg Jones](https://github.com/Tadhgg2002)
- [Anna van Dun](https://github.com/Annnvd)
- [Anna De Martin](https://github.com/annademartin)
- [Ricardo Mohamedhoesein](https://github.com/Mohamedhoesein)



‌
‌

