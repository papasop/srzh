# Structural φ(s) Reconstruction and Zeta Coupling Test

This project explores a novel structural resonance function φ(s), constructed from prime logarithms and harmonic modes, and tests its ability to reconstruct or couple to the Riemann zeta function ζ(s) on the critical line.

## 📁 Files

- `structural_phi_zeta_Kfit.py`  
  Fits a structured φ(s) function to ζ(1/2 + it) using optimization, and computes the structural coupling ratio **K** via a log-log perturbation test.

- `structural_phi_zeta_reconstruction.py`  
  Attempts direct reconstruction of ζ(s) using φ(s), comparing amplitude and phase, and evaluating residuals.

## 📐 Mathematical Model

The core structure function is defined as:

φ(s) = ∑_{n=1}^{N} A_n ⋅ cos(λ_n ⋅ log x + θ_n)


- λₙ = log pₙ (logarithm of primes)  
- Aₙ = −1 / √(n ⋅ λₙ)  
- θₙ: tunable phase offsets  
- s = σ + it; evaluated at Re(s) = 1/2  

## 🧪 Goals

- Reconstruct key ζ(s) values using a structured function φ(s)  
- Optimize phase offsets θₙ and amplitude structure  
- Evaluate **residual error** and **K coupling ratio**:

K(x) = d log|φ(x)| / d log H(x) ≈ 1


A K-value near 1 indicates structural resonance between φ(s) and ζ(s).

## 📊 Sample Output (from `Kfit`)




## ✍️ Author

Y.Y.N. Li  
2025.07.30

## 📜 License

MIT License


