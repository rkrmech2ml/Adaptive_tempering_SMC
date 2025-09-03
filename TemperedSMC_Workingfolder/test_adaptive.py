import numpy as np
#from IS_SMC import potentials
import matplotlib.pyplot as plt

# --- Simulate particles and fake observations ---
np.random.seed(0)
num_particles = 100
true_param = 2.0

particles = np.random.uniform(-5, 5, num_particles)
observation = true_param**2  # synthetic "truth"
simulations = particles**2  # simulation: x^2
potentials =  (simulations - observation)**2  # quadratic loss

# Plot potentials2 and potentials
plt.figure(figsize=(8, 4))
#plt.plot(particles, potentials, 'o', label='potentials2 (quadratic loss)')
plt.plot(particles, potentials, 'x', label='potentials (from IS_SMC)')
plt.xlabel('particles')
plt.ylabel('potential value')
plt.title(' potentials')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
weights = np.ones(num_particles) / num_particles  # uniform weights
beta_k = 0.0 
print(f"Initial potential: {potentials}")
# --- NewBeta function with standard regula falsi ---
def NewBeta(potentials, weights, beta_k, cv_target=0.25):
    def m(delta_beta):
        return np.sum([np.exp(-delta_beta * phi) * w for phi, w in zip(potentials, weights)])

    def s(delta_beta):
        return np.sum([(np.exp(-delta_beta * phi))**2 * w for phi, w in zip(potentials, weights)])

    def f(delta_beta):
        m_val = m(delta_beta)
        s_val = s(delta_beta)
        variance = s_val - m_val**2
        if variance < 0:
            return np.inf
        return cv_target * m_val - np.sqrt(variance)

    def regula_falsi(f, a, b, tol=1e-3, max_iter=50):
        fa = f(a)
        fb = f(b)
        if fa * fb > 0:
            raise ValueError("f(a) and f(b) must have opposite signs")

        for i in range(max_iter):
            c = b - fb * (b - a) / (fb - fa)
            fc = f(c)
            if abs(fc) < tol:
                print(f"Root found at iteration {i+1}")
                return c
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        raise RuntimeError("Regula Falsi did not converge")

    try:
        delta_beta = regula_falsi(f, a=1e-6, b=1.0 - beta_k)
    except Exception as e:
        print(f"[WARN] Regula Falsi failed: {e}. ")
        delta_beta = 0.1

    return delta_beta

# --- Run NewBeta ---
delta_beta = NewBeta(potentials, weights, beta_k)
print(f"Computed Δβ = {delta_beta:.5f}")


# --- Plot f(Δβ) ---
# deltas = np.linspace(0.0, 1.0 - beta_k, 200)
# f_vals = []
# for d in deltas:
#     try:
#         m_val = np.sum([np.exp(-d * phi) * w for phi, w in zip(potentials, weights)])
#         s_val = np.sum([(np.exp(-d * phi))**2 * w for phi, w in zip(potentials, weights)])
#         f_val = 0.25 * m_val - np.sqrt(max(s_val - m_val**2, 0))
#     except Exception as e:
#         f_val = np.nan
#     f_vals.append(f_val)

# plt.figure(figsize=(8, 4))
# plt.plot(deltas, f_vals, label=r"$f(\Delta \beta)$")
# plt.axhline(0, color='black', linestyle='--')
# plt.axvline(delta_beta, color='red', linestyle=':', label=f"Δβ = {delta_beta:.4f}")
# plt.title("Root Finding: f(Δβ) = 0 (CV condition)")
# plt.xlabel("Δβ")
# plt.ylabel("f(Δβ)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
