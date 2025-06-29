import numpy as np





#def NewBeta(potential, w_list, beta):
def CoeffVariation(potential, w_list, beta):
    #potential = (potential - np.min(potential)) / (np.max(potential) - np.min(potential))
    #potential = (potential - np.mean(potential)) / np.std(potential)
    #potential = potential * 10000  # Scale potential values for numerical stability



    def integrand_m(DeltaBeta):
        exponents = [np.exp(-potential[i] * DeltaBeta) * w_list[i] for i in range(len(w_list))]
        return np.sum(exponents)

    def integrand_s(DeltaBeta):
        exponents = [(np.exp(-potential[i] * DeltaBeta) * w_list[i])**2 for i in range(len(w_list))]
        return np.sum(exponents)

    def f(x):
        cv_ = 0.25  # Target coefficient of variation
        m_val = integrand_m(x)
        s_val = integrand_s(x)
        CalcCV =  cv_ * m_val -np.sqrt(max(0.0, s_val - m_val**2))                                                           # target_cv = 0.01
        print(f"f({x:.4f}) = {CalcCV:.6f}")
        return CalcCV

    # def regula_falsi(f, beta, tol=1e-2, max_iter=20):
    #     a = 0.0
    #     b = min(1.0 - beta, 1.0)
    #     fa = f(a)
    #     fb = f(b)
    #     if fa * fb > 0:
    #         raise ValueError("f(a) and f(b) must have opposite signs")

    #     for i in range(max_iter):
    #         c = b - fb * (b - a) / (fb - fa)
    #         fc = f(c)
    #         if abs(fc) < tol:
    #             print(f"Root found at x = {c:.6f} after {i+1} iterations.")
    #             return c
    #         if fa * fc < 0:
    #             b, fb = c, fc
    #             fb *= 0.5
    #         else:
    #             a, fa = c, fc
    #             fa *= 0.5

    #     raise RuntimeError("Regula Falsi did not converge")

    CoeffVariation = f(beta)
    return CoeffVariation



# def NewBeta(potential, w_list, beta):
       
#         #print(f"NewBeta called with potential={potential}, w_list={w_list}, beta={beta}")
#     # Integrand for m(beta)
#         def integrand_m(DeltaBeta):
#             exponents = []
#             val = 0.0
#             for i in range(len(w_list)):
#                 val = np.exp(-potential[i] * DeltaBeta)* w_list[i]
#                 exponents.append(val)
#             #print(f"integrand_m({DeltaBeta}): {exponents}")
#             # mean of all exponentials for current DeltaBeta
#             mean_exp = np.sum(exponents)
#             #print(f"(integrand_m({DeltaBeta}))^2: {mean_exp**2}")
#             return mean_exp

#         # Integrand for s(beta)
#         def integrand_s(DeltaBeta):
#             exponents2 = []
#             val = 0.0
#             for i in range(len(w_list)):
#                 val = np.exp((-(potential[i]  * DeltaBeta)))**2 * w_list[i]
#                 exponents2.append(val)
#             #print(f"integrand_s({DeltaBeta}): {exponents2}")
#             mean_exp2 = np.sum(exponents2)
#             #print(f"integrand_s({DeltaBeta}): {mean_exp2}")
#             return mean_exp2

#         # Function to find root of: f(x) = m(x)^2 * (cv_^2 + 1) - s(x)
#         def f(x):
#             cv_ = 0.01
#             m_val = integrand_m(x)
#             s_val = integrand_s(x)
#             val = cv_ * m_val - np.sqrt(s_val- m_val**2)
#             print(f"f({x}): f={val}")
#             return val
#         # import matplotlib.pyplot as plt
#         x_vals = np.linspace(0, 1, 100)
#         m_vals = [integrand_m(x) for x in x_vals]
#         s_vals = [integrand_s(x) for x in x_vals]
#         y_vals = [f(x) for x in x_vals]

#         # print(f"m_vals: {m_vals}")
#         # print(f"s_vals: {s_vals}")
#         # print(f"y_vals: {y_vals}")

#         # plt.figure(figsize=(10, 6))
#         # plt.plot(x_vals, m_vals, label="m(x)")
#         # plt.plot(x_vals, s_vals, label="s(x)")
#         # plt.plot(x_vals, y_vals, label="f(x)")
#         # plt.xlabel("x")
#         # plt.ylabel("Value")
#         # plt.title("Plot of m(x), s(x), and f(x)")
#         # plt.legend()
#         # plt.grid(True)
#         # plt.show()


#         # Regula Falsi method implementation
#         def regula_falsi(f, beta, tol=1e-2, max_iter=20):
#             a = 0.0
#             b = min(1.0 - beta, 1.0)
#             fa = f(a)
#             fb = f(b)

#             if fa * fb > 0:
#                 raise ValueError("f(a) and f(b) wst have opposite signs")
            

#             for i in range(max_iter):
#                 # Regula Falsi forwla
#                 c = b - fb * (b - a) / (fb - fa)
#                 fc = f(c)

#                 if abs(fc) < tol:
#                     print(f"Root found at x = {c:.6f} after {i+1} iterations.")
#                     return c

#                 if fa * fc < 0:
#                     b, fb = c, fc
#                     fb *= 0.5  # Illinois method
#                 else:
#                     a, fa = c, fc
#                     fa *= 0.5
#             # If loop ends without return, raise error outside the loop
#             raise RuntimeError("Regula Falsi method did not converge")

#         delta = regula_falsi(f, beta)
#         return delta


# def NewBeta(potentials, weights, beta_k, cv_target=0.25):
#     potentials=potentials*10000
#     """
#     Compute Δβ using coefficient of variation (CV) criterion with standard Regula Falsi.

#     Args:
#         potentials: list of Φ values for each particle (negative log-likelihoods)
#         weights: current normalized particle weights (should sum to 1)
#         beta_k: current beta value
#         cv_target: desired coefficient of variation (typically 0.25)

#     Returns:
#         delta_beta: increment to beta_k such that CV condition is met
#     """

#     def m(delta_beta):
#         return np.sum([np.exp(-delta_beta * phi) * w for phi, w in zip(potentials, weights)])

#     def s(delta_beta):
#         return np.sum([(np.exp(-delta_beta * phi))**2 * w for phi, w in zip(potentials, weights)])

#     def f(delta_beta):
#         m_val = m(delta_beta)
#         s_val = s(delta_beta)
#         variance = s_val - m_val**2
#         if variance < 0:
#             return np.inf
#         return cv_target * m_val - np.sqrt(variance)

#     # Standard Regula Falsi
#     def regula_falsi(f, a, b, tol=1e-3, max_iter=50):
#         fa = f(a)
#         fb = f(b)
#         if fa * fb > 0:
#             raise ValueError("f(a) and f(b) must have opposite signs")

#         for i in range(max_iter):
#             c = b - fb * (b - a) / (fb - fa)
#             fc = f(c)

#             if abs(fc) < tol:
#                 return c

#             if fa * fc < 0:
#                 b, fb = c, fc
#             else:
#                 a, fa = c, fc

#         raise RuntimeError("Regula Falsi did not converge")

#     try:
#         delta_beta = regula_falsi(f, a=1e-6, b=1.0 - beta_k)
#     except Exception as e:
#         print(f"[WARN] Regula Falsi failed: {e}. Falling back to Δβ = 0.01")
#         delta_beta = 0.01

#     return delta_beta
# '