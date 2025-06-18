import numpy as np




# Test the f(x) function without running regula falsi
def new_beta_test(potential, w_list, beta):

    # Copy the inner function for testing
    def integrand_m(DeltaBeta):
        exponents = []
        val = 0.0
        for i in range(len(w_list)):
            val = np.exp(-potential[i] * DeltaBeta) * w_list[i]
            val = val * val
            exponents.append(val)
        mean_exp = np.mean(exponents)
        print(f"(integrand_m({DeltaBeta}))^2: {mean_exp}")
        return mean_exp

    def integrand_s(DeltaBeta):
        exponents2 = []
        val = 0.0
        for i in range(len(w_list)):
            val = np.exp((-(potential[i] * DeltaBeta)) ** 2) * w_list[i]
            exponents2.append(val)
        mean_exp2 = np.mean(exponents2)
        print(f"integrand_s({DeltaBeta}): {mean_exp2}")
        return mean_exp2

    def f(x):
        cv_ = 0.25
        m_val = integrand_m(x)
        s_val = integrand_s(x)
        val = cv_ * m_val - np.sqrt(s_val - m_val)
        print(f"f({x}): f={val}")
        return val

    # Test f(x) for a few values

    # Plot f(x) over a range
    import matplotlib.pyplot as plt

    x_vals = np.linspace(0, 20, 100)
    y_vals = [f(x) for x in x_vals]

    plt.plot(x_vals, y_vals, label="f(x)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Plot of f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

# def NewBeta(potential, w_list, beta):
#     # Integrand for m(beta)
#         def integrand_m(DeltaBeta):
#             exponents = []
#             val = 0.0
#             for i in range(len(w_list)):
#                 val = np.exp(-potential[i] * DeltaBeta)* w_list[i]
#                 val = val*val 
#                 exponents.append(val)
#             #print(f"integrand_m({DeltaBeta}): {exponents}")
#             # mean of all exponentials for current DeltaBeta
#             mean_exp = np.mean(exponents)
#             print(f"(integrand_m({DeltaBeta}))^2: {mean_exp}")
#             return mean_exp

#         # Integrand for s(beta)
#         def integrand_s(DeltaBeta):
#             exponents2 = []
#             val = 0.0
#             for i in range(len(w_list)):
#                 val = np.exp((-(potential[i]  * DeltaBeta))**2)* w_list[i]
#                 exponents2.append(val)
#             #print(f"integrand_s({DeltaBeta}): {exponents2}")
#             mean_exp2 = np.mean(exponents2)
#             print(f"integrand_s({DeltaBeta}): {mean_exp2}")
#             return mean_exp2

#         # Function to find root of: f(x) = m(x)^2 * (cv_^2 + 1) - s(x)
#         def f(x):
#             cv_ = 0.9
#             m_val = integrand_m(x)
#             s_val = integrand_s(x)
#             val = cv_ * m_val - np.sqrt(s_val- m_val)
#             print(f"f({x}): f={val}")
#             return val

#         # Regula Falsi method implementation
#         def regula_falsi(f, beta, tol=1e-2, max_iter=2):
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