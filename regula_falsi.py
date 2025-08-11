import numpy as np






def CoeffVariation(potential, w_list, beta,delta):

    def integrand_m(DeltaBeta):
        weights = [np.exp(-potential[i] * DeltaBeta) * w_list[i] for i in range(len(w_list))]
        return np.sum(weights)

    def integrand_s(DeltaBeta):
        weights = [np.exp(-potential[i] * DeltaBeta)**2 * w_list[i] for i in range(len(w_list))]
        return np.sum([w for w in weights])

    def f(x):
         # Target coefficient of variation
        m_val = integrand_m(x)
        s_val = integrand_s(x)
        print(f"m({x}) = {m_val:.6f}, s({x}) = {s_val:.6f}")
        if  (s_val / m_val**2) < 1:
            CalcCV = 0.0
        else:
            CalcCV = np.sqrt((s_val / m_val**2) - 1)  # 
        print(f"CV Variation wrt tempering parameter  = {CalcCV:.6f}")
        return CalcCV

    CoeffVariation = f(delta)
    return CoeffVariation

""" uncomment this section to do the adaptive tempering
    This uses the Regula Falsi method to find the new beta value.

 """

def NewBeta(potential, w_list, beta):
        
        
       
        #print(f"NewBeta called with potential={potential}, w_list={w_list}, beta={beta}")
    # Integrand for m(beta)
        
        def integrand_m(DeltaBeta):
            weights = [np.exp(-potential[i] * DeltaBeta) * w_list[i] for i in range(len(w_list))]
            return np.sum(weights)

        def integrand_s(DeltaBeta):
            weights = [np.exp(-potential[i] * DeltaBeta)**2 * w_list[i] for i in range(len(w_list))]
            return np.sum([w for w in weights])
        # Function to find root of: f(x) = m(x)^2 * (cv_^2 + 1) - s(x)
        def f(x):
            cv_ = 0.25
            m_val = integrand_m(x)
            s_val = integrand_s(x)
            print(f"m({x}) = {m_val:.6f}, s({x}) = {s_val:.6f}")
            print(np.sqrt(np.maximum(0.0,s_val- m_val**2)))
            val = cv_ * m_val - np.sqrt(np.maximum(0.0,s_val- m_val**2))
            print(f"f({x}): f={val}")
            return val

        # Regula Falsi method implementation
        def regula_falsi(f, a, b, tol=1e-3, max_iter=50):
            fa = f(a)
            fb = f(b)


            if fa * fb > 0:
                raise ValueError("f(a) and f(b) must have opposite signs")

            for i in range(max_iter):
                c = b - fb * (b - a) / (fb - fa)
                fc = f(c)

            if abs(fc) < tol:
                print(f"Root found at x = {c:.6f} after {i+1} iterations.")
                return c

            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc

            raise RuntimeError("Regula Falsi did not converge")

        try:
            delta = regula_falsi(f, a=0.0, b=1.0 - beta)
        except Exception as e:
            print(f"[WARN] Regula Falsi failed: {e}. Falling back to Δβ = 0.01")
            delta = 0.01

        return delta


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