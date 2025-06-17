import numpy as np


def NewBeta(potential, w_list, beta):
    # Integrand for m(beta)
        def integrand_m(DeltaBeta):
            exponents = []
            val = 0.0
            for i in range(len(w_list)):
                val = np.exp(-potential[i] * DeltaBeta)* w_list[i]
                val = val*val 
                exponents.append(val)
            #print(f"integrand_m({DeltaBeta}): {exponents}")
            # mean of all exponentials for current DeltaBeta
            mean_exp = np.mean(exponents)
            print(f"(integrand_m({DeltaBeta}))^2: {mean_exp}")
            return mean_exp

        # Integrand for s(beta)
        def integrand_s(DeltaBeta):
            exponents2 = []
            val = 0.0
            for i in range(len(w_list)):
                val = np.exp((-(potential[i]  * DeltaBeta))**2)* w_list[i]
                exponents2.append(val)
            #print(f"integrand_s({DeltaBeta}): {exponents2}")
            mean_exp2 = np.mean(exponents2)
            print(f"integrand_s({DeltaBeta}): {mean_exp2}")
            return mean_exp2

        # Function to find root of: f(x) = m(x)^2 * (cv_^2 + 1) - s(x)
        def f(x):
            cv_ = 0.25
            m_val = integrand_m(x)
            s_val = integrand_s(x)
            val = cv_ * m_val - np.sqrt(s_val- m_val)
            print(f"f({x}): f={val}")
            return val

        # Regula Falsi method implementation
        def regula_falsi(f, beta, tol=1e-2, max_iter=2):
            a = 1e-4
            b = min(1.0 - beta, 1.0)
            fa = f(a)
            fb = f(b)

            if fa * fb > 0:
                raise ValueError("f(a) and f(b) wst have opposite signs")
            

            for i in range(max_iter):
                # Regula Falsi forwla
                c = b - fb * (b - a) / (fb - fa)
                fc = f(c)

                if abs(fc) < tol:
                    print(f"Root found at x = {c:.6f} after {i+1} iterations.")
                    return c

                if fa * fc < 0:
                    b, fb = c, fc
                    fb *= 0.5  # Illinois method
                else:
                    a, fa = c, fc
                    fa *= 0.5
            # If loop ends without return, raise error outside the loop
            raise RuntimeError("Regula Falsi method did not converge")

        delta = regula_falsi(f, beta)
        return delta