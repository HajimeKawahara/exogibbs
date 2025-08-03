import sympy as sp

# define the system of equations for the HCO system
x, k = sp.symbols('x k')                      # variable x and equilibrium constant k
aH, aO = sp.symbols('alpha_H alpha_O')        #

n_CO   = x
n_CH4  = 1 - x
n_H2O  = aO - x
n_H2   = sp.Rational(1,2)*(aH + 2 - 2*aO) - 3*x
n_tot  = sp.Rational(1,2)*aH + 1 - 2*x

F_raw = n_CH4*n_H2O*n_tot**2 - k*n_CO*n_H2**3 

F = sp.expand(8*F_raw) 

poly = sp.Poly(F, x)
print(poly)
coeffs = [c/8 for c in poly.all_coeffs()] 
print(coeffs)