help = """
Takes as argument a polynomail in format X0, X1, X3 ...
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print(help)
    sys.exit(0)

poly_str = sys.argv[1]
poly_coef = [float(c) for c in poly_str.split(', ')]

x = np.linspace(0, 2, 400)
f = np.exp(-1 / (x*x))
poly = poly_coef[0]
for i in range(1, len(poly_coef)):
    poly += poly_coef[i] * x**i

plt.plot(x, poly, label='polynome', linestyle='--', color='blue')
plt.plot(x, f, label='fonction', linestyle='-', color='red')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Approximation Polynomiale")
plt.legend()
plt.grid(True)

plt.show()
