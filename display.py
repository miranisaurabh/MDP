import matplotlib.pyplot as plt
import numpy as np

X = np.array((0.5))
Y= np.array((0.5))
U = np.array((0))
V = np.array((0.5))

fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V,units='xy' ,scale=1)

X = np.array((0.5))
Y= np.array((0.5))
U = np.array((0.5))
V = np.array((0))

q = ax.quiver(X, Y, U, V,units='xy' ,scale=1)

plt.grid()

ax.set_aspect('equal')

plt.xlim(0,8)
plt.ylim(8,0)

plt.title('How to plot a vector in matplotlib ?',fontsize=10)
plt.show()