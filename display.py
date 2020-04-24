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

ax.fill([0,0,1,1], [0,1,1,0])

plt.xlim(0,8)
plt.ylim(8,0)

plt.title('How to plot a vector in matplotlib ?',fontsize=10)
plt.show()

fig, ax = plt.subplots()
grid = np.array([[20, 20, 20, 20, 20, 20,],
 [20,  3.,  2.,  1. , 0., 20,],
 [20,  4. , 3. ,20,  1., 20,],
 [20,  5., 20, 20,  2., 20,],
 [20,  6. , 5. , 4.,  3., 20,],
 [20, 20, 20, 20, 20, 20]])

for i in range(6):
    for j in range(6):
        text = ax.text(j, i, grid[i, j],ha="center", va="center", color="w")
        print(text)

# text(0.5, 0.5, 'matplotlib', horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)


im = ax.imshow(grid)
plt.show()

# plt.plot([1,2,3,4,5])
# plt.show()