import matplotlib.pyplot as plt
import numpy as np

theta = np.linspace(0, 2 * np.pi, 1001)
r = np.sqrt(2)
n = 20

fig = plt.figure(frameon=False)
fig.set_size_inches(20, 20)

ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
ax.set_axis_off()
fig.add_axes(ax)

for x in range(n):
    for y in range(n):
        if (x + y) % 2 == 0:
            ax.plot(x + r * np.cos(theta), y + r * np.sin(theta), "k", linewidth=6)

w = 7
sx = n / 2
sy = n / 2

plt.xlim(sx - w / 2, sx + w / 2)
plt.ylim(sy - w / 2, sy + w / 2)

fig.savefig("circles.png")
