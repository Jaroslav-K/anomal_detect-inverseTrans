import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n_samples = 2000
n_dims = 100

# Generate uniform samples inside the unit sphere
inside_samples = np.random.normal(size=(n_samples, n_dims))
inside_norms = np.linalg.norm(inside_samples, axis=1)
inside_mask = inside_norms != 0
inside_samples = inside_samples[inside_mask] / inside_norms[inside_mask].reshape(-1, 1)

# Generate uniform samples outside the unit sphere
outside_samples = np.random.normal(size=(n_samples, n_dims))
outside_norms = np.linalg.norm(outside_samples, axis=1)
outside_mask = outside_norms > 1
outside_samples = outside_samples[outside_mask]

# Create a 3D plot of the uniform samples
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(inside_samples[:,0], inside_samples[:,1], inside_samples[:,2], s=10, c='r', marker='o')
ax.scatter(outside_samples[:,0], outside_samples[:,1], outside_samples[:,2], s=10, c='b', marker='x')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_box_aspect([1,1,1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add a 3D sphere to the plot
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="gray", alpha=0.5)

plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# n_samples = 2000
# n_dims = 3

# # Generate uniform samples on a hypercube centered at the origin
# samples = np.random.uniform(low=-1, high=1, size=(n_samples, n_dims))

# # Keep only the samples that lie inside the unit sphere
# inside_norms = np.linalg.norm(samples, axis=1)
# inside_mask = inside_norms <= 1
# inside_samples = samples[inside_mask]

# # Keep only the samples that lie outside the unit sphere
# outside_norms = np.linalg.norm(samples, axis=1)
# outside_mask = outside_norms > 1
# outside_samples = samples[outside_mask]

# # Create a 3D plot of the uniform samples
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(inside_samples[:,0], inside_samples[:,1], inside_samples[:,2], s=10, c='r', marker='o')
# ax.scatter(outside_samples[:,0], outside_samples[:,1], outside_samples[:,2], s=10, c='b', marker='x')
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])
# ax.set_box_aspect([1,1,1])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # Add a 3D sphere to the plot
# u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
# x = np.cos(u)*np.sin(v)
# y = np.sin(u)*np.sin(v)
# z = np.cos(v)
# ax.plot_wireframe(x, y, z, color="gray", alpha=0.5)

# plt.show()
