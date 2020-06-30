#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

# Define parameters for Type 1 fonts
fsize = (3,3)
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True

# Figure 1a, division of max(3x+1.5, 2x+1.3, x+0.75, 0) by max(x-0.5, 0).
plt.figure(figsize=fsize, dpi=300)
ax = plt.axes()
ax.plot([0,1,2,3],[0,0.75,1.3,1.5],'b.-',label = 'ENewt(p(x))')
ax.set_ylim(-1.25,2.25)
ax.set_xlabel('Tropical Degree')
ax.set_ylabel('Tropical Coefficient')
ax.plot([1,2],[0.25,-0.75], 'rx-', label = 'ENewt(d(x))')
ax.arrow(1,0.25,0,0.3,**{'width':0.01,'color':'g','linestyle':'-'})
ax.arrow(2,-0.75,0,0.3,**{'width':0.01,'color':'g','linestyle':'-'})
ax.grid(True, **{'alpha':0.5})
ax.legend()
plt.tight_layout()

# Figure 1b, result (quotient + divisor) is max(3x+0.3, 2x+1.3, x+0.75, 0)
plt.figure(figsize=fsize, dpi=300)
ax = plt.axes()
ax.set_xlabel('Tropical Degree')
ax.set_ylabel('Tropical Coefficient')
ax.plot([0,1,2,3],[0,0.75,1.3,0.3],'g.-', label='ENewt(q(x)+d(x))')
ax.set_ylim(-1.25,2.25)
ax.grid(True, **{'alpha':0.5})
ax.legend()
plt.tight_layout()

# Figure 2a, polytope of example network provided in paper, Section 4.1
fig = plt.figure(figsize=fsize, dpi=300)
ax = plt.axes(projection='3d')
ax.view_init(azim=-100)
ax.set_xlim([0,3.5])
ax.set_ylim([0,3.5])
ax.set_zlim([0,4.5])
ax.set_xlabel('Tropical Degree 1')
ax.set_ylabel('Tropical Degree 2')
ax.set_zlabel('Coefficient')

# Create the faces of the polytope
x = [0, 1, 1, 0]
y = [0, 0, 1, 1]
z = [0, 1, 2, 1]
poly = Poly3DCollection([list(zip(x,y,z))], zorder=1)
poly.set_edgecolor('black')
poly.set_facecolor('yellow')
ax.add_collection3d(poly)

x = [1, 1, 2, 2]
y = [0, 1, 2, 1]
z = [1, 2, 3, 2]
poly = Poly3DCollection([list(zip(x,y,z))], zorder=1)
poly.set_edgecolor('black')
poly.set_facecolor('yellow')
ax.add_collection3d(poly)

x = [0, 1, 2, 1]
y = [1, 1, 2, 2]
z = [1, 2, 3, 2]
poly = Poly3DCollection([list(zip(x,y,z))], zorder=1)
poly.set_edgecolor('black')
poly.set_facecolor('yellow')
ax.add_collection3d(poly)

ax.plot3D([2],[1],[2],'bo', zorder=2, label='Vertex 101')
ax.legend(loc='upper left')

# Figure 2b, the other network from the same example as before.
fig = plt.figure(figsize=fsize, dpi=300)
ax = plt.axes(projection='3d')
ax.view_init(azim=-100)
ax.set_xlim([0,3.5])
ax.set_ylim([0,3.5])
ax.set_zlim([0,4.5])
ax.set_xlabel('Tropical Degree 1')
ax.set_ylabel('Tropical Degree 2')
ax.set_zlabel('Coefficient')
x = [0, 2, 2, 0]
y = [0, 0, 1, 1]
z = [0, 2, 3, 1]
poly = Poly3DCollection([list(zip(x,y,z))])
poly.set_edgecolor('black')
poly.set_facecolor('red')
ax.add_collection3d(poly)
x = [2, 2, 3, 3]
y = [0, 1, 2, 1]
z = [2, 3, 4, 3]
poly = Poly3DCollection([list(zip(x,y,z))])
poly.set_edgecolor('black')
poly.set_facecolor('red')
ax.add_collection3d(poly)
x = [0, 2, 3, 1]
y = [1, 1, 2, 2]
z = [1, 3, 4, 2]
poly = Poly3DCollection([list(zip(x,y,z))])
poly.set_edgecolor('black')
poly.set_facecolor('red')
ax.add_collection3d(poly)

ax.plot3D([3],[1],[3],'bo', label='Vertex 101')
ax.legend(loc='upper left')

# Figure 3a, polytope of network with three neurons, example from Section 4.3
fig = plt.figure(figsize=fsize, dpi=300)
ax = plt.axes(projection='3d')
ax.view_init(azim=-100)
ax.set_xlim([0,2.5])
ax.set_ylim([0,2.5])
ax.set_zlim([0,3.5])
ax.set_xlabel('Tropical Degree 1')
ax.set_ylabel('Tropical Degree 2')
ax.set_zlabel('Coefficient')
x = [0, 1, 1, 0]
y = [0, 0, 1, 1]
z = [0, 1, 2, 1]
poly = Poly3DCollection([list(zip(x,y,z))])
poly.set_edgecolor('black')
poly.set_facecolor('yellow')
ax.add_collection3d(poly)
x = [1, 1, 2, 2]
y = [0, 1, 2, 1]
z = [1, 2, 3, 2]
poly = Poly3DCollection([list(zip(x,y,z))])
poly.set_edgecolor('black')
poly.set_facecolor('yellow')
ax.add_collection3d(poly)
x = [0, 1, 2, 1]
y = [1, 1, 2, 2]
z = [1, 2, 3, 2]
poly = Poly3DCollection([list(zip(x,y,z))])
poly.set_edgecolor('black')
poly.set_facecolor('yellow')
ax.add_collection3d(poly)
x = [0, 0, 2, 2]
y = [0, 1, 2, 1]
z = [0, 1, 3, 2]
ax.plot3D(x,y,z,'bo', label='Vertices Kept')
ax.legend()

# Figure 3b, approximation of the previous polytope with a network with fewer
# neurons.
fig = plt.figure(figsize=fsize, dpi=300)
ax = plt.axes(projection='3d')
ax.view_init(azim=-100)
ax.set_xlim([0,2.5])
ax.set_ylim([0,2.5])
ax.set_zlim([0,3.5])
ax.set_xlabel('Tropical Degree 1')
ax.set_ylabel('Tropical Degree 2')
ax.set_zlabel('Coefficient')
x = [0, 0, 2, 2]
y = [0, 1, 2, 1]
z = [0, 1, 3, 2]
poly = Poly3DCollection([list(zip(x,y,z))])
poly.set_edgecolor('black')
poly.set_facecolor('red')
ax.add_collection3d(poly)
ax.plot3D(x,y,z,'bo', label='Vertices Kept')
ax.legend()

# Figure 4a, original network polytope of example.
fig = plt.figure(figsize=fsize, dpi=300)
ax = plt.axes(projection='3d')
ax.view_init(azim=-100)
ax.set_xlim([0,5.5])
ax.set_ylim([0,5.5])
ax.set_zlim([0,3.5])
ax.set_xlabel('Tropical Degree 1')
ax.set_ylabel('Tropical Degree 2')
ax.set_zlabel('Coefficient')

ws = np.array([[1,2,1],[2,1,1],[2,0,0],[0,2,0]])

w_all = [np.zeros((1,3))]
for i1 in range(2):
    for i2 in range(2):
        for i3 in range(2):
            for i4 in range(2):
                if (i1+i2+i3+i4 > 0):
                    w_temp = []
                    if i1>0:
                        w_temp.append(ws[[0],:])
                    if i2>0:
                        w_temp.append(ws[[1],:])
                    if i3>0:
                        w_temp.append(ws[[2],:])
                    if i4>0:
                        w_temp.append(ws[[3],:])
                    w_all.append(sum(w_temp))

w_all.append(np.array([[2,2,5]]))
w_all = np.concatenate(w_all, axis=0)
hull = ConvexHull(w_all, qhull_options='QG16')

f = hull.simplices[hull.good]

for i in range(len(f)):
    x = [w_all[f[i,2],0]]
    y = [w_all[f[i,2],1]]
    z = [w_all[f[i,2],2]]

    for j in range(3):
        w = w_all[f[i,j]]
        x.append(w[0])
        y.append(w[1])
        z.append(w[2])

    poly = Poly3DCollection([list(zip(x,y,z))])
    poly.set_edgecolor('black')
    poly.set_facecolor('blue')
    ax.add_collection3d(poly)

# Figure 4b, polytope approximated via heuristic algorithm.
fig = plt.figure(figsize=fsize, dpi=300)
ax = plt.axes(projection='3d')
ax.view_init(azim=-100)
ax.set_xlim([-2.5,5.5])
ax.set_ylim([-2.5,5.5])
ax.set_zlim([0,3.5])
ax.set_xlabel('Tropical Degree 1')
ax.set_ylabel('Tropical Degree 2')
ax.set_zlabel('Coefficient')

ws = np.array([[5,5,2],[0,-2,0],[-2,0,0]])

w_all = [np.zeros((1,3))]
for i1 in range(2):
    for i2 in range(2):
        for i3 in range(2):
            if (i1+i2+i3 > 0):
                w_temp = []
                if i1>0:
                    w_temp.append(ws[[0],:])
                if i2>0:
                    w_temp.append(ws[[1],:])
                if i3>0:
                    w_temp.append(ws[[2],:])
                w_all.append(sum(w_temp))

w_all.append(np.array([[0,0,5]]))
w_all = np.concatenate(w_all, axis=0)
hull = ConvexHull(w_all, qhull_options='QG8')

f = hull.simplices[hull.good]

for i in range(len(f)):
    x = [w_all[f[i,2],0]]
    y = [w_all[f[i,2],1]]
    z = [w_all[f[i,2],2]]

    for j in range(3):
        w = w_all[f[i,j]]
        x.append(w[0])
        y.append(w[1])
        z.append(w[2])

    poly = Poly3DCollection([list(zip(x,y,z))])
    poly.set_edgecolor('black')
    poly.set_facecolor('red')
    ax.add_collection3d(poly)

# Figure 4c, polytope approximated with stable algorithm.
fig = plt.figure(figsize=fsize, dpi=300)
ax = plt.axes(projection='3d')
ax.view_init(azim=-100)
ax.set_xlim([0,5.5])
ax.set_ylim([0,5.5])
ax.set_zlim([0,3.5])
ax.set_xlabel('Tropical Degree 1')
ax.set_ylabel('Tropical Degree 2')
ax.set_zlabel('Coefficient')

ws = np.array([[0,2,0],[2,0,0],[3,3,2]])

w_all = [np.zeros((1,3))]
for i1 in range(2):
    for i2 in range(2):
        for i3 in range(2):
            if (i1+i2+i3 > 0):
                w_temp = []
                if i1>0:
                    w_temp.append(ws[[0],:])
                if i2>0:
                    w_temp.append(ws[[1],:])
                if i3>0:
                    w_temp.append(ws[[2],:])
                w_all.append(sum(w_temp))

w_all.append(np.array([[0,0,5]]))
w_all = np.concatenate(w_all, axis=0)
hull = ConvexHull(w_all, qhull_options='QG8')

f = hull.simplices[hull.good]

for i in range(len(f)):
    x = [w_all[f[i,2],0]]
    y = [w_all[f[i,2],1]]
    z = [w_all[f[i,2],2]]

    for j in range(3):
        w = w_all[f[i,j]]
        x.append(w[0])
        y.append(w[1])
        z.append(w[2])

    poly = Poly3DCollection([list(zip(x,y,z))])
    poly.set_edgecolor('black')
    poly.set_facecolor('yellow')
    ax.add_collection3d(poly)

# Output all of the above figures.
plt.show()
