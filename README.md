# MeshGradientPy
Compute gradients on mesh and unstructured data objects.

This repository aims to fill a gap: no native Python code was available to compute a particular field gradient on a mesh. Some implementations may exist for a structured grid, but this is the first time gradient can be calculated on unstructured mesh (without external libraries such as Paraview or Pyvista).

While Pyvista does provide this functionality, it gets impossible to make it work with other libraries, such as Machine Learning framework. Therefore, we built this library using Tensorflow (which can be replaced with other Deep Learning framework), allowing one to compute gradient on a mesh while performing gradient descent.

# Libraries 

We used numpy and tensorflow for the computation part and meshio to read/write mesh.

We also developed a multiprocessing version of our functions based on the library Ray. This gives full support to tailored computing power even on large clusters.

# Example 

We can read a mesh using meshio: 
```python3
mesh_pv = meshio.read('example.vtu',"vtu")
```
and compute an AGS matrix with the following: 
```python3 
from meshgradient.matrix import build_AGS_matrix
matrix = build_AGS_matrix(mesh)
```

More informations can be found in this [notebook](https://github.com/DonsetPG/MeshGradientPy/blob/main/Example.ipynb).

### Multiprocessing

Each matrix can also be computed in a multi processed fashion using the '_multiprocessing' function, such as ``` build_AGS_matrix_multiprocess```. 

# Background 

We use three different methods to compute the gradient of a field:
* PCE (Per-Cell Linear Estimation)
* AGS (Average Gradient on Star)
* CON (Connectivity, for gradients on boundaries)

You can use any of these three methods or use our built-in functions to compute gradients (that makes the best of these three methods at the same time)

We based our implementation on the one provided with the paper [Gradient Field Estimation on Triangle Meshes](https://www.researchgate.net/publication/330412652_Gradient_Field_Estimation_on_Triangle_Meshes). We describe the basic principles of the methods below, more details can be found in the paper.

These three methods were built for triangle cells. Any other sort of cells won't be considered. 

### PCE 

This method estimates a constant gradient inside each cell. First, we define a linear interpolation at any point <a href="https://www.codecogs.com/eqnedit.php?latex=p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p" title="p" /></a> in a cell of a function <a href="https://www.codecogs.com/eqnedit.php?latex=f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f" title="f" /></a> with: 

<a href="https://www.codecogs.com/eqnedit.php?latex=f_{\sigma}(p)&space;=&space;\sum&space;_{v_{i}&space;\in&space;\sigma}&space;\lambda&space;_i&space;f_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_{\sigma}(p)&space;=&space;\sum&space;_{v_{i}&space;\in&space;\sigma}&space;\lambda&space;_i&space;f_i" title="f_{\sigma}(p) = \sum _{v_{i} \in \sigma} \lambda _i f_i" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=v_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_i" title="v_i" /></a> are the vertices of the cell and <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda_i" title="\lambda_i" /></a> the barycentric coordinates of <a href="https://www.codecogs.com/eqnedit.php?latex=p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p" title="p" /></a> wrt. the vertices. 

With this estimation, for a triangle with 3 vertices, we have:

<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla&space;f_t&space;=&space;(f_j&space;-&space;f_i)&space;\frac{(v_i&space;-&space;v_k)^{\bot}}{2A}&space;&plus;&space;(f_k&space;-&space;f_i)&space;\frac{(v_j&space;-&space;v_i)^{\bot}}{2A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\nabla&space;f_t&space;=&space;(f_j&space;-&space;f_i)&space;\frac{(v_i&space;-&space;v_k)^{\bot}}{2A}&space;&plus;&space;(f_k&space;-&space;f_i)&space;\frac{(v_j&space;-&space;v_i)^{\bot}}{2A}" title="\nabla f_t = (f_j - f_i) \frac{(v_i - v_k)^{\bot}}{2A} + (f_k - f_i) \frac{(v_j - v_i)^{\bot}}{2A}" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=A" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A" title="A" /></a> is the area of the triangle.

### AGS

Given a node, we can use the PCE method to compute a gradient in each cell of the start of the node <a href="https://www.codecogs.com/eqnedit.php?latex=v" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v" title="v" /></a>, thus having:

<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla&space;f_v&space;=&space;\frac{1}{\sum&space;_{\sigma&space;\in&space;\mathcal{N}(v)}A_\sigma}&space;\sum&space;A_\sigma&space;\nabla&space;f_\sigma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\nabla&space;f_v&space;=&space;\frac{1}{\sum&space;_{\sigma&space;\in&space;\mathcal{N}(v)}A_\sigma}&space;\sum&space;A_\sigma&space;\nabla&space;f_\sigma" title="\nabla f_v = \frac{1}{\sum _{\sigma \in \mathcal{N}(v)}A_\sigma} \sum A_\sigma \nabla f_\sigma" /></a>
