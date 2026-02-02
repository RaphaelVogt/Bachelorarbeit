# Hyperbolic Embeddings of Finite Metric Spaces

Implementation of hyperbolic geometry for efficient embeddings of graph-based data into the Poincaré disk. This project was developed as part of my Bachelor's Thesis at Heidelberg University (Grade: 1.0)

## 📌 Abstract
Traditional Euclidean embeddings often struggle to represent hierarchical data without high distortion. This project utilizes **Riemannian geometry** and the **Poincaré disk model** to embed finite metric spaces, leveraging the fact that hyperbolic space volume grows exponentially with its radius. Additionally different data initialization methods are proposed.

## 🖼 Visualizations

| Bild 1 | Bild 2 | Bild 3 |
|--------|--------|--------|
| ![Bild1](bild1.jpg) | ![Bild2](bild2.jpg) | ![Bild3](bild3.jpg) |

## 🚀 Key Features
* **Manifold Optimization:** Custom implementation of RSGD to maintain points within the Poincaré manifold.
* **GPU Acceleration:** Fully compatible with **PyTorch/CUDA** for high-performance tensor operations.
