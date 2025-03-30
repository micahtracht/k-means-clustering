# K-Means Clustering From Scratch, With Visualizations

This project is a basic implementation of the standard k-means algorithm using only Python and NumPy. No external machine learning libraries were used.

It features multiple visualizations:
- A plot of the final result
- A cost reduction plot over time
- An animation showing the algorithm working
- An elbow-method plot showing cost vs. number of clusters

---

## Features

- Pure Python + NumPy k-means implementation (no scikit-learn)
- Cluster assignment and centroid updates from scratch
- Visualization for:
  - Plotting the result
  - Animating how the result changes over iterations
  - Plotting how the cost changes over iterations
  - Plotting how cost changes with number of centroids

---

## What I Learned

- How k-means works, step by step
- How to create 2D visualizations using Matplotlib
- How to make animations in Matplotlib

---

## TODO

- Add option to save plots as a `.gif`
- Add PCA to support visualizing clustering on higher-dimensional data
- Add dataset importation from CSV

---

## How to Run

Run the script with:

```bash
python kmeans.py
