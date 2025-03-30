K-Means Clustering From Scratch, With Visualizations

This project is a basic implementation of the standard k-means algorithm using only python and numpy. No external ML libraries were used to create this project. It features multiple visualizers (A plot of the result, the cost reduction over time, an animation of the algorithm working, and an elbow-method plot that displays cost vs clusters)

## Features

-Pure Python + Numpy k means implementation (No scikit learn)
-Cluster assignment & centroid updates from scratch
-Visualization for:
    -Plotting the result
    -Animating how the result changes over the iterations
    -Plotting how the cost changes over iterations
    -Plotting how cost changes with number of centroids

## How to Run
Run the file directly:
python kmeans.py

## What I learned

-How k means works, step by step.
-How to create visualizations of 2d plots in matplotlib
-Making animations in matplotlib

## TODO

-Add option to save plots as a .gif
-Add PCA to support visualizing clustering on higher dimensional data.
-Add dataset importation