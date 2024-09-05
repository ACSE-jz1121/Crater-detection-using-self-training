# Project Overview

This repository contains the code and data used for crater detection on Martian surfaces, focusing on different crater sizes using self-training algorithms and multi-resolution data. Below is a description of each file and its purpose in the project.

## Files Description

- **Auto-iterative self-training.ipynb**: This notebook runs the self-training algorithm on the smallest resolution data (416x416 pixels) to detect small craters with diameters smaller than 10 km. It focuses on enhancing detection accuracy for smaller craters by iteratively training the model.

- **Detection_results_L19_0.5.csv**: This file contains the detection results from the Crater Detection Algorithm (CDA) using the L19 metric with an Intersection over Union (IoU) value of 0.5. The data reflects the performance of the detection model on small to medium-sized craters.

- **Multi-resolution detections.ipynb**: This notebook executes the detection process on higher resolution data (1024x1024 and 6144x6144 pixels) to identify larger craters with diameters above 1.5 km. It extends the detection capabilities to larger craters, improving the overall accuracy of the model across various crater sizes.

- **ensemble module.py**: This script handles the ensemble section of the self-training algorithm. It combines detection results from different models to improve the robustness and accuracy of the final crater detection output.

- **dataset_generation.ipynb**: This notebook is used for generating the dataset used in the project, based on existing crater catalogs and image data.


## Data Sources

- **Crater Data**: The crater data used in this project is derived from the Robbins et al. 2020 dataset, which can be accessed at [Robbins et al. Crater Database](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011JE003966).
  
- **THEMIS Images**: The images used for crater detection are derived from the THEMIS (Ther
