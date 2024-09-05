# Merging CSV Files and Adding Columns in Python

This guide provides instructions on how to merge multiple CSV files from a folder and add two new columns: `iou` (Intersection Over Union) and `confidence_score`. The `iou` values will range between 0.1 and 0.98, and the `confidence_score` values will range between 0.5 and 0.99.

## Requirements

- Python 3.x
- Pandas library
- Numpy library

## Installation

If you haven't installed the required libraries, you can do so using pip:

```bash
pip install pandas numpy
