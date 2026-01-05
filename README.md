# Vision-Based Autonomous Navigation Project

This repository presents a vision-based autonomous navigation assistance system developed using classical computer vision techniques. The project targets real-world campus driving conditions and avoids deep learning approaches, focusing instead on robust image processing methods.

## Overview
The system is designed for a pilot route between Boğaziçi University’s North and South Campuses. It addresses real-world challenges such as worn lane markings, strong shadows, illumination changes, and noisy asphalt textures using traditional image processing pipelines.

## Implemented Modules
- Lane Detection  
  Edge detection, region-of-interest masking, Hough Transform, and temporal smoothing.

- Barrier Position Detection  
  Frame differencing, morphological operations, and structural pattern analysis.

- Traffic Light Classification  
  ROI extraction, HSI color space analysis, hue thresholding, intensity verification, and multi-frame voting.

- Obstacle Detection  
  Edge-based detection, ROI filtering, bounding box constraints, and morphological post-processing.

## Technologies
- Python  
- OpenCV  
- NumPy
  
<h2>Repository Structure</h2>

<pre><code>.
├── data/               # Input videos and datasets
├── outputs/            # Generated output videos and visual results
├── report/             # Project report and documentation
└── src/                # Source code for vision-based modules
</code></pre>


## Key Features
- Classical (non-AI) computer vision approach  
- Real on-route video data  
- Robustness-oriented design for uncontrolled environments  
- Modular and experiment-driven implementation  

## Project Context
This work was developed as part of the EE475 – Digital Image Processing course and is inspired by the methodologies presented in Gonzalez & Woods, *Digital Image Processing (4th Edition)*.

## Authors
Mehmet Emin Algül  
Enes Kuzuoğlu
