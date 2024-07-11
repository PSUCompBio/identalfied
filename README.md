# identalfied
surfaced identification


Milestones
1.  Find the position and orientation of the model (DONE)
2.  Find all “peak points” (local maxima) above a height threshold  (DONE)
3.  Partition the model into individual teeth using curvature. (Ongoing)
4.  Identify the type of each tooth found. 
5.  Find the MHB landmark 
6.  Demonstrate program works for any teeth scan. I will provide 2 to test. 



<a id="readme-top"></a>


<h3 align="center">Automated Mouth Guard Creation</h3>

  <p align="center">
    This project takes a 3d scan of a person's teeth and generates a mouth guard with custom controls
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

The project is intended to generate a customizable mouth guard from an teeth stl scan

The code is divided into 2 modules: code to genera
1. Partition teeth and generate a set of usable coordinates (get_landmark_coordinates.py)
2. Use coordinates to generate mouth guard with blender (mg_generator.py)

### Progress

1. Process 3D scan and fixes position and orientation
2. Identify peaks (maxima points)
3. Apply the flood fill algorithm to capture teeth
4. Separate teeth into different blobs
5. Calculate the centroid of each blob
6. Generate edges in Blender according to centroids

### To Do

1. Identify correct tooth labeling for customization (left and right)
2. Give the mesh width and height controls
3. Carve out the teeth scan from the mouth guard

   
### Issues
1. Not versitile. The code should work on all teeth scans
2. Too slow. The code should be optimized for faster testing

<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

Make sure you have python and blender installed on your machine

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/fridge-png/AutomatedMouthGuardCreation.git
   ```
2. Install Python packages
   ```sh
   pip install -r requirements.txt
   ```
3. Running the code
   
   For generating the coordinates:
   ```python
   python get_landmark_coordinates.py
   ```
   For generating the mouth guard:
   ```blender
   blender -b -P mg_generator.py
   ```
   For visualizing the mouth guard:
   ```python
   python mg_visualize.py
   ```

