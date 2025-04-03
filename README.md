# DrosoVAM (Drosophila Video-based Activity Monitor)

DrosoVAM is a Python-based software suite designed to monitor and analyze Drosophila behavior using infrared video recordings. It provides a comprehensive workflow from recording to data analysis, enabling researchers to efficiently study fly activity.

## Purpose

This software aims to automate the process of monitoring Drosophila behavior, providing tools for recording, video clipping, and data analysis derived from DeepLabCut tracking.

## Key Features

* **Recording (pylon\_looper\_Vx.py):**
    * Captures infrared video recordings using a Basler camera.
    * Allows scheduled recording with specified start times and experiment names.
    * Automatically creates directories for video storage.
* **Video Clipping (Crop\_GUI\_Vx.py):**
    * Graphical user interface for defining regions of interest (ROIs) for video clipping.
    * User-defined grid layout for multi-chamber experiments.
    * Precise clipping based on user-selected corner points.
    * Creates a configuration file for the later analysis.
* **Data Analysis (postDLC\_total\_Vx.py):**
    * Analyzes DeepLabCut output data (.csv files).
    * Uses the configuration file created by Crop_GUI_Vx.py.
    * Generates CSV files and plots for various behavioral metrics.

## Installation and Usage

1.  **Recording (pylon_looper_Vx.py):**
    * Ensure a Basler camera is connected to the computer.
    * Install the necessary Basler pylon software.
    * Run `pylon_looper_Vx.py` from the command line or an IDE.
    * Follow the prompts to set the start time and experiment name.

2.  **Video Clipping (Crop_GUI_Vx.py):**
    * Install ffmpeg.
    * Open `Crop_GUI_Vx.py` using Spyder or another Python IDE.
    * Define the path to ffmpeg in the program.
    * Run the program.
    * Specify the number of rows and columns for your experimental setup.
    * Provide the path to the video dataset.
    * Define the top-left and bottom-right corners of the chambers.
    * Leave margins to avoid clipping inside the chambers.
    * Verify the clipping preview and initiate the clipping process.

3. **Analyze the clipped videos using DeepLabCut.**

4.  **Data Analysis (postDLC_total_Vx.py):**
    * Ensure DeepLabCut output .csv files are available.
    * Modify the generated configuration file (from Crop_GUI_Vx.py) to set analysis parameters.
    * Run `postDLC_total_Vx.py` using Spyder or another Python IDE.

## Dependencies

* Python
* ffmpeg
* DeepLabCut

## Examples

Examples of the software's output and usage can be found in the DrosoVAM paper.
