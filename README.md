This repository contains the code to polar-normalize and find coarse masks for iris images.

To visualize the results for a folder of iris images run:

> python visualize_images.py --image_dir <path_to_directory_containing_iris_images> --vis_dir <path_for_saving_visualizations>

To create a CPU-only pytorch conda environment, run:

> conda env create -f environment.yml

Or, install requirements using pip:

> pip install -r requirements.txt
