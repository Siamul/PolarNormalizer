This repository contains the code to polar-normalize and find coarse masks for iris images.

To visualize the results for a folder of iris images run:

> python visualize_images.py --image_dir <path_to_directory_containing_iris_images> --vis_dir <path_for_saving_visualizations>

To create a CPU-only pytorch conda environment, run:

    conda env create -f environment.yml
    
  Or, install requirements using pip:
    
    pip install -r requirements.txt

Read the visualize_images.py file to understand how to use the PolarNormalization object.

Declare the object as:

```
import torch
from polar_normalization import PolarNormalization
polar_normalizer = PolarNormalization(mask_net_path = <mask_net_path>, circle_net_path = <circle_net_path>, eyelid_net_path = <eyelid_net_path>, device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
```

Get pupil and iris circle parameters as:
```
pxyr, ixyr = polar_normalizer.circApprox(<iris image loaded using PIL>)
```
where (pxyr[0], pxyr[1]) is the pupil circle center and pxyr[2] is the pupil radius, and (ixyr[0], ixyr[1]) is the iris circle center and ixyr[2] is the iris radius.

Get iris mask as:
```
iris_mask = polar_normalizer.getIrisMask(<iris image loaded using PIL>)
```
The returned mask will be a numpy array. 


Get iris+sclera mask using:
```
insideeyelid_mask = polar_normalizer.getInsideEyelidMask(<iris image loaded using PIL>)
```
The returned mask here will be a numpy array as well.
