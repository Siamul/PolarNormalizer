from polar_normalization import PolarNormalization
import argparse
import torch
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

def visualize_images(args):
    if not os.path.exists(args.vis_dir):
        os.mkdir(args.vis_dir)
    if not os.path.exists(args.mask_dir):
        os.mkdir(args.mask_dir)
    if not os.path.exists(args.insideeyelid_vis_dir):
        os.mkdir(args.insideeyelid_vis_dir)
    if not os.path.exists(args.insideeyelid_mask_dir):
        os.mkdir(args.insideeyelid_mask_dir)
    polar_normalizer = PolarNormalization(mask_net_path = args.mask_net_path, circle_net_path = args.circle_net_path, eyelid_net_path = args.eyelid_net_path, device = torch.device('cuda') if args.cuda else torch.device('cpu'))
    for imagename in tqdm(os.listdir(args.image_dir)):
        if imagename.endswith(('jpg', 'jpeg', 'bmp', 'png', 'gif', 'pgm', 'tiff')):
            imagepath = os.path.join(args.image_dir, imagename)
            
            image_pil = Image.open(imagepath).convert('L')
            pxyr, ixyr = polar_normalizer.circApprox(image_pil)
            
            mask = polar_normalizer.getIrisMask(image_pil)  # gives a numpy mask
            mask_pil = Image.fromarray(mask.astype(np.uint8), 'L')
            mask_pil.save(os.path.join(args.mask_dir, imagename.split('.')[0] + '.png'))

            image_irismask_vis_pil = polar_normalizer.visualize_image(image_pil, mask, pxyr, ixyr)
            image_irismask_vis_pil.save(os.path.join(args.vis_dir, imagename.split('.')[0] + '.png'))

            insideeyelid_mask = polar_normalizer.getInsideEyelidMask(image_pil)
            insideeyelid_mask_pil = Image.fromarray(insideeyelid_mask.astype(np.uint8), 'L')
            insideeyelid_mask_pil.save(os.path.join(args.insideeyelid_mask_dir, imagename.split('.')[0] + '.png'))

            image_insideeyelidmask_vis_pil = polar_normalizer.visualize_image(image_pil, insideeyelid_mask, pxyr, ixyr)
            image_insideeyelidmask_vis_pil.save(os.path.join(args.insideeyelid_vis_dir, imagename.split('.')[0] + '.png'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg.yaml",
                        help="path of the configuration file")
    parser.add_argument('--mask_net_path', default='./nestedsharedatrousresunet-135-0.026591-maskIoU-0.942362.pth')
    parser.add_argument('--circle_net_path', default='./resnet34-1583-0.045002-maskIoU-0.93717.pth')
    parser.add_argument('--eyelid_net_path', default='./nestedsharedatrousresunetswishgn-256-0.024513-maskIoU-0.968207-eyelid.pth')
    parser.add_argument('--image_dir', default='./test_gray_images')
    parser.add_argument('--vis_dir', default='./vis_dir')
    parser.add_argument('--insideeyelid_vis_dir', default='./insideyelid_vis_dir')
    parser.add_argument('--mask_dir', default='./mask_dir')
    parser.add_argument('--insideeyelid_mask_dir', default='./insideeyelid_mask_dir')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    visualize_images(args)