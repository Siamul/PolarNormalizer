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
    polar_normalizer = PolarNormalization(mask_net_path = args.mask_net_path, circle_net_path = args.circle_net_path, device = torch.device('cuda') if args.cuda else torch.device('cpu'))
    for imagename in tqdm(os.listdir(args.image_dir)):
        if imagename.endswith(('jpg', 'jpeg', 'bmp', 'png', 'gif', 'pgm', 'tiff')):
            imagepath = os.path.join(args.image_dir, imagename)
            image_pil = Image.open(imagepath).convert('L')
            image_vis_pil = polar_normalizer.visualize_image(image_pil)
            image_vis_pil.save(os.path.join(args.vis_dir, imagename.split('.')[0] + '.png'))
            image_mask = polar_normalizer.getMask(image_pil)  # gives a numpy mask
            image_mask_pil = Image.fromarray(image_mask.astype(np.uint8), 'L')
            image_mask_pil.save(os.path.join(args.mask_dir, imagename.split('.')[0] + '.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg.yaml",
                        help="path of the configuration file")
    parser.add_argument('--mask_net_path', default='./nestedsharedatrousresunet-135-0.026591-maskIoU-0.942362.pth')
    parser.add_argument('--circle_net_path', default='./resnet34-1583-0.045002-maskIoU-0.93717.pth')
    parser.add_argument('--image_dir', default='./test_gray_images')
    parser.add_argument('--vis_dir', default='./vis_dir')
    parser.add_argument('--mask_dir', default='./mask_dir')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    visualize_images(args)