from polar_normalization import PolarNormalization
import argparse
import torch
import os
from PIL import Image
from tqdm import tqdm

def visualize_images(args):
    if not os.path.exists(args.vis_dir):
        os.mkdir(args.vis_dir)
    polar_normalizer = PolarNormalization(mask_net_path = args.mask_net_path, circle_net_path = args.circle_net_path, device = torch.device('cpu'))
    for imagename in tqdm(os.listdir(args.image_dir)):
        if imagename.endswith(('jpg', 'jpeg', 'bmp', 'png', 'gif', 'pgm')):
            imagepath = os.path.join(args.image_dir, imagename)
            image_pil = Image.open(imagepath).convert('L')
            image_vis_pil = polar_normalizer.visualize_image(image_pil)
            image_vis_pil.save(os.path.join(args.vis_dir, imagename.split('.')[0] + '.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg.yaml",
                        help="path of the configuration file")
    parser.add_argument('--mask_net_path', default='./nestedsharedatrousresunet-217-0.027828-maskIoU-0.938739.pth')
    parser.add_argument('--circle_net_path', default='./resnet34-1583-0.045002-maskIoU-0.93717.pth')
    parser.add_argument('--image_dir', default='./test_gray_images')
    parser.add_argument('--vis_dir', default='./vis_dir')
    args = parser.parse_args()
    visualize_images(args)