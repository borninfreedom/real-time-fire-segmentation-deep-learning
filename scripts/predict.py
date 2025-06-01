import sys
sys.path.append('.')

import argparse
import os
import numpy as np
import torch
from torch import cuda
from torchvision.transforms import ToTensor, Normalize

from src.data.dataset_handler import load_image_from_zip_by_index
from src.model.model import FireSegmentationModel
from src.training.utils import Checkpoint
from src.prediction.analysis import plot_image_prediction,save_image_prediction


def main():
    # Set the argument parser.
    parser = argparse.ArgumentParser(
        description='Script for validating the results of the fire detection '
        'segmentation model.')

    # Set the script arguments.
    parser.add_argument(
        'image-number', metavar='Image number', type=int,
        help='The number of the image in the zip file for which segmentation '
        'is predicted.')

    parser.add_argument(
        '--images-zip-path', '-imgs', metavar='Images zip path', type=str,
        help='The path of the aerial images of the woodland fires zip file.',
        default=os.path.join('data', 'Images.zip'), nargs='?', required=False)

    parser.add_argument(
        '--checkpoint-file-path','-ckpt', metavar='Checkpoint file path',
        type=str, default=os.path.join('model', 'checkpoints.pth'),
        help='The path of the file where the model checkpoints are loaded.',
        nargs='?', required=False)

    parser.add_argument(
        '--train-mean-std-file-path','-ms', metavar='Mean and std file path',
        type=str, default=os.path.join('model', 'mean-std.npy'),
        help='The file path where the train mean and standard deviation are '
        'loaded', nargs='?', required=False)

    parser.add_argument(
        '--device', '-d', type=str, default=None, nargs='?',
        help='The device to use for training. If not provided, it is set '
            'automatically.', required=False)
    
    parser.add_argument(
        '--save_path', type=str, default=None, nargs='?',
        help='The device to use for training. If not provided, it is set '
            'automatically.', required=False)

    # Get the arguments.
    arguments = parser.parse_args()

    image_index = vars(arguments)['image-number']
    images_zip_path = arguments.images_zip_path
    chekpoint_file_path = arguments.checkpoint_file_path
    train_mean_std_file_path = arguments.train_mean_std_file_path
    device = arguments.device

    # Set the original shape.
    ORIGINAL_SHAPE = (3840, 2160)
    # Set the resize shape.
    RESIZE_SHAPE = (512, 512)
    # Set the device.
    if device is None:
        device = 'cuda' if cuda.is_available() else 'cpu'

    # Get the images and masks.
    print('Loading the image...')
    image = load_image_from_zip_by_index(
        images_zip_path, resize_shape=RESIZE_SHAPE, image_index=image_index)
    print(f'{type(image) = }, {image.shape = }')
    # Set the model.
    model = FireSegmentationModel(RESIZE_SHAPE, device=device)
    print(f'{model = }')
    # Load the best weights of the model.
    checkpoint = Checkpoint(chekpoint_file_path)
    checkpoint.load_best_weights(model)
    model.eval()

    # Load the mean and std of the training set for applying normalization.
    train_mean, train_std = np.load(train_mean_std_file_path)
    print(f'{train_mean = }, {train_std = }')
    
    # Transform the image
    to_tensor = ToTensor()
    normalize = Normalize(mean=train_mean, std=train_std)
    image_tensor = to_tensor(image)
    image_tensor = normalize(image_tensor)

    print('Starting prediction...')
    # Add the batch dimension.
    image_tensor = image_tensor.unsqueeze(0)
    print(f'{image_tensor.shape = }')
    # Move the image to the device.
    image_tensor = image_tensor.to(device)
    # Duplicate the image to make a batch of 2 images to handle
    # batch normalization statistics.
    image_tensor = torch.cat((image_tensor, image_tensor), dim=0)
    print(f'after torch.cat {image_tensor.shape = }')
    # Predict the mask.
    with torch.no_grad():
        predicted_mask = model(image_tensor)
        print(f'{predicted_mask.shape = }')
    # Remove the batch dimension.
    predicted_mask = predicted_mask[0]
    print(f'after 0 index {predicted_mask.shape = }')
    # Get the foreground mask.
    print(f'{predicted_mask.softmax(-3).shape = }, {predicted_mask.softmax(-3).argmax(-3).shape = }')
    predicted_mask = predicted_mask.softmax(-3).argmax(-3)
    
    # Move the mask to the cpu and convert it to numpy.
    predicted_mask = predicted_mask.cpu().numpy()
    print(f'finally {predicted_mask.shape = }')

    # Plot the image and the mask.
    print('Starting plotting...')
    save_image_prediction(image, predicted_mask, resize_shape=ORIGINAL_SHAPE,save_path=arguments.save_path)

if __name__ == '__main__':
    main()
