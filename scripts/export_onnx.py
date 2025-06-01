import sys
sys.path.append('.')

import argparse
import os
import numpy as np
import torch
from torch import cuda
from torchvision.transforms import ToTensor, Normalize
import onnx
import onnxsim
from onnx import shape_inference
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
        '--onnx_save_path', type=str, default=None, nargs='?',
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
        model.eval()
        # predicted_mask = model(image_tensor)

        # print(f'{predicted_mask.shape = }')
        # Export the model to ONNX format.
        print('Exporting the model to ONNX format...')
        torch.onnx.export(
            model, 
            image_tensor, 
            arguments.onnx_save_path, 
            export_params=True, 
            opset_version=11, 
            do_constant_folding=True, 
            input_names=['input'], 
            output_names=['output']
        )
        print(f'Model has been exported to {arguments.onnx_save_path}')

        # 3. 简化 ONNX 模型
        base_name, ext = os.path.splitext(arguments.onnx_save_path)
        simplified_onnx_path = f"{base_name}_sim{ext}"
        model = onnx.load(arguments.onnx_save_path)
        model_simplified, check = onnxsim.simplify(model)
        if check:
            onnx.save(model_simplified, simplified_onnx_path)
            print(f"Simplified ONNX model saved to {simplified_onnx_path}")
        else:
            print("Simplification failed!")

        # 4. 使用 ONNX Shape Inference 推导 Feature Map 的 Shape
        inferred_model = shape_inference.infer_shapes(onnx.load(simplified_onnx_path))
        inferred_onnx_path = f"{base_name}_shape_infer{ext}"
        onnx.save(inferred_model, inferred_onnx_path)
        print("Shape inference completed and model saved as 'restormer_inferred.onnx'")

        # 5. 打印推导出的中间层 Feature Map 的 Shape
        for node in inferred_model.graph.value_info:
            name = node.name
            shape = [
                dim.dim_value if dim.dim_value != 0 else "dynamic"
                for dim in node.type.tensor_type.shape.dim
            ]
            print(f"Feature Map Name: {name}, Shape: {shape}")


if __name__ == '__main__':
    main()




