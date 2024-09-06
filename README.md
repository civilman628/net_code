# 1. Image Resize tool

## Introduction

This image resize tool can help you to do data preparation for model validation. You can resize the validation images into the size, which is same as the model's input size.

You should keep the image ratio to avoid image distortion.
If you do not want to keep ratio and resize the image into an arbitrary size, please set **keep_ratio** to **False**.

Keep ratio means the target output image does not change the source image ratio and is in a square with zero padding, if the width and height of the source image are not equal. 

Most of the popular models use the sizes like 224x224, 299x299 and 416x416.


## Setup

You need to install Pillow package by run this command.

```
pip install Pillow
```


## Parameters

- **width** resize into this width
- **height** resize into this height
- **input_path** a folder for input image files
- **output_path** a folder for output resized image
- **keep_ratio** keep the ratio of the image. if set to false, it may impact model accuracy for classification or detection task



## Run with sample images.

The input and output folder are at the same level as **image_resize.py**. There are 3 sample source images in the input folder.

```
python3 image_resize.py \
    --input_path ./input_folder \
    --output_path ./output_folder
```


## Check output images 

You will see the resized output images in the **output_folder**. All the images are at 416x416 target resolution. 

----

# 2. Image to tensor tool

## Introduction

This tool can help you to convert an image to a tensor that fits the model's input. This tool relies on the above image resize tool. The size depends on the **shape** section in the **inputmeta.yml** file.


## Setup

You need to install pyyaml package by run this command.

```
pip install pyyaml
```


## Parameters

- **input_path** a folder for input image files
- **output_path** a folder for tensor files
- **keep_ratio** keep the ratio of the image. if set to false, it may impact model accuracy for classification or detection task
- **input_meta** a yml file defines the mean, scale, layout, shape and channel reverse information for the output tensor.


## Run with sample images.

The input and output folder are at the same level as **image2tensor.py**. There are 3 sample source images in the input folder.

```
python3 image2tensor.py \
    --input_path ./input_folder \
    --output_path ./output_folder
```


## Check output tensors

You will see the tensor files in the **output_folder**. 