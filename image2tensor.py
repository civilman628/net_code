from PIL import Image
import os
import argparse
import numpy as np
import glob
import yaml
from image_resize import resize

def image2tensor(image, input_meta, keep_ratio):

    '''convert an image to a transposed tensor

    Args:
        image (array):      image array read from the file
        input_meta (dict):  a dictionary for mean, scale, layout and reverse_channel parameter
        keep_ratio (bool):  Whether or not to keep the ratio of image.

    Returns:
        tensor (array):     a tansposed image tensor matches the network input
    '''
    #the default image layout is NHWC
    layout_index = {"n": 0, "h": 1, "w": 2, "c": 3}

    mean = input_meta["input_meta"]["databases"][0]["ports"][0]["preprocess"]["mean"]
    mean = np.array(mean,np.float32)
    
    scale = input_meta["input_meta"]["databases"][0]["ports"][0]["preprocess"]["scale"]
    scale = np.array(scale,np.float32)

    layout = input_meta["input_meta"]["databases"][0]["ports"][0]["layout"]
    layout = list(layout)

    #a bool value for RBG to GBR conversion
    reverse_channel = input_meta["input_meta"]["databases"][0]["ports"][0]["preprocess"]["reverse_channel"]

    shape = input_meta["input_meta"]["databases"][0]["ports"][0]["shape"]

    shape = dict(zip(layout,shape))


    #-------------step 1 resize image -----------------------

    #get the target W and H from inputmeta.yml file
    width = shape['w']
    height = shape['h']

    image = resize(image,width, height,keep_ratio)

    
    #-------------step 2 normalize image ---------------------

    #normalize image with mean and scale
    image = np.array(image,np.float32)
    image = (image - mean) * scale


    #-------------step 3 transpose image  ---------------------
    #expand image to a 4D tensor
    image = np.expand_dims(image, axis=0)

    #get the new layout and tranpose the image to the target layout, like NCHW
    new_layout = [layout_index[layout[0]], layout_index[layout[1]], layout_index[layout[2]], layout_index[layout[3]]]
    image = image.transpose(new_layout)

    #-------------step 4 reverse channel----------------------

    #revese the channel like from RBG to BGR
    if reverse_channel is True:
        index = layout.index('c')
        image = np.flip(image,index)

    return image




def main(config):
    
    #check input and output dir, yaml file
    if not os.path.isdir(config.input_path):
        raise ValueError("Invalid intput dir `"+os.path.abspath(config.input_path)+"`")

    if not os.path.isdir(config.output_path):
        raise ValueError("Invalid output dir `"+os.path.abspath(config.output_path)+"`")

    if not os.path.isfile(config.input_meta):
        raise ValueError("Invalid input meta yaml file `"+os.path.abspath(config.input_meta)+"`")


    with open(config.input_meta,'r') as f:
        input_meta = yaml.load(f)

    #get all supported image files from input folder
    filelist = glob.glob(config.input_path+'/*[.png, .jpg, .bmp, .tif]')
    
    #loop all images in the folder
    for image in filelist:
        print(image)
        try:
            img = Image.open(image)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            tensor = image2tensor(img, input_meta, config.keep_ratio)

            #get file name without extension
            file_name =  os.path.splitext(os.path.basename(image))[0]

            #generate file name for tenor
            path = os.path.join(config.output_path,file_name + '.tensor')
            tensor.tofile(path,'\n')

            print("image tensor is saved at: "+ path +'\n')

        except Exception as e:               
            print("image: " + image + ". has exception: " + str(e))
            exit()

    print("==========DONE==========")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_meta', type=str, default='./yolov4-tiny-416-ca-head_inputmeta.yml', help='a ymal file for preprocessing')
    parser.add_argument('--input_path', type=str, default='./input_folder', help='a folder for input image file')
    parser.add_argument('--output_path', type=str, default='./output_folder', help='a folder for output resized image tensor files')
    parser.add_argument('--keep_ratio', type=bool, default=True, help='keep the ratio of the image. if set to false, it may impact model accuracy for classification or detection task')

    config = parser.parse_args()

    print(config)

    main(config)