from PIL import Image
import os
import argparse
import glob


def resize(image, width, height, keep_ratio):

    '''resize the image into the target size

    We suggest to keep the image ratio to avoid image distortion.
    If you do not want to keep ratio and resize the image into an arbitrary size, please set keep_ratio to False.

    Keep ratio means the target output image does not change the ratio and in a square with zero padding, if width and height of 
    the source image are not equal.
    Most of the popular models use the sizes like 224*224, 299*299 and 416*416

    Args:
        image (array):      image array.
        width (int):        The width of the new image.
        height (int):       The height of the new image.
        keep_ratio (bool):  Whether or not to keep the ratio of image.

    Returns:
        resized_image (array):resized image in the target size.
    '''
    if keep_ratio is False:

        print("WARNING: change image ratio could impact the accuracy of classification and detection")
        resized_image = image.resize((width, height),Image.BICUBIC)
        
    else:

        if width != height:
            print("WARNING: if the target width and height are not equal, please set keep_ratio to False")
            exit()

        # create a square background
        longersize = max(image.size)
        background = Image.new('RGB', (longersize, longersize), "black")

        #paste the image to this square background
        background.paste(image, (int((longersize-image.size[0])/2), int((longersize-image.size[1])/2)))
        image = background

        resized_image = (image.resize((width, height), Image.BICUBIC))

    return resized_image

def main(config):
    
    #check input and output dir
    if not os.path.isdir(config.input_path):
        raise ValueError("Invalid intput dir `"+os.path.abspath(config.input_path)+"`")

    if not os.path.isdir(config.output_path):
        raise ValueError("Invalid output dir `"+os.path.abspath(config.output_path)+"`")

    #get all supported image files from input folder
    filelist = glob.glob(config.input_path+'/*[.png, .jpg, .bmp, .tif]')
    
    #loop all images in the folder
    for image in filelist:
        print(image)
        try:
            img = Image.open(image)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img = resize(img, config.width, config.height, config.keep_ratio)

            #get file name without extension
            file_name =  os.path.splitext(os.path.basename(image))[0]

            #generate new path name and output format is .jpg
            path = os.path.join(config.output_path,file_name + '.jpg')

            #save image with quality at 95
            img.save(path,quality=95)
            print("new resized image save: "+ path +'\n')

        except Exception as e:               
            print("image: " + image + ". has exception: " + str(e))
            exit()

    print("==========DONE==========")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--width', type=int, default=416, help='resize into this width')
    parser.add_argument('--height', type=int, default=416, help='resize into this height')
    parser.add_argument('--input_path', type=str, default='./input_folder', help='a folder for input image file')
    parser.add_argument('--output_path', type=str, default='./output_folder', help='a folder for output resized image')
    parser.add_argument('--keep_ratio', type=bool, default=True, help='keep the ratio of the image. if set to false, it may impact model accuracy for classification or detection task')

    config = parser.parse_args()

    print(config)

    main(config)