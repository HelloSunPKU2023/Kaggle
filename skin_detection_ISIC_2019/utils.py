import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import os
from PIL import Image

categories= ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
sizes =     [0.15,  0.25, 0.15,  0.10, 0.15,  0.05,  0.05,  0.10]

####### Helper functions to load meta data#######
def load_raw_data():
    # set data source
    data_url = '../raw_data/ISIC_2019_Training_GroundTruth.csv'
    data = pd.read_csv(data_url)
    # Added a Cat column for conviniences
    data['Cat']=''

    # Add Cat info of each row
    for c in categories:
        data.loc[data[c]>0, 'Cat']=c

    # Added a label/target column
    data['y']=0
    data['y']=(data.MEL+data.BCC+data.AK+data.SCC).astype(int)

    return data

def get_data_number(data, number):
    result = pd.DataFrame()
    for c in categories:
        result = pd.concat([result, data[data[c]>0].sample(number, random_state=42)], ignore_index=True)
    return result

def get_data_propotional(data, number):
    result = pd.DataFrame()
    numbers=np.multiply(sizes, number).astype(int)
    for i in range(8):
        result = pd.concat(
            [result, data[data[categories[i]]>0].sample(
                numbers[i], random_state=42, replace=True)], ignore_index=True)
    return result

def get_ground_truth_data(number=80):
    """
    when number is less than 800, return the 8 categories evenly.
    when the number is larger than 800 return the 8 categories as per the following percentages:
    NV=25%, BKL=15%, DF=5%, UASC=5%, MEL=15%, BCC=15%, AK=10%, SCC=10%
    if the number is negative, return the entire dataset
    """
    data = load_raw_data()

    if number<0 or number>data.shape[0]:
            result = data
    else:
        if (number<800):
                result =get_data_number(data, number=number//8)
        else:
                result = get_data_propotional(data, number=number)
    # randomize the data
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    return result

def load_processed_balanced_data():
    # set data source
    data_url = '../processed_data/ISIC_2019_Training_GroundTruth_processed_balanced.csv'
    data = pd.read_csv(data_url)
    return data

def load_processed_reduce_NV_data():
    # set data source
    data_url = '../processed_data_reduce_NV/ISIC_2019_Training_GroundTruth_processed_reduced_NV.csv'
    data = pd.read_csv(data_url)
    return data

####### Helper functions to pre-process images#######

def display_image(image: Image, ax, radius=-1):
    """
    Display an image with circle
    """
    # Display the image
    # Create an ImageDraw object
    ax.imshow(image)
    width, height=image.size
    # Plot the circle
    if radius>0:
        center=width//2
        # print(width, center, radius)
        ax.Circle((center, center), radius, color='red')

def display_images(*images):
    """
    Display multiple images
    """
    fig, axes = plt.subplots(1, len(images))

    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis('off')

    plt.show()

def display_image_with_circle(image, radius, ax):
    """
    Display an image with a circle
    """
# Get image dimensions and calculate the center point
    height, width = image.size
    center = (width // 2, height // 2)

    # Create a figure and axis object
    # fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    if radius>0:
        # Create a Circle object
        circle = Circle(center, radius, fill=False, color='#FF0000', linewidth=5)

        # Add the circle to the plot
        ax.add_patch(circle)

    # Set the aspect ratio
    ax.set_aspect('equal')

    # # Display the plot
    # plt.show()

def square_image(image: Image)->Image:
    """
    crop image to sqare
    """
    width, height = image.size
    # Determine the size of the square crop
    size = min(width, height)

    # Calculate the crop coordinates
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size

    # Perform the crop
    squared_image = image.crop((left, top, right, bottom))

    return squared_image

def detect_black_coners(image: Image)->int:
    """
    return the radius of the circle if detected black coner, otherwise, return -1
    This function needs to be improved for better accuracy if needed
    Args:
        image (Image): The image object which may have black corners.

    Returns:
        int: the radius of circle where is the the black corners from
    """
    width, height=image.size
    scope=min(width, height)//4 # the minimum radius to search for
    step=3  # for better searching performance
    margin=0.98

    if width!=height:
        image = square_image(image)

    b_w_image = image.convert('L')
    for i in range(1, scope, step):
        left=i;right=width-i; top=i; bottom=height-i
        color1=b_w_image.getpixel((top, left))
        color2=b_w_image.getpixel((top, right))
        color3=b_w_image.getpixel((bottom, left))
        color4=b_w_image.getpixel((bottom, right))

        if color1+color2+color3+color4>300:
            break
    if i>1:
        radius = int((width//2-i)*np.sqrt(2)*margin)
        # print(f'black corner detected, the radius is {radius}')
        return radius

    # print(f'No black corner detected')
    return -1

def remove_black_corners(image, radius) -> Image:
    """
    This function remove the black corners of an image

    Args:
        image: The image object to which has black corners.
        radius: the readius of the black corners

    Returns:
        image: image object which the black corners have been removed.
    """
    width, height=image.size

    new_width=int(radius*np.sqrt(2))
    x = (width-new_width)//2
    crop_area=(x, x, width-x, width-x)
    image = image.crop(crop_area)

    return image
