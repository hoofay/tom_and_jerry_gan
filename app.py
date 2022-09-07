

# author: daniel hough
# date: sept 2022


# libraries
import streamlit as st
from PIL import Image
import os
import numpy as np
import tensorflow as tf

# intro
st.title("Tom and Jerry GAN")
st.write("This app allows you to use a pre-trained GAN model. A GAN is a generative adverserial network that learns a mapping from some input toa chosen output - in this case, with images. ",
         "It can be applied to a wide range of tasks including synthesising photos from labels, generating coloured photos from black and white images, or even turning sketches into photos.",
         "In this application we have trained our GAN to translate photos into cartoon images. The model has been trained using 20,000 stills from Tom and Jerry cartoons alongside the Flickr30k dataset.")

# file uploader
upload_pic = st.file_uploader("Select your picture file (png,jpg)",
                              type = ['png','jpg'])

# load sample image
input_image = Image.open('ski.jpg')

# choose uploaded image if it exists
def image_choice(upload_pic):
    if upload_pic is not None:
        input_image = Image.open(upload_pic)
        return(input_image)
    else:
        input_image = Image.open('ski.jpg')
        return(input_image)

input_image = image_choice(upload_pic)

# display input image
st.image(input_image, caption = 'Your Photo')

# define scoring function
def main(m_path, img_path):
    imported = tf.saved_model.load(m_path)
    f = imported.signatures['serving_default']
    img = np.array(Image.open(img_path).convert('RGB'))
    img = np.expand_dims(img,0).astype(np.float32) / 127.5 - 1
    out = f(tf.constant(img))['output_1']
    out = ((out.numpy().squeeze() + 1) * 127.5).astype(np.uint8)
    return out

# scoring image choice
def scoring_image(upload_pic):
    if upload_pic is not None:
        scoring_image = upload_pic
        return(scoring_image)
    else:
        scoring_image = 'ski.jpg'
        return(scoring_image)

img_out = scoring_image(upload_pic)    

# run score
output = main('tajg/tj_gan',img_out) # update to chosen image
img = Image.fromarray(output,'RGB')

# display output image
st.write('Your converted image is displayed below:')
st.image(img, caption = 'Your Tom and Jerry picture')

# outro
st.write('Author: Daniel Hough, 2022')
