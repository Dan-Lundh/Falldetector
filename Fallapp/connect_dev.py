# fixing all packages needed
import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import re
import json
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
import numpy as np
from keras._tf_keras.keras.models import load_model
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tensorflow as tf
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import streamlit.components.v1 as components
from streamlit.components.v1 import html
from streamlit_modal import Modal

#modal=Modal("Modal",key='modal')

# setting upp connection to phone to obtain data e.g. accelorameter (PP_CHANNELS)

# web adress http://192.168.0.103

# adress to client (phone)
#_client_adress="http://192.168.68.100"
#_client_adress="http://192.168.1.101"

#url=_client_adress 


modal = Modal(
    "Demo Modal", 
    key="demo-modal",
    
    # Optional
    padding=20,    # default value
    max_width=744  # default value
)



#st.button('test', on_click=open_page, args=('https://www.mozilla.org/',))

# Streamlit app title
st.title("Fall detector throu deep learning MLP")

# Define global variables
buffer_size = 50  # Number of data points to display
xarr, yarr, zarr = np.zeros(buffer_size), np.zeros(buffer_size), np.zeros(buffer_size)

placeholder2 = st.empty()


looper=1

# Placeholder for live plot
plot_placeholder = st.empty()

plot_placehold3 = st.empty()
# Load MLP model
@st.cache_resource
def load_fall_model():
    return load_model("D:/source/Falldetector/Fallapp/model.keras")

model = load_fall_model() 

# model = load_model("D:/source/Falldetector/Fallapp/model.keras")

# Client URL
_client_address = "http://192.168.0.104"  # Replace with actual URL
PP_CHANNELS = ["accX", "accY", "accZ"]  # Example channels




# Draw a filled square

def signal(color='green'):
    fig, ax = plt.subplots(figsize=(2, 0.1))
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    return fig
    # Display in Streamlit
    # st.pyplot(fig)

def fig_to_array(fig):
    fig.canvas.draw()  # Render the figure
    img = np.array(fig.canvas.renderer.buffer_rgba())  # Convert to NumPy array
    image_rgb = img[:, :, :3]
    image_tensor = np.expand_dims(image_rgb, axis=0)
    image_rgb = tf.image.resize(image, (100, 100))

    image_rgb = tf.ensure_shape(image_rgb, (100, 100, 3))
    image_tensor = image_tensor.astype("float32") / 255.0
    return image_tensor  # Shape: (H, W, 4) → RGBA format

def imageproc(image):
    gray_image = rgb2gray(image)

    # Apply thresholding to extract lines (black lines on white background)
    thresh = threshold_otsu(gray_image)
    binary_image = gray_image < thresh  # Converts to black & white (inverted)

    # Plot the processed image
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(binary_image, cmap="binary", interpolation="nearest")  # Black lines on white

    # Remove axis, ticks, and legend
    ax.axis("off")  
    return fig

# signal()

barcolor='green'
while True:
    url = f"{_client_address}/get?" + "&".join(PP_CHANNELS)
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = json.loads(response.content.decode('utf-8'))
            
            x = data["buffer"]['accX']['buffer'][0]
            y = data["buffer"]['accY']['buffer'][0]
            z = data["buffer"]['accZ']['buffer'][0]

            # Append new values and maintain buffer size
            xarr = np.append(xarr, x)[1:]
            yarr = np.append(yarr, y)[1:]
            zarr = np.append(zarr, z)[1:]

            # Plot the updated data
            fig, ax = plt.subplots()
            ax.plot(xarr, label="Acc X", color="red")
            ax.plot(yarr, label="Acc Y", color="green")
            ax.plot(zarr, label="Acc Z", color="blue")

            ax.set_title("Live Accelerometer Data")
            ax.set_xlabel("Time")
            ax.set_ylabel("Acceleration")
            ax.set_ylim(-13,13)
            ax.legend()

            # Update Streamlit plot
            plot_placeholder.pyplot(fig)
            
            # Using get_renderer() for compatibility with newer Matplotlib versions
            canvas = FigureCanvas(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            image = np.array(renderer.buffer_rgba())[:, :, :3]
            
            #image = np.array(renderer.buffer_rgba())[:, :, :3]

            # image_b = imageproc(image)

            image_rgb = tf.image.resize(image, (100, 100))

            image_rgb = tf.ensure_shape(image_rgb, (100, 100, 3))

            # Convert uint8 to float32 and normalize
            image_rgb = tf.cast(image_rgb, tf.float32) / 255.0
            
            ################################
            #figarray=fig_to_array(fig)
            # figarray = imageproc(image_rgb)


            ################################
            
            # Add batch dimension if required by model
            image_rgb = tf.expand_dims(image_rgb, axis=0)  # Shape: (1, 100, 100, 3)


            fig2, ax2 = plt.subplots(figsize=(2, 0.1))
            ax2.add_patch(plt.Rectangle((0, 0), 1, 1, color= barcolor))
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_frame_on(False)

            y_pred = model.predict(image_rgb)
            prediction_score = y_pred[0, 1] * 100
            #value = st.slider("Select a value", 0, 100, 50)
            barcolor = "green" if prediction_score < 90  else "red"

            print(y_pred[0,0],'   ', y_pred[0, 1]*100)
            # if "popup_shown" not in st.session_state:
            #     st.session_state["popup_shown"] = False

            # if prediction_score > 90:

                  
            #     looper=0



            # Update the figure
            #fig2 = signal(color)
            placeholder2.pyplot(fig2)

          
            

            # plot_placehold3.pyplot(imageproc(fig))
            # Set color based on value
            

            #st.markdown("X_batch shape before resizing:", image_rgb.shape)
            plt.close(fig)
            plt.close(fig2)



    except Exception as e:
        st.error(f"Error fetching data: {e}")

    time.sleep(0.1)  # Adjust update interval as needed
    



# Create a figure and axis


# Remove axes
