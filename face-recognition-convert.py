#!/usr/bin/env python
# coding: utf-8

# ## Convert pre-trained weights from CSV files to binary format

# **Important:** before running this notebook, copy the [pre-trained weight CSV files](https://github.com/iwantooxxoox/Keras-OpenFace/tree/master/weights) from the Keras-OpenFace project to the local `weights` directory.

# In[ ]:


from model import create_model
from utils import load_weights


# Instantiate the [nn4.small2](http://cmusatyalab.github.io/openface/models-and-accuracies/#model-definitions) model of the [OpenFace](https://cmusatyalab.github.io/openface/) project as Keras model.

# In[ ]:


nn4_small2 = create_model()


# Load the pre-trained model weights from CSV files.

# In[ ]:


nn4_small2_weights = load_weights()


# Update the Keras model with the loaded weights and save the weights in binary format.

# In[ ]:


for name, w in nn4_small2_weights.items():
    if nn4_small2.get_layer(name) != None:
        nn4_small2.get_layer(name).set_weights(w)

nn4_small2.save_weights('weights/nn4.small2.v1.h5')

