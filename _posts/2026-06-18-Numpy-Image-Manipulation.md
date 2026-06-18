---
layout: post
title: Numpy Image Manipulation
image: "/posts/camaro.jpg"
tags: [Python, Numpy, Scikit, Matplotlib, Image Manipulation]
---

### This project is about using Python and Numpy to manipulate an image of a car by cropping, flipping, and separating color channels:
---

Add the necessary libraries. We will use Numpy for our array manipulation, Scikit-Image to read and save our picture, and Matplotlib to display the results.

```python
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
```

First, we'll load the image of the Camaro, print its matrix values to check the shape, and then show the original image using the "imshow" and "show" commands.

```python
camaro = io.imread("camaro.jpg")
print(camaro)

camaro.shape

plt.imshow(camaro)
plt.show()
```

![alt](/img/posts/camaro.jpg "Camaro")

Now, we'll crop the image. By slicing the array along the y-axis and x-axis, we can trim the picture down to just get the car itself, and then save the resulting image.

```python
# Example of croppring y-axis
cropped = camaro[0:500,:,:]
plt.imshow(cropped)
plt.show()

# Example of croppring x-axis
cropped = camaro[:,400:1000,:]
plt.imshow(cropped)
plt.show()

# To crop the image to just get the car
cropped = camaro[350:1100,200:1400,:]
plt.imshow(cropped)
plt.show()

io.imsave("camaro_cropped.jpg", cropped)
```

![alt](/img/posts/camaro_cropped.jpg "Cropped Camaro")

---
#### What if we wanted to flip the image upside down or mirror it...?

By using a negative step value in our array slicing, we can reverse the pixels to flip the image vertically and horizontally.

```python
# Flip our image

vertical_flip = camaro[::-1,:,:]
plt.imshow(vertical_flip)
plt.show()

io.imsave("camaro_vertically_flipped.jpg", vertical_flip)
```

![alt](/img/posts/camaro_vertically_flipped.jpg "Vertically Flipped Camaro")

```python
horizontal_flip = camaro[:,::-1,:]
plt.imshow(horizontal_flip)
plt.show()

io.imsave("camaro_horizontally_flipped.jpg", horizontal_flip)
```

![alt](/img/posts/camaro_horizontally_flipped.jpg "Horizontally Flipped Camaro")

---
#### Finally, let's separate the image into its core color channels!

We can isolate the Red, Green, and Blue channels by creating a blank array of zeros and copying over one specific color index at a time. Then, we'll stack them all together vertically to create a rainbow effect!

```python
# Color channels

red = np.zeros(camaro.shape, dtype = "uint8")
red[:,:,0] = camaro[:,:,0]
plt.imshow(red)
plt.show()

green = np.zeros(camaro.shape, dtype = "uint8")
green[:,:,1] = camaro[:,:,1]
plt.imshow(green)
plt.show()

blue = np.zeros(camaro.shape, dtype = "uint8")
blue[:,:,2] = camaro[:,:,2]
plt.imshow(blue)
plt.show()

camaro_rainbow = np.vstack((red,green,blue))
plt.imshow(camaro_rainbow)
plt.show()

io.imsave("camaro_rainbow.jpg", camaro_rainbow)
```

![Rainbow Camaro](/img/posts/camaro_rainbow.jpg "Rainbow Camaro")

---

And just like that, we've successfully manipulated our image using Numpy!
