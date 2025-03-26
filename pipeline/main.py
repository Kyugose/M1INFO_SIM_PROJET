import numpy as np

width = 1280  
height = 720

import time

start = time.time()


from graphicPipeline import GraphicPipeline


pipeline = GraphicPipeline(width,height)


from camera import Camera
from projection import Projection


position = np.array([1.1,1.1,1.1])
lookAt = np.array([-0.577,-0.577,-0.577])
up = np.array([0.33333333,  0.33333333, -0.66666667])
right = np.array([-0.57735027,  0.57735027,  0.])

cam = Camera(position, lookAt, up, right)

nearPlane = 0.1
farPlane = 10.0
fov = 1.91986
aspectRatio = width/height

proj = Projection(nearPlane ,farPlane,fov, aspectRatio) 


lightPosition = np.array([10,0,10])

from readply import readply

vertices, triangles = readply('suzanne.ply')


# load and show an image with Pillow
from PIL import Image
from numpy import asarray
# Open the image form working directory
image = asarray(Image.open('suzanne.png'))


data = dict([
  ('viewMatrix',cam.getMatrix()),
  ('projMatrix',proj.getMatrix()),
  ('cameraPosition',position),
  ('lightPosition',lightPosition),
  ('texture', image),
])

start = time.time()

pipeline.draw(vertices, triangles, data)

end = time.time()
print(end - start)

import matplotlib.pyplot as plt
imgplot = plt.imshow(pipeline.image)
plt.show()