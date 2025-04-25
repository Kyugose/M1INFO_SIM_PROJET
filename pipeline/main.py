import sys
import numpy as np
import time
from graphicPipeline import GraphicPipeline
from camera import Camera
from projection import Projection
import matplotlib.pyplot as plt
from readply import readply
from PIL import Image
from numpy import asarray

# Définir les dimensions de l'écran
width = 1280  
height = 720

# Vérifier les arguments de ligne de commande
if len(sys.argv) > 1 and sys.argv[1].lower() == "triangle":
    render_triangle = True
else:
    render_triangle = False

# Initialisation de la caméra et de la projection
position = np.array([1.1,1.1,1.1])
lookAt = np.array([-0.577,-0.577,-0.577])
up = np.array([0.33333333,  0.33333333, -0.66666667])
right = np.array([-0.57735027,  0.57735027,  0.])

cam = Camera(position, lookAt, up, right)

nearPlane = 0.1
farPlane = 10.0
fov = 1.91986
aspectRatio = width / height

proj = Projection(nearPlane, farPlane, fov, aspectRatio)

lightPosition = np.array([10, 0, 10])

image = asarray(Image.open('../resources/suzanne.png'))

data = dict([
  ('viewMatrix', cam.getMatrix()),
  ('projMatrix', proj.getMatrix()),
  ('cameraPosition', position),
  ('lightPosition', lightPosition),
  ('texture', image),
])

if render_triangle:
    # Test MSAA sur un triangle simple
    triangle_vertices = np.array([
        [-5, -5, 0.5, 0, 0, 1, 0, 0, 0, 0, 0], 
        [0.9, -5, 0.5, 0, 0, 1, 0, 0, 0, 1, 0],  
        [0.0, 0.9, 0.5, 0, 0, 1, 0, 0, 0, 0.5, 1]  
    ])
    triangle_indices = np.array([
        [0, 1, 2]
    ])

    width = 320
    height = 180
    pipeline = GraphicPipeline(width, height)

    # Rendu du triangle avec les dimensions définies
    pipeline.draw(triangle_vertices, triangle_indices, data, False, False)
    triangle_no_msaa = pipeline.image.copy()

    pipeline.draw(triangle_vertices, triangle_indices, data, True, False)
    triangle_msaa4x = pipeline.image.copy()

    pipeline.draw(triangle_vertices, triangle_indices, data, False, True)
    triangle_msaa8x = pipeline.image.copy()

    # Affichage des résultats pour le triangle
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(triangle_no_msaa)
    axes[0].set_title("Triangle - Sans MSAA")
    axes[0].axis('off')

    axes[1].imshow(triangle_msaa4x)
    axes[1].set_title("Triangle - Avec MSAA x4")
    axes[1].axis('off')

    axes[2].imshow(triangle_msaa8x)
    axes[2].set_title("Triangle - Avec MSAA x8")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

else:
    vertices, triangles = readply('../resources/suzanne.ply')

    pipeline = GraphicPipeline(width, height)

    # Rendu sans MSAA
    start_no_msaa = time.time()
    pipeline.draw(vertices, triangles, data, False, False)  # MSAA désactivé
    end_no_msaa = time.time()
    print(f"Temps de rendu sans MSAA : {end_no_msaa - start_no_msaa:.4f} secondes")
    image_no_msaa = pipeline.image.copy()

    # Rendu avec MSAA
    start_msaa = time.time()
    pipeline.draw(vertices, triangles, data, True, False)  # MSAA activé
    end_msaa = time.time()
    print(f"Temps de rendu avec MSAA x4: {end_msaa - start_msaa:.4f} secondes")
    image_msaa = pipeline.image.copy()

    # Rendu avec MSAA x8
    start_msaa8x = time.time()
    pipeline.draw(vertices, triangles, data, False, True)  # MSAA x8 activé
    end_msaa8x = time.time()
    print(f"Temps de rendu avec MSAA x8 : {end_msaa8x - start_msaa8x:.4f} secondes")
    image_msaa8x = pipeline.image.copy()

    # Affichage des résultats
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image_no_msaa)
    axes[0].set_title("Sans MSAA")
    axes[0].axis('off')

    axes[1].imshow(image_msaa)
    axes[1].set_title("Avec MSAA x4")
    axes[1].axis('off')

    axes[2].imshow(image_msaa8x)
    axes[2].set_title("Avec MSAA x8")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()