import numpy as np
from numpy import log, arctan2, pi, mean
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
import meshio
import matplotlib as mpl

# Configuración de LaTeX para matplotlib
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "xelatex",        # change this if using xetex or lautex
    "text.usetex": False,                # use LaTeX to write all text
    "font.family": "sans-serif",
    # "font.serif": [],
    "font.sans-serif": ["DejaVu Sans"], # specify the sans-serif font
    "font.monospace": [],
    "axes.labelsize": 8,               # LaTeX default is 10pt font.
    "font.size": 0,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": (3.00, 2.00),     # default fig size of 0.9 textwidth
    "pgf.preamble": r'\usepackage{amsmath},\usepackage{amsthm},\usepackage{amssymb},\usepackage{mathspec},\renewcommand{\familydefault}{\sfdefault},\usepackage[italic]{mathastext}'
    }
mpl.rcParams.update(pgf_with_latex)


def read_plot_mesh(file_path):
    """
    Lee un archivo de malla, extrae los puntos y triángulos, y grafica la malla.

    Parameters:
    file_path (str): La ruta del archivo de malla a leer.

    Returns:
    tuple: Una tupla que contiene:
        - pts (numpy.ndarray): Un array de puntos de la malla.
        - tris_planet (numpy.ndarray): Un array de triángulos de la malla.
        - tris (numpy.ndarray): Un array de triángulos de la malla.
    """
    # Leer el archivo de malla
    mesh = meshio.read(file_path)
    
    # Extraer los puntos y triángulos
    pts = mesh.points
    tris_planet = mesh.cells[0].data
    tris = np.vstack([cells.data for cells in mesh.cells])
    
    # Extraer las coordenadas x e y
    x, y, _ = pts.T
    
    # Graficar la malla
    plt.figure()
    plt.triplot(x, y, tris, lw=0.2)
    plt.axis("image")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis("off")
    plt.savefig('figs/malla.pdf')
    plt.show()

    return pts, tris_planet,tris

def green_pot_2d(r):
    """Green function for Laplace equation
    
    Parameters
    ----------
    r : float
        Distance between the two points.

    Returns
    -------
    phi : float
        Value of the potential.
    """
    return 0.5*np.log(r)/np.pi


def green_field_2d(r, unit_vec):
    """Derivative of the Green function for Laplace equation
    
    Parameters
    ----------
    r : float
        Distance between the two points.
    unit_vec : ndarray, float
        Unit vector from the source point to evaluation
        point.

    Returns
    -------
    E : float
        Flow field.
    """
    nx, ny = unit_vec
    Ex = -0.5*nx/(r * np.pi)
    Ey = -0.5*ny/(r * np.pi)
    return Ex, Ey


def area_tri(coords):
    """Compute the area of a triangle with given coordinates

    Parameters
    ----------
    coords : ndarray, float
        Coordinates for the nodes of the triangle.

    Returns
    -------
    area : float
        Area of the triangle
    """
    mat = coords.copy()
    mat[:, 2] = 1
    return 0.5 * np.abs(np.linalg.det(mat))



def compute_potential_field(pts, tris_planet, area_tri):
    """
    Calcula el potencial y el campo eléctrico para una malla dada.

    Parameters:
    pts (numpy.ndarray): Array de puntos de la malla.
    tris_planet (numpy.ndarray): Array de triángulos de la malla.
    area_tri (function): Función para calcular el área de un triángulo.
    green_pot_2d (function): Función para calcular el potencial de Green en 2D.
    green_field_2d (function): Función para calcular el campo de Green en 2D.

    Returns:
    tuple: Una tupla que contiene:
        - potential (numpy.ndarray): Array de potenciales calculados.
        - field (numpy.ndarray): Array de campos eléctricos calculados.
    """
    x, y, _ = pts.T
    potential = np.zeros_like(x)
    field = np.zeros_like(pts[:, :2])

    for tri in tris_planet:
        coords = pts[tri]
        area = area_tri(coords)
        xm, ym, _ = np.mean(coords, axis=0)
        for cont, pt in enumerate(pts):
            pt_x, pt_y, _ = pt
            vec = np.array([pt_x - xm, pt_y - ym])
            r = np.linalg.norm(vec)
            unit_vec = vec / r
            pot = green_pot_2d(r)
            Ex, Ey = green_field_2d(r, unit_vec)
            potential[cont] += area * pot
            field[cont, 0] += area * Ex
            field[cont, 1] += area * Ey

    return potential, field


def plot_potential_field_star(pts, tris, potential, field):
    """
    Grafica el potencial y el campo eléctrico para una malla dada.

    Parameters:
    pts (numpy.ndarray): Array de puntos de la malla.
    tris (numpy.ndarray): Array de triángulos de la malla.
    potential (numpy.ndarray): Array de potenciales calculados.
    field (numpy.ndarray): Array de campos eléctricos calculados.
    """
    x, y, _ = pts.T

    # Graficar el potencial
    plt.figure()
    plt.tricontourf(x, y, tris, potential,levels=20)
    plt.axis("off")
    plt.axis("image")
    plt.colorbar()
    plt.savefig('figs/potencial_star.pdf')
    
    

    # Graficar la magnitud del campo eléctrico
    plt.figure()
    plt.tricontourf(x, y, tris, np.linalg.norm(field, axis=1), 12, cmap="magma", zorder=4,levels=20)
    plt.colorbar()
    plt.axis("off")
    plt.axis("image")
    plt.savefig('figs/campo_star.pdf')
    plt.axis("image")

def plot_potential_field_circle(pts, tris, potential, field):
    """
    Grafica el potencial y el campo eléctrico para una malla dada.

    Parameters:
    pts (numpy.ndarray): Array de puntos de la malla.
    tris (numpy.ndarray): Array de triángulos de la malla.
    potential (numpy.ndarray): Array de potenciales calculados.
    field (numpy.ndarray): Array de campos eléctricos calculados.
    """
    x, y, _ = pts.T

    # Graficar el potencial
    plt.figure()
    plt.tricontourf(x, y, tris, potential,levels=20)
    plt.axis("off")
    plt.axis("image")
    plt.colorbar()
    plt.savefig('figs/potencial_circle.pdf')
    plt.axis("image")

    # Graficar la magnitud del campo eléctrico
    plt.figure()
    plt.tricontourf(x, y, tris, np.linalg.norm(field, axis=1), 12, cmap="magma", zorder=4,levels=20)
    plt.colorbar()
    plt.axis("off")
    plt.axis("image")
    plt.savefig('figs/campo_circle.pdf')
    plt.axis("image")


def plot_point_source(radius, resolution):
    """
    Grafica el potencial de Green en 2D
    y la magnitud del campo eléctrico para una carga puntual.

    Parameters:
    radius (float): El radio de la máscara.
    resolution (int): La resolución de la malla.
    """
    # Crear una malla de puntos en un rango de -radius a radius
    y, x = radius * np.mgrid[-1:1:complex(0, resolution), -1:1:complex(0, resolution)]

    # Definir el punto central
    pt_x = 0
    pt_y = 0

    # Calcular la distancia desde el punto central
    r = np.sqrt((pt_x - x)**2 + (pt_y - y)**2)

    # Aplicar la máscara circular
    mask = r <= radius

    # Calcular el potencial de Green en 2D
    G = green_pot_2d(r)
    # Remover valores mayores al radio especificado
    G[~mask] = np.nan  

    # Calcular el campo de Green en 2D
    vec = np.array([-x, -y])
    unit_vec = vec / r
    Ex, Ey = green_field_2d(r, unit_vec)
    field = np.stack((Ex, Ey), axis=-1)
    # Remover valores mayores al radio especificado
    field[~mask] = np.nan  

    # Graficar el potencial de Green
    plt.figure()
    plt.contourf(x, y, G, levels=20)
    plt.colorbar()
    plt.axis("image")
    plt.axis("off")
    plt.savefig('figs/potencial_puntual.pdf')
    plt.show()

    # Graficar la magnitud del campo eléctrico
    plt.figure()
    plt.contourf(x, y, np.linalg.norm(field, axis=-1), levels=20, cmap="magma")
    plt.colorbar()
    plt.axis("image")
    plt.axis("off")
    plt.savefig('figs/campo_puntual.pdf')
    plt.show()


