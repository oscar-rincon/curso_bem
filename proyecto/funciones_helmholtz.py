import numpy as np
from numpy import log, pi, mean
from numpy.linalg import norm
from scipy.special import roots_legendre, hankel1

def green_pot(r, k, tol=1e-10):
    """Green's function for 2D Helmholtz equation"""
    r_safe = np.copy(r)
    r_safe[r_safe <= tol] = 1e-10
    return 1j/4 * hankel1(0, k * r_safe)
 
def green_flow(rvec, normal, k, tol=1e-10):
    """Normal derivative of Green's function: ∂G/∂n_q"""
    #rvec[rvec <= tol] = 1e-10 # avoid division by zero
    r = np.linalg.norm(rvec)
    return 1j * k / 4 * hankel1(1, k * r) * rvec.dot(normal)

def green_flow(rvec, normal, k):
   """Normal derivative of Green's function: ∂G/∂n_q"""
   r = np.linalg.norm(rvec, axis=-1)
   r[r == 0] = 1e-10  # avoid division by zero
   dot = np.sum(rvec * normal, axis=-1)
   return -1j * k / 4 * hankel1(1, k * r) * dot / r

def green_pot_0(r, tol=1e-10):
    """Green's function for Laplace in 2D (singular subtraction)"""
    r_safe = np.copy(r)
    #r_safe[r_safe <= tol] = 1e-10
    return 0.5 * log(r_safe) / pi

def interp_coord(coords, npts=8):
    """Interpolate coordinates along element using Gauss points"""
    gs_pts, gs_wts = roots_legendre(npts)
    x = 0.5 * (np.outer(1 - gs_pts, coords[0]) + np.outer(1 + gs_pts, coords[1]))
    jac_det = norm(coords[1] - coords[0]) / 2
    return x, jac_det, gs_wts

def influence_coeff_num(elem, coords, pt_col, k, npts=8):
    """Numerical integration of influence coefficients for element"""
    x, jac_det, gs_wts = interp_coord(coords[elem], npts)
    G = green_pot(np.linalg.norm(x - pt_col, axis=-1), k)
    G0 = green_pot_0(np.linalg.norm(x - pt_col, axis=-1))
    # Normal vector (outward) perpendicular to element
    tangent = coords[elem[1]] - coords[elem[0]]
    tangent /= norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    # dcos = coords[elem[1]] - coords[elem[0]]
    # dcos = dcos / norm(dcos)
    # rotmat = np.array([[dcos[1], -dcos[0]],
    #                   [dcos[0], dcos[1]]])
    # normal = rotmat @ dcos
    H = green_flow(x - pt_col, normal, k)
    G_coeff = np.dot(G, gs_wts) * jac_det
    Gdiff_coeff = np.dot(G-G0, gs_wts) * jac_det
    H_coeff = np.dot(H, gs_wts) * jac_det
    return G_coeff, H_coeff, Gdiff_coeff

def assem(coords, elems, k):
    """Assemble BEM system matrices G and H"""
    nelems = elems.shape[0]
    Gmat = np.zeros((nelems, nelems), dtype=complex)
    Hmat = np.zeros((nelems, nelems), dtype=complex)

    for i, elem_row in enumerate(elems):
        for j, elem_col in enumerate(elems):
            pt_col = mean(coords[elem_col], axis=0)
            Gij, Hij, Gdiff_coeff = influence_coeff_num(elem_row, coords, pt_col, k)
            if i == j:
                L = norm(coords[elem_row[1]] - coords[elem_row[0]])
                Gmat[i, j] = - L/(2*pi)*(log(L/2) - 1)  #+ Gdiff_coeff
                Hmat[i, j] = - 0.5
            else:
                Gmat[i, j] = Gij
                Hmat[i, j] = Hij
    return Gmat, Hmat

def eval_sol(ev_coords, coords, elems, u_boundary, q_boundary, k):
    """Evaluate BEM solution at arbitrary evaluation points"""
    npts = ev_coords.shape[0]
    solution = np.zeros(npts, dtype=complex)
    for i in range(npts):
        pt_col = ev_coords[i]
        for j, elem in enumerate(elems):
            G, H, _ = influence_coeff_num(elem, coords, pt_col, k)
            solution[i] += u_boundary[j] * H - q_boundary[j] * G
    return solution
