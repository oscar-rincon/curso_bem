import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.special import hankel1

def wavenumberToFrequency(k, c = 344.0):
    return 0.5 * k * c / np.pi

def frequencyToWavenumber(f, c = 344.0):
    return 2.0 * np.pi * f / c

def soundPressure(k, phi, t = 0.0, c = 344.0, density = 1.205):
    angularVelocity = k * c
    return (1j * density * angularVelocity  * np.exp(-1.0j*angularVelocity*t)
            * phi).astype(np.complex64)

def SoundMagnitude(pressure):
    return np.log10(np.abs(pressure / 2e-5)) * 20

def AcousticIntensity(pressure, velocity):
    return 0.5 * (np.conj(pressure) * velocity).real

def SignalPhase(pressure):
    return np.arctan2(pressure.imag, pressure.real)

def Normal2D(pointA, pointB):                  
    diff = pointA - pointB                          
    len = norm(diff)                                
    return np.array([diff[1]/len, -diff[0]/len]) 

def ComplexQuad(func, start, end):                                                 
    samples = np.array([[0.980144928249, 5.061426814519E-02],                           
                        [0.898333238707, 0.111190517227],                               
                        [0.762766204958, 0.156853322939],                               
                        [0.591717321248, 0.181341891689],                               
                        [0.408282678752, 0.181341891689],                               
                        [0.237233795042, 0.156853322939],                               
                        [0.101666761293, 0.111190517227],                               
                        [1.985507175123E-02, 5.061426814519E-02]], dtype=np.float32)    
    
    vec = end - start                                                                                                  
    sum = 0.0                                                                                                          
    for n in range(samples.shape[0]):                                                                                  
        x = start + samples[n, 0] * vec                                                                                
        sum += samples[n, 1] * func(x)                                                                                 
    return sum * norm(vec)                                                                                         
    
def ComputeL(k, p, qa, qb, pOnElement):                                                                       
    qab = qb - qa                                                                                                  
    if pOnElement:                                                                                                 
        if k == 0.0:                                                                                               
            ra = norm(p - qa)                                                                                      
            rb = norm(p - qb)                                                                                      
            re = norm(qab)                                                                                         
            return 0.5 / np.pi * (re - (ra * np.log(ra) + rb * np.log(rb)))                                        
        else:                                                                                                      
            def func(x):                                                                                           
                R = norm(p - x)                                                                                    
                return 0.5 / np.pi * np.log(R) + 0.25j * hankel1(0, k * R)                                         
            return ComplexQuad(func, qa, p) + ComplexQuad(func, p, qa) \
                 + ComputeL(0.0, p, qa, qb, True)                                                              
    else:                                                                                                          
        if k == 0.0:                                                                                               
            return -0.5 / np.pi * ComplexQuad(lambda q: np.log(norm(p - q)), qa, qb)                           
        else:                                                                                                      
            return 0.25j * ComplexQuad(lambda q: hankel1(0, k * norm(p - q)), qa, qb)                          
    return 0.0                                                                                                     
                                                                                                                   
def ComputeM(k, p, qa, qb, pOnElement):                                                                       
    qab = qb - qa                                                                                                  
    vecq = Normal2D(qa, qb)                                                                                    
    if pOnElement:                                                                                                 
        return 0.0                                                                                                 
    else:                                                                                                          
        if k == 0.0:                                                                                               
            def func(x):                                                                                           
                r = p - x                                                                                          
                return np.dot(r, vecq) / np.dot(r, r)                                                              
            return -0.5 / np.pi * ComplexQuad(func, qa, qb)                                                    
        else:                                                                                                      
            def func(x):                                                                                           
                r = p - x                                                                                          
                R = norm(r)                                                                                        
                return hankel1(1, k * R) * np.dot(r, vecq) / R                                                     
            return 0.25j * k * ComplexQuad(func, qa, qb)                                                       
    return 0.0                                                                                                 
                                                                                                                   
def ComputeMt(k, p, vecp, qa, qb, pOnElement):                                                                
    qab = qb - qa                                                                                                  
    if pOnElement:                                                                                                 
        return 0.0                                                                                                 
    else:                                                                                                          
        if k == 0.0:                                                                                               
            def func(x):                                                                                           
                r = p - x                                                                                          
                return np.dot(r, vecp) / np.dot(r, r)                                                              
            return -0.5 / np.pi * ComplexQuad(func, qa, qb)                                                    
        else:                                                                                                      
            def func(x):                                                                                           
                r = p - x                                                                                          
                R = norm(r)                                                                                        
                return hankel1(1, k * R) * np.dot(r, vecp) / R                                                     
            return -0.25j * k * ComplexQuad(func, qa, qb)                                                      
                                                                                                                   
def ComputeN(k, p, vecp, qa, qb, pOnElement):                                                                 
    qab = qb- qa                                                                                                   
    if pOnElement:                                                                                                 
        ra = norm(p - qa)                                                                                          
        rb = norm(p - qb)                                                                                          
        re = norm(qab)                                                                                             
        if k == 0.0:                                                                                               
            return -(1.0 / ra + 1.0 / rb) / (re * 2.0 * np.pi) * re                                                
        else:                                                                                                      
            vecq = Normal2D(qa, qb)                                                                            
            k2 = k * k                                                                                             
            def func(x):                                                                                           
                r = p - x                                                                                          
                R2 = np.dot(r, r)                                                                                  
                R = np.sqrt(R2)                                                                                    
                drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / R2                                                 
                dpnu = np.dot(vecp, vecq)                                                                          
                c1 =  0.25j * k / R * hankel1(1, k * R)                                  - 0.5 / (np.pi * R2)      
                c2 =  0.50j * k / R * hankel1(1, k * R) - 0.25j * k2 * hankel1(0, k * R) - 1.0 / (np.pi * R2)      
                c3 = -0.25  * k2 * np.log(R) / np.pi                                                               
                return c1 * dpnu + c2 * drdudrdn + c3                                                              
            return ComputeN(0.0, p, vecp, qa, qb, True) - 0.5 * k2 * ComputeL(0.0, p, qa, qb, True) \
                 + ComplexQuad(func, qa, p) + ComplexQuad(func, p, qb)                                     
    else:                                                                                                          
        sum = 0.0j                                                                                                 
        vecq = Normal2D(qa, qb)                                                                                
        un = np.dot(vecp, vecq)                                                                                    
        if k == 0.0:                                                                                               
            def func(x):                                                                                           
                r = p - x                                                                                          
                R2 = np.dot(r, r)                                                                                  
                drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / R2                                                 
                return (un + 2.0 * drdudrdn) / R2                                                                  
            return 0.5 / np.pi * ComplexQuad(func, qa, qb)                                                     
        else:                                                                                                      
            def func(x):                                                                                           
                r = p - x                                                                                          
                drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / np.dot(r, r)                                       
                R = norm(r)                                                                                        
                return hankel1(1, k * R) / R * (un + 2.0 * drdudrdn) - k * hankel1(0, k * R) * drdudrdn            
            return 0.25j * k * ComplexQuad(func, qa, qb)                                                       

def SolveLinearEquation(Ai, Bi, ci, alpha, beta, f):
    A = np.copy(Ai)
    B = np.copy(Bi)
    c = np.copy(ci)

    x = np.empty(c.size, dtype=complex)
    y = np.empty(c.size, dtype=complex)

    gamma = np.linalg.norm(B, np.inf) / np.linalg.norm(A, np.inf)
    swapXY = np.empty(c.size, dtype=bool)
    for i in range(c.size):
        if np.abs(beta[i]) < gamma * np.abs(alpha[i]):
            swapXY[i] = False
        else:
            swapXY[i] = True

    for i in range(c.size):
        if swapXY[i]:
            for j in range(alpha.size):
                c[j] += f[i] * B[j,i] / beta[i]
                B[j, i] = -alpha[i] * B[j, i] / beta[i]
        else:
            for j in range(alpha.size):
                c[j] -= f[i] * A[j, i] / alpha[i]
                A[j, i] = -beta[i] * A[j, i] / alpha[i]

    A -= B
    y = np.linalg.solve(A, c)

    for i in range(c.size):
        if swapXY[i]:
            x[i] = (f[i] - alpha[i] * y[i]) / beta[i]
        else:
            x[i] = (f[i] - beta[i] * y[i]) / alpha[i]

    for i in range(c.size):
        if swapXY[i]:
            temp = x[i]
            x[i] = y[i]
            y[i] = temp

    return x, y

def computeBoundaryMatrices(k, mu, aVertex, aElement, orientation='interior'):
    A = np.empty((aElement.shape[0], aElement.shape[0]), dtype=complex)
    B = np.empty(A.shape, dtype=complex)

    for i in range(aElement.shape[0]):
        pa = aVertex[aElement[i, 0]]
        pb = aVertex[aElement[i, 1]]
        pab = pb - pa
        center = 0.5 * (pa + pb)
        centerNormal = Normal2D(pa, pb)
        for j in range(aElement.shape[0]):
            qa = aVertex[aElement[j, 0]]
            qb = aVertex[aElement[j, 1]]

            elementL  = ComputeL(k, center, qa, qb, i==j)
            elementM  = ComputeM(k, center, qa, qb, i==j)
            elementMt = ComputeMt(k, center, centerNormal, qa, qb, i==j)
            elementN  = ComputeN(k, center, centerNormal, qa, qb, i==j)
            
            A[i, j] = elementL + mu * elementMt
            B[i, j] = elementM + mu * elementN

        if orientation == 'interior':
            # interior variant, signs are reversed for exterior
            A[i,i] -= 0.5 * mu
            B[i,i] += 0.5
        elif orientation == 'exterior':
            A[i,i] += 0.5 * mu
            B[i,i] -= 0.5
        else:
            assert False, 'Invalid orientation: {}'.format(orientation)
            
    return A, B

def BoundarySolution(c, density, k, aPhi, aV):
    res = f"Density of medium:      {density} kg/m^3\n"
    res += f"Speed of sound:         {c} m/s\n"
    res += f"Wavenumber (Frequency): {k} ({wavenumberToFrequency(k)} Hz)\n\n"
    res += "index          Potential                   Pressure                    Velocity              Intensity\n"

    for i in range(aPhi.size):
        pressure = soundPressure(k, aPhi[i], t=0.0, c=344.0, density=1.205)
        intensity = AcousticIntensity(pressure, aV[i])
        res += f"{i+1:5d}  {aPhi[i].real: 1.4e}+ {aPhi[i].imag: 1.4e}i   {pressure.real: 1.4e}+ {pressure.imag: 1.4e}i   {aV[i].real: 1.4e}+ {aV[i].imag: 1.4e}i    {intensity: 1.4e}\n"
    
    return res

def solveInteriorBoundary(k, alpha, beta, f, phi, v, aVertex, aElement, c_=0, density=0, mu = None):
    mu = (1j / (k + 1))
    assert f.size == aElement.shape[0]
    A, B = computeBoundaryMatrices(k, mu, aVertex, aElement, orientation = 'interior')
    c = np.empty(aElement.shape[0], dtype=complex)
    for i in range(aElement.shape[0]):
        # Note, the only difference between the interior solver and this
        # one is the sign of the assignment below.
        c[i] = phi[i] + mu * v[i]

    phi, v = SolveLinearEquation(B, A, c, alpha, beta, f)
    res = BoundarySolution(c_, density, k, phi, v)
    #print(res)
    return  v, phi

def solveSamples(k, aV, aPhi, aIncidentPhi, aSamples, aVertex, aElement, orientation='interior'):
    assert aIncidentPhi.shape == aSamples.shape[:-1], \
        "Incident phi vector and sample points vector must match"

    aResult = np.empty(aSamples.shape[0], dtype=complex)

    for i in range(aIncidentPhi.size):
        p  = aSamples[i]
        sum = aIncidentPhi[i]
        for j in range(aPhi.size):
            qa = aVertex[aElement[j, 0]]
            qb = aVertex[aElement[j, 1]]

            elementL  = ComputeL(k, p, qa, qb, False)
            elementM  = ComputeM(k, p, qa, qb, False)
            if orientation == 'interior':
                sum += elementL * aV[j] - elementM * aPhi[j]
            elif orientation == 'exterior':
                sum -= elementL * aV[j] - elementM * aPhi[j]
            else:
                assert False, 'Invalid orientation: {}'.format(orientation)
        aResult[i] = sum
    return aResult

def solveInterior(k, aV, aPhi, aIncidentInteriorPhi, aInteriorPoints, aVertex, aElement):
    return solveSamples(k, aV, aPhi, aIncidentInteriorPhi, aInteriorPoints, aVertex, aElement, 'interior')


def solveSamples(k, aV, aPhi, aIncidentPhi, aSamples, aVertex, aElement, orientation):
    assert aIncidentPhi.shape == aSamples.shape[:-1], \
        "Incident phi vector and sample points vector must match"

    aResult = np.empty(aSamples.shape[0], dtype=complex)

    for i in range(aIncidentPhi.size):
        p  = aSamples[i]
        sum = aIncidentPhi[i]
        for j in range(aPhi.size):
            qa = aVertex[aElement[j, 0]]
            qb = aVertex[aElement[j, 1]]

            elementL  = ComputeL(k, p, qa, qb, False)
            elementM  = ComputeM(k, p, qa, qb, False)
            if orientation == 'interior':
                sum += elementL * aV[j] - elementM * aPhi[j]
            elif orientation == 'exterior':
                sum -= elementL * aV[j] - elementM * aPhi[j]
            else:
                assert False, 'Invalid orientation: {}'.format(orientation)
        aResult[i] = sum
    return aResult

def solveInterior(k, aV, aPhi, aIncidentInteriorPhi, aInteriorPoints, aVertex, aElement):
    return solveSamples(k, aV, aPhi, aIncidentInteriorPhi, aInteriorPoints, aVertex, aElement, 'interior')

def printInteriorSolution(k, c, density, aPhiInterior):
    print("\nSound pressure at the sample points\n")
    print("index          Potential                    Pressure               Magnitude         Phase\n")
    for i in range(aPhiInterior.size):
        pressure = soundPressure(k, aPhiInterior[i], c=c, density=density)
        magnitude = SoundMagnitude(pressure)
        phase = SignalPhase(pressure)
        print("{:5d}  {: 1.4e}+ {: 1.4e}i   {: 1.4e}+ {: 1.4e}i    {: 1.4e} dB       {:1.4f}".format( \
            i+1, aPhiInterior[i].real, aPhiInterior[i].imag, pressure.real, pressure.imag, magnitude, phase))
        
def Square():
    aVertex = np.array([[0.00, 0.0000], [0.00, 0.0125], [0.00, 0.0250], [0.00, 0.0375],
                         [0.00, 0.0500], [0.00, 0.0625], [0.00, 0.0750], [0.00, 0.0875],
                         
                         [0.0000, 0.10], [0.0125, 0.10], [0.0250, 0.10], [0.0375, 0.10],
                         [0.0500, 0.10], [0.0625, 0.10], [0.0750, 0.10], [0.0875, 0.10],
                         
                         [0.10, 0.1000], [0.10, 0.0875], [0.10, 0.0750], [0.10, 0.0625],
                         [0.10, 0.0500], [0.10, 0.0375], [0.10, 0.0250], [0.10, 0.0125],
                         
                         [0.1000, 0.00], [0.0875, 0.00], [0.0750, 0.00], [0.0625, 0.00],
                         [0.0500, 0.00], [0.0375, 0.00], [0.0250, 0.00], [0.0125, 0.00]], dtype=np.float32)

    aEdge = np.array([[ 0,  1], [ 1,  2], [ 2,  3], [ 3,  4],
                      [ 4,  5], [ 5,  6], [ 6,  7], [ 7,  8],
                      
                      [ 8,  9], [ 9, 10], [10, 11], [11, 12],
                      [12, 13], [13, 14], [14, 15], [15, 16],
                      
                      [16, 17], [17, 18], [18, 19], [19, 20],
                      [20, 21], [21, 22], [22, 23], [23, 24],
                      
                      [24, 25], [25, 26], [26, 27], [27, 28],
                      [28, 29], [29, 30], [30, 31], [31,  0]], dtype=np.int32)

    return aVertex, aEdge
def Square_n(n=10, length=0.1):

    h = length / n

    # Generar puntos por lado (sin repetir esquinas)
    left   = [(0.0, i * h) for i in range(n)]                      # 0 → n-1
    top    = [(i * h, length) for i in range(n)]                   # n → 2n-1
    right  = [(length, length - i * h) for i in range(n)]          # 2n → 3n-1
    bottom = [(length - i * h, 0.0) for i in range(n)]             # 3n → 4n-1

    # Concatenar en orden deseado
    aVertex = np.array(left + top + right + bottom, dtype=np.float32)

    # Crear aristas conectando consecutivamente + cierre del contorno
    num_vertices = 4 * n
    aEdge = np.array([[i, (i + 1) % num_vertices] for i in range(num_vertices)], dtype=np.int32)

    return aVertex, aEdge

def generateInteriorPoints_test_problem_2(Nx=10, Ny=10, length=0.1):

    # Evitar incluir los bordes: desplazamos un poco desde 0 hasta length
    x = np.linspace(length / (Nx + 1), length * Nx / (Nx + 1), Nx)
    y = np.linspace(length / (Ny + 1), length * Ny / (Ny + 1), Ny)

    X, Y = np.meshgrid(x, y)
    interiorPoints = np.column_stack([X.ravel(), Y.ravel()])
    return interiorPoints.astype(np.float32)


def phi_test_problem_1_2(p1, p2, k):
    factor = k / np.sqrt(2)
    return np.sin(factor * p1) * np.sin(factor * p2)


def plot_oriented_edges(vertices, elementos, title="Sentido de las aristas en el contorno"):

    plt.figure(figsize=(5, 5))

    # Dibujar flechas para cada arista
    for i, (start, end) in enumerate(elementos):
        p1, p2 = vertices[start], vertices[end]
        dx, dy = p2 - p1
        plt.arrow(p1[0], p1[1], dx, dy, head_width=0.002, length_includes_head=True, 
                  color='blue', alpha=0.7)

        # Etiqueta en el centro de la arista
        cx, cy = (p1 + p2) / 2
        plt.text(cx, cy, f'{i}', color='purple', fontsize=8, ha='center', va='center')

    # Dibujar nodos y sus índices
    for i, (x, y) in enumerate(vertices):
        plt.plot(x, y, 'ko')
        plt.text(x, y, f'{i}', color='red', fontsize=9, ha='left', va='bottom')

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def plot_edge_scalar_field(centros, vertices, elementos, f, cmap="magma", title=""):
 
    plt.figure(figsize=(5, 5))
    colormap = plt.get_cmap(cmap)

    # Dibujar los elementos como líneas grises
    for i, edge in enumerate(elementos):
        p1, p2 = vertices[edge[0]], vertices[edge[1]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', linewidth=2)

    # Dibujar los centros coloreados según f
    sc = plt.scatter(centros[:, 0], centros[:, 1], c=f.real, cmap=colormap, edgecolors='k', zorder=3)

    # Barra de color
    plt.colorbar(sc, label="Condición de contorno f", orientation="vertical", pad=0.02, aspect=20, shrink=0.8)
    plt.gca().set_aspect('equal')
    plt.xlabel("x")
    plt.ylabel("y")
    if title:
        plt.title(title)
    plt.grid(True)
    plt.show()

def plot_solutions(exact_sol, num_sol, interiorPoints):
 
    # Crear figura con tres subgráficos lado a lado
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)

    # Solución numérica
    tcf1 = axs[0].tricontourf(interiorPoints[:, 0], interiorPoints[:, 1], num_sol, levels=50, cmap='magma')
    fig.colorbar(tcf1, ax=axs[0], label=r'$u(p)$')
    axs[0].set_title("Solución Numérica")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    # Solución exacta
    tcf2 = axs[1].tricontourf(interiorPoints[:, 0], interiorPoints[:, 1], exact_sol, levels=50, cmap='magma')
    fig.colorbar(tcf2, ax=axs[1], label=r'$u(p)$')
    axs[1].set_title("Solución Exacta")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")

    # Error absoluto
    error = np.abs(num_sol - exact_sol)
    tcf3 = axs[2].tricontourf(interiorPoints[:, 0], interiorPoints[:, 1], error, levels=50, cmap='magma')
    fig.colorbar(tcf3, ax=axs[2], label=r'Error $|u_{num} - u_{exact}|$')
    axs[2].set_title("Error")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")

    plt.show()


