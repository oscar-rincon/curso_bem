# ---------------------------------------------------------------------------
# Copyright (C) 2017 Frank Jargstorff
#
# This file is part of the AcousticBEM library.
# AcousticBEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AcousticBEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AcousticBEM.  If not, see <http://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------
import numpy as np
from BoundaryData import *
from Geometry import *

class HelmholtzSolver(object):
    def __init__(self, aVertex = None, aElement = None, c = 344.0, density = 1.205):
        assert not (aVertex is None), "Cannot construct InteriorHelmholtzProblem2D without valid vertex array."
        self.aVertex = aVertex
        assert not (aElement is None), "Cannot construct InteriorHelmholtzProblem2D without valid element array."
        self.aElement = aElement
        self.c       = c
        self.density = density

        '''
        # compute the centers of the discrete elements (depending on kind, i.e line segment or triangle).
        # The solver computes the velocity potential at at these center points. 
        if (self.aElement.shape[1] ==  2):
            self.aCenters = 0.5 * (self.aVertex[self.aElement[:, 0]] + self.aVertex[aElement[:, 1]])
        elif (self.aElement.shape[1] == 3):
            self.aCenters = 1.0/3.0 * (self.aVertex[self.aElement[:, 0]] +\
                                       self.aVertex[self.aElement[:, 1]] +\
                                       self.aVertex[self.aElement[:, 2]])
        '''

    def __repr__(self):
        result = "HelmholtzSolover("
        result += "  aVertex = " + repr(self.aVertex) + ", "
        result += "  aElement = " + repr(self.aElement) + ", "
        result += "  c = " + repr(self.c) + ", "
        result += "  rho = " + repr(self.rho) + ")"
        return result

    def solveExteriorBoundary(self, k, boundaryCondition, boundaryIncidence, mu = None):
        mu = mu or (1j / (k + 1))
        assert boundaryCondition.f.size == self.aElement.shape[0]
        A, B = self.computeBoundaryMatricesExterior(k, mu)
        c = np.empty(self.aElement.shape[0], dtype=complex)
        for i in range(self.aElement.shape[0]):
            # Note, the only difference between the interior solver and this
            # one is the sign of the assignment below.
            c[i] = -(boundaryIncidence.phi[i] + mu * boundaryIncidence.v[i])

        phi, v = self.SolveLinearEquation(B, A, c,
                                          boundaryCondition.alpha,
                                          boundaryCondition.beta,
                                          boundaryCondition.f)
        return BoundarySolution(self, k, phi, v)

    def solveInteriorBoundary(self, k, boundaryCondition, boundaryIncidence, mu = None):
        mu = mu or (1j / (k + 1))
        assert boundaryCondition.f.size == self.aElement.shape[0]
        A, B = self.computeBoundaryMatricesInterior(k, mu)
        c = np.empty(self.aElement.shape[0], dtype=complex)
        for i in range(self.aElement.shape[0]):
            # Note, the only difference between the interior solver and this
            # one is the sign of the assignment below.
            c[i] = boundaryIncidence.phi[i] + mu * boundaryIncidence.v[i]

        phi, v = self.SolveLinearEquation(B, A, c,
                                          boundaryCondition.alpha,
                                          boundaryCondition.beta,
                                          boundaryCondition.f)
        return BoundarySolution(self, k, phi, v)

    
    @classmethod
    def SolveLinearEquation(cls, Ai, Bi, ci, alpha, beta, f):
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


def printInteriorSolution(solution, aPhiInterior):
    print("\nSound pressure at the sample points\n")
    print("index          Potential                    Pressure               Magnitude         Phase\n")
    for i in range(aPhiInterior.size):
        pressure = soundPressure(solution.k, aPhiInterior[i], c=solution.parent.c, density=solution.parent.density)
        magnitude = SoundMagnitude(pressure)
        phase = SignalPhase(pressure)
        print("{:5d}  {: 1.4e}+ {: 1.4e}i   {: 1.4e}+ {: 1.4e}i    {: 1.4e} dB       {:1.4f}".format( \
            i+1, aPhiInterior[i].real, aPhiInterior[i].imag, pressure.real, pressure.imag, magnitude, phase))

def printInteriorSolution(solution, aPhiInterior):
    print("\nSound pressure at the sample points\n")
    print("index          Potential                    Pressure               Magnitude         Phase\n")
    for i in range(aPhiInterior.size):
        pressure = soundPressure(solution.k, aPhiInterior[i], c=solution.parent.c, density=solution.parent.density)
        magnitude = SoundMagnitude(pressure)
        phase = SignalPhase(pressure)
        print("{:5d}  {: 1.4e}+ {: 1.4e}i   {: 1.4e}+ {: 1.4e}i    {: 1.4e} dB       {:1.4f}".format( \
            i+1, aPhiInterior[i].real, aPhiInterior[i].imag, pressure.real, pressure.imag, magnitude, phase))
