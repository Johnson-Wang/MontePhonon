#coding:utf-8
__author__ = 'xwangan'
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from structure import NanoMeshStructure

class OrderedNanoMesh(NanoMeshStructure):
    "functionx f = a * x + b * y + c * z + d, each row in the self.boundaries represent a tuple of (a, b , c, d)"
    def __init__(self, *args, **kwargs):
        NanoMeshStructure.__init__(self, *args, **kwargs)
        porosity = self.holex * self.holey / (self.periodx * self.periody)
        self.porosity = porosity
        self.correction_factor = (1 - porosity) / (1 + porosity)
        self.info = "ordered nanomesh"

    def construct_boundaries(self):
        self.boundaries = np.array([
            [1, 0, 0, self.periodx / 2], # phantom
            [1, 0, 0, -self.periodx / 2.], # phantom
            [0, 1, 0, self.periody / 2.], # phantom
            [0, 1, 0, -self.periody / 2.], # phantom
            [0, 0, 1, self.thickness / 2], # real
            [0, 0, 1, -self.thickness / 2], #real
            [1, 0, 0, -self.holex / 2],  # real
            [1, 0, 0, self.holex / 2], # real
            [0, 1, 0, self.holey / 2.],  # real
            [0, 1, 0, -self.holey / 2.]], # real
            dtype=np.double)

        self.boundary_map = np.array([1, 0, 3, 2, 4, 5, 6, 7, 8, 9], dtype=np.int) # phantom boundary(totally specular)
        # if a boundary is a real boundary, it maps to itself, otherwise (e.g. periodic) it maps to the other end of the unitcell
        self.boundary_map_operation = np.array([[self.periodx, 0, 0],
                                                [-self.periodx, 0, 0],
                                                [0, self.periody, 0],
                                                [0, -self.periody, 0],
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]],
                                               dtype=np.double)
        # once a point reaches a periodic boundary it is draged back by one unitcell length.

    def pos_irt_boundary(self, points):
        "position in regards to boundaries, i.e. 0 (on), 1 (in) and -1 (out)"
        relative_positions = np.zeros(len(points), dtype=np.int)
        ithickness = np.abs(points[:,2]) - self.thickness / 2
        # positions in regards to the thickness
        iperiodx = np.abs(points[:, 0]) - self.periodx / 2
        # positions in regards to peroidx
        iperiody = np.abs(points[:,1]) - self.periody / 2
        iholex = np.abs(points[:, 0]) - self.holex / 2
        # positions in regards to holex boundaries
        iholey = np.abs(points[:, 1]) - self.holey / 2
        # positions in regards to holey boundaries
        is_out1 = (ithickness > self.tolerance).__or__(iperiodx > self.tolerance).__or__(iperiody > self.tolerance)
        # self.tolerance is an infinitesmal number (e.g. 1e-8)
        # outside the thinfilm (z) or outside  periodx (x) or outside periody (y)
        is_out2 =  (iholex < -self.tolerance).__and__(iholey < -self.tolerance)
        # inside holex (x) and inside holey (y)
        relative_positions[np.where(is_out1.__or__(is_out2))] = -1
        #points satisfying  is_out1 or is out2 are categorized as out
        is_in1 = (ithickness < -self.tolerance).__and__(iperiodx < -self.tolerance).__and__(iperiody < -self.tolerance)
        #inside the outmost boundaries
        is_in2 = (iholex > self.tolerance).__or__(iholey > self.tolerance)
        #outside the inner boundaries
        relative_positions[np.where(is_in1.__and__(is_in2))] = 1 # in
        # otherwise: on
        return relative_positions

    def plot_boundaries(self):
        print "Phonons in %s" %self.info
        self.fig = plt.figure()
        ax = self.fig.gca()
        ax.set_title('Structure of the ordered nanomesh thin film (top view)')
        ax.hlines(self.periody/2, -self.periodx/2, self.periodx/2, colors='r')
        ax.hlines(-self.periody/2, -self.periodx/2, self.periodx/2, colors='r')
        ax.vlines(-self.periodx/2, -self.periody/2, self.periody/2, colors='r')
        ax.vlines(self.periodx/2, -self.periody/2, self.periody/2, colors='r')
        ax.text(0, self.periody / 2 * 0.9, "%5.2f"%(self.periodx))
        ax.text(self.periodx / 2 * 0.9, 0, "%5.2f"%(self.periody))

        ax.hlines(self.holey/2, -self.holex/2, self.holex/2, colors='b')
        ax.hlines(-self.holey/2, -self.holex/2, self.holex/2, colors='b')
        ax.text(0, self.holey / 2 * 1.1, "%5.1f"%(self.holex))
        ax.text(self.holex / 2 * 1.1, 0, "%5.1f"%(self.holey))

        ax.vlines(-self.holex/2, -self.holey/2, self.holey/2, colors='b')
        ax.vlines(self.holex/2, -self.holey/2, self.holey/2, colors='b')
        ax.set_xlim((-self.periodx/1.5, self.periodx/1.5))
        ax.set_ylim((-self.periody/1.5, self.periody/1.5))
        ax.set_aspect('equal')
        plt.show()


class StaggeredNanoMesh(NanoMeshStructure):
    "functionx f = a * x + b * y + c * z + d, each row in the self.boundaries represent a tuple of (a, b , c, d)"
    def __init__(self, *args, **kwargs):
        NanoMeshStructure.__init__(self, *args, **kwargs)
        porosity = 2 * self.holex * self.holey / (self.periodx * self.periody)
        self.porosity = porosity
        self.correction_factor = (1 - porosity) / (1 + porosity)
        self.info = "staggered nanomesh"

    def construct_boundaries(self):
        self.boundaries = np.array([
            [1, 0, 0,  self.periodx / 2.], # phantom
            [1, 0, 0, -self.periodx / 2.], # phantom
            [0, 1, 0,  self.periody / 2.], # phantom
            [0, 1, 0, -self.periody / 2.], # phantom
            [0, 0, 1,  self.thickness / 2.], # real
            [0, 0, 1, -self.thickness / 2.], # real
            [1, 0, 0, -(self.periodx / 4. + self.holex / 2.)],  # real
            [1, 0, 0, -(self.periodx / 4. - self.holex / 2.)], # real
            [1, 0, 0,  (self.periodx / 4. + self.holex / 2.)],  # real
            [1, 0, 0,  (self.periodx / 4. - self.holex / 2.)], # real
            [0, 1, 0, -(self.periody / 4. + self.holey / 2.)],  # real
            [0, 1, 0, -(self.periody / 4. - self.holey / 2.)], # real
            [0, 1, 0,  (self.periody / 4. + self.holey / 2.)],  # real
            [0, 1, 0,  (self.periody / 4. - self.holey / 2.)]], # real
            dtype=np.double)

        self.boundary_map = np.array([1, 0, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.int) # phantom boundary(totally specular)
        # if a boundary is a real boundary, it maps to itself, otherwise (e.g. periodic) it maps to the other end of the unitcell
        self.boundary_map_operation = np.array([[self.periodx, 0, 0],
                                                [-self.periodx, 0, 0],
                                                [0, self.periody, 0],
                                                [0, -self.periody, 0],
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]],
                                               dtype=np.double)
        # once a point reaches a periodic boundary it is draged back by one unitcell length.

    def pos_irt_boundary(self, points):
        "position in regards to boundaries, i.e. 0 (on), 1 (in) and -1 (out)"
        relative_positions = np.zeros(len(points), dtype=np.int)
        ithickness = np.abs(points[:,2]) - self.thickness / 2.
        # positions in regards to the thickness
        iperiodx = np.abs(points[:, 0]) - self.periodx / 2.
        # positions in regards to peroidx
        iperiody = np.abs(points[:,1]) - self.periody / 2.
        # positions in regards to peroidy
        iholex1 = np.abs(points[:, 0] + self.periodx / 4.) - self.holex / 2.
        iholey1 = np.abs(points[:, 1] - self.periody / 4.) - self.holey / 2.

        # positions in regards to holex boundaries
        iholex2 = np.abs(points[:, 0] - self.periodx / 4.) - self.holex / 2.
        iholey2 = np.abs(points[:, 1] + self.periody / 4.) - self.holey / 2.
        # positions in regards to holey boundaries
        is_out1 = (ithickness > self.tolerance).__or__(iperiodx > self.tolerance).__or__(iperiody > self.tolerance)
        # self.tolerance is an infinitesmal number (e.g. 1e-8)
        # outside the thinfilm (z) or outside  periodx (x) or outside periody (y)
        is_out2 =  (iholex1 < -self.tolerance).__and__(iholey1 < -self.tolerance)
        is_out3 =  (iholex2 < -self.tolerance).__and__(iholey2 < -self.tolerance)
        # inside holex (x) and inside holey (y)
        relative_positions[np.where((is_out1).__or__(is_out2).__or__(is_out3))] = -1
        #points satisfying  is_out1 or is out2 are categorized as out
        is_in1 = (ithickness < -self.tolerance).__and__(iperiodx < -self.tolerance).__and__(iperiody < -self.tolerance)
        #inside the outmost boundaries
        is_in2 = (iholex1 > self.tolerance).__or__(iholey1 > self.tolerance)
        is_in3 = (iholex2 > self.tolerance).__or__(iholey2 > self.tolerance)
        #outside the inner boundaries
        relative_positions[np.where(is_in1.__and__(is_in2).__and__(is_in3))] = 1 # in
        # otherwise: on
        return relative_positions

    def plot_boundaries(self):
        print "Phonons in %s" %self.info
        self.fig = plt.figure()
        ax = self.fig.gca()
        ax.set_title('Structure of the staggered nanomesh thin film (top view)')
        ax.hlines(self.periody/2, -self.periodx/2, self.periodx/2, colors='r')
        ax.hlines(-self.periody/2, -self.periodx/2, self.periodx/2, colors='r')
        ax.vlines(-self.periodx/2, -self.periody/2, self.periody/2, colors='r')
        ax.vlines(self.periodx/2, -self.periody/2, self.periody/2, colors='r')
        ax.text(0, self.periody / 2 * 0.9, "%5.1f"%(self.periodx))
        ax.text(self.periodx / 2 * 0.9, 0, "%5.1f"%(self.periody))
        #outer boundary

        ax.hlines(self.periody / 4 - self.holey/2, -(self.periodx / 4 + self.holex/2), -(self.periodx / 4 - self.holex/2), colors='b')
        ax.hlines(self.periody / 4 + self.holey/2, -(self.periodx / 4 + self.holex/2), -(self.periodx / 4 - self.holex/2), colors='b')
        ax.vlines(-(self.periodx / 4 - self.holex/2), self.periody / 4 - self.holey/2, self.periody / 4 + self.holey/2, colors='b')
        ax.vlines(-(self.periodx / 4 + self.holex/2), self.periody / 4 - self.holey/2, self.periody / 4 + self.holey/2, colors='b')
        ax.text(-self.periodx / 4, self.periody / 4 - self.holey/2 * 1.5, "%5.1f"%(self.holex))
        ax.text(-(self.periodx / 4 - self.holex / 2 * 1.3), self.periody / 4, "%5.1f"%(self.holey))
        # hole1

        ax.hlines(-(self.periody / 4 - self.holey/2), self.periodx / 4 - self.holex/2, self.periodx / 4 + self.holex/2, colors='b')
        ax.hlines(-(self.periody / 4 + self.holey/2), self.periodx / 4 - self.holex/2, self.periodx / 4 + self.holex/2, colors='b')
        ax.vlines(self.periodx / 4 - self.holex/2, -(self.periody / 4 + self.holey/2), -(self.periody / 4 - self.holey/2), colors='b')
        ax.vlines(self.periodx / 4 + self.holex/2, -(self.periody / 4 + self.holey/2), -(self.periody / 4 - self.holey/2), colors='b')
        #hole2

        ax.set_xlim((-self.periodx/1.5, self.periodx/1.5))
        ax.set_ylim((-self.periody/1.5, self.periody/1.5))
        ax.set_aspect('equal')
        # plt.show(block=False)

class ThinFilm(NanoMeshStructure):
    # A inheritor of the superclass NanoMeshStructure defined in structure.py
    def __init__(self, *args, **kwargs):
        NanoMeshStructure.__init__(self, *args, **kwargs)
        self.info = "thin film"

    def construct_boundaries(self):
        "functionx f = a * x + b * y + c * z + d, each row in the self.boundaries represent a tuple of (a, b , c, d)"
        self.boundaries = np.array([
            [1, 0, 0, self.periodx / 2], # phantom
            [1, 0, 0, -self.periodx / 2.], # phantom
            [0, 1, 0, self.periody / 2.], # phantom
            [0, 1, 0, -self.periody / 2.], # phantom
            [0, 0, 1, self.thickness / 2], # real
            [0, 0, 1, -self.thickness / 2]], # real
            dtype=np.double)

        self.boundary_map = np.array([1, 0, 3, 2, 4, 5], dtype=np.int) # phantom boundary(totally specular)
        # if a boundary is a real boundary, it maps to itself, otherwise (e.g. periodic) it maps to the other end of the unitcell
        self.boundary_map_operation = np.array([[self.periodx, 0, 0],
                                                [-self.periodx, 0, 0],
                                                [0, self.periody, 0],
                                                [0, -self.periody, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]],
                                               dtype=np.double)
        # once a point reaches a periodic boundary it is draged back by one unitcell length.

    def pos_irt_boundary(self, points):
        "position in regards to boundaries, i.e. 0 (on), 1 (in) and -1 (out)"
        relative_positions = np.zeros(len(points), dtype=np.int)
        ithickness = np.abs(points[:,2]) - self.thickness / 2
        # positions in regards to the thickness
        iperiodx =np.abs(points[:,0]) - self.periodx / 2
        # positions in regards to periodx
        iperiody = np.abs(points[:,1]) - self.periody / 2
        # positions in regards to periody
        out = (ithickness > self.tolerance).__or__(iperiodx > self.tolerance).__or__(iperiody > self.tolerance)
        inn = (ithickness < -self.tolerance).__and__(iperiodx < -self.tolerance).__and__(iperiody < -self.tolerance)
        # self.tolerance is an infinitesmal number (e.g. 1e-8)
        relative_positions[np.where(out)] = -1
        relative_positions[np.where(inn)] = 1
        return relative_positions

    def plot_boundaries(self, is_3D=True):
        print "Phonons in %s" %self.info
        if not is_3D:
            self.fig = plt.figure()
            ax = self.fig.gca()
            ax.set_title('Structure of the thin film (top view)')
            ax.hlines(self.periody/2, -self.periodx/2, self.periodx/2, colors='r')
            ax.hlines(-self.periody/2, -self.periodx/2, self.periodx/2, colors='r')
            ax.vlines(-self.periodx/2, -self.periody/2, self.periody/2, colors='r')
            ax.vlines(self.periodx/2, -self.periody/2, self.periody/2, colors='r')
            ax.set_xlim((-self.periodx/1.5, self.periodx/1.5))
            ax.set_ylim((-self.periody/1.5, self.periody/1.5))
            plt.show()
        else:
            self.fig = plt.figure()
            ax = self.fig.add_subplot(111, projection='3d')
            x = [-self.periodx / 2, self.periodx / 2]
            y = [-self.periody / 2, self.periody / 2]
            z = [-self.thickness / 2, self.thickness / 2]
            X, Y  = np.meshgrid(x, y)
            ax.plot_surface(X, Y,self.thickness/2, alpha=0.5)
            ax.plot_surface(X, Y,-self.thickness/2, alpha=0.5)
            X, Z = np.meshgrid(x, z)
            ax.plot_surface(X, self.periody / 2, Z, alpha=0.5)
            ax.plot_surface(X, -self.periody / 2, Z, alpha=0.5)
            Y, Z = np.meshgrid(y, z)
            ax.plot_surface(self.periodx / 2, Y, Z, alpha=0.5)
            ax.plot_surface(-self.periodx / 2, Y, Z, alpha=0.5)
            plt.draw()

class NanoWire(ThinFilm):
    # A inheritor of the superclass NanoMeshStructure defined in structure.py
    def __init__(self, *args, **kwargs):
        ThinFilm.__init__(self, *args, **kwargs)
        self.correction_factor = 1.
        self.info = "square nanowire"

    def construct_boundaries(self):
        "functionx f = a * x + b * y + c * z + d, each row in the self.boundaries represent a tuple of (a, b , c, d)"
        self.boundaries = np.array([
            [1, 0, 0, self.periodx / 2], # phantom
            [1, 0, 0, -self.periodx / 2.], # phantom
            [0, 1, 0, self.periody / 2.], # real
            [0, 1, 0, -self.periody / 2.], # real
            [0, 0, 1, self.thickness / 2], # real
            [0, 0, 1, -self.thickness / 2]], # real
            dtype=np.double)

        self.boundary_map = np.array([1, 0, 2, 3, 4, 5], dtype=np.int) # phantom boundary(totally specular)
        # if a boundary is a real boundary, it maps to itself, otherwise (e.g. periodic) it maps to the other end of the unitcell
        self.boundary_map_operation = np.array([[self.periodx, 0, 0],
                                                [-self.periodx, 0, 0],
                                                [0, 0, 0], #real
                                                [0, 0, 0], #real
                                                [0, 0, 0],
                                                [0, 0, 0]],
                                               dtype=np.double)
        # once a point reaches a periodic boundary it is draged back by one unitcell length.

class Bulk(NanoMeshStructure):
    "functionx f = a * x + b * y + c * z + d, each row in the self.boundaries represent a tuple of (a, b , c, d)"
    def __init__(self, *args, **kwargs):
        NanoMeshStructure.__init__(self, *args, **kwargs)
        self.info = "bulk structure"

    def construct_boundaries(self):
        self.boundaries = np.array([
            [1, 0, 0, self.periodx / 2], #phantom
            [1, 0, 0, -self.periodx / 2.],#phantom
            [0, 1, 0, self.periody / 2.],#phantom
            [0, 1, 0, -self.periody / 2.],#phantom
            [0, 0, 1, self.thickness / 2],#phantom
            [0, 0, 1, -self.thickness / 2]],#phantom
            dtype=np.double)

        self.boundary_map = np.array([1, 0, 3, 2, 5, 4]) # phantom boundary
        self.boundary_map_operation = np.array([[self.periodx, 0, 0],#phantom
                                                [-self.periodx, 0, 0],#phantom
                                                [0, self.periody, 0],#phantom
                                                [0, -self.periody, 0],#phantom
                                                [0, 0, self.thickness],#phantom
                                                [0, 0, -self.thickness]])#phantom

    def pos_irt_boundary(self, points):
        "position in regards to boundaries, i.e. 0 (on), 1 (in) and -1 (out), the same as that defined in ThinFilm"
        relative_positions = np.zeros(len(points), dtype=np.int)
        ithickness = np.abs(points[:,2]) - self.thickness / 2
        iperiodx =np.abs(points[:,0]) - self.periodx / 2
        iperiody = np.abs(points[:,1]) - self.periody / 2
        out = (ithickness > self.tolerance).__or__(iperiodx > self.tolerance).__or__(iperiody > self.tolerance)
        inn = (ithickness < -self.tolerance).__and__(iperiodx < -self.tolerance).__and__(iperiody < -self.tolerance)
        relative_positions[np.where(out)] = -1
        relative_positions[np.where(inn)] = 1
        return relative_positions

    def plot_boundaries(self):
        print "Phonons in %s" %self.info
        self.fig = plt.figure()
        ax = self.fig.gca()
        ax.set_title('Structure of the system (top view)')
        ax.hlines(self.periody/2, -self.periodx/2, self.periodx/2, colors='r')
        ax.hlines(-self.periody/2, -self.periodx/2, self.periodx/2, colors='r')
        ax.vlines(-self.periodx/2, -self.periody/2, self.periody/2, colors='r')
        ax.vlines(self.periodx/2, -self.periody/2, self.periody/2, colors='r')
        ax.set_xlim((-self.periodx/1.5, self.periodx/1.5))
        ax.set_ylim((-self.periody/1.5, self.periody/1.5))
        # plt.draw()


class Cube(Bulk):
    "functionx f = a * x + b * y + c * z + d, each row in the self.boundaries represent a tuple of (a, b , c, d)"
    def __init__(self, *args, **kwargs):
        Bulk.__init__(self, *args, **kwargs)
        self.info = "cubic structure"

    def construct_boundaries(self):
        self.boundaries = np.array([
            [1, 0, 0, self.periodx / 2], #phantom
            [1, 0, 0, -self.periodx / 2.],#phantom
            [0, 1, 0, self.periody / 2.],#phantom
            [0, 1, 0, -self.periody / 2.],#phantom
            [0, 0, 1, self.thickness / 2],#phantom
            [0, 0, 1, -self.thickness / 2]],#phantom
            dtype=np.double)

        self.boundary_map = np.array([0, 1, 2, 3, 4, 5])
        self.boundary_map_operation = np.array([[0, 0, 0],#real
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]])

if __name__ == "__main__":
    from structure import random_emit, random_emit_boundary, boundary_collision

    nanomesh_order = OrderedNanoMesh(thickness=150, # Thickness (nm)
                                              holex=160,
                                              holey=160,
                                              periodx=600,
                                              periody=600,
                                              tolerance=1e-8,
                                              specularity=0)
    bulk = Bulk(thickness=400000, # Thickness (nm)
                         periodx=4000000,
                         periody=4000000, # reduce the number of phantom boundaries during transport
                         tolerance=1e-6)

    nanomesh_stagger = StaggeredNanoMesh(thickness=150, # Thickness (nm)
                                         holex=160,
                                         holey=160,
                                         periodx=1200, # The period is double the value of ordered case
                                         periody=600,
                                         tolerance=1e-8,
                                         specularity=0)
    thin_film = ThinFilm(thickness=150, # Thickness (nm)
                         periodx=600000,
                         periody=600000, # reduce the number of phantom boundaries during transport
                         tolerance=1e-6,
                         specularity=0)

    nanowire = NanoWire(thickness=150, # Thickness (nm)
                        periodx=60000000,# reduce the number of phantom boundaries during transport
                        periody=150,
                        tolerance=1e-6,
                        specularity=0)
    nanomesh = nanowire
    nanomesh.set_specularity(0)
    nphonon = 1000
    pos = nanomesh.random_position_boundary(nphonon, 0)
    arrow = random_emit_boundary(nphonon, nanomesh.boundaries[0])
    fig = plt.figure(figsize=plt.figaspect(1.5)*1.5)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(arrow[:,0], arrow[:,1], arrow[:,2])
    plt.show()

