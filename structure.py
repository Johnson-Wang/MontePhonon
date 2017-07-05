__author__ = 'xwangan'
import numpy as np
# np.random.seed(0) # for debug
np.seterr(divide="ignore")
import matplotlib.pyplot as plt

def get_distance_from_boundaries(points, arrows, boundaries, tolerance=1e-8):
    """Get distance with all boundaries according to the equation
    L = F(P) / dot(n, l) * norm(l)
    P2 = P + F(P) / dot(n,l) * l
    inputs:
        points [num_phonons, 3]: points of the emitting phonons
        arrows [num_phonons, 3]: directions of the emitting phonns
        boundaries [num_boundaries, 4]: the parameters (a,b,c,d) of a boundary function F = ax + by + cz + d
        tolerance: tolerance for checking if a point is already on the boundary
    """
    assert (np.abs(np.sum(arrows ** 2, axis=-1) - 1) < 1e-5).all() # direction is a unit vetor
    points4 = np.hstack((points, np.ones(len(points))[:,np.newaxis]))
    # make the points the form of [x, y, z, 1] for the convenience of dot product with the boundary parameters
    fp = - np.dot(boundaries, points4.T)
    #the value of F(x,y,z), with shape of [num_boundaries, num_points]
    # The negative sign denotes a points lies the other side of the normal vector [a,b,c] of the face
    fp = np.where(np.abs(fp) < tolerance, 0, fp)
    # Force a point very close to a boundary to be on the boundary
    distance = fp / np.dot(boundaries[:,:-1], arrows.T) # only positive distance is acceptable
    # The negative sign in fp denotes a phonon is pointing to the boundary
    intersect = distance[:, :, np.newaxis] * arrows[np.newaxis] + points
    # intersect point
    return distance, intersect

def random_emit(N=1, cone_axis=2):
    #choose x-axis as the cone axis
    # theta: azimuthal angle on xy plane (0-2pi)
    # phi: polar angle with the positive z axis (0-pi)
    #Refer to http://mathworld.wolfram.com/SpherePointPicking.html for the random point picking problem on a sphere
    if cone_axis is not None:
        theta, v = np.random.rand(2, N)
        theta *= 2 * np.pi
        cosphi = 2 * v - 1
        sinphi = np.sqrt(1 - cosphi ** 2)
        if cone_axis == 2: # z-axis
            angle = np.array([np.cos(theta) * sinphi, np.sin(theta) * sinphi, cosphi]).T
        elif cone_axis == 1: # y-axis
            angle = np.array([np.sin(theta) * sinphi, cosphi, np.cos(theta) * sinphi]).T
        elif cone_axis == 0:
            angle = np.array([cosphi, np.cos(theta) * sinphi, np.sin(theta) * sinphi]).T
        else:
            return
    else:
        angle = np.random.randn(3, N)
        angle = angle / np.sqrt(np.sum(angle ** 2, axis=0))
        angle = angle.T
    return angle

def random_emit_half_space(N=1, cone_axis=2):
    #choose x-axis as the cone axis
    # theta: azimuthal angle on xy plane (0-2pi)
    # phi: polar angle with the positive z axis (0-pi)
    #Refer to http://mathworld.wolfram.com/SpherePointPicking.html for the random point picking problem on a sphere
    if cone_axis is not None:
        theta, cosphi = np.random.rand(2, N)
        theta *= 2 * np.pi
        sinphi = np.sqrt(1 - cosphi ** 2)
        if cone_axis == 2: # z-axis
            angle = np.array([np.cos(theta) * sinphi, np.sin(theta) * sinphi, cosphi]).T
        elif cone_axis == 1: # y-axis
            angle = np.array([np.sin(theta) * sinphi, cosphi, np.cos(theta) * sinphi]).T
        elif cone_axis == 0:
            angle = np.array([cosphi, np.cos(theta) * sinphi, np.sin(theta) * sinphi]).T
        else:
            return
    else:
        angle = np.random.randn(3, N)
        angle = angle / np.sqrt(np.sum(angle ** 2, axis=0))
        angle = angle.T
    return angle

def random_emit_boundary(N, boundary):
    """N: number of phonons
    boundary: boundary function [a, b, c, d]
    """
    cone = np.where(boundary[:3] != 0)[0][0]
    angles = random_emit_half_space(N, cone_axis=cone)
    return angles

def boundary_reflect_specular(boundaries, incident_arrows):
    """boundaries: the boundaries that each phonon collide with, shape: [num_phonon, 4]
    incident_arrows: the incident arrows of each phonon, shape: [num_phonon, 3]
    """
    a,b,c = boundaries[:, :3].T
    reflection = np.array([[1 - 2 * a ** 2, -2 * a * b, -2 * a * c],
                           [-2 * a * b, 1 - 2 * b ** 2, -2 * b * c],
                           [-2 * a * c, -2 * b * c, 1 - 2 * c ** 2]])
    # reflection is a stacked matrix with shape [3, 3, num_phonon]
    return np.einsum("ijn, nj->ni", reflection, incident_arrows)

def boundary_reflect_diffuse(boundaries, incident_arrows):
    """boundaries: the boundaries that each phonon collide with, shape: [num_phonon, 4]
    incident_arrows: the incident arrows of each phonon, shape: [num_phonon, 3]
    """
    num_phonon = len(incident_arrows)
    angle_diffuse = random_emit(num_phonon, 0)
    # check whether the incident arrow is in the same direction with the emission angle i.r.t. the surface
    in_face_collinear = (np.einsum('ni, ni->n', incident_arrows, boundaries[:,:3]) > 0)
    out_face_collinear = (np.einsum("ni, ni->n", angle_diffuse, boundaries[:,:3]) > 0)

    in_out_collinear = np.where(in_face_collinear == out_face_collinear)
    if len(in_out_collinear[0]) > 0:
        angle_diffuse[in_out_collinear] = - angle_diffuse[in_out_collinear]
    return angle_diffuse

def boundary_reflect(boundaries, incident_arrows, specularity=0):
    """boundaries: the boundaries that each phonon collide with, shape: [num_phonon, 4]
    incident_arrows: the incident arrows of each phonon, shape: [num_phonon, 3]
    """
    num_phonon = len(incident_arrows)
    reflect_arrows = np.zeros_like(incident_arrows)
    if specularity == 0:
        reflect_arrows[:] = boundary_reflect_diffuse(boundaries, incident_arrows) ##diffuse case
        is_diffuse = np.ones(len(incident_arrows), dtype=bool)
    elif specularity == 1:
        reflect_arrows[:] = boundary_reflect_specular(boundaries, incident_arrows) ##specular case
        is_diffuse = np.zeros(len(incident_arrows), dtype=bool)
    else:
        randn = np.random.rand(num_phonon)
        is_diffuse = (randn > specularity)
        where_diffuse = np.where(is_diffuse)
        where_specular = np.where(is_diffuse == False)
        reflect_arrows[where_diffuse] = boundary_reflect_diffuse(boundaries[where_diffuse], incident_arrows[where_diffuse]) ##diffuse case
        reflect_arrows[where_specular] = boundary_reflect_specular(boundaries[where_specular], incident_arrows[where_specular]) ##diffuse case
    return reflect_arrows, is_diffuse

def boundary_transmit_diffuse(boundaries, incident_arrows):
    """boundaries: the boundaries that each phonon collide with, shape: [num_phonon, 4]
    incident_arrows: the incident arrows of each phonon, shape: [num_phonon, 3]
    """
    num_phonon = len(incident_arrows)
    angle_diffuse = random_emit(num_phonon)
    # check whether the incident arrow is in the same direction with the emission angle i.r.t. the surface
    in_face_collinear = (np.einsum('ni, ni->n', incident_arrows, boundaries[:,:3]) > 0)
    out_face_collinear = (np.einsum("ni, ni->n", angle_diffuse, boundaries[:,:3]) > 0)
    in_out_not_collinear = np.where(in_face_collinear != out_face_collinear)
    if len(in_out_not_collinear[0] > 0):
        angle_diffuse[in_out_not_collinear] = - angle_diffuse[in_out_not_collinear]
    return angle_diffuse

def boundary_transmit_specular(boundaries, incident_arrows):
    """boundaries: the boundaries that each phonon collide with, shape: [num_phonon, 4]
    incident_arrows: the incident arrows of each phonon, shape: [num_phonon, 3]
    """
    return incident_arrows

def boundary_transmit(boundaries, incident_arrows, specularity=1):
    """boundaries: the boundaries that each phonon collide with, shape: [num_phonon, 4]
    incident_arrows: the incident arrows of each phonon, shape: [num_phonon, 3]
    """
    num_phonon = len(incident_arrows)
    tansmit_arrows = np.zeros_like(incident_arrows)
    if specularity == 0:
        tansmit_arrows[:] = boundary_transmit_diffuse(boundaries, incident_arrows) ##diffuse case
        is_diffuse = np.ones(len(incident_arrows), dtype=bool)
    elif specularity == 1:
        tansmit_arrows[:] = boundary_transmit_specular(boundaries, incident_arrows) ##specular case
        is_diffuse = np.zeros(len(incident_arrows), dtype=bool)
    else:
        randn = np.random.rand(num_phonon)
        is_diffuse = (randn > specularity)
        if not len(is_diffuse) == 0:
            where_diffuse = np.where(is_diffuse)
            where_specular = np.where(is_diffuse == False)
            tansmit_arrows[where_diffuse] = boundary_transmit_diffuse(boundaries[where_diffuse], incident_arrows[where_diffuse]) ##diffuse case
            tansmit_arrows[where_specular] = boundary_transmit_specular(boundaries[where_specular], incident_arrows[where_specular]) ##diffuse case
    return tansmit_arrows, is_diffuse

def boundary_collision(boundaries, incident_arrows, behaviors=None, specularity = 1):
    """Boundary collisions given the boundary functions and  incident arrows
            Inputs:
                boundaries [num_phonons, 4]: the boundaries that each phonon collide
                incident_arrows [num_phonons, 3]: the incident arrows of each phonon
                behaviors [num_phonons]: showing whether boundaries behave as reflection (0), transmission (1) or periodic (2).
            Outputs:
                reemit_angles [num_phonons, 3]: reemitting direction of phonons after colliding with the boundary
                is_diffuse_boundary [num_phonons]: True if a diffuse scattering happens (only a diffuse scattering is considered as a real scattering)
    """
    reemit_angles = np.zeros_like(incident_arrows)
    is_diffuse_boundary = np.zeros(len(incident_arrows), dtype=bool)
    if behaviors is None:
        # Default: refelection with the given specularity
        reemit_angles[:], is_diffuse_boundary[:] = boundary_reflect(boundaries,
                                                                    incident_arrows,
                                                                    specularity=specularity)
    else:
        compos_transmit = np.where(behaviors == 1)
        compos_reflect = np.where(behaviors == 0)
        compos_periodic = np.where(behaviors == 2)
        reemit_angles[compos_periodic],is_diffuse_boundary[compos_periodic] =\
            boundary_transmit(boundaries[compos_periodic],
                              incident_arrows[compos_periodic],
                              specularity=1)# periodic boundaries

        reemit_angles[compos_transmit], is_diffuse_boundary[compos_transmit] =\
            boundary_transmit(boundaries[compos_transmit],
                              incident_arrows[compos_transmit],
                              specularity=specularity) # transmitting boundaries

        reemit_angles[compos_reflect], is_diffuse_boundary[compos_reflect] =\
            boundary_reflect(boundaries[compos_reflect],
                             incident_arrows[compos_reflect],
                             specularity=specularity) # reflecting boundaries

    return reemit_angles, is_diffuse_boundary

class Box():
    def __init__(self, boundaries, is_entity=True): # boundaries: tuple (x0, x1, y0, y1, z0, z1)
        self. is_entity = is_entity
        self.x0, self.x1, self.y0, self.y1, self.z0, self.z1 = boundaries[0]

    def intersection_point(self, point, direction): # point:[x,y,z], direction[vx, vy, vz]
        pass


class Boundary():
    def __init__(self, parameters = None, is_transmit=False, specularity=0):
        self.parameters = parameters
        self.is_transmit = is_transmit
        if is_transmit:
            self.specularity = 1
        else:
            self.specularity = specularity

    def set_map(self, boundary2, map_vector):
        self.map = boundary2
        self.map_vector = map_vector


class NanoMeshStructure():
    def __init__(self, thickness=None, holex=None, holey=None, periodx=None, periody=None, specularity=0, tolerance=1e-8):
        self.thickness = np.double(thickness)
        self.holex = np.double(holex)
        self.holey = np.double(holey)
        self.periodx = np.double(periodx)
        self.periody = np.double(periody)
        self.specularity = specularity
        self.fig = None
        self.boundary_map = None
        self.boundaries = None
        self.boundary_map_operation = None
        self.tolerance=tolerance
        self.info = ""
        self.construct_boundaries()
        self.correction_factor = 1.0

    def construct_boundaries(self):
        "defined in the descendant class; functionx f = a * x + b * y + c * z + d, each row in the self.boundaries represent a tuple of (a, b , c, d)"
        pass

    def plot_boundaries(self, is_3D=False):
        "plot the boundaries using pyplot, defined in the descendant class"
        pass

    def set_specularity(self, specularity):
        self.specularity = specularity

    def pos_irt_boundary(self, point):
        "judge the position of point in regards to the boundaries defined in the descendant class"
        pass

    def random_position(self, N):
        "Generate N random positions inside the structure"
        positions = (np.random.rand(N,3) - 0.5) * np.array([self.periodx, self.periody, self.thickness])
        inboundary = (self.pos_irt_boundary(positions) == 1)
        if not inboundary.all():
            # Reckon with the on/out-boundary cases
            unassigned = np.where(inboundary == False)
            positions[unassigned] = self.random_position(len(unassigned[0]))
            # Recursive function for only the unassigned phonons
            inboundary[unassigned] = (self.pos_irt_boundary(positions[unassigned]) == 1) # 0: on boundary; 1: in; -1: out
        return positions

    def random_position_boundary(self, N, boundary_index=0):
        "Generate N random positions inside the structure"
        if boundary_index == 0:
            positions = (np.random.rand(N,3) - 0.5) * np.array([0, self.periody, self.thickness])
            positions -= np.array([self.periodx / 2., 0, 0])
        elif boundary_index == 1:
            positions = (np.random.rand(N,3) - 0.5) * np.array([0, self.periody, self.thickness])
            positions += np.array([self.periodx / 2., 0, 0])
        if boundary_index == 2:
            positions = (np.random.rand(N,3) - 0.5) * np.array([self.periodx, 0, self.thickness])
            positions -= np.array([0, self.periody / 2., 0])
        elif boundary_index == 3:
            positions = (np.random.rand(N,3) - 0.5) * np.array([self.periodx, 0, self.thickness])
            positions += np.array([0, self.periody / 2., 0])
        if boundary_index == 4:
            positions = (np.random.rand(N,3) - 0.5) * np.array([self.periodx, self.periody, 0])
            positions -= np.array([0, 0, self.thickness / 2.])
        elif boundary_index == 5:
            positions = (np.random.rand(N,3) - 0.5) * np.array([self.periodx, self.periody, 0])
            positions += np.array([0, 0, self.thickness / 2.])
        return positions

    def find_next_boundary(self, points, arrows):
        """points: [num_phonons, 3] -- points where phonon are reemitted
                    arrows [num_phonons, 3] -- directions of the emitting phonons
                    return:
                        distance [num_phonons] -- distance a phonon travels to hit a boundary (regardless of real or phantom boundary)
                        points2 [num_phonons, 3] -- intersects of the reemitted phonon with the boundary
                        arrows_next [num_phonons, 3] -- direction of phonons after colliding with boundaries
                        is_diffuse_boundary [num_phonons] -- True if a phonon hit a real surface or another phonon, otherwise False
        """
        dists, intersects_allb = get_distance_from_boundaries(points, arrows, self.boundaries)
        for i in range(len(self.boundaries)):
            d = dists[i]
            d[np.where(d < self.tolerance)] = np.inf # the original emitting position
            d[np.where(self.pos_irt_boundary(intersects_allb[i]) != 0)] = np.inf # not on boundary
        distance = dists.min(axis=0)# the true distance with the boundaries (minimum)
        index_boundary = dists.argmin(axis=0)
        intersects = np.array([intersects_allb[a, i] for i, a in enumerate(index_boundary)], dtype=np.double)
        assert (distance != np.inf).all(), "Random error encountered, please run the program again"
        return distance, index_boundary, intersects

    def new_emit_from_boundary(self, index_boundary, intersects, orig_arrows):
        # points2 = np.array([intersects[a, i] for i, a in enumerate(index_boundary)], dtype=np.double)
        points2 = intersects
        # intersection points
        boundary_maps = self.boundary_map[index_boundary]
        points2 += self.boundary_map_operation[index_boundary] # for phantom boundaries, points are dragged back
        boundary_behaviors = np.where(boundary_maps == index_boundary, 0, 2) # 0: reflections, 2: periodic boundary conditions
        # for phonons hitting the periodic boundaries, several inner interations are needed before a real collision
        arrows_next, is_diffuse_boundary = boundary_collision(self.boundaries[boundary_maps],
                                                              orig_arrows,
                                                              behaviors=boundary_behaviors,
                                                              specularity=self.specularity)
        return points2, arrows_next, is_diffuse_boundary


