#coding:utf-8
# __author__ = 'xwangan'
import numpy as np
from structure import NanoMeshStructure, random_emit, random_emit_boundary
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class PhononCollisions():
    def __init__(self,
                 mfp =None,
                 structure=NanoMeshStructure,
                 steps=1,
                 num = 1000,
                 trace_plot=False):
        self.mfp = mfp
        self.structure= structure
        self.steps = steps
        self.step = 0
        self.num = num
        self.n = 0
        self.is_trace_plot = trace_plot
        self.allocate_values()
        self.generate_initial_positions()

    def allocate_values(self):
        self.free_paths = np.zeros((self.steps, self.num), dtype=np.double)
        self.free_path_vectors = np.zeros((self.steps, self.num, 3), dtype=np.double)
        self.arrows = np.zeros_like(self.free_path_vectors)
        self.positions = np.zeros_like(self.arrows)

    def generate_initial_positions(self):
        self.pos = self.structure.random_position(self.num)
        self.arrow = random_emit(self.num)
        self.pos_init = self.pos.copy()
        self.arrow_init = self.arrow.copy()

    def __iter__(self):
        return self

    def next(self):
        if self.step == self.steps:
            # fig = self.structure.fig
            # ax = fig.gca()
            # ax.scatter(self.pos[::10,0], self.pos[::10,1])
            # fig.show()
            raise StopIteration
        self.positions[self.step] = self.pos.copy()
        self.arrows[self.step] = self.arrow.copy()
        #starting next iteration
        mfp_3ph = - np.log(np.random.rand(self.num)) * self.mfp
        # mfp_3ph = np.ones(self.num, dtype='double') * self.mfp
        self.mfp_3ph = mfp_3ph
        self.distance = self.free_paths[self.step]
        self.distvector = self.free_path_vectors[self.step]
        try:
            self.find_next_diffuse_collision()
        except AssertionError: # in case of unpredicted errors, run the step again
            self.free_paths[self.step] = 0
            self.free_path_vectors[self.step] = 0
            self.pos[:] = self.positions[self.step].copy()
            self.arrow[:] = self.arrows[self.step].copy()
            return self
        self.step += 1
        return self

    def find_next_diffuse_collision(self, index_to_calculate=None):
        if index_to_calculate is None:
            index_to_calculate = slice(None)
            index_original = np.arange(len(self.pos))
        else:
            index_original = index_to_calculate
        structure = self.structure
        distance = self.distance[index_to_calculate]
        mfp_3ph = self.mfp_3ph[index_to_calculate]
        arrows = self.arrow[index_to_calculate]
        points = self.pos[index_to_calculate]
        distance_bound, iBoundary, intersects = structure.find_next_boundary(points, arrows)
        points_next, arrows_next, is_diffuse_boundary = structure.new_emit_from_boundary(iBoundary, intersects, arrows)

        distance[:] += distance_bound
        distvector = distance_bound[:,np.newaxis] * arrows
        where_3ph = np.where(distance > mfp_3ph + 100 * structure.tolerance)

        distvector[where_3ph] -= (distance[where_3ph] - mfp_3ph[where_3ph])[:, np.newaxis] * arrows[where_3ph]

        distance[where_3ph] = mfp_3ph[where_3ph]
        points_next[where_3ph] = points[where_3ph] + distvector[where_3ph]
        arrows_next[where_3ph] = random_emit(len(where_3ph[0]))

        self.distance[index_to_calculate] = distance
        self.arrow[index_to_calculate] = arrows_next
        self.pos[index_to_calculate] = points_next
        self.distvector[index_to_calculate] += distvector
        #Assign to the original values container

        is_diffuse_boundary[where_3ph] = True
        where_phantom = np.where(is_diffuse_boundary == False)[0]
        if len(where_phantom) != 0:
            # self.nhits[index_original[where_phantom]]
            ip = index_original[where_phantom] # index of phantom boundaries
            is_diffuse_boundary_iter = self.find_next_diffuse_collision(ip)
            is_diffuse_boundary[where_phantom] = is_diffuse_boundary_iter
        assert (structure.pos_irt_boundary(points_next) != -1).all()
        return is_diffuse_boundary

    def plot_trace(self, info="", is_3D=False):
        if not self.is_trace_plot:
            return
        if self.structure.fig is None:
            ax = self.structure.plot_boundaries(is_3D = is_3D)
        else:
            ax = self.structure.fig.gca()
        if not is_3D:
            scatter = ax.scatter(self.pos[0], self.pos[1], s=50)
            arrow = ax.arrow(self.pos[0],
                       self.pos[1],
                       self.arrow[0]*100,
                       self.arrow[1]*100,
                       head_width=10,
                       head_length=20,
                       fc='k', ec='k')
        else:
            scatter = ax.scatter3D(self.pos[0], self.pos[1], self.pos[2], s=50)


        if info != "":
            txt = ax.text(self.pos[0], self.pos[1], info, fontsize=15)
        ax.figure.canvas.draw()
        scatter.remove()
        if info != "":
            txt.remove()
        arrow.remove()

class RayTracing():
    def __init__(self,
                 mfp =None,
                 structure=NanoMeshStructure,
                 nperiod=10,
                 num = 1000,
                 emit_boundary=0,
                 steps=800, # maximum collision steps allowed. (Those uncollected phonons are treated as uncertainty)
                 trace_plot=False):
        self.mfp = mfp
        self.structure= structure
        self.num = num
        self.nperiod = nperiod
        self.bemit = emit_boundary
        self.steps=steps
        self.is_trace_plot = trace_plot

    def allocate_values(self):
        self.distance = np.zeros(self.num, dtype=np.double)
        self.distvector = np.zeros((self.num, 3), dtype=np.double)
        self.nhits = np.zeros(self.num, dtype=np.int)
        # mfp_3ph = - self.mfp * np.log(np.random.rand(self.num))
        mfp_3ph = np.ones(self.num, dtype='double') * self.mfp
        self.mfp_3ph = mfp_3ph
        # self.init_pos = self.structure.random_position_boundary(self.num, 0)
        # self.init_arrow = random_emit_boundary(self.num, self.structure.boundaries[0])
        # self.pos = self.init_pos.copy()
        # self.arrow = self.init_arrow.copy()

    def init_positions(self):
        self.allocate_values()
        self.init_pos = self.structure.random_position_boundary(self.num, 0)
        self.init_arrow = random_emit_boundary(self.num, self.structure.boundaries[0])
        self.pos = self.init_pos.copy()
        self.arrow = self.init_arrow.copy()

    def plot_trace(self, info="", is_3D=False):
        if not self.is_trace_plot:
            return
        if self.structure.fig is None:
            ax = self.structure.plot_boundaries(is_3D = is_3D)
        else:
            ax = self.structure.fig.gca()
        if not is_3D:
            scatter = ax.scatter(self.pos[:,0], self.pos[:,1], s=50)
            arrow = ax.arrow(self.pos[:,0],
                       self.pos[:,1],
                       self.arrow[0]*100,
                       self.arrow[1]*100,
                       head_width=10,
                       head_length=20,
                       fc='k', ec='k')
        else:
            print self.pos
            scatter = ax.scatter3D(self.pos[:,0], self.pos[:,1], self.pos[:,2], s=50)
            quiver = ax.quiver(self.pos[:,0], self.pos[:,1], self.pos[:,2],
                               self.arrow[:, 0], self.arrow[:,1], self.arrow[:,2])


        if info != "":
            txt = ax.text(self.pos[0], self.pos[1], info, fontsize=15)
        ax.figure.canvas.draw()
        plt.show(False)
        plt.pause(0.5)
        scatter.remove()
        quiver.remove()


    def run(self):
        index_to_calculate = np.arange(len(self.pos))
        while True:
            self.plot_trace(is_3D=True)
            structure = self.structure
            distance = self.distance[index_to_calculate]
            mfp_3ph = self.mfp_3ph[index_to_calculate]
            arrows = self.arrow[index_to_calculate]
            points = self.pos[index_to_calculate]
            try:
                distance_bound, iBoundary, intersects = structure.find_next_boundary(points, arrows)
            except AssertionError:
                self.distance[index_to_calculate_last] = distance_last
                self.mfp_3ph[index_to_calculate_last] = mfp_3ph_last
                self.arrow[index_to_calculate_last] = arrows_last
                self.pos[index_to_calculate_last] = points_last
                self.nhits[index_to_calculate_last] = nhit_last
                index_to_calculate = index_to_calculate_last
                continue
            # setting a milestone in case of recalculation
            distance_last = distance.copy()
            mfp_3ph_last = mfp_3ph.copy()
            arrows_last = arrows.copy()
            points_last = points.copy()
            index_to_calculate_last = index_to_calculate.copy()
            nhit_last = self.nhits[index_to_calculate].copy()
            points_next, arrows_next, is_diffuse_boundary = structure.new_emit_from_boundary(iBoundary, intersects, arrows)
            distance[:] += distance_bound
            distvector = distance_bound[:,np.newaxis] * arrows
            where_3ph = np.where(distance > mfp_3ph + 100 * structure.tolerance)

            distvector[where_3ph] -= (distance[where_3ph] - mfp_3ph[where_3ph])[:, np.newaxis] * arrows[where_3ph]
            distance[where_3ph] = mfp_3ph[where_3ph]
            points_next[where_3ph] = points[where_3ph] + distvector[where_3ph]
            arrows_next[where_3ph] = random_emit(len(where_3ph[0]))

            is_diffuse_boundary[where_3ph] = True
            distance[np.where(is_diffuse_boundary)] = 0
            mfp_3ph[np.where(is_diffuse_boundary)] = self.mfp #* (- np.log(np.random.rand(is_diffuse_boundary.sum()))) #modified
            self.distance[index_to_calculate] = distance
            self.arrow[index_to_calculate] = arrows_next
            self.pos[index_to_calculate] = points_next
            self.distvector[index_to_calculate] += distvector
            self.mfp_3ph[index_to_calculate] = mfp_3ph
            #Assign to the original values container
            if ((self.nhits[index_to_calculate]==-1).__or__(self.nhits[index_to_calculate]==self.nperiod)).any():
                print
            self.nhits[index_to_calculate[np.where(iBoundary == 0)]] -= 1
            self.nhits[index_to_calculate[np.where(iBoundary == 1)]] += 1

            # for ph, is_diffuse in enumerate(is_diffuse_boundary):
            #     if not is_diffuse:
            #         i, j = index_to_calculate[ph], iBoundary[ph]
            #         if j == 0:
            #             self.nhits[i] -= 1
            #         elif j == 1:
            #             self.nhits[i] += 1
            if (np.abs(self.nhits) > 1).any():
                print
            new_indx = np.where((self.nhits != -1).__and__(self.nhits != self.nperiod))[0]
            if len(new_indx) == 0 or self.steps == 0:
                break
            self.steps -= 1
            index_to_calculate = new_indx
            assert (structure.pos_irt_boundary(points_next) != -1).all()

if __name__ == "__main__":
    from strlib import ThinFilm, NanoWire, OrderedNanoMesh, StaggeredNanoMesh, Cube
    mode = "ray_tracing"

    if mode == "ray_tracing":
        # nanomesh = OrderedNanoMesh(thickness=366, # Thickness (nm)
        #                            periodx=1100, # The period is double the value of ordered case
        #                            periody=1100, # The period is double the value of ordered case
        #                            holex=1100-250,
        #                            holey=1100-250,
        #                            tolerance=1e-8,
        #                            specularity=0)
        nanomesh = NanoWire(thickness=1, # Thickness (nm)
                               periodx=20, # The period is double the value of ordered case
                               periody=1, # The period is double the value of ordered case
                               tolerance=1e-8,
                               specularity=0)
        nanomesh.plot_boundaries()
        collisions = RayTracing(mfp=1e6, structure=nanomesh, nperiod=1, steps=2000, num=1, trace_plot=True)
        collisions.init_positions()
        # plt.figure()
        # plt.hist(collisions.pos[:, 1], 100, alpha=0.75)
        # plt.show()
        collisions.run()
        nback = (collisions.nhits==-1)
        npass = (collisions.nhits==collisions.nperiod)
        # plt.figure()
        # plt.hist(collisions.pos[np.where(nback)[0], 1], 100, alpha=0.75)
        # plt.show()
        # plt.figure()
        # plt.hist(collisions.pos[np.where(npass)[0], 1], 50, alpha=0.75)
        # plt.show()


        uncollected = (collisions.nhits!=collisions.nperiod).__and__(collisions.nhits!=-1)
        L = collisions.nperiod * nanomesh.periodx
        lam_nano = np.average(collisions.init_arrow[:,0] * npass) * L * 3./2.
        npass = (collisions.nhits!=-1)
        lam_nano2 = np.average(collisions.init_arrow[:,0] * npass) * L * 3./2.
        print "L: %.2f, l_nano: %15.5f +/-%-15.5f" %(L, (lam_nano+lam_nano2)/2, np.abs(lam_nano - lam_nano2) / 2)

    elif mode == "mfp_count":
        nanomesh = Cube(thickness=10, # Thickness (nm)
                            periodx=1000, # The period is double the value of ordered case
                            periody=10, # The period is double the value of ordered case
                            tolerance=1e-8,
                            specularity=0)
        # nanomesh = OrderedNanoMesh(thickness=60, # Thickness (nm)
        #                            periodx=50, # The period is double the value of ordered case
        #                            periody=100, # The period is double the value of ordered case
        #                            holex=50-45,
        #                            holey=100-45,
        #                            tolerance=1e-8,
        #                            specularity=0)
        # nanomesh = StaggeredNanoMesh(thickness=55, # Thickness (nm)
        #                              periodx=200, # The period is double the value of ordered case
        #                              periody=100, # The period is double the value of ordered case
        #                              holex=55,
        #                              holey=55,
        #                              tolerance=1e-8,
        #                              specularity=0)
        nanomesh.plot_boundaries()
        phononcoll = PhononCollisions(mfp=4e6, structure=nanomesh, num=200000)
        for c in phononcoll:
            pass
        lam_nano = 3 * np.average(phononcoll.arrow_init * phononcoll.free_path_vectors, axis=(0,1))
        print lam_nano


        # print 3./4. * (np.log(2) + 1./3. + 1./2.) * (nanomesh.periodx - nanomesh.holex)
        # Kn = phononcoll.mfp / nanomesh.thickness
        # def ratio_inplane(Kn, Num=17751):
        #     "Both inplane and outplane are derived from Fuchs 1937"
        #     thetas = np.linspace(0, np.pi, num=Num, endpoint=True)
        #     abscos = np.abs(np.cos(thetas))
        #     sin = np.sin(thetas)
        #     term1 = sin ** 3 * abscos * Kn
        #     term2 = 1 - np.exp(-1 / abscos / Kn)
        #     integration = np.pi * np.average(term1 * term2)
        #     return 1 - 3. / 4. * integration
        # def ratio_outplane(Kn, Num=17751):
        #     thetas = np.linspace(0, np.pi, num=Num, endpoint=True)
        #     abscos = np.abs(np.cos(thetas))
        #     sin = np.sin(thetas)
        #     term1 = sin * abscos ** 3 * Kn
        #     term2 = 1 - np.exp(-1 / abscos / Kn)
        #     integration = np.pi * np.average(term1 * term2)
        #     return 1 - 3. / 2. * integration
        # print ratio_inplane(Kn) * phononcoll.mfp, ratio_outplane(Kn) * phononcoll.mfp
