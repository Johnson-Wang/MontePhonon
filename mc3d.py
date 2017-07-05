#coding:utf-8
#!/usr/bin/env python
__author__ = 'xwangan'
import os, sys
import numpy as np
from phonons import PhononCollisions, RayTracing
from strlib import ThinFilm, Bulk, StaggeredNanoMesh, OrderedNanoMesh, NanoWire,Cube

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--seg", dest="seg", type="int", default="300",
                  help="Number of mean free path values segmented for integration")
parser.add_option("--ncol", dest="ncol", type="int", default=1,
                  help="Number of diffuse collisions each phonon experiences")
parser.add_option("--nph", dest="nph", type="int", default=10000,
                  help="Number of phonons randomly emitted from anywhere in the system")
parser.add_option("-p", action="store_true", dest="is_plot", default=False,
                  help="plot the boundaries")
parser.add_option("-m", action="store_true", dest="is_mute", default=False,
                  help="Muting mode (as little information as possible)")
parser.add_option("-r", "--ray", "--ray_trace", action="store_true", dest="is_ray", default=False,
                  help="Ray tracing (Calculating transmision coefficient)")
parser.add_option("--structure", dest="structure", type="string", default="",
                  help="Choose the structure from the available structure library")
parser.add_option("--px", dest="px", type="float", default=600,
                  help="Boundary stretch (pitch) in x axis (either real or phantom, depending on the system)")
parser.add_option("--py", dest="py", type="float", default=600,
                  help="Boundary stretch (pitch) in y axis (either real or phantom, depending on the system)")
parser.add_option("--pz", dest="pz", type="float", default=150,
                  help="Boundary stretch (pitch) in z axis (either real or phantom, depending on the system)")
parser.add_option("--hx", dest="hx", type="float", default=140,
                  help="hole size in x axis (real boundaries)")
parser.add_option("--hy", dest="hy", type="float", default=140,
                  help="hole size in y axis (real boundaries)")
parser.add_option("--sp", dest="specularity", type="float", default=0,
                  help="Specularity of each real boundary")
parser.add_option("--seed", dest="seed", type="int", default=10,
                  help="seed for generating random numbers (0 makes the results predictable)")
parser.add_option("--tolerance", dest="tolerance", type="float", default=1e-8,
                  help="Tolerance for checking if a point is on the boundary (nm)")
(options, args) = parser.parse_args()

if len(args) == 0:
    print "Please assign a filename for MFP contribution first!"
    sys.exit(1)
filename = args[0]
if not os.path.isfile(filename):
    print "File %s not found!" %filename
    sys.exit(1)

## Starting point of the program
seg = options.seg  # Number of mean free path values segmented for integration
nph = options.nph # Number of phonons of same mfp with random initial emission
specularity = options.specularity
ncol = options.ncol # Number of collisions each phonon experiences
is_ray = options.is_ray
if not options.is_mute:
    print "Number of randomised MFP values: %d" %seg
    print "Number of phonons of same mfp with random initial emission: %d" %nph
    print "Maximum number of collisions each phonon experiences: %d" %ncol
    print "Specularity: %5.2f" %specularity
    print "File of MFP cumulative contribution to thermal conductivity: %s" %filename
if is_ray:
    print "RAY TRACING mode triggered"
struct = options.structure.strip().lower()[0]

if struct == "s":
    nanomesh = StaggeredNanoMesh(thickness=options.pz, # Thickness (nm)
                                 holex=options.hx,
                                 holey=options.hy,
                                 periodx=options.px, # The period is double the value of ordered case
                                 periody=options.py,
                                 tolerance=options.tolerance,
                                 specularity=specularity)
elif struct == "o":
    nanomesh = OrderedNanoMesh(thickness=options.pz, # Thickness (nm)
                               holex=options.hx,
                               holey=options.hy,
                               periodx=options.px, # The period is double the value of ordered case
                               periody=options.py,
                               tolerance=options.tolerance,
                               specularity=specularity)
elif struct == "t":
    nanomesh = ThinFilm(thickness=options.pz, # Thickness (nm)
                        periodx=options.px, # The period is double the value of ordered case
                        periody=options.py,
                        tolerance=options.tolerance,
                        specularity=specularity)
elif struct == "b":
    nanomesh = Bulk(thickness=options.pz, # Thickness (nm)
                    periodx=options.px, # The period is double the value of ordered case
                    periody=options.py,
                    tolerance=options.tolerance)
elif struct == "c":
    nanomesh = Cube(thickness=options.pz, # Thickness (nm)
                    periodx=options.px, # The period is double the value of ordered case
                    periody=options.py,
                    tolerance=options.tolerance)
elif struct == "n":
    nanomesh = NanoWire(thickness=options.pz, # Thickness (nm)
                        periodx=options.px, # The period is double the value of ordered case
                        periody=options.py,
                        tolerance=options.tolerance,
                        specularity=specularity)
else:
    print "The nanomesh system can only be one of the following:"
    print "ThinFilm, Bulk, StaggeredNanoMesh, OrderedNanoMesh, NanoWire"
    sys.exit(1)

if options.is_plot:
    nanomesh.plot_boundaries()

mfpcount=np.fromfile(filename, sep=" ").reshape(-1,2)
#mfpcout has two columns, first one is mfp values (nm); second one is cumulative thermal conductivity
kappa = mfpcount[:,1].max()
mfpcount[:,1] /= kappa
# normalization of cumulative kappa to 1
mfp = np.zeros(seg, dtype=np.double)
# Build mfp values
randoms = np.random.rand(seg)
# building N random numbers from 0 to 1
for i in range(seg):
    rand = randoms[i]
    argmin = np.argmin(np.abs(mfpcount[:,1] - rand))
    # the mfp index corresponding to the given normalized cumulative kappa
    k = (mfpcount[argmin+1, 1] - mfpcount[argmin-1, 1]) / (mfpcount[argmin+1, 0] - mfpcount[argmin-1, 0])
    mfp[i] = (mfpcount[argmin, 0] +  (rand - mfpcount[argmin,1]) / k)
    # linear insertion
mfp = np.sort(mfp)
# randomize mfp with the probability distribution function determined by the mfp cumulative function
#the structure can be any of the predefined ones,
#including OrderedNanoMeshStructure, BulkStructure and StaggeredNanoMeshStructure
# defined in the strlib.py
factors = np.zeros((seg,3), dtype=np.double)
for i in np.arange(seg):
    if not options.is_mute:
        print "#%5d(/%-5d)----MFP: %10.5e"%(i, seg, mfp[i])
    collisions = PhononCollisions(mfp=mfp[i], structure=nanomesh, steps=ncol, num=nph, trace_plot=False)
    for p in collisions:
        pass # self iterator runs the collision processes. The process is located in PhononCollisions.next()
    factors[i] = 3 * np.average(collisions.arrow_init * collisions.free_path_vectors, axis=(0,1)) / mfp[i]
    factors[i] *= nanomesh.correction_factor
    # factors[i] = 1 / (1 + 4 * mfp[i] / 3. / nanoxmesh.thickness)
if is_ray:
    pass
else:
    print "Thermal conductivity in %s:" %nanomesh.info
    print "%10.6f %10.6f %10.6f" %tuple(np.average(factors, axis=0) * kappa)
    print "Percentage in regards to the bulk material:"
    print "%10.6f %10.6f %10.6f" %tuple(np.average(factors, axis=0))