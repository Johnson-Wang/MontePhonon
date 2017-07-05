1. The program needs only the MFP cumulative contribution to thermal conductivity based on the paper  [F. Yang and C. Dames, Phys. Rev. B 87, 035437 (2013)].
2. Prerequisite installed program and packages:
    a. python 2.7 and above
    b. python packages (numpy, matplotlib)
3. files:
    mc3d.py -- Main entrance
    structure.py -- The base class of the structures possessing most common features (thickness, period and phonon bouncing functions)
    stdlib.py -- structure library for different structures inherited from the base class. The functions to determine whether a point lies inside the structure are defined in each class
    phonons.py -- a class named Phonons is defined. 
4. usage
e.g. 
    a. Run the simulation for the Staggered case
      python mc3d.py --seg=300 --nph=10000 --structure=Stagger --px=1200 --py=600 --pz=150 --hx=160 --hy=160  ./data/M_K-T300-s0.0-sum.dat
    b. Run the simulation for the Ordered case
      python mc3d.py --seg=300 --nph=10000 --structure=Ordered --px=600 --py=600 --pz=150 --hx=160 --hy=160  ./data/M_K-T300-s0.0-sum.dat
    c. Help on the arguments and parameters
      python mc3d.py -h

5. Run on MEZ503
   a. Download and install the software "security shell client"
   b. Setup the machine MEZ503
     host: mez503.ust.h
     usrname: xwangan
     password: xwangan
   c. go the the directory by runing command "cd /home/xwangan/file/nanomesh/"
   d. run the commands shown in #4

    