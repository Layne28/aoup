import numpy as np
import argparse
import os

def main():

    #Get optional arguments
    parser = argparse.ArgumentParser(description='Compute an active Ornstein-Uhlenbeck particle trajectory')
    parser.add_argument('--D0', default=1.0, help='Diffusion constant')
    parser.add_argument('--tau', default=2.0, help='Persistence time')
    parser.add_argument('--seed', default=123, help='Random seed')
    parser.add_argument('--nsteps', default=100000, help='Number of timesteps')
    parser.add_argument('--dt', default=1e-3, help='Timestep')
    parser.add_argument('--output_dir', default='data', help='Folder to output trajectories')
    parser.add_argument('--do_random_initial_velocity', default=0, help='1 for random initial velocity, otherwise points in z direction')
    args = parser.parse_args()

    #Set parameters
    D0 = float(args.D0)
    tau = float(args.tau)
    seed = int(args.seed)
    nsteps = int(args.nsteps)
    dt = float(args.dt)
    output_dir = args.output_dir
    do_random_initial_velocity = int(args.do_random_initial_velocity)

    #Make arrays for storing trajectory
    pos = np.zeros((nsteps, 3))
    self_prop_vel = np.zeros((nsteps, 3))
    disp = np.zeros((nsteps, 3)) #displacement at each timestep

    #Create arrays for instantaneous values of position (r) and self-propulsion (p)
    r = np.zeros(3)
    p = np.array([0.0,0.0,np.sqrt(D0/tau)])
    if do_random_initial_velocity==1:
        p = np.random.normal(loc=0.0, scale=np.sqrt(D0/tau), size=3)

    #Propagate dynamics
    for n in range(nsteps):
        pos[n,:] = r
        self_prop_vel[n,:] = p
        r += dt*p #update position
        p += -(dt/tau)*p + (np.sqrt(2*D0)/tau)*np.random.normal(loc=0.0, scale=np.sqrt(dt), size=3) #update self-propulsion
        disp[n,:] = r-pos[n,:]

    #Save trajectory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(output_dir + '/traj_seed=%d.txt' % seed, np.c_[np.linspace(0,(nsteps-1)*dt, nsteps), pos, disp], 
               header = 'D0=%f, tau=%f, dt=%f\nColumns: time, x, y, z, dx, dy, dz' % (D0, tau, dt))

if __name__=='__main__':
    main()