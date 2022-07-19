import numpy as np
from numpy import genfromtxt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt


class lower_convex_hull:

    # points: input points -- 2D only!
    def __init__(self,points):
        self.hull = ConvexHull(points)  # hull object
        self.boundaries = points[self.hull.vertices] # the points at convex hull


    def get_lower_hull(self):
        # find the min x in self.boundaries
        # then go one round, scipy always gives
        # anti-clockwise direction

        xmin_idx = np.argmin(self.boundaries,axis=0)[0]
        lower_hull = []
        lower_hull.append(self.boundaries[xmin_idx,:])

        cur_idx = xmin_idx
        pre_x  =  lower_hull[0][0]
        for i in range(len(self.boundaries)):
            cur_idx = (cur_idx + 1)%len(self.boundaries)
            cur_x = self.boundaries[cur_idx][0]
            if cur_x < pre_x: break
            pre_x = cur_x
            lower_hull.append(self.boundaries[cur_idx])

        return np.stack(lower_hull,axis=0)



if __name__=='__main__':

    # use below line if data needs to put read in from file
    #my_data = genfromtxt('t.dat', delimiter=' ')
    number_of_points= 32
    np.random.seed(331)
    points = np.random.random((2048, 2)) 
    lhull = lower_convex_hull(points) # create the convex hull object
    lower_hull = lhull.get_lower_hull()
 
    # here, plot the whole convex hull
    plt.plot(points[:,0], points[:,1], 'o')
    for simplex in lhull.hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
 
    plt.savefig('hull.png')

    # clear figure and then plot the lower hull only
    plt.figure().clear()
    plt.scatter(lower_hull[:, 0], lower_hull[:, 1], cmap='hot')
    #plt.plot(lower_hull)
    plt.savefig('lower.png')








