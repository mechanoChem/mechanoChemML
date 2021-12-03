import numpy as np
from copy import deepcopy

# See, for example,  Robert Smith, "Efficient Monte Carlo Procedures for Generating Points Uniformly Distributed Over Bounded Regions,"
# Operations Research , Nov. - Dec., 1984, Vol. 32, No. 6 (Nov. - Dec., 1984), pp. 1296-1308 [https://www.jstor.org/stable/170949]

# For BilliardWalk, see B. T. Polyak & E. N. Gryazina, "Billiard walk - a new sampling algorithm for control and optimization"
# IFAC 2014, 978-3-902823-62-5/2014

def in_bounds(x,n_planes,c_planes):

    return np.all(n_planes.dot(x) <= c_planes)

def random_direction(dim):

    # Return a random direction in "dim" dimensions
    
    d = 2*np.random.rand(dim) - 1
    return d/np.linalg.norm(d)

def coordinate_direction(dim):

    # Return a randomly selected coordinate direction

    d = np.zeros(dim)
    d[np.random.randint(0,dim)] = 1
    return d

def intersection(n,c,x,d):

    # Return the distance t that produces an
    # intersection between the given
    # hyper-plane (defined by vector n and scalar c)
    # and line (defined by point x and vector d)

    if n.dot(d) == 0:
        print('problem')
        return None
    else:
        return (c - n.dot(x))/n.dot(d)

def nearest_intersections(n_planes,c_planes,x,d):

    # Find the two nearest intersections (in the positive
    # and negative directions) for the given line
    # (point x and direction d) and the set of planes

    neg_t = None
    pos_t = None
    neg_i = None
    pos_i = None
    count_zeros = 0
    for j in range(len(c_planes)):
        t = intersection(n_planes[j],c_planes[j],x,d)
        if t>1e-15:
            if (pos_t is None or t<pos_t):
                pos_t = t
                pos_i = j
        elif t<-1e-15:
            if (neg_t is None or t>neg_t):
                neg_t = t
                neg_i = j
        else:
            count_zeros += 1

    # If still "None" at this point, there weren't any planes in
    # the postive (or negative) direction

    # If count_zeros >= 2, then the point is at an intersection of
    # two or more planes

    return neg_t, pos_t, neg_i, pos_i, count_zeros

def inward_normal(x,n,c,xp):

    # Given an interior point x, an intersecting plane
    # defined by vector n and scalar c, and the intersecting
    # point xp, return the inward unit normal to the plane

    n = n/np.linalg.norm(n)
    l = xp - x # point outward from interior point

    if l.dot(n) > 0: # if cos(theta) > 0, i.e. theta < pi/2
        n *= -1

    return n
    
def hitandrun(x0,n_planes,c_planes,N_points):

    # Initialize (currently, assume given point is in domain) 
    x = x0
    dim = len(x0)

    points = np.zeros((N_points,dim))
    for i in range(N_points):

        # Get random direction
        d = random_direction(dim)
        #d = coordinate_direction(dim)
        
        # Find intersections with all planes; take the most restrictive
        neg_t, pos_t, neg_i, pos_i, count_zeros = nearest_intersections(n_planes,c_planes,x,d)
                
        # Take random point in the domain [neg_t,pos_t]
        x = x + (np.random.rand()*(pos_t - neg_t) + neg_t)*d
        points[i] = x
        
    return points

def billiardwalk(x0,n_planes,c_planes,N_points,tau):

    # Initialize (currently, assume given point is in domain) 
    x = x0
    xprev = x0
    dim = len(x0)
    R = 10*dim

    center = np.zeros(dim)
    center[0] = 0.5

    boundary = []
    points = np.zeros((N_points,dim))
    for i in range(N_points):

        while(True):
            # Generate trajectory length
            l = -tau*np.log(np.random.rand())
        
            # Get random direction
            d = random_direction(dim)

            # Do one step of the billiard walk
            for r in range(R):
        
                # Find intersection with closest plane
                neg_t, pos_t, neg_i, pos_i, count_zeros = nearest_intersections(n_planes,c_planes,x,d)
            
                # Check if the trajectory hits the intersection (only use positive direction)
                if pos_t is None or not in_bounds(x,n_planes,c_planes):
                    # Somehow went outside of bounds, or something
                    r = R-1
                    break
                elif (l <= pos_t):
                    x = x + l*d
                    if not in_bounds(x,n_planes,c_planes):
                        r = R-1 # try again
                    break
                elif count_zeros > 1:
                    # Hit nonsmooth boundary, i.e. intersection of multiple planes
                    r = R-1
                    break
                else:
                    x = x + pos_t*d
                    if in_bounds(x,n_planes,c_planes):
                        boundary.append(deepcopy(x))
                    n = inward_normal(center,n_planes[pos_i],c_planes[pos_i],x)
                    l = l - pos_t
                    d = d - 2*d.dot(n)*n
            if r < (R-1):
                break
            else:
                # reset to previous point, repeat
                x = xprev
                    
        points[i] = deepcopy(x)
        xprev = deepcopy(x)
        
    return points, np.array(boundary)
