import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d


# ==========================================================
# LOAD DATA
# ==========================================================

mesh = pv.read("backFTLE_output.vtk")

pts = mesh.points[:, :2]

FTLE = mesh.point_data["FTLE"]
eigvec2 = mesh.point_data["LCS_vector"]
lambda_max = mesh.point_data["lambda_max"]
lambda_min = mesh.point_data["lambda_min"]


# ==========================================================
# TRIANGULATION
# ==========================================================

tri = Delaunay(pts)


# ==========================================================
# BARYCENTRIC INTERPOLATION
# ==========================================================

def barycentric_interpolate(point, field):

    simplex = tri.find_simplex(point)

    if simplex == -1:
        return None

    vertices = tri.simplices[simplex]
    T = tri.transform[simplex]

    delta = point - T[2]
    bary = np.dot(T[:2], delta)

    w1, w2 = bary
    w3 = 1 - w1 - w2

    weights = np.array([w1, w2, w3])

    values = field[vertices]

    return np.dot(weights, values)


# ==========================================================
# DOMAIN SETTINGS
# ==========================================================

xmin, xmax = -2.5, 10
ymin, ymax = -3, 3

center = np.array([0.0,0.0])
radius = 0.5001

domain = max(xmax-xmin, ymax-ymin)

step = 0.0003 * domain

seed_threshold = 0.6 * np.max(FTLE)


# ==========================================================
# RIDGE SEED DETECTION
# ==========================================================

ridge_points = []

n_lines = 500
n_samples = 1200

ys = np.linspace(ymin, ymax, n_lines)

for y in ys:

    xs = np.linspace(xmin, xmax, n_samples)

    ftle_vals = []

    for x in xs:

        # skip points inside cylinder
        if np.linalg.norm(np.array([x,y]) - center) < radius:
            ftle_vals.append(np.nan)
            continue

        val = barycentric_interpolate(np.array([x,y]), FTLE)

        ftle_vals.append(val if val is not None else np.nan)

    ftle_vals = np.array(ftle_vals)

    for i in range(1, n_samples-1):

        if np.isnan(ftle_vals[i]):
            continue

        if ftle_vals[i] > ftle_vals[i-1] and ftle_vals[i] > ftle_vals[i+1]:

            if ftle_vals[i] < seed_threshold:
                continue

            x1,x2,x3 = xs[i-1], xs[i], xs[i+1]
            y1,y2,y3 = ftle_vals[i-1], ftle_vals[i], ftle_vals[i+1]

            denom = y1 - 2*y2 + y3

            if abs(denom) < 1e-12:
                continue

            x_peak = x2 + 0.5*(y1-y3)/denom * (x3-x1)/2

            ridge_points.append([x_peak,y])

ridge_points = np.array(ridge_points)

print("Seed points:", len(ridge_points))


# ==========================================================
# RK4 STEP
# ==========================================================

def rk4_step(p, h, direction=1):

    def vec(x):

        v = barycentric_interpolate(x, eigvec2)

        if v is None:
            return None

        v = v / np.linalg.norm(v)

        if v[0] < 0:
            v = -v

        return direction * v


    k1 = vec(p)
    if k1 is None: return None

    k2 = vec(p + 0.5*h*k1)
    if k2 is None: return None

    k3 = vec(p + 0.5*h*k2)
    if k3 is None: return None

    k4 = vec(p + h*k3)
    if k4 is None: return None

    step_vec = (k1 + 2*k2 + 2*k3 + k4)/6

    return p + h*step_vec


# ==========================================================
# RIDGE MARCHING
# ==========================================================

def march_ridge(seed, direction):

    ridge = [seed]
    p = seed

    for _ in range(6000):

        p_new = rk4_step(p, step, direction)

        if p_new is None:
            break

        # stop if outside domain
        if not (xmin <= p_new[0] <= xmax and ymin <= p_new[1] <= ymax):
            break

        # stop if inside cylinder
        if np.linalg.norm(p_new - center) < radius:
            break

        lam_max = barycentric_interpolate(p_new, lambda_max)
        lam_min = barycentric_interpolate(p_new, lambda_min)

        if lam_max is None or lam_min is None:
            break

        anisotropy = abs(lam_max - lam_min) / lam_max

        if anisotropy < 0.05:
            break

        ridge.append(p_new)

        p = p_new

    return np.array(ridge)

# ==========================================================
# RIDGE SMOOTHING
# ==========================================================

def smooth_ridge(ridge,
                 gaussian_sigma=3,
                 spline_smoothness=0.5,
                 refine_factor=4):

    if len(ridge) < 10:
        return ridge

    # ------------------------------------------
    # Separate coordinates
    # ------------------------------------------

    x = ridge[:,0]
    y = ridge[:,1]

    # ------------------------------------------
    # Gaussian smoothing of discretised points
    # ------------------------------------------

    x_smooth = gaussian_filter1d(x, sigma=gaussian_sigma)
    y_smooth = gaussian_filter1d(y, sigma=gaussian_sigma)

    # ------------------------------------------
    # Arc-length parameter
    # ------------------------------------------

    ds = np.sqrt(np.diff(x_smooth)**2 + np.diff(y_smooth)**2)

    s = np.zeros(len(x_smooth))

    s[1:] = np.cumsum(ds)

    # avoid duplicate points
    unique = np.diff(s, prepend=-1) > 1e-10

    x_smooth = x_smooth[unique]
    y_smooth = y_smooth[unique]
    s = s[unique]

    if len(s) < 5:
        return ridge

    # normalize parameter
    s = s / s[-1]

    # ------------------------------------------
    # Cubic spline fitting
    # ------------------------------------------

    tck, _ = splprep([x_smooth, y_smooth],
                     u=s,
                     k=3,
                     s=spline_smoothness)

    s_fine = np.linspace(0,1,len(ridge)*refine_factor)

    x_fine, y_fine = splev(s_fine, tck)

    return np.column_stack((x_fine, y_fine))


# ==========================================================
# EXTRACT RIDGES
# ==========================================================

all_ridges = []

for seed in ridge_points:

    # skip seeds near boundaries
    if not (xmin+step < seed[0] < xmax-step and ymin+step < seed[1] < ymax-step):
        continue

    if np.linalg.norm(seed-center) < radius:
        continue

    forward = march_ridge(seed, 1)
    backward = march_ridge(seed, -1)

    ridge = np.vstack((backward[::-1], forward))

    if len(forward) > 10 or len(backward) > 10:
        all_ridges.append(ridge)


print("Ridges extracted:", len(all_ridges))

# ==========================================================
# SELECT LONGEST RIDGES
# ==========================================================

# ==========================================================
# SELECT TOP-N LONGEST RIDGES
# ==========================================================

def ridge_length(ridge):
    diffs = np.diff(ridge, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))

N = 3   # change to 2, 3, etc.

if len(all_ridges) == 0:
    top_ridges = []
else:
    lengths = np.array([ridge_length(r) for r in all_ridges])

    # sort indices by descending length
    sorted_idx = np.argsort(-lengths)

    # pick top N
    top_indices = sorted_idx[:N]

    top_ridges = [smooth_ridge(all_ridges[i])
              for i in top_indices]

print(f"Top {N} ridge lengths:", lengths[top_indices])

# ==========================================================
# VISUALIZATION
# ==========================================================

plt.figure(figsize=(7,5))

plt.tricontourf(pts[:,0], pts[:,1], FTLE, levels=50)

for ridge in top_ridges:
    plt.plot(ridge[:,0], ridge[:,1], color="blue", linewidth=2)

plt.scatter(ridge_points[:,0], ridge_points[:,1], s=5, color="red")

circle = plt.Circle(center, radius, color='white')
plt.gca().add_patch(circle)

plt.title("FTLE LCS Longest Ridge(s)")
plt.xlabel("x")
plt.ylabel("y")

plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

plt.gca().set_aspect('equal', adjustable='box')

plt.show()