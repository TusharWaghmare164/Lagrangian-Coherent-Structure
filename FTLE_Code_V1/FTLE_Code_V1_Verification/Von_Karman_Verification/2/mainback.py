import os
import numpy as np
import pyvista as pv
import pandas as pd
from mpi4py import MPI
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
# from scipy.interpolate import CloughTocher2DInterpolator


# ================ The FTLE Calculating class starts from here ================


class FTLE_calculate:

    # INITIAL DEFINING
    def __init__(self,pts,T,V1,V2,circleCentre):
        
        self.pts = pts
        self.T = T
        self.initVel = V1
        self.lastVel = V2
        self.circleCentre = circleCentre
        self.tree = cKDTree(pts)
        self.tri = Delaunay(pts)

        # Precompute affine transforms
        self.transform = self.tri.transform
        self.simplices = self.tri.simplices

    # VELOCITY PREDICTING SPATIAL
    def guessVel_fast(self, pts):

        pts = np.atleast_2d(pts)

        simplex = self.tri.find_simplex(pts)
        X = self.transform[simplex, :2]
        Y = pts - self.transform[simplex, 2]

        bary = np.einsum('ijk,ik->ij', X, Y)
        bary_coords = np.c_[bary, 1 - bary.sum(axis=1)]

        nodes = self.simplices[simplex]

        u1 = self.initVel[nodes]   # shape (N,3,2)
        u2 = self.lastVel[nodes]

        # Weighted sum (vectorized)
        U1 = np.einsum('ij,ijk->ik', bary_coords, u1)
        U2 = np.einsum('ij,ijk->ik', bary_coords, u2)

        return U1, U2
        

    # VELOCITY PREDICTING
    def Vel(self, pts, t):

        pts = np.atleast_2d(pts)

        u1, u2 = self.guessVel_fast(pts)

        alpha = (t / self.T)

        return (alpha * u2) + ((1 - alpha) * u1)
    

    # STEP INTEGRATION  
    # =========================
    # NEW ADAPTIVE RK4 STEP
    # =========================
    # STEP INTEGRATION  
    def RK4_step(self, pts, t, h):

        pts = np.asarray(pts, dtype=float)

        k1 = np.atleast_2d(self.Vel(pts, t))
        k2 = np.atleast_2d(self.Vel(pts + 0.5*h*k1, t + 0.5*h))
        k3 = np.atleast_2d(self.Vel(pts + 0.5*h*k2, t + 0.5*h))
        k4 = np.atleast_2d(self.Vel(pts + h*k3, t + h))

        pts_next = pts + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        return pts_next


    # FULL INTEGRATION
    def RK4_integrate(self, pts0, T, t0=0, h=-0.001):

        pts = np.asarray(pts0, dtype=float)

        t = t0
        steps = int(np.ceil(abs(T / h)))   # ✅ FIX

        for _ in range(steps):
            pts = self.RK4_step(pts, t, h)
            t += h

        return pts
    

    # NORMAL JACOBIAN WHEN PTS ARE FAR ENOUGH FROM CYLINDER
    def Jacobian(self, pt, dn):
        """
        Optimized Jacobian (no KDTree call inside, minimal allocations)
        dn = distances from KDTree.query(pt, k=2)
        """

        # Select spacing
        d = dn[1] if dn[0] == 0 else dn[0]

        if d < 1e-12:
            raise ValueError(f"Degenerate spacing detected at point {pt} {d}")

        h = 0.5 * d

        # Preallocate cloud (avoid np.array([...]))
        cloud = np.empty((4, 2), dtype=float)

        cloud[0, 0] = pt[0] + h; cloud[0, 1] = pt[1]
        cloud[1, 0] = pt[0] - h; cloud[1, 1] = pt[1]
        cloud[2, 0] = pt[0];     cloud[2, 1] = pt[1] + h
        cloud[3, 0] = pt[0];     cloud[3, 1] = pt[1] - h

        # Integrate all 4 points together (vectorized RK4)
        F = self.RK4_integrate(cloud, self.T)

        # Unpack
        F_p1 = F[0]
        F_m1 = F[1]
        F_p2 = F[2]
        F_m2 = F[3]

        inv_d = 1.0 / d

        # Compute columns of Jacobian
        a11 = (F_p1 - F_m1) * inv_d
        a12 = (F_p2 - F_m2) * inv_d

        # Assemble J (2x2)
        J = np.empty((2, 2), dtype=float)
        J[:, 0] = a11
        J[:, 1] = a12

        return J
    

    # SPECIAL JACOBIAN WHEN PTS'S CDS APPROX PTS CAN BE GO INSIDE OF CYLINDER AND BECOME PHYSICALLY NON EXISTING 
    def Jacobian_special(self, pt, dn):
        """
        Special Jacobian near cylinder
        dn passed from precomputed KDTree
        """

        # Robust spacing
        if dn[0] < 1e-2:
            d = 1e-2
        elif dn[0] == 0:
            d = dn[1]
        else:
            d = dn[0]

        # Direction e1 (radial)
        e1 = pt - self.circleCentre
        norm_e1 = np.sqrt(e1[0]*e1[0] + e1[1]*e1[1])

        # Avoid division by zero
        if norm_e1 < 1e-14:
            norm_e1 = 1e-14

        scale = d / norm_e1
        e1 = e1 * scale

        # Perpendicular direction
        e2 = np.array([-e1[1], e1[0]])

        # Build transform matrix manually
        Tmat = np.empty((2, 2), dtype=float)
        Tmat[:, 0] = e1
        Tmat[:, 1] = e2

        invT = np.linalg.inv(Tmat)

        # Preallocate cloud
        cloud = np.empty((4, 2), dtype=float)

        cloud[0] = pt + 0.5 * e1
        cloud[1] = pt
        cloud[2] = pt + e2
        cloud[3] = pt - e2

        # Integrate
        F = self.RK4_integrate(cloud, self.T)

        F_p1 = F[0]
        F_0  = F[1]
        F_p2 = F[2]
        F_m2 = F[3]

        inv_d = 1.0 / d

        # Columns in transformed space
        a11 = 2.0 * (F_p1 - F_0) * inv_d
        a12 = (F_p2 - F_m2) * (0.5 * inv_d)

        # Assemble and transform back
        J_local = np.empty((2, 2), dtype=float)
        J_local[:, 0] = a11
        J_local[:, 1] = a12

        J = invT @ J_local

        return J
    
    # COUNTING FTLE BY MAKING CGST AND TAKING IT'S SPECTRAL MODE
    def FTLE(self, pt, T, flag, dn):
        """
        Fully optimized FTLE
        - no KDTree call inside
        - fast eigenvalue computation
        """

        if flag == 1:
            dF = self.Jacobian_special(pt, dn)
        else:
            dF = self.Jacobian(pt, dn)

        # C = dF^T dF
        C00 = dF[0,0]*dF[0,0] + dF[1,0]*dF[1,0]
        C01 = dF[0,0]*dF[0,1] + dF[1,0]*dF[1,1]
        C11 = dF[0,1]*dF[0,1] + dF[1,1]*dF[1,1]

        # Fast eigenvalue (2x2 symmetric)
        l,v = np.linalg.eig(np.array([[C00, C01],[C01,C11]]))

        lambda_max = np.max(l)
        lambda_min = np.min(l)

        LCS_vector = v[:,np.argmin(l)]

        if lambda_max < 1e-12:
            lambda_max = 1e-12

        FTLE = np.log(lambda_max) / (2.0 * T)

        return [FTLE,lambda_max,lambda_min,LCS_vector[0],LCS_vector[1]]


# ================ The MPI Operating class starts from here ================


class MPI_OPERATION:

    def __init__(self, pts, ftle, circleCentre, circleRadius, batch_size=4000):

        self.pts = pts
        self.ftle = ftle
        self.circleCentre = np.array(circleCentre)
        self.circleRadius = circleRadius

        self.batch_size = batch_size
        self.N = len(pts)
        self.nBatch = int(np.ceil(self.N / batch_size))

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        os.makedirs("cache_back", exist_ok=True)


    # -----------------------------
    # detect finished batches
    # -----------------------------
    def finished_batches(self):

        done = []

        for f in os.listdir("cache_back"):

            if f.startswith("batch_") and f.endswith(".npy"):

                idx = int(f.split("_")[1].split(".")[0])
                done.append(idx)

        return set(done)


    # -----------------------------
    # worker computation
    # -----------------------------
    def compute_batch(self, batch_id):

        start = batch_id * self.batch_size
        end = min(start + self.batch_size, self.N)

        pts_batch = self.pts[start:end]

        tol = 0.001 * self.circleRadius

        r = np.linalg.norm(pts_batch - self.circleCentre, axis=1)
        flags = (np.abs(r - self.circleRadius) <= tol).astype(int)

        results = np.empty((len(pts_batch),5))

        dists, _ = self.ftle.tree.query(pts_batch, k=2)

        for i in range(len(pts_batch)):
            results[i] = self.ftle.FTLE(pts_batch[i], self.ftle.T, flags[i], dists[i])

        np.save(f"cache_back/batch_{batch_id:05d}.npy", results)


    # -----------------------------
    # worker loop
    # -----------------------------
    def worker_run(self):

        finished = self.finished_batches()

        for batch_id in range(self.nBatch):

            if batch_id in finished:
                continue

            if batch_id % self.size != self.rank:
                continue

            self.compute_batch(batch_id)


    # -----------------------------
    # master rebuild result
    # -----------------------------
    def assemble_results(self):

        FTLE_field = np.zeros((self.N,5))

        for batch_id in range(self.nBatch):

            start = batch_id * self.batch_size
            end = min(start + self.batch_size, self.N)

            path = f"cache_back/batch_{batch_id:05d}.npy"

            # missing file
            if not os.path.exists(path):
                print(f"Missing cache {path}, recomputing...")
                self.compute_batch(batch_id)

            try:
                data = np.load(path)

            except Exception:
                print(f"Corrupted cache detected: {path}, recomputing...")
                self.compute_batch(batch_id)
                data = np.load(path)

            FTLE_field[start:end] = data

        return FTLE_field


    # -----------------------------
    # write vtk output
    # -----------------------------
    def write_vtk(self, template_mesh, FTLE_field):

        mesh = template_mesh.copy()
        pts_all = self.ftle.pts

        FTLE = np.zeros(len(pts_all))
        lambda_max = np.zeros(len(pts_all))
        lambda_min = np.zeros(len(pts_all))
        
        LCS_DirVec = np.zeros((len(pts_all),2))

        if (len(FTLE_field) < len(pts_all)):
            idx = np.array([np.where((pts_all == p).all(axis=1))[0][0] for p in self.pts])
            FTLE[idx] = FTLE_field[:,0]
            lambda_max[idx] = FTLE_field[:,1]
            lambda_min[idx] = FTLE_field[:,2]
            LCS_DirVec[idx] = FTLE_field[:,3:]


        mesh.point_data["FTLE"] = FTLE
        mesh.point_data["lambda_max"] = lambda_max
        mesh.point_data["lambda_min"] = lambda_min
        mesh.point_data["LCS_vector"] = LCS_DirVec

        mesh.save("backFTLE_output.vtk", binary=False)


#  ================ Operational Code Start here ================ 

# ================= MPI initialization =================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ================= Load FULL dataset =================
if rank == 0:

    # for vtk format
    A1 = pv.read("SS_2.vtk")
    A2 = pv.read("SS_1.vtk")

    pts_all = A1.points[:, :2]
    V1_all = A1.point_data["V"][:, :2]
    V2_all = A2.point_data["V"][:, :2]

    # # for vtu format
    # A1 = pv.read("SS_0.vtu")
    # A2 = pv.read("SS_1.vtu")

    # pts_all = A1.points[:, :2]
    # V1_all = A1.point_data["V"][:, :2]
    # V2_all = A2.point_data["V"][:, :2]

    # for csv format
    # A1 = np.array(pd.read_csv("SS_0.csv"))
    # A2 = np.array(pd.read_csv("SS_1.csv"))
    # M = pv.read("send/")

    # pts_all = A1[:, :2]
    # V1_all = A1[:, 3:-1]
    # V2_all = A2[:, 3:-1]

    T = 1.196

else:
    pts_all = None
    V1_all = None
    V2_all = None
    T = None
    A1 = None


# ================= Broadcast FULL dataset =================
pts_all = comm.bcast(pts_all, root=0)
V1_all = comm.bcast(V1_all, root=0)
V2_all = comm.bcast(V2_all, root=0)
T = comm.bcast(T, root=0)


circleCentre = [0,0]
circleRadius = 0.5


# ================= Build FTLE class (needs FULL mesh) =================
ftle = FTLE_calculate(pts_all, T, V1_all, V2_all, circleCentre)


# ================= Mask region for particle computation =================
x_min, x_max = -2.5, 10
y_min, y_max = -3, 3

mask = (
    (pts_all[:,0] >= x_min) & (pts_all[:,0] <= x_max) &
    (pts_all[:,1] >= y_min) & (pts_all[:,1] <= y_max) &
    ((pts_all[:,0]**2 + pts_all[:,1]**2) > (circleRadius)**2)
)

pts = pts_all[mask]


# ================= Free large arrays from main scope =================
del pts_all, V1_all, V2_all


# ================= Create MPI controller =================
mpi_job = MPI_OPERATION(
    pts,
    ftle,
    circleCentre,
    circleRadius,
    batch_size=2000
)


# ================= Run MPI workers =================
mpi_job.worker_run()


# ================= Synchronize processes =================
comm.Barrier()


# ================= Assemble results (master only) =================
if rank == 0:

    FTLE_field = mpi_job.assemble_results()

    mpi_job.write_vtk(A1, FTLE_field)

    print("FTLE computation finished.")