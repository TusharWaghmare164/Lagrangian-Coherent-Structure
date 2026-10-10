import os
import numpy as np
import pyvista as pv
from mpi4py import MPI
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay


# ============================================================
# FTLE MODE SELECTION
# ============================================================

selection = "bf"
batches = 5000

# "f"  -> forward FTLE
# "b"  -> backward FTLE
# "bf" -> backward then forward
# "fb" -> forward then backward
# "p"  -> parallel forward + backward


# ============================================================
# FTLE CLASS
# ============================================================

class FTLE_calculate:

    def __init__(self, pts, T, V_all, dt):

        self.pts = pts
        self.V_all = V_all

        self.T = T
        self.dt = dt

        self.Nt = V_all.shape[0]

        self.tree = cKDTree(pts)

        self.tri = Delaunay(pts)

        self.transform = self.tri.transform
        self.simplices = self.tri.simplices


    # ========================================================
    # VELOCITY INTERPOLATION
    # ========================================================

    def Vel(self, pts, t):

        pts = np.atleast_2d(pts)

        k = int(np.clip(np.floor(t / self.dt), 0, self.Nt - 2))

        tau = (t - k * self.dt) / self.dt

        simplex = self.tri.find_simplex(pts)

        X = self.transform[simplex, :2]
        Y = pts - self.transform[simplex, 2]

        bary = np.einsum('ijk,ik->ij', X, Y)

        bary_coords = np.c_[bary, 1 - bary.sum(axis=1)]

        nodes = self.simplices[simplex]

        V1 = self.V_all[k][nodes]
        V2 = self.V_all[k + 1][nodes]

        U1 = np.einsum('ij,ijk->ik', bary_coords, V1)
        U2 = np.einsum('ij,ijk->ik', bary_coords, V2)

        return (1 - tau) * U1 + tau * U2


    # ========================================================
    # RK4 STEP
    # ========================================================

    def RK4_step(self, pts, t, h):

        pts = np.asarray(pts, dtype=float)

        k1 = np.atleast_2d(self.Vel(pts, t))

        k2 = np.atleast_2d(
            self.Vel(pts + 0.5 * h * k1, t + 0.5 * h)
        )

        k3 = np.atleast_2d(
            self.Vel(pts + 0.5 * h * k2, t + 0.5 * h)
        )

        k4 = np.atleast_2d(
            self.Vel(pts + h * k3, t + h)
        )

        pts_next = pts + (h / 6.0) * (
            k1 + 2 * k2 + 2 * k3 + k4
        )

        return pts_next


    # ========================================================
    # FULL INTEGRATION
    # ========================================================

    def RK4_integrate(self, pts0, t_start, t_end, h):

        pts = np.asarray(pts0, dtype=float)

        t = t_start

        nsteps = int(np.ceil(abs((t_end - t_start) / h)))

        for _ in range(nsteps):

            if h > 0 and t >= t_end:
                break

            if h < 0 and t <= t_end:
                break

            pts = self.RK4_step(pts, t, h)

            t += h

        return pts


    # ========================================================
    # JACOBIAN
    # ========================================================

    def Jacobian(self, pt, dn, mode='forward'):

        d = dn[1] if dn[0] == 0 else dn[0]

        if d < 1e-12:
            raise ValueError(
                f"Degenerate spacing detected at point {pt}"
            )

        h_fd = 0.5 * d

        cloud = np.empty((4, 2), dtype=float)

        cloud[0] = [pt[0] + h_fd, pt[1]]
        cloud[1] = [pt[0] - h_fd, pt[1]]

        cloud[2] = [pt[0], pt[1] + h_fd]
        cloud[3] = [pt[0], pt[1] - h_fd]

        if mode == 'forward':

            F = self.RK4_integrate(
                cloud,
                t_start=0,
                t_end=self.T,
                h=0.001
            )

        elif mode == 'backward':

            F = self.RK4_integrate(
                cloud,
                t_start=self.T,
                t_end=0,
                h=-0.001
            )

        F_p1 = F[0]
        F_m1 = F[1]

        F_p2 = F[2]
        F_m2 = F[3]

        inv_d = 1.0 / d

        a11 = (F_p1 - F_m1) * inv_d
        a12 = (F_p2 - F_m2) * inv_d

        J = np.empty((2, 2), dtype=float)

        J[:, 0] = a11
        J[:, 1] = a12

        return J


    # ========================================================
    # FTLE
    # ========================================================

    def FTLE(self, pt, T, dn, mode='forward'):

        dF = self.Jacobian(pt, dn, mode)

        C00 = dF[0,0]*dF[0,0] + dF[1,0]*dF[1,0]
        C01 = dF[0,0]*dF[0,1] + dF[1,0]*dF[1,1]
        C11 = dF[0,1]*dF[0,1] + dF[1,1]*dF[1,1]

        C = np.array([
            [C00, C01],
            [C01, C11]
        ])

        l, v = np.linalg.eig(C)

        lambda_max = np.max(l)
        lambda_min = np.min(l)

        LCS_vector = v[:, np.argmin(l)]

        lambda_max = max(lambda_max, 1e-12)

        FTLE = np.log(lambda_max) / (2.0 * abs(T))

        return [
            FTLE,
            lambda_max,
            lambda_min,
            LCS_vector[0],
            LCS_vector[1]
        ]


# ============================================================
# MPI OPERATION CLASS
# ============================================================

class MPI_OPERATION:

    def __init__(self, pts, ftle, mode='forward', batch_size=2000):

        self.pts = pts

        self.ftle = ftle

        self.mode = mode

        self.batch_size = batch_size

        self.N = len(pts)

        self.nBatch = int(np.ceil(self.N / batch_size))

        self.comm = MPI.COMM_WORLD

        self.rank = self.comm.Get_rank()

        self.size = self.comm.Get_size()

        os.makedirs("cache", exist_ok=True)


    # ========================================================
    # FINISHED BATCHES
    # ========================================================

    def finished_batches(self):

        done = []

        for f in os.listdir("cache"):

            if (
                f.startswith(f"{self.mode}_batch_")
                and f.endswith(".npy")
            ):

                idx = int(f.split("_")[-1].split(".")[0])

                done.append(idx)

        return set(done)


    # ========================================================
    # COMPUTE BATCH
    # ========================================================

    def compute_batch(self, batch_id):

        start = batch_id * self.batch_size

        end = min(start + self.batch_size, self.N)

        pts_batch = self.pts[start:end]

        results = np.empty((len(pts_batch), 5))

        dists, _ = self.ftle.tree.query(pts_batch, k=2)

        for i in range(len(pts_batch)):

            results[i] = self.ftle.FTLE(
                pts_batch[i],
                self.ftle.T,
                dists[i],
                mode=self.mode
            )

        np.save(
            f"cache/{self.mode}_batch_{batch_id:05d}.npy",
            results
        )


    # ========================================================
    # WORKER LOOP
    # ========================================================

    def worker_run(self):

        finished = self.finished_batches()

        for batch_id in range(self.nBatch):

            if batch_id in finished:
                continue

            if batch_id % self.size != self.rank:
                continue

            self.compute_batch(batch_id)


    # ========================================================
    # ASSEMBLE RESULTS
    # ========================================================

    def assemble_results(self):

        FTLE_field = np.zeros((self.N, 5))

        for batch_id in range(self.nBatch):

            start = batch_id * self.batch_size

            end = min(start + self.batch_size, self.N)

            path = (
                f"cache/{self.mode}_batch_{batch_id:05d}.npy"
            )

            if not os.path.exists(path):

                print(f"Missing cache {path}")

                self.compute_batch(batch_id)

            try:

                data = np.load(path)

            except Exception:

                print(f"Corrupted cache {path}")

                self.compute_batch(batch_id)

                data = np.load(path)

            FTLE_field[start:end] = data

        return FTLE_field


    # ========================================================
    # WRITE VTK
    # ========================================================

    def write_vtk(self, template_mesh, FTLE_field, filename):

        mesh = template_mesh.copy()

        mesh.point_data["FTLE"] = FTLE_field[:,0]

        mesh.point_data["lambda_max"] = FTLE_field[:,1]

        mesh.point_data["lambda_min"] = FTLE_field[:,2]

        vec = np.zeros((len(FTLE_field), 3))

        vec[:, :2] = FTLE_field[:, 3:]

        mesh.point_data["LCS_vector"] = vec

        mesh.save(filename, binary=False)


# ============================================================
# MPI INITIALIZATION
# ============================================================

comm = MPI.COMM_WORLD

rank = comm.Get_rank()

size = comm.Get_size()


# ============================================================
# LOAD DATA
# ============================================================

if rank == 0:

    files = sorted([
        f for f in os.listdir()
        if f.endswith(".vtk")
    ])

    meshes = [pv.read(f) for f in files]

    pts_all = meshes[0].points[:, :2]

    V_all = np.array([
        m.point_data["V"][:, :2]
        for m in meshes
    ])

    template_mesh = meshes[0]

    dt = 4

    T = dt * (len(files) - 1)

else:

    pts_all = None
    V_all = None
    template_mesh = None
    dt = None
    T = None


# ============================================================
# BROADCAST
# ============================================================

pts_all = comm.bcast(pts_all, root=0)

V_all = comm.bcast(V_all, root=0)

template_mesh = comm.bcast(template_mesh, root=0)

dt = comm.bcast(dt, root=0)

T = comm.bcast(T, root=0)


# ============================================================
# BUILD FTLE OBJECT
# ============================================================

ftle = FTLE_calculate(
    pts_all,
    T,
    V_all,
    dt
)


# ============================================================
# EXECUTION FUNCTION
# ============================================================

def run_case(mode, filename):

    mpi_job = MPI_OPERATION(
        pts_all,
        ftle,
        mode=mode,
        batch_size=2000
    )

    mpi_job.worker_run()

    comm.Barrier()

    if rank == 0:

        FTLE_field = mpi_job.assemble_results()

        mpi_job.write_vtk(
            template_mesh,
            FTLE_field,
            filename
        )

        print(f"{filename} exported successfully.")


# ============================================================
# MODE CONTROL
# ============================================================

if selection == "f":

    run_case("forward", "forward_ftle.vtk")

elif selection == "b":

    run_case("backward", "backward_ftle.vtk")

elif selection == "bf":

    run_case("backward", "backward_ftle.vtk")

    comm.Barrier()

    run_case("forward", "forward_ftle.vtk")

elif selection == "fb":

    run_case("forward", "forward_ftle.vtk")

    comm.Barrier()

    run_case("backward", "backward_ftle.vtk")

elif selection == "p":

    if rank % 2 == 0:

        mode = "forward"

        filename = "forward_ftle.vtk"

    else:

        mode = "backward"

        filename = "backward_ftle.vtk"

    mpi_job = MPI_OPERATION(
        pts_all,
        ftle,
        mode=mode,
        batch_size= batches
    )

    mpi_job.worker_run()

    comm.Barrier()

    if rank == 0:

        print("Parallel FTLE computation finished.")