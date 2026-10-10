
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp


# ============================================================
# 1. Read VTK and reconstruct the 2D FTLE grid
# ============================================================

def read_ftle_vtk(filename, ftle_name="FTLE"):
    mesh = pv.read(filename)

    points = mesh.points
    x = points[:, 0]
    y = points[:, 1]

    if ftle_name not in mesh.point_data:
        raise KeyError(f"{ftle_name!r} not found in point_data")

    ftle = np.asarray(mesh.point_data[ftle_name])

    if ftle.ndim != 1:
        raise ValueError("FTLE must be a scalar point-data array")

    # Determine unique coordinates
    xu = np.unique(x)
    yu = np.unique(y)

    nx, ny = len(xu), len(yu)

    if nx * ny != len(points):
        raise ValueError(
            "The VTK points do not form a complete rectangular grid."
        )

    # Map each point to its grid index
    ix = np.searchsorted(xu, x)
    iy = np.searchsorted(yu, y)

    # Check for duplicate or missing grid locations
    flat = iy * nx + ix

    if len(np.unique(flat)) != len(points):
        raise ValueError("Duplicate grid locations found")

    field = np.full((ny, nx), np.nan)
    field[iy, ix] = ftle

    if not np.all(np.isfinite(field)):
        raise ValueError("Missing or invalid FTLE values")

    dx = np.diff(xu)
    dy = np.diff(yu)

    if not np.allclose(dx, dx[0]) or not np.allclose(dy, dy[0]):
        raise ValueError(
            "This implementation requires uniform grid spacing"
        )

    return xu, yu, field


# ============================================================
# 2. Calculate the Hessian and extract candidate ridges
# ============================================================

def detect_ridges(x, y, field, sigma=0.0,
                  min_curvature=0.0,
                  gradient_tolerance=0.1):
    """
    Hessian-based 2D ridge detection.

    sigma:
        Gaussian smoothing scale in grid cells.
        0 means no smoothing.

    min_curvature:
        Minimum magnitude of negative normal curvature.

    gradient_tolerance:
        Relative tolerance for the gradient component
        perpendicular to the ridge.
    """

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    if sigma > 0:
        f = gaussian_filter(field, sigma=sigma, mode="nearest")
    else:
        f = field.copy()

    # First derivatives
    fy, fx = np.gradient(f, dy, dx)

    # Second derivatives
    fyy, fyx = np.gradient(fy, dy, dx)
    fxy, fxx = np.gradient(fx, dy, dx)

    fxy = 0.5 * (fxy + fyx)

    # Hessian eigenvalues and eigenvectors
    H = np.empty((*f.shape, 2, 2))
    H[..., 0, 0] = fxx
    H[..., 0, 1] = fxy
    H[..., 1, 0] = fxy
    H[..., 1, 1] = fyy

    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Eigenvalues are sorted in ascending order.
    # The first eigenvector is the strongest negative
    # curvature direction, normal to the ridge.
    lambda_n = eigenvalues[..., 0]
    normal = eigenvectors[..., :, 0]
    tangent = eigenvectors[..., :, 1]

    # Gradient in the normal direction
    grad_normal = fx * normal[..., 0] + fy * normal[..., 1]

    # Scale-aware gradient tolerance
    grad_magnitude = np.hypot(fx, fy)
    threshold = gradient_tolerance * (
        grad_magnitude + 1e-12
    )

    ridge = (
        (lambda_n < -min_curvature)
        & (np.abs(grad_normal) <= threshold)
        & np.isfinite(f)
    )

    # Avoid unreliable boundary derivatives
    ridge[[0, -1], :] = False
    ridge[:, [0, -1]] = False

    return {
        "field": f,
        "ridge": ridge,
        "normal": normal,
        "tangent": tangent,
        "lambda_n": lambda_n,
        "gradient": np.stack((fx, fy), axis=-1)
    }


# ============================================================
# 3. Multiscale ridge detection
# ============================================================

def multiscale_ridges(
    x, y, field,
    scales=(0.0, 0.75, 1.5, 2.5),
    smoothing=True,
    min_curvature=0.0,
    gradient_tolerance=0.1,
    min_votes=2
):
    """
    Detect ridges at several Gaussian scales.

    If smoothing=False, all scales are replaced by 0.
    min_votes controls how many scales must detect a ridge.
    """

    if smoothing:
        used_scales = tuple(scales)
    else:
        used_scales = (0.0,)

    results = []

    for scale in used_scales:
        result = detect_ridges(
            x, y, field,
            sigma=scale,
            min_curvature=min_curvature,
            gradient_tolerance=gradient_tolerance
        )
        results.append(result)

    votes = np.sum(
        [r["ridge"] for r in results],
        axis=0
    )

    consensus = votes >= min_votes

    # Use the middle scale as the tangent field
    middle = results[len(results) // 2]

    return {
        "results": results,
        "votes": votes,
        "ridge": consensus,
        "tangent": middle["tangent"],
        "field": middle["field"],
        "scale_used": used_scales
    }


# ============================================================
# 4. Interpolate the ridge tangent field for RK45
# ============================================================

def make_tangent_interpolators(x, y, tangent):
    tx = RegularGridInterpolator(
        (y, x), tangent[..., 0],
        bounds_error=False, fill_value=np.nan
    )
    ty = RegularGridInterpolator(
        (y, x), tangent[..., 1],
        bounds_error=False, fill_value=np.nan
    )

    return tx, ty


def trace_ridge(x0, y0, x, y, tangent,
                direction=1, max_length=1.0,
                max_step=None):
    """
    Integrate a ridge curve using RK45.

    The vector field is the local tangent direction.
    The curve is parameterized by arc length.
    """

    tx, ty = make_tangent_interpolators(x, y, tangent)

    if max_step is None:
        max_step = min(x[1] - x[0], y[1] - y[0])

    def rhs(s, position):
        px, py = position

        vx = float(tx((py, px)))
        vy = float(ty((py, px)))

        if not np.isfinite(vx + vy):
            return [0.0, 0.0]

        # Normalize to make s approximately arc length
        norm = np.hypot(vx, vy)

        if norm < 1e-12:
            return [0.0, 0.0]

        return [
            direction * vx / norm,
            direction * vy / norm
        ]

    def outside(s, position):
        px, py = position
        return min(
            px - x[0], x[-1] - px,
            py - y[0], y[-1] - py
        )

    outside.terminal = True
    outside.direction = -1

    solution = solve_ivp(
        rhs,
        (0.0, max_length),
        (x0, y0),
        method="RK45",
        max_step=max_step,
        rtol=1e-5,
        atol=1e-7,
        events=outside
    )

    return solution.y.T


# ============================================================
# 5. Trace curves from selected ridge seed points
# ============================================================

def trace_all_ridges(x, y, ridge_result,
                     seed_spacing=10,
                     max_length=1.0):
    ridge = ridge_result["ridge"]
    tangent = ridge_result["tangent"]

    # Select seed points on the ridge mask
    iy, ix = np.where(ridge)
    candidates = list(zip(iy, ix))

    seeds = []
    for j, i in candidates:
        if all(
            (i - si)**2 + (j - sj)**2 >= seed_spacing**2
            for sj, si in seeds
        ):
            seeds.append((j, i))

    curves = []

    for j, i in seeds:
        x0, y0 = x[i], y[j]

        forward = trace_ridge(
            x0, y0, x, y, tangent,
            direction=1,
            max_length=max_length
        )

        backward = trace_ridge(
            x0, y0, x, y, tangent,
            direction=-1,
            max_length=max_length
        )

        # Join both directions at the seed
        curve = np.vstack((
            backward[::-1],
            forward[1:]
        ))

        if len(curve) > 2:
            curves.append(curve)

    return curves


# ============================================================
# 6. Plot FTLE, consensus mask, and traced curves
# ============================================================

def plot_ridges(x, y, field, result, curves=None):
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(10, 7))

    contour = ax.contourf(
        X, Y, field,
        levels=40,
        cmap="viridis"
    )

    fig.colorbar(contour, ax=ax, label="FTLE")

    ax.contour(
        X, Y, result["ridge"].astype(float),
        levels=[0.5],
        colors="white",
        linewidths=0.7
    )

    if curves is not None:
        for curve in curves:
            ax.plot(
                curve[:, 0], curve[:, 1],
                color="red", linewidth=1.2
            )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title("Multiscale FTLE ridges")
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. Run
# ============================================================

if __name__ == "__main__":
    x, y, ftle = read_ftle_vtk(
        "your_file.vtk",
        ftle_name="FTLE"
    )

    result = multiscale_ridges(
        x, y, ftle,
        scales=(0.0, 0.75, 1.5, 2.5),
        smoothing=True,
        min_curvature=0.0,
        gradient_tolerance=0.1,
        min_votes=2
    )

    curves = trace_all_ridges(
        x, y, result,
        seed_spacing=10,
        max_length=1.0
    )

    plot_ridges(x, y, ftle, result, curves)
