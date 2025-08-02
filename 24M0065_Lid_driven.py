import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Benchmark Data: Ghia et al. (Re = 100)
# ------------------------------
# These are benchmark solutions used to validate CFD results
ghia_y = np.array([
    1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
    0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000
])
ghia_u = np.array([
    1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332,
   -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434,
   -0.04775, -0.04192, -0.03717,  0.00000
])
ghia_x = np.array([
    1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
    0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000
])
ghia_v = np.array([
    0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914,
   -0.22445, -0.24533,  0.05454,  0.17527,  0.17507,  0.16077,
    0.12317,  0.10890,  0.10091,  0.09233,  0.00000
])

# ------------------------------
# Simulation Parameters
# ------------------------------
L = 1.0              # Length of cavity (1m x 1m)
I, J = 31, 31        # Grid points in x and y directions
Re = 100             # Reynolds number
nu = 1.0 / Re        # Non-dimensional kinematic viscosity
dx = L / (I - 1)     # Grid spacing in x
dy = L / (J - 1)     # Grid spacing in y

# Time-stepping stability parameters
sigma_c = 0.4        # Courant number (convection)
sigma_d = 0.6        # Stability for diffusion

# Convergence criteria
tol_u = 1e-8         # Tolerance for u velocity
tol_v = 1e-8         # Tolerance for v velocity
tol_psi = 1e-2       # Tolerance for stream function residual
max_iter = 100000    # Maximum number of iterations

# Creating mesh and initializing fields
x = np.linspace(0, L, I)
y = np.linspace(0, L, J)
X, Y = np.meshgrid(x, y)

psi = np.full((I, J), 100.0)  # Stream function initialized to constant
omega = np.zeros((I, J))      # Vorticity
u = np.zeros((I, J))          # u-velocity
v = np.zeros((I, J))          # v-velocity

# ------------------------------
# Boundary Conditions
# ------------------------------
def boundary_conditions(psi, omega, u_top=1.0):
    """
    Applying stream function and vorticity boundary conditions.
    Top lid moves with u=1; all other walls are stationary.
    """
    psi[:, -1] = 100.0
    omega[:, -1] = -2*(psi[:, -2] - psi[:, -1])/dy**2 - 2*u_top/dy
    psi[:, 0] = 100.0
    omega[:, 0] = -2*(psi[:, 1] - psi[:, 0])/dy**2
    psi[0, :] = 100.0
    omega[0, :] = -2*(psi[1, :] - psi[0, :])/dx**2
    psi[-1, :] = 100.0
    omega[-1, :] = -2*(psi[-2, :] - psi[-1, :])/dx**2

# ------------------------------
# Solving Poisson Equation for Stream Function
# ------------------------------
def poisson_equation(psi, omega):
    """
    Iteratively solving the Poisson equation: ∇²ψ = -ω
    using Gauss-Seidel until the RMS residual is below tol_psi.
    """
    res = 1.0
    while res > tol_psi:
        psi_old = psi.copy()
        psi[1:-1, 1:-1] = 0.25 * (
            psi[2:, 1:-1] + psi[:-2, 1:-1] +
            psi[1:-1, 2:] + psi[1:-1, :-2] +
            dx**2 * omega[1:-1, 1:-1]
        )
        # Computing residual for convergence
        R = (
            (psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dx**2 +
            (psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dy**2 +
            omega[1:-1, 1:-1]
        )
        res = np.sqrt(np.mean(R**2))  # RMS residual
    return psi

# ------------------------------
# Computing Velocity from Stream Function
# ------------------------------
def velocity_update(psi):
    """
    Calculating velocities from stream function:
    u = ∂ψ/∂y, v = -∂ψ/∂x using central differences.
    """
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    u[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * dy)
    v[1:-1, :] = -(psi[2:, :] - psi[:-2, :]) / (2 * dx)
    u[:, 0] = 0.0; u[:, -1] = 1.0
    u[0, :] = 0.0; u[-1, :] = 0.0
    v[:, 0] = 0.0; v[:, -1] = 0.0
    v[0, :] = 0.0; v[-1, :] = 0.0
    return u, v

# ------------------------------
# Vorticity Transport Equation
# ------------------------------
def vorticity_transport(omega, u, v, dt):
    """
    Solving the vorticity transport equation:
    dω/dt + u dω/dx + v dω/dy = ν ∇²ω
    using upwind for convection and central for diffusion.
    """
    omega_new = omega.copy()
    for i in range(1, I-1):
        for j in range(1, J-1):
            dωdx = (omega[i, j] - omega[i-1, j])/dx if u[i, j] > 0 else (omega[i+1, j] - omega[i, j])/dx
            dωdy = (omega[i, j] - omega[i, j-1])/dy if v[i, j] > 0 else (omega[i, j+1] - omega[i, j])/dy
            diffusion = nu * (
                (omega[i+1, j] - 2*omega[i, j] + omega[i-1, j]) / dx**2 +
                (omega[i, j+1] - 2*omega[i, j] + omega[i, j-1]) / dy**2
            )
            convection = -(u[i, j]*dωdx + v[i, j]*dωdy)
            omega_new[i, j] = omega[i, j] + dt * (diffusion + convection)
    return omega_new

# ------------------------------
# Time Integration Loop
# ------------------------------
u_hist, v_hist = [], []
rms_u_history, rms_v_history = [], []

for t in range(max_iter):
    boundary_conditions(psi, omega)
    psi = poisson_equation(psi, omega)
    u, v = velocity_update(psi)

    # Adaptive time-step from convection and diffusion limits
    umax, vmax = np.max(np.abs(u)), np.max(np.abs(v))
    dtc = sigma_c * dx * dy / (umax * dy + vmax * dx + 1e-8)
    dtd = sigma_d / (2 * nu * ((1/dx**2) + (1/dy**2)))
    dt = min(dtc, dtd)

    # Updating vorticity
    omega_new = vorticity_transport(omega, u, v, dt)

    # Computing RMS error on interior only
    if u_hist:
        du = u[1:-1, 1:-1] - u_hist[-1][1:-1, 1:-1]
        dv = v[1:-1, 1:-1] - v_hist[-1][1:-1, 1:-1]
        rms_u = np.sqrt(np.mean(du**2))
        rms_v = np.sqrt(np.mean(dv**2))
    else:
        rms_u = 1.0
        rms_v = 1.0

    # Storing results
    rms_u_history.append(rms_u)
    rms_v_history.append(rms_v)
    u_hist.append(u.copy())
    v_hist.append(v.copy())
    omega = omega_new

    # Convergence print
    if t % 100 == 0:
        print(f"Iter {t}: RMS_u={rms_u:.2e}, RMS_v={rms_v:.2e}")
    if rms_u < tol_u and rms_v < tol_v:
        print(f"Converged at iteration {t}")
        break

# ------------------------------
# Visualization Section
# ------------------------------

# 1. Stream Function Contours
plt.contourf(X, Y, psi.T, 50, cmap='plasma')
plt.colorbar()
plt.title("Stream Function Contours")
plt.xlabel("x"); plt.ylabel("y")
plt.axis('equal'); plt.show()

# 2. Streamlines
plt.streamplot(X, Y, u.T, v.T, density=2, linewidth=1, arrowsize=1, color='black')
plt.title("Streamlines")
plt.xlabel("x"); plt.ylabel("y")
plt.axis('equal'); plt.show()

# 3a. u-velocity (no comparison)
plt.plot(u[int(I/2), :], y, label="CFD Only")
plt.title("u-velocity (Vertical Midline)")
plt.xlabel("u"); plt.ylabel("y")
plt.legend(); plt.grid(); plt.show()

# 4a. v-velocity (no comparison)
plt.plot(x, v[:, int(J/2)], label="CFD Only")
plt.title("v-velocity (Horizontal Midline)")
plt.xlabel("x"); plt.ylabel("v")
plt.legend(); plt.grid(); plt.show()

# 3b. u-velocity (with Ghia et al.)
plt.plot(u[int(I/2), :], y, label="CFD")
plt.plot(ghia_u, ghia_y, 'o', label="Ghia et al.")
plt.title("u-velocity (Vertical Midline, Ghia Comparison)")
plt.xlabel("u"); plt.ylabel("y")
plt.legend(); plt.grid(); plt.show()

# 4b. v-velocity (with Ghia et al.)
plt.plot(x, v[:, int(J/2)], label="CFD")
plt.plot(ghia_x, ghia_v, 'o', label="Ghia et al.")
plt.title("v-velocity (Horizontal Midline, Ghia Comparison)")
plt.xlabel("x"); plt.ylabel("v")
plt.legend(); plt.grid(); plt.show()

# 5. Convergence History
plt.semilogy(range(1, len(rms_u_history)+1), rms_u_history, label="RMS_u")
plt.semilogy(range(1, len(rms_v_history)+1), rms_v_history, label="RMS_v")
plt.title("Convergence History")
plt.xlabel("Iteration")
plt.ylabel("RMS Residual (log scale)")
plt.legend(); plt.grid(); plt.show()
