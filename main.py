import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.io import wavfile
from IPython.display import Audio
import numba
import time

Nx = 101
L = 0.65

Nt = 500000
T = 5

f = 110
damping = 1e-4
l = 1e-5

c = 2 * f * L
x = np.linspace(0, L, Nx)
dx = np.diff(x)[0]
dt = T / Nt


@numba.jit("f8[:,:](f8[:,:], f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def solve_undamped(sol, N_t, N_x, d_t, d_x, c):
    sol = sol.copy()
    for t in range(1, N_t - 1):
        for j in range(1, N_x - 1):
            factor = (c * d_t / d_x) ** 2
            p1 = sol[t][j + 1] - 2 * sol[t][j] + sol[t][j - 1]
            p2 = 2 * sol[t][j] - sol[t - 1][j]
            sol[t + 1][j] = factor * p1 + p2
    return sol


@numba.jit("f8[:,:](f8[:,:], f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def solve_damped(sol, N_t, N_x, d_t, d_x, c, gamma):
    sol = sol.copy()
    for t in range(1, N_t - 1):
        for j in range(1, N_x - 1):
            f1 = (c * d_t / d_x) ** 2
            f2 = -gamma * c ** 2 * dt
            p1 = sol[t][j + 1] - 2 * sol[t][j] + sol[t][j - 1]
            p2 = sol[t][j] - sol[t - 1][j]
            p3 = 2 * sol[t][j] - sol[t - 1][j]
            sol[t + 1][j] = f1 * p1 + f2 * p2 + p3
    return sol


@numba.jit("f8[:,:](f8[:,:], f8, f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def solve_stiff(sol, N_t, N_x, d_t, d_x, c, gamma, l):
    sol = sol.copy()
    for t in range(1, N_t - 1):
        for j in range(2, N_x - 2):
            f1 = (c * d_t / d_x) ** 2
            f2 = -gamma * c ** 2 * dt
            f4 = -(l*c*dt/dx**2)**2
            p1 = sol[t][j + 1] - 2 * sol[t][j] + sol[t][j - 1]
            p2 = sol[t][j] - sol[t - 1][j]
            p3 = 2 * sol[t][j] - sol[t - 1][j]
            p4 = sol[t][j-2] - 4 * sol[t][j-1] + \
                 4 * sol[t][j] - 4 * sol[t][j+1] + sol[t][j+2]
            sol[t + 1][j] = f1 * p1 + f2 * p2 + p3 + f4 * p4
    return sol


def animate_solution(solution, frame_interval, file_name, frames=500, fps=20):
    def animate(i):
        ax.clear()
        for sol in solution:
            ax.plot(x, sol[i * frame_interval])
        ax.set_ylim(-0.01, 0.01)
        ax.text(0.05, 0.95, f"Time: {i * dt * 1000:.3f}ms",
                transform=ax.transAxes, fontsize=14,
                verticalalignment='top')

    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(-0.01, 0.01)
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=50)
    ani.save(file_name, writer='pillow', fps=fps)


def fourier_coefficient(solution, n):
    sin = np.sin(n * np.pi * x)
    return np.multiply(sin, solution).sum(axis=1)


def save_audio(solution, sample_rate, file_name, harmonics=10):
    if harmonics:
        wave = np.sum([fourier_coefficient(solution, i + 1)
                       for i in range(harmonics)], axis=0)
    else:
        wave = np.sum(solution, axis=1)

    wavfile.write(file_name, sample_rate, wave.astype(np.float32))


def main():
    ya = np.linspace(0, 0.01, 25)
    yb = np.linspace(0.01, 0, 76)
    y0 = np.concatenate([ya, yb])

    sol = np.zeros((Nt, Nx))
    sol[0] = y0
    sol[1] = y0

    sol1 = solve_stiff(sol, Nt, Nx, dt, dx, c, damping, l)
    sol2 = solve_damped(sol, Nt, Nx, dt, dx, c, damping)
    sol3 = solve_undamped(sol, Nt, Nx, dt, dx, c)


    # animate_solution([sol1, sol2, sol3], 10, "animations/combination.gif")

    # sample_rate = int(1/dt/2)
    # harmonics = 30
    # audio_name = f"audio/stiff-30-harmonics.wav"
    # save_audio(sol1[::2], sample_rate, audio_name, harmonics=harmonics)
    # Audio(audio_name)


if __name__ == '__main__':
    main()
