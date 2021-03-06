import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

'''
The function synthetic_velocity_v02 takes two inputs
1) s, int, random seed
2) t, 1d array of time values that must be in the range [0, 150]

The function will output a 1d array of velocities corresponding the the input times
'''


def synthetic_velocity_v02(s, t):
    # Set seed for repeatability
    np.random.seed(s)

    # Storage for coordinates
    coord = np.zeros((6, 2))

    # Point 1
    coord[0, 0] = 0
    coord[0, 1] = 0

    # Point 2
    coord[1, 0] = 49 * np.random.random_sample(1) + 1
    coord[1, 1] = 0

    # Point 3
    coord[2, 0] = coord[1, 0] + 9.9 * np.random.random_sample(1) + 0.1
    coord[2, 1] = 1000 * np.random.random_sample(1) + 500

    # Point 4
    coord[3, 0] = coord[2, 0] + 39.9 * np.random.random_sample(1) + 0.1
    coord[3, 1] = (0.1 * np.random.random_sample(1) + 0.95) * coord[2, 1]

    # Point 5
    coord[4, 0] = coord[3, 0] + 20 * np.random.random_sample(1) + 10
    coord[4, 1] = (0.5 * np.random.random_sample(1) + 0.25) * coord[3, 1]

    # Point 6
    coord[5, 0] = coord[4, 0] + 10 * np.random.random_sample(1) + 5
    coord[5, 1] = (0.65 * np.random.random_sample(1) + 0.25) * \
                  (coord[3, 1] - coord[4, 1]) + coord[4, 1]

    # Space left after point 6
    R = 150 - coord[5, 0]

    # Interval for each of the next points
    I = coord[5, 0] - coord[4, 0]

    # Number of points left
    N = np.floor(R / I)

    # Assign coordinates for remaining points
    for point in range(int(N)):
        x = ((coord[5, 0] - coord[4, 0]) * (point + 1)) + coord[5, 0]
        if (point % 2) == 0:
            y = coord[4, 1]
        else:
            y = coord[5, 1]
        coord = np.vstack((coord, np.array([x, y])))

    # Create final coordinate at t=150ns
    x_end = 150
    r = (150 - coord[-1, 0]) / (coord[5, 0] - coord[4, 0])
    if coord[-1, 1] == coord[4, 1]:
        y_end = r * (coord[5, 1] - coord[4, 1]) + coord[4, 1]
    else:
        y_end = coord[5, 1] - r * (coord[5, 1] - coord[4, 1])
    coord = np.vstack((coord, np.array([x_end, y_end])))

    # Generate more data points with interpolation
    f = interp1d(coord[:, 0], coord[:, 1])
    x_interp = np.arange(0, 150.1, 0.1)
    coord_interp = np.zeros((x_interp.shape[0], 2))
    coord_interp[:, 0] = x_interp
    coord_interp[:, 1] = f(coord_interp[:, 0])

    # Smooth the curve with a Savitzky-Golay filter
    sg = savgol_filter(coord_interp[:, 1], 59, 1)
    coord_sg = np.zeros(coord_interp.shape)
    coord_sg[:, 0] = coord_interp[:, 0]
    coord_sg[:, 1] = sg

    # Interpolate on the sg data to get the velocity at the requested points
    v_func = interp1d(coord_sg[:, 0], coord_sg[:, 1])
    v = v_func(t)

    # Set any negative points from the filtering to be zero
    v[v < 0] = 0

    return v


# Example usage 
fs = 0.5
time = np.arange(0, 150 + fs, fs)
seeds = np.array([1, 2, 3, 4, 5])
fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=250)

for seed in seeds:
    v = synthetic_velocity_v02(seed, time)
    ax.plot(time, v, '-', linewidth=3)

ax.set_xlim([0, 150])
ax.set_ylim([-40, 1600])
ax.set_xlabel('Time (ns)')
ax.set_ylabel('Velocity (m/s)')
plt.show()
