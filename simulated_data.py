from typing import Callable

import numpy as np
import os

# def get_ode_temperature_change(T: np.ndarray[np.float64],
#                                a: Callable,
#                                out_radiation_coeff: np.float64, out_radiation_intercept: np.float64) -> np.float64:
#     return a(T) - (out_radiation_coeff * T + out_radiation_intercept)

albedo_threshold = 5

def albedo(T):
    global albedo_threshold
    res = np.zeros_like(T)

    res[T > albedo_threshold] = 1.0
    T_normalised = T / albedo_threshold
    mask = np.logical_and(0 < T, T < albedo_threshold)
    res[mask] = 3 * T_normalised[mask]**2 - 2 * T_normalised[mask]**3

    return res

def outgoing_radiation(T):
    # I know it's unphysical but I can do whatever I want, you're not the boss of me!
    return np.minimum((T + 1) / (albedo_threshold + 2), np.log(np.maximum(T, 1)))

def generate_noise(num_samples, sample_size, means, sigma_sq, rho):
    cov = np.array([[sigma_sq, rho * sigma_sq], [rho * sigma_sq, sigma_sq]])
    samples = np.random.multivariate_normal(means, cov, (num_samples, sample_size))
    return samples

def generate_0d_data(initial_temperature, num_samples, ensemble_size, path):
    np.random.seed(1995)

    sigma_sq = 2.
    means = np.zeros((2,), dtype=float)
    rho = 0.6

    noise = generate_noise(num_samples, ensemble_size, means, sigma_sq, rho)

    res = np.zeros((num_samples, ensemble_size, 3), dtype=np.float64)
    res[0, :, 0] = initial_temperature
    res[0, :, 1] = albedo(res[0, :, 0])
    res[0, :, 2] = outgoing_radiation(res[0, :, 0])
    res[0, :, 1:3] += noise[0]

    for i in range(1, num_samples):
        res[i, :, 0] = res[i - 1, :, 0] + res[i - 1, :, 1] - res[i - 1, :, 2]  # T + incoming_radiation - outgoing_radiation
        res[i, :, 1] = albedo(res[i, :, 0])
        res[i, :, 2] = outgoing_radiation(res[i, :, 0])
        res[i, :, 1:3] += noise[i]

    np.savez(path,
             T=res[:, :, 0], incoming=res[:, :, 1], outgoing=res[:, :, 2],
             incoming_noise=noise[:, :, 0], outgoing_noise=noise[:, :, 1])

if __name__ == '__main__':
    ensemble_size = 1000
    num_samples = 1000
    initial_temperature = 1400
    generate_0d_data(initial_temperature, num_samples, ensemble_size, os.path.join(os.getcwd(), "data"))


