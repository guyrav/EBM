import numpy as np
import pandas as pd
import numba


albedo_threshold = 5


@numba.njit
def generate_noise_terms(sigma, rho, sqrt_1_minus_rho_sq):
    x = np.random.randn()
    y = np.random.randn()

    return sigma * x, sigma * (rho * x + sqrt_1_minus_rho_sq * y)


@numba.njit
def albedo_simple(T):
    if T > albedo_threshold:
        return 1.0
    if T < 0:
        return 0.0
    else:
        T_normalised = T / albedo_threshold
        return 3 * T_normalised**2 - 2 * T_normalised**3


@numba.njit
def outgoing_radiation_linear(T):
    return (T + 1) / (albedo_threshold + 2)


@numba.njit
def simulate(T0, dt, n_steps, save_every_steps, sigma, rho):
    sqrt_1_minus_rho_sq = np.sqrt(1 - rho**2)
    sigma_times_sqrt_dt = sigma * np.sqrt(dt)

    n_ensemble = T0.size
    n_save = n_steps // save_every_steps + 1

    T = T0.copy()
    out_T = np.empty((n_save, n_ensemble))
    out_T[0, :] = T
    out_incoming = np.empty((n_save, n_ensemble))
    out_incoming[0, :] = 0
    out_outgoing = np.empty((n_save, n_ensemble))
    out_outgoing[0, :] = 0
    out_noise_incoming = np.empty((n_save, n_ensemble))
    out_noise_incoming[0, :] = 0
    out_noise_outgoing = np.empty((n_save, n_ensemble))
    out_noise_outgoing[0, :] = 0

    sqrt_dt = np.sqrt(dt)
    save_idx = 1

    incoming_noise_sum = np.zeros(n_ensemble)
    outgoing_noise_sum = np.zeros(n_ensemble)
    incoming_sum = np.zeros(n_ensemble)
    outgoing_sum = np.zeros(n_ensemble)

    for n in range(n_steps):
        for i in range(n_ensemble):
            # Compute
            noise_incoming, noise_outgoing = generate_noise_terms(sigma_times_sqrt_dt, rho, sqrt_1_minus_rho_sq)
            incoming = dt * albedo_simple(T[i])
            outgoing = dt * outgoing_radiation_linear(T[i])

            # Accumulate
            incoming_noise_sum += noise_incoming
            outgoing_noise_sum += noise_outgoing
            incoming_sum += incoming
            outgoing_sum += outgoing

            # Update
            T[i] += (incoming - outgoing + noise_incoming - noise_outgoing)

        if n % save_every_steps == 0:
            out_T[save_idx, :] = T
            out_incoming[save_idx, :] = incoming_sum
            out_outgoing[save_idx, :] = outgoing_sum
            out_noise_incoming[save_idx, :] = incoming_noise_sum
            out_noise_outgoing[save_idx, :] = outgoing_noise_sum

            incoming_sum[:] = 0
            outgoing_sum[:] = 0
            incoming_noise_sum[:] = 0
            outgoing_noise_sum[:] = 0

            save_idx += 1

    return out_T, out_incoming, out_outgoing, out_noise_incoming, out_noise_outgoing


def add_data(df, dt, timestamps, T, incoming, outgoing,
             noise_incoming, noise_outgoing):
    n_times, n_ensemble = T.shape

    new_df = pd.DataFrame({
        "T": T.ravel(),
        "time": np.repeat(timestamps, n_ensemble),
        "dt": dt,
        "ensemble": np.tile(np.arange(n_ensemble), n_times),
        "incoming": incoming.ravel(),
        "outgoing": outgoing.ravel(),
        "incoming_noise": noise_incoming.ravel(),
        "outgoing_noise": noise_outgoing.ravel(),
    })

    if df is None:
        return new_df

    return pd.concat([df, new_df], ignore_index=True)



def main():
    # Note: for arbitrary dt values we'll need to save by testing "time > n seconds"
    # and then record also the elapsed time since last save to later normalise the change and noise terms.
    dts = [1., 1./2, 1./4, 1./8, 1./16, 1./32, 1./64, 1./128, 1./256, 1./512, 1./1024]
    T0 = np.array([6.])  # Implicit ensemble size of 1
    total_time = 1000
    sigma = 1
    rho = 0.6
    df = None
    timestamps = np.arange(total_time + 1)

    np.random.seed(1995)
    for dt in dts:
        n_steps = int(total_time / dt)
        save_every_steps = int(1 / dt)
        print(f"Simulating with dt = {dt}...", end=' ')
        T, incoming, outgoing, noise_incoming, noise_outgoing = simulate(T0, dt, n_steps, save_every_steps, sigma, rho)
        print("done.")
        df = add_data(df, dt, timestamps, T, incoming, outgoing, noise_incoming, noise_outgoing)

    df["sigma"] = sigma
    df["rho"] = rho
    df.to_parquet(f"./data/data_sigma={sigma}_rho={rho}.parquet")


if __name__ == "__main__":
    main()
