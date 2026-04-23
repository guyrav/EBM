import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_boxplot, geom_hline, stat_summary, theme_minimal, scale_color_manual, labs, \
    geom_jitter
from npeet import entropy_estimators as ee


def read_data(filename):
    data = np.load(filename)

    n_ensemble, n_sample = data[data.files[0]].shape

    df = pd.DataFrame({
        "ensemble": np.repeat(np.arange(n_ensemble), n_sample),
        "sample": np.tile(np.arange(n_sample), n_ensemble),
    })

    for name in data.files:
        df[name] = data[name].ravel()

    return df


def add_regime_column(df, far_range, near_range):
    df["regime"] = None
    df.loc[df["sample"].between(*far_range), "regime"] = "far"
    df.loc[df["sample"].between(*near_range), "regime"] = "near"
    return df


def detrend_moving_average(x, window):
    trend = (
        pd.Series(x)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    return x - trend


def compute_mi(g, detrend_incoming=False, window=51):
    incoming = g["incoming"].to_numpy()
    outgoing = g["outgoing"].to_numpy()

    if detrend_incoming:
        incoming = detrend_moving_average(incoming, window=window)

    return ee.mi(incoming, outgoing, k=3)


def plot_mi(mi_df, ref_value, title, subtitle):
    p = (
            ggplot(mi_df, aes(x="regime", y="mi"))
            + geom_boxplot()
            + geom_jitter(width=0.1, alpha=0.2)
            + stat_summary(
                aes(color='"Mean"'),
                fun_y=np.mean,
                geom="point",
                size=2
            )
            + scale_color_manual(values={"Mean": "red"})
            + geom_hline(yintercept=ref_value, linetype="dashed", color="red")
            + labs(title=title, subtitle=subtitle, x="Regime", y="MI", color="")
            + theme_minimal()
    )

    p.show()


def main():
    sigma = 2
    rho = 0.6
    ref_value = -0.5 * np.log2(1 - rho ** 2)

    print("Reading data...")
    df = read_data(f"data_sigma={sigma}_rho={rho:.1f}.npz")

    near_range = (0, 250)
    far_range = (500, 1000)

    df = add_regime_column(df, far_range=far_range, near_range=near_range)

    print("Computing MI...")
    # near and far, without detrending
    result_base = (
        df[df["regime"].isin(["near", "far"])]
        .groupby(["ensemble", "regime"])
        .apply(lambda g: compute_mi(g, detrend_incoming=False))
        .reset_index(name="mi")
    )

    # far, with detrended incoming
    result_far_detrended = (
        df[df["regime"] == "far"]
        .groupby("ensemble")
        .apply(lambda g: compute_mi(g, detrend_incoming=True, window=51))
        .reset_index(name="mi")
    )

    result_far_detrended["regime"] = "far_detrended"

    # combine all three
    result_all = pd.concat([result_base, result_far_detrended], ignore_index=True)

    result_all["regime"] = pd.Categorical(
        result_all["regime"],
        categories=["near", "far", "far_detrended"],
        ordered=True
    )

    # boxplot
    print("Plotting...")
    plot_mi(result_all, ref_value, "MI of incoming heat and outgoing radiation", f"σ={sigma}, ρ={rho:.1f}")


if __name__ == "__main__":
    main()
