import numpy as np
import pandas as pd
import plotnine as gg


def plot_ensemble_cloud(df, sigma, rho):
    columns = ["T", "incoming", "outgoing"]

    long_df = df[df["sample"] > 250]

    long_df = long_df.melt(
        id_vars=["ensemble", "sample"],
        value_vars=columns,
        var_name="variable",
        value_name="value"
    )

    mean_df = (
        long_df
        .groupby(["sample", "variable"], as_index=False)["value"]
        .mean()
    )

    cloud_df = long_df.copy()
    cloud_df["plot_value"] = cloud_df["value"]

    cloud_df.loc[
        (cloud_df["variable"] == "T")
        & ((cloud_df["plot_value"] < -5) | (cloud_df["plot_value"] > 10)),
        "plot_value"
    ] = np.nan

    ref_df = pd.DataFrame({
        "variable": ["T", "T"],
        "y": [0, 5]
    })

    p = (
        gg.ggplot()
        # individual ensemble members: faint cloud
        + gg.geom_line(
            cloud_df,
            gg.aes(x="sample", y="plot_value", group="ensemble"),
            alpha=0.05,
            size=0.3
        )
        # ensemble mean: highlighted
        + gg.geom_line(
            mean_df,
            gg.aes(x="sample", y="value"),
            color="red",
            size=1.0
        )
        + gg.geom_hline(
            data=ref_df,
            mapping=gg.aes(yintercept="y"),
            linetype="dashed",
            color="blue"
        )
        + gg.facet_wrap("~ variable", scales="free_y", ncol=1)
        + gg.labs(
            title="Ensemble trajectories",
            subtitle=f"(σ={sigma}, ρ={rho:.1f})",
            x="Sample",
            y="Value"
        )
        + gg.theme_minimal()
    )

    p.show()


def read_data(filename):
    data = np.load(filename)

    n_sample, n_ensemble = data[data.files[0]].shape

    df = pd.DataFrame({
        "sample": np.repeat(np.arange(n_sample), n_ensemble),
        "ensemble": np.tile(np.arange(n_ensemble), n_sample),
    })

    for name in data.files:
        df[name] = data[name].ravel()

    return df


def main():
    sigma = 2
    rho = 0.6

    df = read_data(f"data_sigma={sigma}_rho={rho:.1f}.npz")
    plot_ensemble_cloud(df, sigma, rho)


if __name__ == "__main__":
    main()
