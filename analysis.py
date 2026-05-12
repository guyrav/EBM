import numpy as np
import pandas as pd
import plotnine as gg
from npeet import entropy_estimators as ee


def compute_series_mi_per_dt(df, column_1="incoming", column_2="outgoing", k=3):
    # 1. Compute MI for each dt and each ensemble member
    mi_by_dt_ensemble = (
        df.groupby(["dt", "ensemble"])
        .apply(lambda g: ee.mi(g[column_1].to_numpy(), g[column_2].to_numpy(), k=k))
        .reset_index(name="mi")
    )
    return mi_by_dt_ensemble


def compute_ensemble_mi_per_dt(times, df, column_1="incoming", column_2="outgoing", k=3):
    df_filtered = df[df["time"].isin(times)]

    mi_by_dt_time = (
        df_filtered
        .groupby(["dt", "time"])
        .apply(lambda g: ee.mi(
            g[column_1].to_numpy(),
            g[column_2].to_numpy(),
            k=k
        ))
        .reset_index(name="mi")
    )

    return mi_by_dt_time


def plot_series_mi_vs_dt_relative(mi_by_dt_ensemble, subtitle):
    dt_min = mi_by_dt_ensemble["dt"].min()

    ref = (
        mi_by_dt_ensemble[mi_by_dt_ensemble["dt"] == dt_min]
        [["ensemble", "mi"]]
        .rename(columns={"mi": "mi_ref"})
    )

    mi_plot_df = mi_by_dt_ensemble.merge(ref, on="ensemble", how="left")

    mi_plot_df["mi_diff"] = mi_plot_df["mi"] - mi_plot_df["mi_ref"]
    mi_plot_df["abs_mi_diff"] = mi_plot_df["mi_diff"].abs()

    mi_plot_df = mi_plot_df[mi_plot_df["dt"] != dt_min].copy()
    single_member_df = mi_plot_df[mi_plot_df["ensemble"] == 0]

    p = (
        gg.ggplot(mi_plot_df, gg.aes(x="factor(dt)", y="abs_mi_diff"))
        + gg.geom_boxplot()

        # Overlay single ensemble member
        + gg.geom_line(
            single_member_df,
            gg.aes(group=1),
            color="red"
        )
        + gg.geom_point(
            single_member_df,
            color="red",
            size=2
        )

        + gg.scale_y_log10()
        + gg.theme_minimal()
        + gg.labs(
            x="dt",
            y="|MI(dt) - MI(dt_min)|",
            title="Convergence of MI estimate as dt decreases",
            subtitle=subtitle
        )
    )

    p.show()


def plot_series_mi_vs_dt_relative_regression(mi_by_dt_ensemble, subtitle):
    dt_min = mi_by_dt_ensemble["dt"].min()

    ref = (
        mi_by_dt_ensemble[mi_by_dt_ensemble["dt"] == dt_min]
        [["ensemble", "mi"]]
        .rename(columns={"mi": "mi_ref"})
    )

    mi_plot_df = mi_by_dt_ensemble.merge(ref, on="ensemble", how="left")

    mi_plot_df["mi_diff"] = mi_plot_df["mi"] - mi_plot_df["mi_ref"]
    mi_plot_df["abs_mi_diff"] = mi_plot_df["mi_diff"].abs()

    mi_plot_df = mi_plot_df[mi_plot_df["dt"] != dt_min]
    mi_plot_df["log_dt"] = np.log10(mi_plot_df["dt"])
    mi_plot_df["log_abs_mi_diff"] = np.log10(mi_plot_df["abs_mi_diff"])

    p_regression = (
        gg.ggplot(mi_plot_df, gg.aes(x="log_dt", y="log_abs_mi_diff"))
        + gg.geom_point(alpha=0.5)
        + gg.geom_smooth(method="lm", se=False, color="red")
        + gg.theme_minimal()
        + gg.labs(
            x="dt",
            y="|MI(dt) - MI(dt_min)|",
            title="Convergence of MI estimate as dt decreases",
            subtitle=subtitle
        )
    )

    p_regression.show()


def plot_ensemble_mi_vs_dt(mi_by_dt_time, subtitle):
    mi_by_dt_time["time_label"] = pd.Categorical(
        mi_by_dt_time["time"].map(lambda t: f"time = {t} s"),
        categories=[f"time = {t} s" for t in sorted(mi_by_dt_time["time"].unique())],
        ordered=True
    )

    p = (
        gg.ggplot(mi_by_dt_time, gg.aes(x="dt", y="mi"))
        + gg.geom_point()
        + gg.geom_line()
        + gg.facet_wrap("~time_label")
        + gg.scale_x_log10()
        + gg.theme_minimal()
        + gg.labs(
            x="dt",
            y="MI(incoming, outgoing)",
            title="Convergence of ensemble MI estimate as dt decreases",
            subtitle=subtitle
        )
    )

    p.show()


def plot_ensemble_mi_vs_dt_relative(mi_by_dt_time, subtitle):
    dt_min = mi_by_dt_time["dt"].min()

    ref = (
        mi_by_dt_time[mi_by_dt_time["dt"] == dt_min]
        [["time", "mi"]]
        .rename(columns={"mi": "mi_ref"})
    )

    mi_plot_df = mi_by_dt_time.merge(ref, on="time", how="left")

    mi_plot_df["mi_diff"] = mi_plot_df["mi"] - mi_plot_df["mi_ref"]
    mi_plot_df["abs_mi_diff"] = mi_plot_df["mi_diff"].abs()

    mi_plot_df = mi_plot_df[mi_plot_df["dt"] != dt_min].copy()

    p = (
        gg.ggplot(mi_plot_df, gg.aes(x="dt", y="abs_mi_diff"))
        + gg.geom_point()
        + gg.geom_line()
        + gg.facet_wrap("~time")
        + gg.scale_x_log10()
        + gg.scale_y_log10()
        + gg.theme_minimal()
        + gg.labs(
            x="dt",
            y="|MI(dt) - MI(dt_min)|",
            title="Convergence of ensemble MI estimate as dt decreases",
            subtitle=subtitle
        )
    )

    p.show()


def main():
    sigma = 1
    rho = 0.6
    # ref_value = -0.5 * np.log2(1 - rho ** 2)

    # timestamps = [10, 150, 300, 450, 600, 750, 900]
    cutoff = 400

    print("Reading data...")
    df = pd.read_parquet(f"./data/data_sigma={sigma}_rho={rho}.parquet")

    print("Computing MI...")
    # mi_by_dt = compute_ensemble_mi_per_dt(timestamps, df, k=3)
    mi_by_dt_ensemble = compute_series_mi_per_dt(df[df["time"] > cutoff])

    # boxplot
    print("Plotting...")
    # plot_ensemble_mi_vs_dt(mi_by_dt, f"Ensemble size = {df["ensemble"].nunique()}, Noise σ={sigma}, ρ={rho:.1f}")
    plot_series_mi_vs_dt_relative_regression(mi_by_dt_ensemble, f"sigma={sigma}, rho={rho}, cutoff_time={cutoff}")


if __name__ == "__main__":
    main()
