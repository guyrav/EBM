import numpy as np
import pandas as pd
import plotnine as gg
from npeet import entropy_estimators as ee


def compute_mi_per_dt(df, column_1="incoming", column_2="outgoing", k=3):
    # 1. Compute MI for each dt
    mi_by_dt = (
        df.groupby("dt")
        .apply(lambda g: ee.mi(g[column_1].to_numpy(), g[column_2].to_numpy(), k=k))
        .reset_index(name="mi")
    )
    return mi_by_dt


def plot_mi_vs_dt(mi_by_dt, subtitle):
    # 2. Compare each value to the smallest dt
    dt_min = mi_by_dt["dt"].min()
    mi_ref = mi_by_dt.loc[mi_by_dt["dt"] == dt_min, "mi"].iloc[0]
    mi_by_dt["mi_diff"] = mi_by_dt["mi"] - mi_ref
    mi_by_dt["abs_mi_diff"] = mi_by_dt["mi_diff"].abs()
    # remove reference row, whose difference is zero
    mi_diff_df = mi_by_dt[mi_by_dt["dt"] != dt_min].copy()
    p = (
            gg.ggplot(mi_diff_df, gg.aes(x="dt", y="abs_mi_diff"))
            + gg.geom_point()
            + gg.geom_line()
            + gg.scale_x_log10()
            + gg.scale_y_log10()
            + gg.labs(
                x="dt",
                y="|MI(dt) - MI(dt_min)|",
                title="Convergence of MI estimate as dt decreases"
            )
    )

    p.show()


def main():
    sigma = 1
    rho = 0.6
    # ref_value = -0.5 * np.log2(1 - rho ** 2)

    transient_time_cutoff = 400

    print("Reading data...")
    df = pd.read_parquet(f"./data/data_sigma={sigma}_rho={rho}.parquet")

    print("Computing MI...")
    mi_by_dt = compute_mi_per_dt(df[df["time"] > transient_time_cutoff], k=3)

    # boxplot
    print("Plotting...")
    plot_mi_vs_dt(mi_by_dt, f"σ={sigma}, ρ={rho:.1f}")


if __name__ == "__main__":
    main()
