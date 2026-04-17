import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1) AGGREGATION FUNCTION
# ------------------------------------------------------------
def compute_mean_steps_per_signal(
    df,
    reference_col,
    sensor_col="sensor_id",
    step_col="cycle_step_index",
    time_col="cycle_cumulative_time_ms",
    heater_col="step_heater_duration",
    name_col="name"
):
    """
    Computes mean and std per signal (name), sensor and step.
    Returns a dataframe ready for radar plotting.
    """

    mean_steps = (
        df
        .groupby([
            name_col,
            sensor_col,
            step_col,
            time_col,
            heater_col
        ], as_index=False)
        .agg(
            mean=(reference_col, "mean"),
            std=(reference_col, "std")
        )
    )

    return mean_steps


# ------------------------------------------------------------
# 2) RADAR PLOT FUNCTION
# ------------------------------------------------------------
def plot_multi_signal_radar(
    df,
    reference_col,
    sensor_order,
    steps_per_sensor=10,
    radial_limit=1.05,
    title="Radar Plot — Sensor Blocks Shaded",
    figsize=(10, 10)
):
    """
    Computes mean_steps internally and creates a multi-signal radar plot.
    Each unique df['name'] becomes one radar trace.
    """

    # ---- compute aggregated data ----
    mean_steps = compute_mean_steps_per_signal(df, reference_col)

    total_points = len(sensor_order) * steps_per_sensor

    # ---- create angles ----
    angles = np.linspace(0, 2 * np.pi, total_points, endpoint=False)
    angles = np.concatenate((angles, angles[:1]))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # ---------------------------------------------------------
    # Plot each signal
    # ---------------------------------------------------------
    for signal in mean_steps["name"].unique():

        radar_values = []
        signal_df = mean_steps[mean_steps["name"] == signal]

        for sensor in sensor_order:

            sensor_data = (
                signal_df[signal_df["sensor_id"] == sensor]
                .sort_values("cycle_step_index")
            )

            for step in range(steps_per_sensor):

                step_val = sensor_data[
                    sensor_data["cycle_step_index"] == step
                ]["mean"]

                radar_values.append(
                    step_val.mean() if len(step_val) > 0 else 0
                )

        radar_values += radar_values[:1]

        ax.plot(angles, radar_values, linewidth=2, label=str(signal))
        ax.fill(angles, radar_values, alpha=0.15)

    # ---------------------------------------------------------
    # Reference circle at radius = 1
    # ---------------------------------------------------------
    ax.plot(
        angles,
        np.ones_like(angles),
        linestyle="--",
        color="black",
        alpha=0.5
    )

    # ---------------------------------------------------------
    # Step labels
    # ---------------------------------------------------------
    labels = []
    for _ in sensor_order:
        for step in range(1, steps_per_sensor + 1):
            labels.append(f"St{step}")
    labels.append(labels[0])

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=7)

    # ---------------------------------------------------------
    # Sensor block labels
    # ---------------------------------------------------------
    sector_width = 2 * np.pi / len(sensor_order)

    for i, sensor in enumerate(sensor_order):
        center_angle = (i + 0.5) * sector_width
        ax.text(
            center_angle,
            1.18,
            f"S{sensor}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold"
        )

    # ---------------------------------------------------------
    # Dashed separators between sensors
    # ---------------------------------------------------------
    for i in range(len(sensor_order)):
        angle = i * (2 * np.pi / total_points) * steps_per_sensor
        ax.plot(
            [angle, angle],
            [0, radial_limit],
            linestyle="--",
            color="gray",
            alpha=0.5
        )

    ax.set_ylim(0, radial_limit)

    plt.title(title, pad=40)
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()

    return fig, ax