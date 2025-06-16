import os
import argparse
import numpy as np
from PIL import Image
import skimage.color
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# --- Configuration ---
OUTPUT_DIR = "image_analysis_output"
VIZ_SAMPLE_SIZE = 25000
CHART_DPI = 300  # Set DPI for high-resolution 2D charts
CHART_BACKGROUND_COLOR = "#808080"  # 50% gray


def calculate_metrics(original_img_arr, compressed_img_arr):
    """Calculates MSE, PSNR, and SSIM between two images."""
    if original_img_arr.shape != compressed_img_arr.shape:
        raise ValueError("Input images must have the same dimensions.")
    mse = mean_squared_error(original_img_arr, compressed_img_arr)
    psnr = peak_signal_noise_ratio(original_img_arr, compressed_img_arr, data_range=255)
    ssim = structural_similarity(
        original_img_arr, compressed_img_arr, channel_axis=-1, data_range=255
    )
    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim}


def plot_2d_hsv_difference(original_rgb, compressed_rgb, chart_type, output_path):
    """Generates high-resolution 2D static plots of HSV differences with standardized axes."""
    hsv_orig = skimage.color.rgb2hsv(original_rgb)
    hsv_comp = skimage.color.rgb2hsv(compressed_rgb)

    h_orig, s_orig, v_orig = hsv_orig.reshape(-1, 3).T
    h_comp, s_comp, v_comp = hsv_comp.reshape(-1, 3).T
    if len(h_orig) > VIZ_SAMPLE_SIZE:
        indices = np.random.choice(len(h_orig), VIZ_SAMPLE_SIZE, replace=False)
        h_orig, s_orig, v_orig = h_orig[indices], s_orig[indices], v_orig[indices]
        h_comp, s_comp, v_comp = h_comp[indices], s_comp[indices], v_comp[indices]

    delta_h = h_comp - h_orig
    delta_h[delta_h > 0.5] -= 1.0
    delta_h[delta_h < -0.5] += 1.0
    delta_s = s_comp - s_orig
    delta_v = v_comp - v_orig

    # --- Plotting Setup ---
    # Use a light text color that works well on a gray background
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(CHART_BACKGROUND_COLOR)
    ax.set_facecolor(CHART_BACKGROUND_COLOR)

    chart_params = {
        "sv_vs_h_delta": (
            v_orig,
            s_orig,
            delta_h,
            "Saturation vs. Brightness (Colored by Hue Change)",
            "Original Brightness (Value)",
            "Original Saturation",
            "Hue Change (ΔH)",
            "twilight_shifted",
            (-0.1, 0.1),
        ),
        "hs_vs_v_delta": (
            s_orig,
            h_orig,
            delta_v,
            "Hue vs. Saturation (Colored by Brightness Change)",
            "Original Saturation",
            "Original Hue (0-1 scale)",
            "Brightness Change (ΔV)",
            "coolwarm",
            (-0.2, 0.2),
        ),
        "hv_vs_s_delta": (
            v_orig,
            h_orig,
            delta_s,
            "Hue vs. Brightness (Colored by Saturation Change)",
            "Original Brightness (Value)",
            "Original Hue (0-1 scale)",
            "Saturation Change (ΔS)",
            "coolwarm",
            (-0.2, 0.2),
        ),
    }
    x, y, c, title, xlabel, ylabel, clabel, cmap, clim = chart_params[chart_type]

    scatter = ax.scatter(x, y, c=c, cmap=cmap, s=15, alpha=0.6, edgecolors="none")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(clabel, fontsize=12)
    scatter.set_clim(clim)
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # --- Enforce Full Axis Ranges ---
    if "Saturation" in xlabel:
        ax.set_xlim(0, 1)
    if "Saturation" in ylabel:
        ax.set_ylim(0, 1)
    if "Brightness" in xlabel:
        ax.set_xlim(0, 1)
    if "Brightness" in ylabel:
        ax.set_ylim(0, 1)
    if "Hue" in xlabel:
        ax.set_xlim(0, 1)
    if "Hue" in ylabel:
        ax.set_ylim(0, 1)

    plt.tight_layout()
    # Save with specified DPI for higher resolution
    plt.savefig(output_path, dpi=CHART_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved 2D chart: {output_path}")


def plot_3d_interactive(rgb_orig, rgb_comp, plot_type, output_path):
    """Creates an interactive 3D plot using Plotly with standardized axes and background."""
    print(f"Generating 3D plot ({plot_type})... This might take a moment.")

    flat_rgb_orig = rgb_orig.reshape(-1, 3)
    if len(flat_rgb_orig) > VIZ_SAMPLE_SIZE:
        indices = np.random.choice(len(flat_rgb_orig), VIZ_SAMPLE_SIZE, replace=False)
        sampled_rgb_orig = flat_rgb_orig[indices]
    else:
        indices = np.arange(len(flat_rgb_orig))
        sampled_rgb_orig = flat_rgb_orig

    hsv_orig = skimage.color.rgb2hsv(sampled_rgb_orig.reshape(-1, 1, 3)).reshape(-1, 3)
    h_orig, s_orig, v_orig = hsv_orig.T
    marker_colors = [f"rgb({r},{g},{b})" for r, g, b in sampled_rgb_orig]

    fig = go.Figure()
    scene_layout = {}

    if plot_type == "distribution":
        x = s_orig * np.cos(h_orig * 2 * np.pi)
        y = s_orig * np.sin(h_orig * 2 * np.pi)
        z = v_orig

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=3, color=marker_colors, opacity=0.7),
                hovertext=[f"H:{h:.2f}, S:{s:.2f}, V:{v:.2f}" for h, s, v in hsv_orig],
                hoverinfo="text",
            )
        )

        scene_layout = dict(
            xaxis_title="Saturation (X-projection)",
            yaxis_title="Saturation (Y-projection)",
            zaxis_title="Value / Brightness",
            # Enforce full range on axes
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[0, 1]),
            aspectmode="cube",
        )
        title = "Interactive 3D Color Distribution (HSV Cylinder)"

    elif plot_type == "difference":
        flat_rgb_comp = rgb_comp.reshape(-1, 3)
        sampled_rgb_comp = flat_rgb_comp[indices]
        hsv_comp = skimage.color.rgb2hsv(sampled_rgb_comp.reshape(-1, 1, 3)).reshape(
            -1, 3
        )
        h_comp, s_comp, v_comp = hsv_comp.T

        delta_h = h_comp - h_orig
        delta_h[delta_h > 0.5] -= 1.0
        delta_h[delta_h < -0.5] += 1.0
        delta_s = s_comp - s_orig
        delta_v = v_comp - v_orig

        fig.add_trace(
            go.Scatter3d(
                x=delta_s,
                y=delta_v,
                z=delta_h,
                mode="markers",
                marker=dict(size=3, color=marker_colors, opacity=0.7),
                hovertext=[
                    f"Original H:{h:.2f},S:{s:.2f},V:{v:.2f}<br>ΔS:{ds:.3f},ΔV:{dv:.3f},ΔH:{dh:.3f}"
                    for h, s, v, ds, dv, dh in zip(
                        h_orig, s_orig, v_orig, delta_s, delta_v, delta_h
                    )
                ],
                hoverinfo="text",
            )
        )

        scene_layout = dict(
            xaxis_title="Saturation Change (ΔS)",
            yaxis_title="Brightness Change (ΔV)",
            zaxis_title="Hue Change (ΔH)",
            # Set axis range to encompass most logical errors. Full range (-1 to 1) often makes plot too sparse.
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-0.5, 0.5]),
            aspectmode="cube",
        )
        title = "Interactive 3D Color Difference Plot"

    fig.update_layout(
        title_text=title,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=scene_layout,
        paper_bgcolor=CHART_BACKGROUND_COLOR,  # Background of the entire chart area
        font=dict(color="white"),  # Text color for title, axes, etc.
    )

    fig.write_html(output_path)
    print(f"Saved interactive 3D chart: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="""Calculate quality metrics and visualize differences between an original image and its compressed versions.
Generates 2D static charts and 3D interactive HTML plots.""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "original_image", help="Path to the original, high-quality image."
    )
    parser.add_argument(
        "compressed_images",
        nargs="*",
        help="Path(s) to compressed/modified images. If none, only analyzes the original.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.original_image):
        print(f"Error: Original image not found at '{args.original_image}'")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    try:
        original_pil = Image.open(args.original_image).convert("RGB")
        original_array = np.array(original_pil)
    except Exception as e:
        print(f"Error loading original image '{args.original_image}': {e}")
        return

    print("\n--- Analyzing Original Image ---")
    orig_basename = os.path.splitext(os.path.basename(args.original_image))[0]
    plot_3d_interactive(
        original_array,
        None,
        "distribution",
        os.path.join(OUTPUT_DIR, f"{orig_basename}_3d_distribution.html"),
    )

    if not args.compressed_images:
        print("\nNo compressed images provided. Analysis complete.")
        return

    print("\n--- Comparing Against Compressed Images ---")
    print(f"{'File':<25} | {'MSE':<12} | {'PSNR (dB)':<12} | {'SSIM':<12}")
    print("-" * 70)

    for compressed_path in args.compressed_images:
        if not os.path.exists(compressed_path):
            print(f"Warning: Skipping non-existent file '{compressed_path}'")
            continue
        try:
            compressed_pil = Image.open(compressed_path).convert("RGB")
            compressed_array = np.array(compressed_pil)

            metrics = calculate_metrics(original_array, compressed_array)
            filename = os.path.basename(compressed_path)
            print(
                f"{filename:<25} | {metrics['MSE']:<12.2f} | {metrics['PSNR']:<12.2f} | {metrics['SSIM']:<12.4f}"
            )

            base_filename = os.path.splitext(filename)[0]

            plot_2d_hsv_difference(
                original_array,
                compressed_array,
                "sv_vs_h_delta",
                os.path.join(OUTPUT_DIR, f"{base_filename}_2d_SvV_HueDelta.png"),
            )
            plot_2d_hsv_difference(
                original_array,
                compressed_array,
                "hs_vs_v_delta",
                os.path.join(OUTPUT_DIR, f"{base_filename}_2d_HvS_ValDelta.png"),
            )
            plot_2d_hsv_difference(
                original_array,
                compressed_array,
                "hv_vs_s_delta",
                os.path.join(OUTPUT_DIR, f"{base_filename}_2d_HvV_SatDelta.png"),
            )

            plot_3d_interactive(
                original_array,
                compressed_array,
                "difference",
                os.path.join(OUTPUT_DIR, f"{base_filename}_3d_difference.html"),
            )
            print("-" * 70)

        except ValueError as e:
            print(f"Skipping '{compressed_path}': {e}")
        except Exception as e:
            print(f"Could not process file '{compressed_path}': {e}")


if __name__ == "__main__":
    main()
