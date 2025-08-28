import numpy as np
from skimage import measure
from shapely.geometry import LineString
from shapely import speedups
import matplotlib.pyplot as plt
import os

if speedups.available:
    speedups.enable()

def extract_shoreline(mask):
    """
    Extracts the main shoreline contour from a binary mask.
    """
    binary_mask = mask.squeeze().cpu().numpy()
    contours = measure.find_contours(binary_mask, 0.5)
    if not contours:
        return None
    longest = max(contours, key=len)
    return np.array(longest)  # shape: [N, 2] in (y, x)

def compute_normals(points, delta=5):
    normals = []
    N = len(points)

    for i in range(N):
        if i < delta:
            p0, p1 = points[i], points[min(i + delta, N - 1)]
        elif i > N - delta - 1:
            p0, p1 = points[max(i - delta, 0)], points[i]
        else:
            p0, p1 = points[i - delta], points[i + delta]

        dx = p1[0] - p0[0]  # x
        dy = p1[1] - p0[1]  # y
        normal = np.array([-dy, dx])
        norm = np.linalg.norm(normal)
        normals.append(normal / norm if norm > 0 else np.array([0.0, 0.0]))

    return np.array(normals)



def create_transects(shoreline, normals, length_m=200, resolution=20, step=5):
    """
    Creates transects centered on the shoreline.
    """
    half_pixels = (length_m // 2) // resolution
    transects = []
    for i in range(0, len(shoreline), step):
        p, n = shoreline[i], normals[i]
        start = p - n * half_pixels
        end = p + n * half_pixels
        transects.append((start, end))
    return np.array(transects)  # shape: [M, 2, 2]

from shapely.geometry import Point, MultiPoint, LineString, GeometryCollection

from shapely.geometry import LineString
from scipy.signal import savgol_filter


def resample_shoreline(shoreline, spacing=5.0, smooth=True, window_length=11, polyorder=2):
    """
    Resample a shoreline contour to have points spaced equally along its length.
    Optionally smooths the contour to remove noise.

    Args:
        shoreline (np.ndarray): shape (N, 2) in (y, x) format.
        spacing (float): Desired spacing between resampled points in pixels.
        smooth (bool): Whether to apply Savitzky-Golay smoothing.
        window_length (int): Length of the filter window (must be odd and > polyorder).
        polyorder (int): Polynomial order for Savitzky-Golay filter.

    Returns:
        np.ndarray: shape (M, 2) resampled shoreline in (y, x) format.
    """
    # Optional smoothing
    if smooth:
        y = savgol_filter(shoreline[:, 0], window_length=window_length, polyorder=polyorder, mode='interp')
        x = savgol_filter(shoreline[:, 1], window_length=window_length, polyorder=polyorder, mode='interp')
        shoreline = np.stack([y, x], axis=1)

    # Convert to LineString: reverse to (x, y)
    line = LineString(shoreline[:, ::-1])

    # Total length of the line
    total_length = line.length
    num_points = max(2, int(total_length // spacing))

    # Sample points along the line
    resampled = [line.interpolate(dist).coords[0] for dist in np.linspace(0, total_length, num_points)]

    # Convert back to (y, x)
    return np.array(resampled)[:, ::-1]



from shapely.geometry import Point, MultiPoint, GeometryCollection, LineString
import numpy as np

def compute_shoreline_shift(transects, pred_shoreline):
    """
    Computes signed distance along each transect from the GT shoreline point 
    (midpoint of the transect) to the intersection with the predicted shoreline.
    
    Negative shift means predicted shoreline is "behind" GT (underprediction/erosion).
    Positive shift means predicted shoreline is "ahead" of GT (overprediction/accretion).

    Args:
        transects: array of shape (M, 2, 2), each with start and end points (y, x).
        pred_shoreline: array of shape (N, 2) predicted shoreline points (y, x).

    Returns:
        shifts: list of signed distances for each transect.
    """
    if len(pred_shoreline) < 2:
        return [np.nan] * len(transects)

    pred_line = LineString(pred_shoreline[:, ::-1])  # convert (y,x) → (x,y)
    shifts = []

    for start, end in transects:
        line = LineString([start[::-1], end[::-1]])  # (x,y) line from start to end
        half_length = line.length / 2
        intersection = line.intersection(pred_line)

        if intersection.is_empty:
            shifts.append(np.nan)
        elif isinstance(intersection, Point):
            dist = line.project(intersection)
            signed_dist = dist - half_length
            shifts.append(signed_dist)
        elif isinstance(intersection, MultiPoint):
            dists = [line.project(pt) for pt in intersection.geoms]
            closest_dist = min(dists, key=lambda d: abs(d - half_length))
            signed_dist = closest_dist - half_length
            shifts.append(signed_dist)
        elif isinstance(intersection, GeometryCollection):
            points = [geom for geom in intersection.geoms if isinstance(geom, Point)]
            if points:
                dists = [line.project(pt) for pt in points]
                closest_dist = min(dists, key=lambda d: abs(d - half_length))
                signed_dist = closest_dist - half_length
                shifts.append(signed_dist)
            else:
                shifts.append(np.nan)
        else:
            shifts.append(np.nan)

    return shifts



def shoreline_displacement_analysis(target_t, pred_t, resolution=20):
    """
    Returns displacement distances for each transect between prediction and target mask.
    """
    gt_shoreline = extract_shoreline(target_t)
    pred_shoreline = extract_shoreline(pred_t)
    if gt_shoreline is None or pred_shoreline is None:
        return []
    
    gt_shoreline = resample_shoreline(gt_shoreline, spacing=5.0)  # adjust spacing as needed
    pred_shoreline = resample_shoreline(pred_shoreline, spacing=5.0)

    normals = compute_normals(gt_shoreline)
    transects = create_transects(gt_shoreline, normals, resolution=resolution)
    shifts = compute_shoreline_shift(transects, pred_shoreline)
    return np.array(shifts) * resolution  # in meters

from matplotlib.patches import Patch

def plot_transects_and_shorelines(gt_mask, pred_mask, transects, results_dir, sample_idx=None, title="Transects and Shorelines"):
    import matplotlib.pyplot as plt
    import os

    gt_shoreline = extract_shoreline(gt_mask)
    pred_shoreline = extract_shoreline(pred_mask)
    shifts = compute_shoreline_shift(transects, pred_shoreline)
    print(f"Shifts for sample {sample_idx}: {shifts}")
    #print mean shift value
    mean_shift = np.nanmean(shifts)
    print(f"Mean shift for sample {sample_idx}: {mean_shift:.2f} m")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Show the mask background
    mask_np = gt_mask.squeeze().cpu().numpy()
    ax.imshow(mask_np, cmap='gray', origin='lower')

    # Plot shorelines
    if gt_shoreline is not None:
        ax.plot(gt_shoreline[:, 1], gt_shoreline[:, 0], color='#e0c080', label="GT Shoreline", linewidth=2)
    if pred_shoreline is not None:
        ax.plot(pred_shoreline[:, 1], pred_shoreline[:, 0], color='#1f78b4', label="Predicted Shoreline", linewidth=2)

    # Plot transects with shift values
    for i, (start, end) in enumerate(transects):
        ax.plot([start[1], end[1]], [start[0], end[0]], 'r-', linewidth=1)

        # Display shift value at the midpoint of each transect
        mid_y = (start[0] + end[0]) / 2
        mid_x = (start[1] + end[1]) / 2
        shift_val = shifts[i]
        if not np.isnan(shift_val):
            ax.text(mid_x, mid_y, f"{shift_val:.1f}m", color='black', fontsize=8,
                    ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='black', lw=0.5))

    # Axis and title
    ax.set_title(f"{title} - Sample {sample_idx}", fontsize=14)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.axis("equal")

    # Styled legend (similar to your other plot)
    legend_elements = [
        Patch(facecolor='#e0c080', edgecolor='black', label='GT Shoreline'),
        Patch(facecolor='#1f78b4', edgecolor='black', label='Predicted Shoreline'),
        Patch(facecolor='red', edgecolor='black', label='Transects')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, framealpha=0.9, edgecolor='gray')

    # Save figure
    plt.tight_layout()
    if sample_idx is not None:
        plt.savefig(os.path.join(results_dir, f"{title.replace(' ', '_')}_sample_{sample_idx}.png"), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(results_dir, f"{title.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    plt.close()

