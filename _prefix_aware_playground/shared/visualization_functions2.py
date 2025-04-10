import json
import os
import glob
import random
from textwrap import wrap
from typing import Optional, Dict, List, Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter

# --- Global Settings & Constants ---

# Plotting Style (applied globally when this module is imported)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook")
sns.set_palette("colorblind") # Use a colorblind-friendly palette

# Default Colors for plots with multiple lines/bars
# Using seaborn's colorblind palette directly for consistency
DEFAULT_COLORS = sns.color_palette("colorblind")

# Default Figure Size
DEFAULT_FIGSIZE = (12, 4)

# Map strategy prefixes to full names for titles/legends
STRATEGY_NAMES = {
    "random": "Random",
    "round": "Round Robin",
    "pow": "Power of 2",
    "prefix": "Prefix Aware"
}

# Metrics dictionary for plot_metric_by_strategy
AVAILABLE_BENCHMARK_METRICS = {
    "duration": "Benchmark Duration (s)",
    # "completed": "Completed Requests",
    # "request_throughput": "Request Throughput (req/s)",
    # "input_throughput": "Input Throughput (tokens/s)",
    # "output_throughput": "Output Throughput (tokens/s)",
    "mean_ttft_ms": "Mean TTFT (ms)",
    "median_ttft_ms": "Median TTFT (ms)",
    "std_ttft_ms": "Std Dev TTFT (ms)",
    "p99_ttft_ms": "P99 TTFT (ms)",
    "mean_tpot_ms": "Mean TPOT (ms)",
    "median_tpot_ms": "Median TPOT (ms)",
    "std_tpot_ms": "Std Dev TPOT (ms)",
    "p99_tpot_ms": "P99 TPOT (ms)",
    # "mean_itl_ms": "Mean ITL (ms)",
    # "median_itl_ms": "Median ITL (ms)",
    # "std_itl_ms": "Std Dev ITL (ms)",
    # "p99_itl_ms": "P99 ITL (ms)",
    # "mean_e2e_latency_ms": "Mean E2E Latency (ms)",
    # "median_e2e_latency_ms": "Median E2E Latency (ms)",
}

# --- Helper Function ---

def _load_json_data(json_file_path: str) -> Dict[str, Any]:
    """Loads data from a JSON file."""
    print(f"[INFO] Loading JSON from: {json_file_path}")
    try:
        with open(json_file_path, 'r') as f:
            raw_data = json.load(f)
        return raw_data
    except FileNotFoundError:
        print(f"[ERROR] File not found: {json_file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"[ERROR] Could not decode JSON from: {json_file_path}")
        return {}

# --- Plotting Functions ---

def plot_metric(json_file_path: str, metric_name: str, title: Optional[str] = None):
    """
    Plot a vLLM gauge or counter metric over time for each replica, with average.

    Args:
        json_file_path: Path to the JSON file with vllm_metrics data.
        metric_name: Name of the metric to plot (e.g., 'ray_vllm:gpu_cache_usage_perc').
        title: Optional plot title suffix.
    """
    raw_data = _load_json_data(json_file_path)
    if not raw_data:
        return

    times = sorted([float(t) for t in raw_data.keys()])
    print(f"[INFO] Loaded {len(times)} timestamps.")

    # Find replica IDs that report the metric
    replica_ids = set()
    for t_str in raw_data:
        for rid, metrics in raw_data[t_str].items():
            if any(metric_name in k for k in metrics.keys()):
                replica_ids.add(rid)

    replica_ids = sorted(list(replica_ids))
    print(f"[INFO] Found {len(replica_ids)} replicas with metric '{metric_name}'.")

    if not replica_ids:
        print(f"[WARN] No replicas contain the specified metric '{metric_name}'.")
        return

    plt.figure(figsize=DEFAULT_FIGSIZE)
    all_replica_metrics = []

    for i, replica_id in enumerate(replica_ids):
        y_vals = []
        for t in times:
            t_str = str(t)
            snapshot = raw_data.get(t_str, {}).get(replica_id, {})
            # Find the value for the metric, handling potential full key variation
            matching_values = [v for k, v in snapshot.items() if metric_name in k]
            val = matching_values[0] if matching_values else 0.0
            y_vals.append(val)

        # Basic check if data seems valid
        if all(v == 0 for v in y_vals):
            print(f"[WARN] All values are zero for replica {replica_id}.")
        # else: # Reduce verbosity
        #     print(f"[INFO] Replica {replica_id} has non-zero data points.")

        all_replica_metrics.append(y_vals)
        plt.plot(times, y_vals, marker='o', markersize=1, linestyle='-', linewidth=0.5,
                 color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)], label=f'Replica ID: {replica_id}', alpha=0.6)

    # Plot average
    if all_replica_metrics:
        try:
            avg_vals = np.mean(all_replica_metrics, axis=0)
            plt.plot(times, avg_vals, marker='s', markersize=2, linestyle='-', linewidth=2,
                     color='black', label='Average', alpha=0.8, zorder=10)
            print(f"[INFO] Plotted average line.")
        except Exception as e:
            print(f"[WARN] Could not compute or plot average: {e}")
    else:
        print("[WARN] No replica data available to compute average.")

    plot_title = f'Metric: {metric_name}'
    if title:
        plot_title += f" – {title}"
    plt.title(plot_title)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Metric Value')
    plt.grid(True, alpha=0.5)

    # Ensure legend is shown only if there are items to show
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(loc='upper right')
    else:
        print("[WARN] No data plotted. Check metric name and data file.")

    # Improve Y-axis ticker for integer metrics if applicable (heuristic)
    if all(isinstance(val, (int, float)) and val == int(val) for metrics in all_replica_metrics for val in metrics):
         plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()


def plot_average_vllm_metrics(json_file_path: str, base_metric_name: str, title: Optional[str] = None, window_sec: Optional[float] = None):
    """
    Plot the average of vLLM *_sum / *_count metrics over time for each replica.

    Args:
        json_file_path: Path to the JSON file containing vLLM metrics.
        base_metric_name: Metric name prefix (e.g. 'time_to_first_token' for 'ray_vllm:time_to_first_token_seconds_*').
        title: Optional plot title suffix.
        window_sec: Optional window size in seconds for moving average. If None, no windowing is applied.
    """
    raw_data = _load_json_data(json_file_path)
    if not raw_data:
        return

    times = sorted([float(t) for t in raw_data.keys()])
    print(f"[INFO] Loaded {len(times)} timestamps.")

    sum_metric_part = f"{base_metric_name}_sum"
    count_metric_part = f"{base_metric_name}_count"

    # Find replicas reporting this metric pair
    replica_ids = set()
    for t_str in raw_data:
        for rid, metrics in raw_data[t_str].items():
            keys = metrics.keys()
            if any(sum_metric_part in k for k in keys) and any(count_metric_part in k for k in keys):
                 replica_ids.add(rid)

    replica_ids = sorted(list(replica_ids))
    print(f"[INFO] Found {len(replica_ids)} replicas reporting '{base_metric_name}' sum/count pair.")

    if not replica_ids:
        print(f"[WARN] No replicas found reporting both '{sum_metric_part}' and '{count_metric_part}'.")
        return

    plt.figure(figsize=DEFAULT_FIGSIZE)
    all_replica_averages = []

    for i, replica_id in enumerate(replica_ids):
        y_vals = []
        # Pre-find full metric names for this replica if possible (assuming they don't change)
        first_t = str(times[0])
        first_snapshot = raw_data.get(first_t, {}).get(replica_id, {})
        full_sum_key = next((k for k in first_snapshot if sum_metric_part in k), None)
        full_count_key = next((k for k in first_snapshot if count_metric_part in k), None)

        if not full_sum_key or not full_count_key:
             print(f"[WARN] Could not find full keys for {sum_metric_part}/{count_metric_part} for replica {replica_id} at t={first_t}. Skipping replica.")
             continue


        last_sum = 0
        last_count = 0

        for t_idx, t in enumerate(times):
            t_str = str(t)
            snapshot = raw_data.get(t_str, {}).get(replica_id, {})
            current_sum = snapshot.get(full_sum_key, last_sum)
            current_count = snapshot.get(full_count_key, last_count)

            if window_sec is not None:
                # Find the time 'window_sec' ago
                window_start_time = t - window_sec
                # Find the index of the earliest time within the window
                start_idx = 0
                while start_idx < t_idx and times[start_idx] < window_start_time:
                    start_idx += 1

                if start_idx >= t_idx: # Window too small or at the beginning
                    avg_val = 0.0
                else:
                    start_t_str = str(times[start_idx])
                    start_snapshot = raw_data.get(start_t_str, {}).get(replica_id, {})
                    start_sum = start_snapshot.get(full_sum_key, 0)
                    start_count = start_snapshot.get(full_count_key, 0)

                    sum_diff = current_sum - start_sum
                    count_diff = current_count - start_count

                    avg_val = (sum_diff / count_diff) if count_diff > 0 else 0.0
            else:
                # Use cumulative difference from the *previous* timestamp for point-in-time rate
                sum_diff = current_sum - last_sum
                count_diff = current_count - last_count
                avg_val = (sum_diff / count_diff) if count_diff > 0 else 0.0

            y_vals.append(avg_val)
            last_sum = current_sum
            last_count = current_count


        if all(abs(v) < 1e-9 for v in y_vals): # Check for near-zero values
            print(f"[WARN] All calculated average values are zero for replica {replica_id}")
        # else: # Reduce verbosity
        #     print(f"[INFO] Replica {replica_id} has non-zero average values.")

        all_replica_averages.append(y_vals)
        plt.plot(times, y_vals, marker='o', markersize=1, linestyle='-', linewidth=0.5,
                 color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)], label=f'Replica ID: {replica_id}', alpha=0.6)

    # Plot average across replicas
    if all_replica_averages:
        try:
            # Pad shorter arrays if windowing caused length differences (unlikely with current logic but safer)
            max_len = max(len(arr) for arr in all_replica_averages)
            padded_averages = [np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=np.nan) for arr in all_replica_averages]
            avg_across_replicas = np.nanmean(padded_averages, axis=0)

            plt.plot(times, avg_across_replicas, marker='s', markersize=2, linestyle='-', linewidth=2,
                     color='black', label='Average Across Replicas', alpha=0.8, zorder=10)
            print(f"[INFO] Plotted average across replicas.")
        except Exception as e:
            print(f"[WARN] Could not compute or plot average across replicas: {e}")

    plt.xlabel("Time (seconds)")
    plt.ylabel(f"Average {base_metric_name.replace('_', ' ').title()}")
    plot_title = f"Average {base_metric_name.replace('_', ' ').title()} per Replica"
    if window_sec is not None:
        plot_title += f" ({window_sec}s Moving Average)"
    if title:
        plot_title += f" – {title}"
    plt.title(plot_title)
    plt.grid(True, alpha=0.5)

    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(loc='upper right')
    else:
        print("[WARN] No data plotted.")

    # Improve Y-axis ticker
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    # plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f')) # Optional: format y-axis labels

    plt.tight_layout()
    plt.show()


def plot_metric_by_strategy(csv_file_path: str, metric: str):
    """
    Plot a specific metric from benchmark results grouped by routing strategy.

    Args:
        csv_file_path: Path to the CSV file with benchmark results.
        metric: The column name of the metric to plot (must exist in AVAILABLE_BENCHMARK_METRICS).
    """
    if metric not in AVAILABLE_BENCHMARK_METRICS:
        print(f"[ERROR] Metric '{metric}' not found in available metrics: {list(AVAILABLE_BENCHMARK_METRICS.keys())}")
        return

    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {csv_file_path}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to read CSV {csv_file_path}: {e}")
        return

    # Basic data cleaning (remove header rows if accidentally included)
    df = df[df['gpu_type'] != 'gpu_type']
    if df.empty:
        print("[WARN] DataFrame is empty after initial cleaning.")
        return

    # Ensure metric column is numeric
    try:
        df[metric] = pd.to_numeric(df[metric], errors='raise')
    except (ValueError, TypeError) as e:
        print(f"[ERROR] Could not convert metric column '{metric}' to numeric: {e}")
        # Attempt to coerce, logging NaNs
        initial_len = len(df)
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
        nan_count = df[metric].isna().sum()
        if nan_count > 0:
             print(f"[WARN] Coerced '{metric}' to numeric, resulting in {nan_count}/{initial_len} NaN values.")
        if df[metric].notna().sum() == 0:
            print(f"[ERROR] Metric column '{metric}' contains no valid numeric data after coercion.")
            return


    # Prepare labels and order
    df['benchmark_label'] = df['benchmark_label'].astype(str).str.replace("_", " ")
    df['wrapped_label'] = df['benchmark_label'].apply(lambda x: '\n'.join(wrap(x, 15))) # Wrap long labels
    strategy_order = list(STRATEGY_NAMES.keys()) # Use defined order
    original_label_order = sorted(df['wrapped_label'].unique()) # Consistent ordering of benchmarks

    # Check consistency of results per strategy (optional, for context)
    try:
        strategy_counts = df.groupby(['wrapped_label', 'scheduler_strategy']).size()
        num_seeds_per_group = strategy_counts.min() # Assumes min represents the intended number of seeds
        if strategy_counts.nunique() > 1:
            print(f"[WARN] Inconsistent number of results per strategy/benchmark group. Min runs/group: {num_seeds_per_group}")
        else:
            print(f"[INFO] Found {num_seeds_per_group} runs per strategy/benchmark group.")
    except KeyError:
        print("[WARN] Could not verify number of seeds/group - columns 'wrapped_label' or 'scheduler_strategy' might be missing.")
        num_seeds_per_group = 1 # Assume 1 if check fails

    # Calculate mean for plotting
    # Group by the wrapped label and strategy, calculate mean, handle potential missing groups
    avg_df = df.groupby(['wrapped_label', 'scheduler_strategy'])[metric].mean().unstack().reindex(index=original_label_order, columns=strategy_order).stack(dropna=False).reset_index(name=metric)
    avg_df = avg_df.dropna(subset=[metric]) # Remove combinations that didn't exist in the data

    if avg_df.empty:
        print("[WARN] No data remaining after grouping and averaging.")
        return

    # Create plot
    plt.figure(figsize=(max(10, len(original_label_order) * 1.5), 5)) # Adjust width based on number of benchmarks

    ax = sns.barplot(x='wrapped_label', y=metric, hue='scheduler_strategy',
                     hue_order=strategy_order, data=avg_df,
                     order=original_label_order, palette=DEFAULT_COLORS) # Use defined palette

    # Add value labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=8, padding=3)

    # Customize plot
    metric_display_name = AVAILABLE_BENCHMARK_METRICS[metric]
    plot_title = f"{metric_display_name} by Routing Strategy"
    if num_seeds_per_group > 1:
        plot_title += f" (averaged over {num_seeds_per_group} runs)"

    plt.title(plot_title)
    plt.xlabel('Benchmark Label')
    plt.ylabel(metric_display_name)
    plt.xticks(rotation=0) # Keep labels horizontal if possible

    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [STRATEGY_NAMES.get(label, label) for label in labels],
              title="Routing Strategy",
              bbox_to_anchor=(1.02, 1), # Place legend outside plot
              loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend
    plt.show()


def plot_load_distribution(json_file_path: str, title: Optional[str] = None, max_replicas_to_plot: int = 4):
    """
    Plot the load distribution over time for each replica on a single chart.

    Args:
        json_file_path: Path to the JSON file containing load distribution data.
        title: Optional plot title suffix.
        max_replicas_to_plot: Maximum number of individual replica lines to plot.
    """
    load_data = _load_json_data(json_file_path)
    if not load_data:
        return

    # Convert times to float and sort, filter out empty timestamps
    times = sorted([float(t) for t, data in load_data.items() if data])
    if not times:
        print("[WARN] No valid timestamps found in the data.")
        return

    # Find all replica IDs present at the first valid timestamp
    first_valid_time_str = str(times[0])
    replicas = sorted(list(load_data[first_valid_time_str].keys()))
    if not replicas:
        print("[WARN] No replicas found at the first timestamp.")
        return

    print(f"[INFO] Found {len(replicas)} replicas. Plotting individual lines for first {min(len(replicas), max_replicas_to_plot)}.")

    plt.figure(figsize=DEFAULT_FIGSIZE)
    all_replica_loads = []

    # Plot each replica's load (up to max_replicas_to_plot)
    for i, replica_id in enumerate(replicas):
        replica_loads = [load_data.get(str(t), {}).get(replica_id, 0) for t in times]
        all_replica_loads.append(replica_loads) # Collect data for all replicas for average

        if i < max_replicas_to_plot:
            plt.plot(times, replica_loads, marker='o', markersize=1, linestyle='-', linewidth=0.5,
                     color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)], label=f'Replica ID: {replica_id}', alpha=0.6)

    # Calculate and plot the average load across ALL replicas
    avg_loads = None
    avg_load_value = 0
    if all_replica_loads:
        try:
            avg_loads = np.mean(all_replica_loads, axis=0)
            plt.plot(times, avg_loads, marker='s', markersize=2, linestyle='-', linewidth=2,
                     color='black', label=f'Average Load ({len(replicas)} replicas)', alpha=0.7, zorder=10)

            # Calculate average load during perceived "high-load" period (heuristic: when avg > 5)
            high_load_indices = np.where(avg_loads > 5)[0]
            if high_load_indices.size > 0:
                # Find contiguous high-load blocks if needed, or just average the high points
                avg_load_value = np.mean(avg_loads[high_load_indices])
            else:
                 avg_load_value = np.mean(avg_loads) # If never above 5, just use overall average

            print(f"[INFO] Plotted average load. Calculated representative avg load: {avg_load_value:.2f}")
        except Exception as e:
             print(f"[WARN] Could not compute or plot average load: {e}")

    # Set title
    plot_title = 'Load Distribution'
    if title:
        plot_title += f': {title}'
    if avg_loads is not None:
        plot_title += f' (Avg Load: {avg_load_value:.2f})'
    plt.title(plot_title)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Number of Concurrent Requests')
    plt.grid(True, alpha=0.5)

    # Determine reasonable y-axis limits
    max_load = np.max(all_replica_loads) if all_replica_loads else 20
    y_limit = max(20, max_load * 1.1) # Ensure at least 0-20, or slightly above max observed
    plt.ylim(0, y_limit)

    # Set y-axis ticks (integers, sensible spacing)
    ax = plt.gca()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins='auto'))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(1)) # Minor ticks every 1 request

    # Add legend if necessary
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
         # Place legend outside plot area if many items
        if len(handles) > 4:
             plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        else:
            plt.legend(loc='upper right')


    plt.tight_layout(rect=[0, 0, 0.9, 1] if len(handles) > 4 else [0,0,1,1]) # Adjust layout if legend is outside
    plt.show()


def plot_prefix_match_rate(json_file_path: str, title: Optional[str] = None):
    """
    Plot the prefix match rates for each replica on a single chart.

    Args:
        json_file_path: Path to the JSON file containing prefix match rates data.
                         Expected format: {"replica_id_1": [rate1, rate2,...], ...}
        title: Optional plot title suffix.
    """
    data = _load_json_data(json_file_path)
    if not data or not isinstance(data, dict):
        print("[WARN] Invalid or empty data loaded. Expected dict format.")
        return

    plt.figure(figsize=DEFAULT_FIGSIZE)

    all_match_rates_flat = []
    all_replica_rates_padded = []
    max_length = 0
    replica_ids = sorted(data.keys())

    print(f"[INFO] Found data for {len(replica_ids)} replicas.")

    # Plot individual replica points and collect data for average
    for i, replica_id in enumerate(replica_ids):
        match_rates = data[replica_id]
        if not isinstance(match_rates, list):
            print(f"[WARN] Data for replica {replica_id} is not a list. Skipping.")
            continue

        indices = np.arange(len(match_rates))
        max_length = max(max_length, len(match_rates))
        all_match_rates_flat.extend(match_rates)

        # Pad rates with NaN for consistent length averaging later
        padded_rates = match_rates + [np.nan] * (max_length - len(match_rates))
        all_replica_rates_padded.append(padded_rates)

        # Add slight vertical jitter to points near 0 or 1 for visibility
        jitter = np.random.uniform(-0.015, 0.015, size=len(match_rates))
        jittered_rates = np.clip(np.array(match_rates) + jitter, -0.02, 1.02) # Keep within bounds

        # Calculate replica stats
        valid_rates = [r for r in match_rates if isinstance(r, (int, float))] # Filter out non-numeric if any
        replica_avg_rate = np.mean(valid_rates) if valid_rates else 0
        # Define "match" heuristic (e.g., rate > 0.1)
        match_threshold = 0.1
        replica_matches = sum(1 for rate in valid_rates if rate > match_threshold)
        replica_total = len(valid_rates)
        replica_match_perc = (replica_matches / replica_total * 100) if replica_total > 0 else 0

        plt.scatter(indices, jittered_rates,
                    color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                    alpha=0.5, s=8, # Slightly larger points
                    label=f'Replica {replica_id}: Avg={replica_avg_rate:.2f}, Matches={replica_matches}/{replica_total} ({replica_match_perc:.1f}%)')

    # Calculate and plot the overall average hit rate
    overall_avg_rate = np.mean(all_match_rates_flat) if all_match_rates_flat else 0
    overall_matches = sum(1 for rate in all_match_rates_flat if rate > match_threshold)
    overall_total = len(all_match_rates_flat)
    overall_match_perc = (overall_matches / overall_total * 100) if overall_total > 0 else 0

    print(f"[INFO] Overall Avg Hit Rate: {overall_avg_rate:.2f}, Matches: {overall_matches}/{overall_total} ({overall_match_perc:.1f}%)")

    if all_replica_rates_padded and max_length > 0:
         try:
            # Calculate average across replicas, ignoring NaNs from padding
            avg_rates = np.nanmean(all_replica_rates_padded, axis=0)
            avg_indices = np.arange(max_length)

            # Smooth the average line if enough data points
            smoothed_avg = avg_rates # Default to non-smoothed
            if max_length > 10: # Only smooth if data is sufficient
                # Savitzky-Golay filter parameters
                window_length = min(max(5, max_length // 8 * 2 + 1), 51) # Odd, reasonable fraction of length
                polyorder = min(3, window_length - 2) # Must be < window_length, sensible default
                if polyorder < 1: polyorder = 1 # Ensure polyorder is at least 1

                # Interpolate NaNs before smoothing if needed
                nan_mask = np.isnan(avg_rates)
                if np.any(nan_mask):
                    interp_x = avg_indices[~nan_mask]
                    interp_y = avg_rates[~nan_mask]
                    if len(interp_x) > polyorder : # Check if enough points to interpolate and smooth
                        avg_rates[nan_mask] = np.interp(avg_indices[nan_mask], interp_x, interp_y)
                        # Only smooth non-NaN sections if interpolation didn't fill all gaps
                        if not np.isnan(avg_rates).all():
                             smoothed_avg = savgol_filter(avg_rates, window_length, polyorder, mode='interp')
                        else:
                             print("[WARN] Could not interpolate NaNs for smoothing average rate.")
                    else:
                        print("[WARN] Not enough non-NaN points to interpolate/smooth average rate.")

                elif len(avg_rates) > window_length: # No NaNs, apply filter directly if enough points
                    smoothed_avg = savgol_filter(avg_rates, window_length, polyorder, mode='interp')
                else:
                    print("[INFO] Not smoothing average rate (too few data points).")


            plt.plot(avg_indices, np.clip(smoothed_avg, 0, 1), # Clip smoothed line to [0, 1]
                     color='black', linewidth=2.5,
                     label=f'Smoothed Avg Rate: {overall_avg_rate:.2f}',
                     zorder=10) # Ensure average line is on top
         except Exception as e:
              print(f"[WARN] Could not compute or plot average/smoothed hit rate: {e}")


    # Set plot title and labels
    plot_title = 'Prefix Match Rates'
    if title:
        plot_title += f': {title}'
    plt.title(plot_title)
    plt.xlabel('Request Index (Approx.)')
    plt.ylabel('Match Rate (0-1)')

    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)

    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        # Place legend outside if too many items
        if len(handles) > 5:
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        else:
            plt.legend(loc='best', fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.85, 1] if len(handles) > 5 else [0,0,1,1])
    plt.show()


# --- Batch Plotting Functions ---

def plot_all_load_distributions(load_dist_dir: str):
    """
    Plot load distributions for all JSON files in the specified directory.

    Args:
        load_dist_dir (str): Directory containing load distribution JSON files.
                             Files are expected to be named like '{strategy_prefix}_*.json'.
    """
    print(f"\n--- Plotting All Load Distributions in: {load_dist_dir} ---")
    json_files = glob.glob(os.path.join(load_dist_dir, "*.json"))
    if not json_files:
        print("[WARN] No JSON files found in the directory.")
        return

    # Group files by strategy prefix found in the filename
    strategy_files: Dict[str, List[str]] = {}
    for file_path in json_files:
        filename = os.path.basename(file_path)
        strategy_prefix = filename.split('_')[0] # Assumes prefix is before first underscore

        if strategy_prefix not in strategy_files:
            strategy_files[strategy_prefix] = []
        strategy_files[strategy_prefix].append(file_path)

    # Sort files within each strategy group
    for strategy in strategy_files:
        strategy_files[strategy].sort()

    # Plot files in a defined order of strategies
    plot_order = list(STRATEGY_NAMES.keys())
    for strategy_prefix in plot_order:
        if strategy_prefix in strategy_files:
            print(f"\nProcessing Strategy: {STRATEGY_NAMES.get(strategy_prefix, strategy_prefix)}...")
            for file_path in strategy_files[strategy_prefix]:
                strategy_name = STRATEGY_NAMES.get(strategy_prefix, strategy_prefix.capitalize())
                # Extract a more specific title part from filename if possible
                # e.g., remove prefix and suffix
                specific_title = os.path.basename(file_path).replace(strategy_prefix + '_', '').replace('.json', '')
                plot_load_distribution(file_path, title=f"{strategy_name} ({specific_title})")


def plot_all_prefix_match_rates(match_rates_dir: str):
    """
    Plot prefix match rates for all JSON files in the specified directory.

    Args:
        match_rates_dir (str): Directory containing prefix match rates JSON files.
                               Files are expected to be named like '{strategy_prefix}_*.json'.
    """
    print(f"\n--- Plotting All Prefix Match Rates in: {match_rates_dir} ---")
    json_files = glob.glob(os.path.join(match_rates_dir, "*.json"))
    if not json_files:
        print("[WARN] No JSON files found in the directory.")
        return

    # Group files by strategy prefix
    strategy_files: Dict[str, List[str]] = {}
    for file_path in json_files:
        filename = os.path.basename(file_path)
        strategy_prefix = filename.split('_')[0]

        if strategy_prefix not in strategy_files:
            strategy_files[strategy_prefix] = []
        strategy_files[strategy_prefix].append(file_path)

    # Sort files within each group
    for strategy in strategy_files:
        strategy_files[strategy].sort()

    # Plot files in defined strategy order
    plot_order = list(STRATEGY_NAMES.keys())
    for strategy_prefix in plot_order:
        if strategy_prefix in strategy_files:
            print(f"\nProcessing Strategy: {STRATEGY_NAMES.get(strategy_prefix, strategy_prefix)}...")
            for file_path in strategy_files[strategy_prefix]:
                strategy_name = STRATEGY_NAMES.get(strategy_prefix, strategy_prefix.capitalize())
                specific_title = os.path.basename(file_path).replace(strategy_prefix + '_', '').replace('.json', '')
                plot_prefix_match_rate(file_path, title=f"{strategy_name} ({specific_title})")

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    print("Plotting utility module loaded.")
    print("Example Usage (uncomment and replace paths):")
    print("\nPlotting a specific metric:")
    plot_metric('/home/ray/default/work/ray/_prefix_aware_playground/inside_ray/results/vllm_metrics/random_1744240419.json', 'ray_vllm:gpu_cache_usage_perc', title='My Experiment')

    print("\nPlotting average TTFT:")
    plot_average_vllm_metrics('/home/ray/default/work/ray/_prefix_aware_playground/inside_ray/results/vllm_metrics/random_1744240419.json', 'ray_vllm:time_to_first_token_seconds', title='TTFT Analysis', window_sec=10.0)

    print("\nPlotting benchmark results from CSV:")
    plot_metric_by_strategy('/home/ray/default/work/ray/_prefix_aware_playground/inside_ray/results/chosen_sweep_results.csv', 'mean_ttft_ms')
    plot_metric_by_strategy('/home/ray/default/work/ray/_prefix_aware_playground/inside_ray/results/chosen_sweep_results.csv', 'p99_tpot_ms')

    print("\nPlotting load distribution:")
    plot_load_distribution('/home/ray/default/work/ray/_prefix_aware_playground/inside_ray/results/load_distributions/random_1744240418.json', title='Prefix Aware Strategy')

    print("\nPlotting prefix match rate:")
    plot_prefix_match_rate('/home/ray/default/work/ray/_prefix_aware_playground/inside_ray/results/prefix_match_rates/random_1744240419.json', title='Prefix Aware Strategy')

    print("\nPlotting all load distributions in a directory:")
    plot_all_load_distributions('/home/ray/default/work/ray/_prefix_aware_playground/inside_ray/results/load_distributions/hardcode_tree_and_probe_queues')

    print("\nPlotting all prefix match rates in a directory:")
    plot_all_prefix_match_rates('/home/ray/default/work/ray/_prefix_aware_playground/inside_ray/results/prefix_match_rates')