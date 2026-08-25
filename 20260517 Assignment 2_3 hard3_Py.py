import pandas as pd
import numpy as np

# Снимаем ограничение на количество столбцов
pd.set_option('display.max_columns', 25)
# Увеличиваем общую ширину вывода (чтобы столбцы не переносились на новую строку)
pd.set_option('display.width', 200)
# Возврат:
# pd.set_option('display.max_rows', None)
# pd.reset_option('display.max_rows')

import matplotlib as mpl
# mpl.use('Qt5Agg')  # Переключаем с agg на интерактивный Qt5
import matplotlib.pyplot as plt

# ==============================================================================
# ==================== Assignment 3 for .py (hard 3) ===========================
# ====== using cid = fig.canvas.mpl_connect("button_press_event", onclick) =====
# ==============================================================================

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650),
                   np.random.normal(43000,100000,3650),
                   np.random.normal(43500,140000,3650),
                   np.random.normal(48000,70000,3650)],
                  index=[1992,1993,1994,1995])

# Your Code Here

import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

years = df.index.astype(str)  # to strings to use in labels
means = df.mean(axis=1)  # means (bar heights) - Series
stds = df.std(axis=1)  # std dev - Series
n = df.shape[1]  # number of elements (3650)

# Confidence Interval (CI) 95% calculation:
t_crit = stats.t.ppf(0.975, df=n - 1)
sem = stds / np.sqrt(n)  # standard error of mean
ci_half = t_crit * sem  # half of CI
# print('ci_half=\n', ci_half)  # just for checking


# Get the colormap (Blue -> White -> Red)
cmap = plt.get_cmap('seismic')

# Build the base plot using your variables
fig, ax = plt.subplots(figsize=(9, 6))
# Adjust the main plot area to make room for the colorbar
plt.subplots_adjust(right=0.80)

# bar diagram with error intervals (yerr)
bars = ax.bar(years, means, yerr=ci_half, capsize=20, color="lightgray", edgecolor="black")

ax.set_title("Means with Confidence Intervals vs. Selected Level\n"
             "(Click on the Graph to Set the New Y-Level)", fontsize=14)
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("Value", fontsize=14)
# plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5000))

# Grid and Y-axis tick customization
# Major ticks every 10,000, Minor ticks every 5,000
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10000))
ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5000))
# Format numbers on Y-axis with thousands separators (e.g., 40,000)
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
# Enable horizontal grid lines for major ticks (styled subtle and behind the bars)
ax.grid(axis='y', which='major', color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
ax.set_axisbelow(True)  # Ensures grid lines are drawn behind the bars

# Create an initial horizontal line and text label (visible immediately)
target_y = 40000
line = ax.axhline(y=target_y, color="darkgreen", linestyle="--", linewidth=2)
text_label = ax.text(0.01, target_y + 1000, f"Y = {target_y:,.0f}", color="darkgreen", weight="bold")


# Function to calculate bar colors using the t-distribution CDF
def update_bar_colors(current_y):
    for i, bar in enumerate(bars):
        mean = means.iloc[i]
        se = sem.iloc[i]  # Use standard error to evaluate the mean's probability

        # Calculate the t-statistic for the selected Y relative to the group mean
        t_stat = (mean - current_y) / se

        # Convert t-statistic to a probability value between 0.0 and 1.0
        # If mean == current_y, prob = 0.5 (White color)
        # If mean >> current_y, prob approaches 1.0 (Red color)
        # If mean << current_y, prob approaches 0.0 (Blue color)
        prob = stats.t.cdf(t_stat, df=n - 1)

        # Set the dynamic background color for the bar from the cmap palette
        bar.set_facecolor(cmap(prob))

# Color the bars for the initial target_y value right at the start
update_bar_colors(target_y)

# Add the Colorbar legend on the right side of the figure
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar_ax = fig.add_axes([0.84, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Probability of Mean > Y', fontsize=11)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(['0.0 (Min)', '0.5 (Med)', '1.0 (Max)'])


# Define the callback function for mouse clicks
def onclick(event):
    # Check if the click happened inside the plot axes
    if event.ydata is None:
        return

    # Get the exact Y-coordinate of the click
    target_y = event.ydata

    # Update horizontal line and text label positions
    line.set_ydata([target_y])
    text_label.set_position((0.01, target_y + 1000))
    text_label.set_text(f"Y = {target_y:,.0f}")

    # Recalculate and update bar colors dynamically on click
    update_bar_colors(target_y)

    # Force a redraw of the canvas
    fig.canvas.draw_idle()


# Connect the click event ('button_press_event') to the onclick function
cid = fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()



















