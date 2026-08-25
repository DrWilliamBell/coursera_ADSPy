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
# ==================== Assignment 3 for .py (hard 4) ===========================
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
print('stds =\n', stds)  # just for checking

# Confidence Interval (CI) 95% calculation:
t_crit = stats.t.ppf(0.975, df=n - 1)
print('t_crit =\n', t_crit)  # just for checking
sem = stds / np.sqrt(n)  # standard error of mean
print('sem =\n', sem)  # just for checking
ci_half = t_crit * sem  # half of CI
print('ci_half =\n', ci_half)  # just for checking

# Define initial range for Y values
target_y_low = 30000
target_y_high = 41000

# Get the colormap (Blue -> White -> Red)
cmap = plt.get_cmap('seismic')

# Build the base plot using your variables
fig, ax = plt.subplots(figsize=(9, 6))
# Adjust the main plot area to make room for the colorbar
plt.subplots_adjust(right=0.80)

# bar diagram with error intervals (yerr)
bars = ax.bar(years, means, yerr=ci_half, capsize=20, color="lightgray", edgecolor="black")

ax.set_title("Means with Confidence Intervals vs. Y-Range\n"
             "(Move levels by mouse to change the Range)", fontsize=14)
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("Value", fontsize=14)
# plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5000))  # for 5000 ticls on Y axis

# Grid and Y-axis tick customization
# Major ticks every 10,000, Minor ticks every 5,000
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10000))
ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5000))
# Format numbers on Y-axis with thousands separators (e.g., 40,000)
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
# Enable horizontal grid lines for major ticks (styled subtle and behind the bars)
ax.grid(axis='y', which='major', color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
ax.set_axisbelow(True)  # Ensures grid lines are drawn behind the bars

# Create TWO lines and a shaded span area between them
line_low = ax.axhline(y=target_y_low, color="darkgreen", linestyle="--", linewidth=2)
line_high = ax.axhline(y=target_y_high, color="darkgreen", linestyle="--", linewidth=2)
# Shaded visual corridor
span = ax.axhspan(target_y_low, target_y_high, color='green', alpha=0.15)

# Text labels for both thresholds
text_low = ax.text(0.02, target_y_low - 2000, f"Y_low = {target_y_low:,.0f}", color="darkgreen", weight="bold")
text_high = ax.text(0.02, target_y_high + 1000, f"Y_high = {target_y_high:,.0f}", color="darkgreen", weight="bold")


# Calculate bar colors based on range overlap probability
def update_bar_colors(y_low, y_high):
    # Ensure y_low is always the smaller value for correct math
    ymin, ymax = min(y_low, y_high), max(y_low, y_high)

    for i, bar in enumerate(bars):
        mean = means.iloc[i]
        se = sem.iloc[i]

        # Calculate t-statistics for both thresholds
        # (how many sems in distance to mean)
        t_high = (ymax - mean) / se
        t_low = (mean - ymin) / se

        # Вероятность попасть в хвосты (вылететь выше/ниже диапазонов):
        tail_high = stats.t.sf(t_high, df=n - 1)
        tail_low = stats.t.sf(t_low, df=n - 1)

        # Probability that the mean falls within the chosen range
        prob_in_range = 1 - (tail_high + tail_low)

        # ИНФОРМАЦИЯ НА БАРАХ ДЛЯ АНАЛИЗА И ПОНИМАНИЯ
        # -------------------------------------------
        # Сохраняем строки для вывода на бары
        text_data = (
            f"t_high: {t_high:.2f}\n"
            f"t_low: {t_low:.2f}\n"
            f"tail_high: {tail_high:.3f}\n"
            f"tail_low: {tail_low:.3f}\n"
            f"prob: {prob_in_range:.3f}"
        )
        # Вывод текста на бары
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # Координата X: центр бара
            height * 0.3,  # Координата Y: середина высоты бара
            text_data,
            ha='center',  # Выравнивание по горизонтали
            va='center',  # Выравнивание по вертикали
            fontsize=7,
            color='black',
            weight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
            # Подложка для читаемости
        )
        # ---------------------------------------------

        # Color interpretation: High probability in range -> Red, Outside -> Blue
        bar.set_facecolor(cmap(prob_in_range))

# New colors for the bars
update_bar_colors(target_y_low, target_y_high)

# Add the Colorbar legend on the right side of the figure
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar_ax = fig.add_axes([0.84, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Probability of Mean Inside the Range', fontsize=11)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(['0.0 (Min)', '0.5 (Med)', '1.0 (Max)'])

# Track dragging state for multiple objects
active_line = None  # Will store either 'low' or 'high' when dragging starts

def on_press(event):
    global active_line
    if event.inaxes != ax:
        return

    y_click = event.ydata
    current_low = line_low.get_ydata()[0]
    current_high = line_high.get_ydata()[0]

    # Check which line is closer to the user's click (click zone tolerance: 3000 units)
    if abs(y_click - current_low) < 3000:
        active_line = 'low'
    elif abs(y_click - current_high) < 3000:
        active_line = 'high'


def on_motion(event):
    global active_line
    if active_line is None or event.ydata is None:
        return

    new_y = event.ydata

    # Update the position of the line that is currently being dragged
    if active_line == 'low':
        line_low.set_ydata([new_y])
        text_low.set_position((0.02, new_y - 2000))
        text_low.set_text(f"Y_low = {new_y:,.0f}")
    elif active_line == 'high':
        line_high.set_ydata([new_y])
        text_high.set_position((0.02, new_y + 1000))
        text_high.set_text(f"Y_high = {new_y:,.0f}")

    # Get current positions of both lines to update the shaded area and colors
    y_l = line_low.get_ydata()[0]
    y_h = line_high.get_ydata()[0]

    # Dynamically update the shaded corridor boundaries
    # We remove the old span and draw a new one to prevent graphic glitching
    global span
    span.remove()
    span = ax.axhspan(y_l, y_h, color='green', alpha=0.15)

    # Recalculate colors for the new custom range configuration
    update_bar_colors(y_l, y_h)
    fig.canvas.draw_idle()


def on_release(event):
    global active_line
    active_line = None


cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
cid_release = fig.canvas.mpl_connect('button_release_event', on_release)

plt.show()




