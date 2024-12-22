import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, MaxNLocator


def mjd_to_date(mjd):
    """Konvertiere MJD in ein Kalenderdatum."""
    mjd_start_date = datetime.datetime(1858, 11, 17)  # MJD Startdatum
    return mjd_start_date + datetime.timedelta(days=mjd)


def format_month_day(mjd, pos):
    """Formatter für die obere Achse, der MJD in Monats- und Tagesformat umwandelt."""
    date = mjd_to_date(mjd)
    return date.strftime('%b %d')  # Format: 'Aug 01', 'Sep 15', ...


def format_relative_days(mjd, pos):
    """Formatter für relative Tage (mit erstem Wert als Referenz)."""
    base_mjd = 57581.66  # Startwert (erster MJD)
    relative_day = mjd - base_mjd
    return f"{int(relative_day)}"  # Zeige nur relative Tage


def get_type_methodes(data_type):
    if data_type == 'lightcurves':
        return configure_axis_lightcurves, finalize_figure_lightcurves
    elif data_type == 'ccfs':
        return configure_axis_ccfs, finalize_figure_ccfs
    else:
        raise ValueError(f"Wrong datatype. Expected 'lightcurves' or 'ccfs', got '{data_type}'.")


def plot_1d_data_in_groups(data, x_key, y_key, xlabel='X-axis', ylabel='Y-axis', shared_y=False,
                           yerr_name=None, title=None, save_only=False,
                           output_dir=None, color_dict=None, rows=4, cols=2, data_type='lightcurves'):
    """
    Plot multiple 1D data sets in groups.

    Parameters:
        data_type:
        shared_y:
        y_key:
        x_key:
        data (dict): Dictionary containing the data for the plots.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        yerr_name (str): Optional name for the error bars.
        title (str): Title of the plot.
        save_only (bool): Whether to save the plots without displaying them.
        output_dir (str or Path): Directory to save the plots.
        color_dict (dict): Optional dictionary with colors for each data series.
        rows (int): Number of rows in the subplot grid.
        cols (int): Number of columns in the subplot grid.
    """

    # Get the appropriate methods based on the data type
    configure_axis, finalize_figure = get_type_methodes(data_type)

    # Main plotting loop
    for current_data, group_index in prepare_data(data, x_key, y_key, yerr_name, rows, cols):
        fig, axes = plt.subplots(rows, cols, figsize=(8, 12), sharex=True, sharey=shared_y)
        fig.subplots_adjust(hspace=0, wspace=0)

        for i, (line_name, line_data) in enumerate(current_data):
            row, col = divmod(i, cols)
            ax = axes[row, col]

            x_values = np.array(line_data.get(x_key, []))
            y_values = np.array(line_data.get(y_key, []))
            yerr_values = np.array(line_data.get(yerr_name, [])) if yerr_name else None
            color = color_dict.get(line_name, 'black') if color_dict else 'black'

            # Call the correct configure_axis based on data_type
            configure_axis(ax, row, col, ylabel, color, x_values, y_values, yerr_values, line_name)

        # Call the correct finalize_figure based on data_type
        finalize_figure(fig, axes, title=title, group_index=group_index,
                        save_only=save_only, output_dir=output_dir)


def prepare_data(data, xlabel, ylabel, yerr_name, rows, cols):
    """Prepare data and ensure all groups are filled."""
    total_plots = len(data)
    num_groups = (total_plots + (rows * cols) - 1) // (rows * cols)  # Calculate groups based on grid size
    data_items = list(data.items())

    for group_index in range(num_groups):
        start_index = group_index * (rows * cols)
        end_index = min(start_index + (rows * cols), total_plots)
        current_data = data_items[start_index:end_index]

        # Fill missing plots with empty placeholders
        while len(current_data) < (rows * cols):
            current_data.append((f'Empty {len(current_data) + 1}',
                                 {xlabel: np.array([]), ylabel: np.array([]),
                                  yerr_name: np.array([]) if yerr_name else np.array([])}))
        yield current_data, group_index


def configure_axis_lightcurves(ax, row, col, ylabel, color, x_values, y_values, yerr_values, line_name):
    """Configure individual subplot axes."""
    if x_values.size > 0 and y_values.size > 0:
        if yerr_values is not None:
            ax.errorbar(x_values, y_values, yerr=yerr_values,
                        fmt='.:', capsize=3, markersize=4, label=f'{line_name}', color=color)
        else:
            ax.plot(x_values, y_values, label=f'{line_name}', color=color)

        ax.legend(fontsize=8, loc='upper right')

    if col == 0:
        ax.set_ylabel(ylabel)
    else:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(ylabel)

    if row < 3:
        ax.set_xticklabels([])

    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(format_relative_days))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, linestyle='--', linewidth=0.5)

    if row == 0:
        ax_top = ax.secondary_xaxis('top')
        ax_top.xaxis.set_major_locator(MultipleLocator(4))
        ax_top.xaxis.set_major_formatter(FuncFormatter(format_month_day))
        ax_top.tick_params(axis='x', rotation=45, labelsize=8)


def finalize_figure_lightcurves(fig, axes, title, group_index, save_only, output_dir):
    """Finalize figure layout and save or show."""

    check_for_empty_rows(axes, fig, x_label='Relative Days')

    if title:
        fig.suptitle(f'{title} - Group {group_index + 1}', fontsize=14, y=0.95)

    if save_only and output_dir:
        save_path = output_dir / f"{title.replace(' ', '_')}_group_{group_index + 1}.pdf"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def check_for_empty_rows(axes, fig, x_label, formating=True):
    base_mjd = 57581.66
    # Löschen leerer Reihen
    for row in range(4):
        if all(not axes[row, col].has_data() for col in range(2)):
            for col in range(2):
                fig.delaxes(axes[row, col])
    # Ermittlung der untersten verbleibenden Reihe
    remaining_rows = [row for row in range(4) if any(axes[row, col].has_data() for col in range(2))]
    if remaining_rows:  # Überprüfen, ob noch Reihen existieren
        lowest_row = max(remaining_rows)

        # X-Achsenbeschriftungen und Ticks setzen
        for row in remaining_rows:
            for col in range(2):
                if axes[row, col].has_data():  # Stelle sicher, dass die Achse existiert und Daten hat
                    axes[row, col].xaxis.set_major_locator(MultipleLocator(2))  # Ticks festlegen

                    if formating:
                        axes[row, col].xaxis.set_major_formatter(FuncFormatter(format_relative_days))  # Formatierung
                    else:
                        axes[row, col].xaxis.set_major_formatter(
                            plt.FuncFormatter(lambda x, pos: f"{x}"))  # Standard-Werte anzeigen

                    axes[row, col].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

                    if row == lowest_row:  # Nur in der untersten Reihe Beschriftungen und Ticks anzeigen
                        axes[row, col].tick_params(axis='x', which='both', direction='inout', labelbottom=True)
    len_remaining_rows = len(remaining_rows)
    text_heigth = 0.04 + (4 - len_remaining_rows) * 0.20
    fig.text(0.95, text_heigth, f"Base: {base_mjd:.2f} MJD", ha='right', fontsize=10)
    fig.text(0.5, text_heigth, x_label, ha='center', fontsize=12)


def configure_axis_ccfs(ax, row, col, ylabel, color, x_values, y_values, yerr_values, line_name):
    """Configure individual subplot axes."""
    if x_values.size > 0 and y_values.size > 0:
        if yerr_values is not None:
            ax.errorbar(x_values, y_values, yerr=yerr_values,
                        fmt='.:', capsize=3, markersize=4, label=f'{line_name}', color=color)
        else:
            ax.plot(x_values, y_values, label=f'{line_name}', color=color)

        ax.legend(fontsize=8, loc='upper right')

    if col == 0:
        ax.set_ylabel(ylabel)
    else:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(ylabel)

    if row < 3:
        ax.set_xticklabels([])

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, linestyle='--', linewidth=0.5)

    if row == 0:
        ax_top = ax.secondary_xaxis('top')
        ax_top.xaxis.set_major_locator(MultipleLocator(2))
        ax_top.tick_params(axis='x')


def finalize_figure_ccfs(fig, axes, title, group_index, save_only, output_dir):
    """Finalize figure layout and save or show."""

    check_for_empty_rows(axes, fig, x_label='Time Lag', formating=False)

    if title:
        fig.suptitle(f'{title} - Group {group_index + 1}', fontsize=14, y=0.95)

    if save_only and output_dir:
        save_path = output_dir / f"{title.replace(' ', '_')}_group_{group_index + 1}.pdf"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
