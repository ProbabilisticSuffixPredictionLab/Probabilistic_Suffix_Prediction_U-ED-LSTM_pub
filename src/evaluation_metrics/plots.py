import numpy as np
import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
import io

def plot_pits(data, x_label='', caption='', pgf=False):
    if pgf:
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",  # Use pdflatex or lualatex
            "font.family": "serif",       # Match LaTeX font
            "text.usetex": True,          # Enable LaTeX rendering
            "pgf.rcfonts": False,         # Disable rc settings override
        })
    plt.rcParams.update({
        'font.size': 8,          # General font size
        'axes.titlesize': 15,     # Title font size
        'axes.labelsize': 26,     # Axis label size
        'xtick.labelsize': 20,    # X-axis tick labels
        'ytick.labelsize': 20,    # Y-axis tick labels
        'xtick.major.width' : 0.2,
        'ytick.major.width' : 0.2,
        'legend.fontsize': 20,    # Legend font size
        'lines.linewidth': 0.7,  # Thinner lines
        'lines.markersize': 3    # Smaller markers
    })
    d = [v['prob'] for v in data.values()]
    proba_bin_edges = np.linspace(0, 1, 30)
    proba_counts, proba_bin_edges = np.histogram(d, bins=proba_bin_edges, density=True)
    proba_bin_centers = (proba_bin_edges[:-1] + proba_bin_edges[1:]) / 2

    m = [v['mean'] for v in data.values()]
    mean_bin_edges = np.linspace(0, 1, 3)
    mean_counts, mean_bin_edges = np.histogram(m, bins=mean_bin_edges, density=True)
    mean_bin_centers = (mean_bin_edges[:-1] + mean_bin_edges[1:]) / 2


    fig, ax1 = plt.subplots()

    if caption:
        plt.title(r'\textbf{' + caption + '}', fontsize=25)
    

    #ax2 = ax1.twinx()
    #ax1.bar(mean_bin_centers, mean_counts, width=np.diff(mean_bin_edges),
    #        color='orange', alpha=0.6)


    ax1.bar(proba_bin_centers, proba_counts, width=np.diff(proba_bin_edges),
            color='blue', alpha=1)
    
    ax1.axhline(y=1, color='grey', linestyle='--', label="y = 1", linewidth=2,
                alpha=1)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('p')
    # Add vertical line at y = 1
    #ax1.axhline(y=1, color='blue', linestyle='--', label="y = 1",
    #            alpha=0.2)


    # Labels and title
    plt.xlabel(f"PIT | {x_label}")
    plt.ylabel("Probability density")
    #plt.title("Histogram with Probability Distribution and Vertical Line")

    # Add legend
    plt.legend()
    fig.tight_layout()


    # Show plot
    if pgf:
        pgf_stream = io.BytesIO()
        fig.savefig(pgf_stream, format='pgf', bbox_inches='tight', pad_inches=0)
        pgf_data = pgf_stream.getvalue()
        pgf_stream.close()
        return pgf_data
    else:
        plt.show()


def plot_res(res, c, columns, caption='', pgf=False):
    mpl.rcdefaults()
    if pgf:
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",  # Use pdflatex or lualatex
            "font.family": "serif",       # Match LaTeX font
            "text.usetex": True,          # Enable LaTeX rendering
            "pgf.rcfonts": False,         # Disable rc settings override
        })

    plt.rcParams.update({
        'font.size': 6,          # General font size
        'axes.titlesize': 7,     # Title font size
        'axes.labelsize': 9,     # Axis label size
        'xtick.labelsize': 7,    # X-axis tick labels
        'ytick.labelsize': 7,    # Y-axis tick labels
        'xtick.major.width' : 0.2,
        'ytick.major.width' : 0.2,
        'legend.fontsize': 7,    # Legend font size
        'lines.linewidth': 0.7,  # Thinner lines
        'lines.markersize': 3    # Smaller markers
    })

    n_rows = int((len(res.keys())*2 + 1) / columns)
    fig, axes = plt.subplots(n_rows, columns, figsize=(4, 2 * n_rows), dpi=300,
                            gridspec_kw={'width_ratios' : [1]*columns, 'height_ratios' : [1]*n_rows},
                )
    if caption:
        fig.text(0.15, 0.85, r'\underline{\smash{\textbf{' + caption + '}}}', ha='center', va='center', fontsize=18)

    #fig.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)
    axes = axes.flatten(order='F')

    axes_enumerator = enumerate(axes)
    for v, i in {'prefix' : 1, 'suffix' : 2}.items():
        new_order = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(list)))

        for (metric_name, label) , metric_results in res.items():
            for info, result in metric_results.items():
                for result_key, result_value in result.items():
                    new_order[(metric_name, label)][result_key][info[i]].append(result[result_key])


        if v == 'prefix':
            background_data = c[0]
        else:
            background_data = c[1]

        # MSE, Levensthein, ...
        for j, (metric_name, label) in enumerate(new_order.keys()):
            y_label = label
            
            #fig, ax1 = plt.subplots(figsize=(6, 4), dpi=100)
            #ax1.set_title(metric_name)
            _, ax1 = next(axes_enumerator)
            if v == 'prefix':
                ax1.set_ylabel(y_label, labelpad=0.0)#, color="dimgray")
                #ax1.set_ylim(bottom=0)



            for spine in ax1.spines.values():
                spine.set_visible(False)


            # Change tick labels and tick strokes to grey
            #ax1.tick_params(axis="both", colors="dimgrey", pad=0.5)
            #ax1.set_xlabel("", color="dimgray")

            # name: mean / prob
            # data: dict[tuple, list]
            i = 0
            for name, data in new_order[(metric_name, label)].items():
                if name == 'mean':
                    label_name = 'most-likely suffix'
                elif name == 'prob':
                    label_name = 'mean probabilistic suffix'
                ax1.margins(x=0)
                if len(data) and isinstance(data[next(iter(data))][0], tuple):
                    # data is tuple (mean, (min, max))
                    sorted_keys = sorted(data.keys())
                    
                    # Probabilistic blue sample line:
                    ax1.plot(sorted_keys,
                        [np.mean([m for m, _ in data[k]]) for k in sorted_keys],
                        marker='x', linestyle='-', label=label_name,
                        color='C0')
                    
                    #ax1.plot(sorted_keys,
                    #        [np.mean([b for _, (b, t) in data[k]]) for k in data.keys()],
                    #        marker=None, linestyle='--', alpha=0.8, label=label_name + ' bottom',
                    #        color='C0')
                    #ax1.plot(sorted_keys,
                    #        [np.mean([t for _, (b, t) in data[k]]) for k in data.keys()],
                    #        marker=None, linestyle='--', alpha=0.8, label=label_name + ' top',
                    #        color='C0')
                    
                    ax1.fill_between(sorted_keys,
                                     [np.mean([b for _, (b, t) in data[k]]) for k in sorted_keys],
                                     [np.mean([t for _, (b, t) in data[k]]) for k in sorted_keys],
                                     color='blue', alpha=0.1, label="IQR Range")
                else:
                    # data is only mean
                    if 'most-likely' in label_name:
                        kwargs = {'color' : 'C1', 'marker' : 'o'}
                    else:
                        kwargs = {'color' : 'C0', 'marker' : 'x'}
                    sorted_keys = sorted(data.keys())
                    
                    ax1.plot(sorted_keys,
                            [np.mean(data[k]) for k in sorted_keys],
                            linestyle='-', label=label_name,
                            **kwargs)



            ax2 = ax1.twinx()
            for spine in ax2.spines.values():
                spine.set_linewidth(0.2)
        
            ax2.spines['top'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            #ax2.tick_params(colors="dimgrey", pad=0.2)

            # Plot the background data first
            sorted_bg_keys = sorted(background_data.keys())
            sorted_bg_values = [background_data[k] for k in sorted_bg_keys]
            ax2.plot(sorted_bg_keys, 
                sorted_bg_values, 
                linestyle='--', color='gray', label='\# instances' if pgf else '# instances')
            ax2.fill_between(sorted_bg_keys, sorted_bg_values, color='gray', alpha=0.3)
            #plt.grid(True)
            if v == 'suffix':
                ax2.set_ylabel('instances', labelpad=0.0)#, color="dimgray")
            #plt.legend()

            if v == 'prefix':
                ax1.set_ylim(bottom=0)
                ax2.set_ylim(bottom=0)
                ax1.set_xlim(left=0)
                ax2.set_xlim(left=0)
                ax2.set_yticks([])
                ax2.spines['right'].set_visible(False)
            elif v == 'suffix':
                y_top = max(ax1.get_ylim()[1], axes[j].get_ylim()[1])
                ax1.set_ylim((0, y_top))
                ax1.set_xlim(left=0)
                ax2.set_xlim(left=0)
                axes[j].set_ylim((0, y_top))
                ax1.tick_params(left=False, labelleft=False)
                ax2.set_ylim(bottom=0)

            
            ax1.set_xlabel(v + ' len', labelpad=0.2)
            # Show plot
            # Move ax2 to the background
            ax1.set_zorder(2)  # Bring ax1 to the front
            ax2.set_zorder(1)  # Push ax2 to the back
            ax1.patch.set_visible(False)  # Hide ax1 background so fill is visible

            if ax1 == axes[0]:
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
                           frameon=True)
            pos = ax1.get_position()
            ax1.set_position([pos.x0 + 0.01, pos.y0 + 0.01, pos.width * 0.98, pos.height * 0.98])
            

    #handles1, labels1 = ax1.get_legend_handles_labels()
    #handles2, labels2 = ax2.get_legend_handles_labels()
    #fig.legend(handles1 + handles2, labels1 + labels2, loc='lower center')
    #plt.margins(0,0.3)
    plt.subplots_adjust(left=0, right=1, top=0.85, bottom=0, hspace=0.2, wspace = 0.05)

    if pgf:
        pgf_stream = io.BytesIO()
        fig.savefig(pgf_stream, format='pgf', bbox_inches='tight', pad_inches=0.1)
        pgf_data = pgf_stream.getvalue()
        pgf_stream.close()
        return pgf_data
    else:
        plt.show()