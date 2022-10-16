
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from celluloid import Camera
import matplotlib.cm as cm
import numpy as np
import os


def xz_plot(Feaff, Fe, Fi, muVn, t, x, z):
    """
    2D plots of Fe_aff, Fe, Fi and muVn against time, for a single network at (x,z).
    """

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    axs[0, 0].plot(Feaff[:t, x, z])
    axs[0, 0].set_title(
        '$\\nu_e^{aff}$ for pixel (' + str(x) + ', ' + str(z) + ')')
    axs[0, 0].set(xlabel='t (ms)', ylabel='$\\nu_e^{aff}(x, t)$')

    axs[0, 1].plot(Fe[:t, x, z], 'tab:orange')
    axs[0, 1].set_title('$\\nu_e$ for pixel (' + str(x) + ', ' + str(z) + ')')
    axs[0, 1].set(xlabel='t (ms)', ylabel='$\\nu_e(x, t)$')

    axs[1, 0].plot(Fi[:t, x, z], 'tab:green')
    axs[1, 0].set_title('$\\nu_i$ for pixel (' + str(x) + ', ' + str(z) + ')')
    axs[1, 0].set(xlabel='t (ms)', ylabel='$\\nu_i(x, t)$')

    axs[1, 1].plot(muVn[:t, x, z], 'tab:red')
    axs[1, 1].set_title(
        '$\\mu_V^{N}$ for pixel (' + str(x) + ', ' + str(z) + ')')
    axs[1, 1].set(xlabel='t (ms)', ylabel='$\\mu_V^{N}(x, t)$')

    plt.setp(axs, xticks=np.linspace(0, t, 5),
             xticklabels=np.linspace(0, t/2, 5, dtype=int))
    fig.tight_layout()
    plt.show()

    return


def xz_combined(Feaff, Fe, Fi, muVn, length, x, z):
    """
    2D plot that combines the different plots of Fe_aff, Fe, Fi and muVn against time on the same plot,
    for a single network at (x,z).
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.title(
        '$\\nu_e^{aff}$, $\\nu_e$, $\\nu_i$ and $\\mu_V^{N}$ for pixel (' + str(x) + ', ' + str(z) + ')')
    axes = [ax, ax.twinx(), ax.twinx(), ax.twinx()]

    fig.subplots_adjust(right=0.75)
    axes[-2].spines['right'].set_position(('axes', 1.3))
    axes[-2].set_frame_on(True)
    axes[-2].patch.set_visible(False)
    axes[-1].spines['right'].set_position(('axes', 1.6))
    axes[-1].set_frame_on(True)
    axes[-1].patch.set_visible(False)

    axes[0].plot(Feaff[:length, x, z], color='Blue')
    axes[0].set_ylabel('$\\nu_e^{aff}(x, t)$', color='Blue')
    axes[0].set_xlabel('t (ms)')
    axes[0].tick_params(axis='y', colors='Blue')

    axes[1].plot(Fe[:length, x, z], color='Orange')
    axes[1].set_ylabel('$\\nu_e(x, t)$', color='Orange')
    axes[1].set_xlabel('t (ms)')
    axes[1].tick_params(axis='y', colors='Orange')

    axes[2].plot(Fi[:length, x, z], color='Green')
    axes[2].set_ylabel('$\\nu_i(x, t)$', color='Green')
    axes[2].set_xlabel('t (ms)')
    axes[2].tick_params(axis='y', colors='Green')

    axes[3].plot(muVn[:length, x, z], color='Red')
    axes[3].set_ylabel('$\\mu_V^{N}(x, t)$', color='Red')
    axes[3].set_xlabel('t (ms)')
    axes[3].tick_params(axis='y', colors='Red')

    plt.setp(
        ax,
        xticks=np.linspace(0, length, 5),
        xticklabels=np.linspace(0, length/2, 5, dtype=int)
    )
    fig.tight_layout()
    plt.show()

    return


def xz_movie(Feaff, Fe, Fi, muVn, X, Z, length, fps=10, title='output'):
    """
    Movie of contour plots of Fe_aff, Fe, Fi and muVn in the (x,z) plane. 
    """

    def colorbar_format(x, pos):
        a = '{:.3f}'.format(x)
        return format(a)

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 0].set_title('$\\nu_e^{aff}$')
    axs[0, 0].set(xlabel='X (mm)', ylabel='Z (mm)')
    axs[0, 1].set_title('$\\nu_e$')
    axs[0, 1].set(xlabel='X (mm)', ylabel='Z (mm)')
    axs[1, 0].set_title('$\\nu_i$')
    axs[1, 0].set(xlabel='X (mm)', ylabel='Z (mm)')
    axs[1, 1].set_title('$\\mu_V^{N}$')
    axs[1, 1].set(xlabel='X (mm)', ylabel='Z (mm)')

    camera = Camera(fig)

    for i in range(0, length, fps):
        cbar0 = axs[0, 0].contourf(X, Z, Feaff[i, :, :].T,
                                   np.linspace(Feaff.min(), Feaff.max(), 20),
                                   cmap=cm.viridis)
        cbar1 = axs[0, 1].contourf(X, Z, Fe[i, :, :].T,
                                   np.linspace(Fe.min(), Fe.max(), 20),
                                   cmap=cm.viridis)
        cbar2 = axs[1, 0].contourf(X, Z, Fi[i, :, :].T,
                                   np.linspace(Fi.min(), Fi.max(), 20),
                                   cmap=cm.viridis)
        cbar3 = axs[1, 1].contourf(X, Z, muVn[i, :, :].T,
                                   np.linspace(muVn.min(), muVn.max(), 20),
                                   cmap=cm.viridis)
        camera.snap()

    anim = camera.animate()

    fig.colorbar(cbar0, ax=axs[0, 0],
                 format=ticker.FuncFormatter(colorbar_format))
    fig.colorbar(cbar1, ax=axs[0, 1],
                 format=ticker.FuncFormatter(colorbar_format))
    fig.colorbar(cbar2, ax=axs[1, 0],
                 format=ticker.FuncFormatter(colorbar_format))
    fig.colorbar(cbar3, ax=axs[1, 1],
                 format=ticker.FuncFormatter(colorbar_format))

    fig.tight_layout()

    # Saving movie
    absolute_path = os.path.realpath(os.path.dirname(__file__))
    relative_file_path = f'../results/movies/{title}.mp4'
    path = os.path.join(absolute_path, relative_file_path)
    anim.save(path)
    print(f'Movie saved in {path}.')

    plt.close(fig)

    return


def show_rand_conn(random_conn_params):
    """
    Plots the random connections in the 'Torus with random elements Model'.
    """

    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(111)
    colors = cm.rainbow(np.linspace(
        0, 1, random_conn_params['nb_random_conn']))

    for x_pix, z_pix, x_neigh, z_neigh, c in zip(
            random_conn_params['x_pixel'], random_conn_params['z_pixel'],
            random_conn_params['x_neigh'], random_conn_params['z_neigh'], colors):
        ax.scatter(x_pix, z_pix, color=c)
        ax.scatter(x_neigh, z_neigh, color=c)
        ax.plot([x_pix, x_neigh], [z_pix, z_neigh], color=c, linewidth=1)

    plt.title('Random connectivity')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Z (pixel)')
    plt.show()

    return
