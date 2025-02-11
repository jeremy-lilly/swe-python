import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import torch as tn
import torchtt as tt


def calc_compression(tt):
    entries = sum([tn.numel(core) for core in tt.cores])
    compression = entries / np.prod(np.array(tt.N))

    return compression, entries
# END calc_compression()


def get_compression_data(data):
    nvars = len(data)
    ntimeLevels = data[0]['var'].shape[0]

    compression = []
    err = []
    for i in range(nvars):
        compression.append(np.zeros(ntimeLevels))
        err.append(np.zeros(ntimeLevels))
    # END for
    
    for level in range(ntimeLevels):
        for ind in range(nvars):
            var = data[ind]['var'][level, :]
            shape = data[ind]['shape']
            eps = data[ind]['eps']
        
            var_tt = tt.TT(var, shape, eps=eps)
        
            compression[ind][level], _ = calc_compression(var_tt)
        
            err[ind][level] = (np.sqrt(np.sum((var_tt.full().flatten() - var)**2)) /
                               (np.sqrt(np.sum(var**2)) + 1e-32))
        # END for
    # END for
    return compression, err
# END get_compression_data()


def make_eps_plot(data, compression, err, figtitle, savefile):
    nvars = len(data)
    ntimeLevels = data[0]['var'].shape[0]
    
    time_step = 480
    save_freq = 4 * 60 * 60 / time_step
    time = time_step * save_freq * np.arange(ntimeLevels) / 60 / 60 / 24
    ones = np.ones(ntimeLevels)
    
    fig, axes = plt.subplots(2, 1, tight_layout=True)

    for ind in range(nvars):
        axes[0].plot(time, compression[ind], 'o-', label=f'{data[ind]['name']}, eps={data[ind]['eps']}')
    # END for
    axes[0].plot(time, ones, ':k')
    axes[0].set(ylabel='compression')
    axes[0].legend()

    for ind in range(nvars):
        axes[1].plot(time, err[ind], 'x--', label=f'{data[ind]['name']}, eps={data[ind]['eps']}')
    # END for
    axes[1].set(yscale='log',
                xlabel='time (days)',
                ylabel='normalized frob error')
    axes[1].legend()
    
    fig.suptitle(figtitle)
    plt.savefig(savefile)
    plt.show()
# END make_eps_plot()


def order_by_angle(ds, root, adjacent):
    root_lat = ds.variables['latCell'][root]
    if root_lat > 0:
        root_pt = np.array([ds.variables['xCell'][root],
                            ds.variables['zCell'][root]])
        root_vec = np.array([0, -1])
    else:    
        root_pt = np.array([ds.variables['xCell'][root],
                            ds.variables['yCell'][root]])
        root_vec = np.array([1, 0])
    # END if

    angles = np.zeros(adjacent.size)
    for i, ind in enumerate(adjacent):
        if root_lat > 0:
            pt = np.array([ds.variables['xCell'][ind],
                           ds.variables['zCell'][ind]])
        else:
            pt = np.array([ds.variables['xCell'][ind],
                           ds.variables['yCell'][ind]])
        vec = pt - root_pt
        
        angle = np.arccos(np.dot(root_vec, vec) / np.linalg.norm(vec))
        if vec[1] < 0 and root_lat <= 0:
            angle = 2 * np.pi - angle
        # END if
        angles[i] = angle
    # END for
    
    sorted_inds = np.argsort(angles)
    return adjacent[sorted_inds], angles[sorted_inds]
# END order_by_angle


def get_spiral_cell_order(filename):
    ds = nc.Dataset(filename)
    cells_on_cell = ds.variables['cellsOnCell']
    
    
    visited = np.zeros(ds.dimensions['nCells'].size, dtype=int)
    halo = [[], []]
    
    start_ind = 0
    visited[start_ind] = 1
    sorted_inds = [start_ind]
    halo[0].append(start_ind)
    
    iteration = 0
    while halo[0] or halo[1]:
        cur_halo = iteration % 2
        other_halo = (iteration + 1) % 2
    
        # sweep top half of cells adjacent to the cell
        # at the end of the current halo
        root = halo[cur_halo][-1]
        
        root_lat = ds.variables['latCell'][root]
        if root_lat > 0:
            threshold = np.pi / 2
        else:
            threshold = np.pi
        
        adjacent = np.array(cells_on_cell[root][cells_on_cell[root] != 0]) - 1
        adjacent, angles = order_by_angle(ds, root, adjacent)
        for ind, angle in zip(adjacent, angles):
            if angle < threshold and not visited[ind]:
                #print(f'1root = {root}, ind = {ind}')
                visited[ind] = 1
                sorted_inds.append(ind)
                halo[other_halo].append(ind)
            # END if
        # END for
    
        # sweep all cells adjacent to the cells in 
        # interior of the current halo
        for root in halo[cur_halo][:-1]:
            adjacent = np.array(cells_on_cell[root][cells_on_cell[root] != 0]) - 1
            adjacent, angles = order_by_angle(ds, root, adjacent)
            #print(f'root = {root}\n{adjacent}\n{angles}')
            for ind, angle in zip(adjacent, angles):
                if not visited[ind]:
                    #print(f'2root = {root}, ind = {ind}')
                    visited[ind] = 1
                    sorted_inds.append(ind)
                    halo[other_halo].append(ind)
                # END if
            # END for
        # END for
        
        # sweep bottom half of cells adjacent to the cell
        # at the end of the current halo
        root = halo[cur_halo][-1]
        adjacent = np.array(cells_on_cell[root][cells_on_cell[root] != 0]) - 1
        adjacent, angles = order_by_angle(ds, root, adjacent)
        for ind, angle in zip(adjacent, angles):
            if angle >= threshold and not visited[ind]:
                #print(f'3root = {root}, ind = {ind}')
                visited[ind] = 1
                sorted_inds.append(ind)
                halo[other_halo].append(ind)
            # END if
        # END for
    
        #print(f'iteration = {iteration}')
        #print(sorted_inds)
        #print(halo[cur_halo])
        #print(halo[other_halo])
        #print('')
        halo[cur_halo][:] = []
        iteration += 1
    
        #if iteration == 3: break
    # END while
    spiral_cell_inds = np.array(sorted_inds)
    #print(spiral_cell_inds)
    
    ds.close()

    return spiral_cell_inds
# END get_spiral_order()


def get_bfs_cell_order(filename):
    ds = nc.Dataset(filename)
    
    cells_on_cell = ds.variables['cellsOnCell']
    visited = np.zeros(ds.dimensions['nCells'].size, dtype=int)
    root_ind = 0
    
    sorted_cell_inds = []
    queue = [root_ind]
    visited[root_ind] = 1
    while queue:
        cur_ind = queue.pop(0)
        sorted_cell_inds.append(cur_ind)
    
        adjacent_inds = np.array(cells_on_cell[cur_ind][cells_on_cell[cur_ind] != 0]) - 1
        for ind in adjacent_inds:
            if not visited[ind]:
                queue.append(ind)
                visited[ind] = 1
            # END if
        # END for
    # END while   
    bfs_cell_inds = np.array(sorted_cell_inds)
    #print(bfs_cell_inds)
    
    ds.close()
    return bfs_cell_inds
# END get_bfs_order()
