import matplotlib.pyplot as plt

def plot_surface(optimiser, bounds):
    grid = np.array(np.meshgrid(np.linspace(bounds[0][0],bounds[0][1], 100),
                                np.linspace(bounds[1][0],bounds[1][1],100)))
    grid = np.swapaxes(grid,0,2)
    orig_shape = grid.shape
    ys = optimiser.predict(grid.reshape(-1,2))
    ys = ys.reshape(orig_shape[:2])
    plt.imshow(ys)
    plt.show()
