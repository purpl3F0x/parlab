import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess
import glob
import numpy as np
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Set methd to use


# # run the simulation from 0 to 512 iterations
def run_task(i):
    subprocess.run(f"mpirun -np 4 ./jacobi_mpi 256 256 2 2 {i}", shell=True)
    subprocess.run(f"mpirun -np 4 ./seidel_mpi 256 256 2 2 {i}", shell=True)
    subprocess.run(f"mpirun -np 4 ./redblack_mpi 256 256 2 2 {i}", shell=True)


# tasks = range(0, 500, 1)
# with ThreadPoolExecutor(max_workers=4) as executor:
#     executor.map(lambda i: run_task(i), tasks)

# tasks = range(500, 1000, 20)
# with ThreadPoolExecutor(max_workers=4) as executor:
#     executor.map(lambda i: run_task(i), tasks)

# tasks = range(1000, 2000, 50)
# with ThreadPoolExecutor(max_workers=4) as executor:
#     executor.map(lambda i: run_task(i), tasks)

# tasks = range(2000, 5000, 100)
# with ThreadPoolExecutor(max_workers=4) as executor:
#     executor.map(lambda i: run_task(i), tasks)

# tasks = range(5000, 10000, 200)
# with ThreadPoolExecutor(max_workers=4) as executor:
#     executor.map(lambda i: run_task(i), tasks)

# tasks = range(10000, 68000, 500)
# with ThreadPoolExecutor(max_workers=4) as executor:
#     executor.map(lambda i: run_task(i), tasks)


images_jacobi = glob.glob("./images/*Jacobi*")
images_gauss = glob.glob("./images/*Seidel*")
images_red = glob.glob("./images/*Red*")

# sort images by iteration number
images_jacobi.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
images_gauss.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
images_red.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

plt.rcParams.update({"font.size": 16})

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12.8, 7.2))

im = ax[0].imshow(np.loadtxt(images_jacobi[0]), cmap="jet")
ax[0].set_title(f"Jacobi")
im2 = ax[1].imshow(np.loadtxt(images_gauss[0]), cmap="jet")
ax[1].set_title(f"Gauss-Seidel")
im3 = ax[2].imshow(np.loadtxt(images_red[0]), cmap="jet")
ax[2].set_title(f"Red-Black")

fig.suptitle("Heat Transfer Simulation Iteration: 0")


def update(frame):
    iter = images_jacobi[frame].split("_")[-1].split(".")[0]
    im.set_array(np.loadtxt(images_jacobi[frame]))
    im2.set_array(np.loadtxt(images_gauss[frame]))
    im3.set_array(np.loadtxt(images_red[frame]))

    fig.suptitle(f"Heat Transfer Simulation Iteration: {iter}")
    return im, fig


ani = animation.FuncAnimation(fig, update, interval=1, blit=True)


FFwriter = animation.FFMpegWriter(fps=12)
ani.save(f"heat_transfer.mp4", writer=FFwriter)
# plt.show()
