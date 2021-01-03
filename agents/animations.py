import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import phyre

random.seed(0)


def animateSimulatedTask(images):
    print(len(images))
    num_across = 5
    task_img0 = images
    # taskid = tasks[40]
    height = int(math.ceil(len(task_img0) / num_across))
    fig, axs = plt.subplots(height, num_across, figsize=(20, 15))
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.2)

    # We can visualize the simulation at each timestep.

    for i, (ax, image) in enumerate(zip(axs.flatten(), task_img0)):
        # Convert the simulation observation to images.
        if image is None:
            continue
        img = phyre.observations_to_float_rgb(image)
        ax.imshow(img)
        ax.title.set_text(f'Timestep {i}')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    plt.show()

    frames = []
    for taskimgi in task_img0:
        if taskimgi is None:
            continue
        frames.append(phyre.observations_to_uint8_rgb(taskimgi))
    from array2gif import write_gif
    timestr = time.strftime("%Y%m%d-%H%M%S")
    write_gif(np.asarray(frames), 'rgbbgr.gif' + timestr, fps=8)
