import numpy as np
import os
from phyre import SimulationStatus
import ScoreFunction as sf
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import phyre


def concatenateSamples(isUpdate=False):
    arr = []
    count = 0
    path = "/media/kyra/Elements/phyre/agents/samples/"
    if isUpdate:
        path = "/media/kyra/Elements/phyre/agents/UpdatedSamples/"
    for file in os.listdir(path):
        count = count + 1
        print(count)
        temp = np.load(path + file, allow_pickle=True)
        for i in range(temp.shape[0]):
            arr.append(temp[i])
    arr = np.asarray([arr])        
    print("mh")
    path = "/media/kyra/Elements/phyre/agents/ConcSamples"
    np.save(os.path.join(path, "Concat"), arr)

concatenateSamples()
def concatenateTestSamples():
    arr = []
    path = "/home/kyra/Desktop/phyre/agents/TestSamples/"
    for file in os.listdir(path):
        temp = np.load(path + file, allow_pickle=True)
        for i in range(temp.shape[0]):
            arr.append(temp[i])
    return arr


def prepareSamples():
    print("here")
    path = "/media/kyra/Elements/phyre/agents/ConcSamples/Concat"
    samples = np.load(path, allow_pickle=True)
    images = []
    action_list = []
    label = []
    for i in range(len(samples)):
        image = samples[i].images[0]
        rgb_image = phyre.observations_to_float_rgb(image)
        rgb_image = resize(rgb_image, (128, 128))
        # image2 = resize(samples[i].images[-1], (64, 64,1))
        images.append(rgb_image)
        # images.append(image2)
        action = samples[i].featurized_objects.features[0][2]
        actions = np.array([action[0], action[1], action[3]])
        action_list.append(actions)
        # if flag == "score":
        #     score = sf.ScoreFunctionValue(samples[i])
        #     label.append(score)
        # else:
        if (samples[i].status == SimulationStatus.SOLVED):
            label.append(1)
        else:
            label.append(0)

    label = np.asarray(label)
    images = np.asarray(images)
    action = np.asarray(action_list)
    return images, action, label


def prepareTestSamples():
    samples = concatenateTestSamples()
    action_list = []
    image_list = []
    for i in range(len(samples)):
        image = samples[i][0].images[0]
        action = samples[i][0].featurized_objects.features[0][2]
        rgb_image = phyre.observations_to_float_rgb(image)
        rgb_image = resize(rgb_image, (128, 128))
        image_list.append(np.array([rgb_image]))
        actions = np.array([action[0], action[1], action[3]])
        action_list.append(np.array([actions]))

    action = np.asarray(action_list)
    image = np.asarray(image_list)
    return action, image


def prepareSamplesWithScore(is_update=False):
    print("here")
    path = "/media/kyra/Elements/phyre/agents/ConcSamples/Concat"
    samples = np.load(path, allow_pickle=True)
    images = []
    action_list = []
    label = []
    for i in range(len(samples)):
        score = samples[i][1]
        image = samples[i][0].images[0]
        rgb_image = phyre.observations_to_float_rgb(image)
        rgb_image = resize(rgb_image, (128, 128))
        images.append(rgb_image)
        # image2 = resize(samples[i].images[-1], (64, 64,1))
        # images.append(rgb_image)
        # images.append(image2)
        action = samples[i][0].featurized_objects.features[0][2]
        actions = np.array([action[0], action[1], action[3]])
        action_list.append(actions)
        # if flag == "score":
        #     score = sf.ScoreFunctionValue(samples[i])
        #     label.append(score)
        # else:
        label.append(score)

    label = np.asarray(label)
    images = np.asarray(images)
    action = np.asarray(action_list)
    return images, action, label


def prepareTestSamplesWithScore():
    samples = concatenateSamples()
    images = []
    action_list = []
    label = []
    for i in range(len(samples)):
        score = sf.ScoreFunctionValue(samples[i])
        image = samples[i].images[0]
        rgb_image = phyre.observations_to_float_rgb(image)
        rgb_image = resize(rgb_image, (128, 128))
        images.append(rgb_image)
        # image2 = resize(samples[i].images[-1], (64, 64,1))
        # images.append(rgb_image)
        # images.append(image2)
        action = samples[i].featurized_objects.features[0][2]
        actions = np.array([action[0], action[1], action[3]])
        action_list.append(actions)
        # if flag == "score":
        #     score = sf.ScoreFunctionValue(samples[i])
        #     label.append(score)
        # else:
        label.append(score)

    label = np.asarray(label)
    images = np.asarray(images)
    action = np.asarray(action_list)
    return images, action, label


def prepareForScore():
    samples = concatenateSamples()
    return samples


def prepareTestActions():
    pass
    # simulator = phyre.initialize_simulator(tasks, tier)
    # action = simulator.sample()


# images, action, label = prepareSamplesWithScore()
# print(label)
