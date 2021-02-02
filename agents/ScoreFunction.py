# delete white trash
from phyre import SimulationStatus
import numpy as np
import animations as a

def getImpuls(simulation_result):
    radius = simulation_result.featurized_objects.diameters[2]
    x0 =  simulation_result.featurized_objects.xs[0][2]
    y0 = simulation_result.featurized_objects.ys[0][2]

    x1 = simulation_result.featurized_objects.xs[-1][2]
    y1 = simulation_result.featurized_objects.ys[-1][2]
    strecke =np.sqrt((y0-y1)**2 + (x0-x1)**2)
    time = len(simulation_result.images)
    velocity = time * strecke
    mass = radius **2 * np.pi
    impuls = mass * velocity

    return impuls


def getDistnaceBeginningEnd(simulation_result):
    dist_start = simulation_result.featurized_objects.xs[0][0]
    dist_end = simulation_result.featurized_objects.xs[-1][0]
    dist_start_goal = simulation_result.featurized_objects.xs[0][1]
    dist_end_goal = simulation_result.featurized_objects.xs[-1][1]

    rel_dist = 0
    if(abs(dist_start - dist_end_goal) < abs(dist_end - dist_end_goal) ):
        rel_dist = -1
    else:
        rel_dist = 1

    return rel_dist


#def checkInvolvement(simulation_result):

def ScoreFunctionValue(simulation_result):
    score = 0
    impuls_score = 0
    distance_score=0
    time_score=0
    ##check distance object_green
    final_image = simulation_result
    distance = getDistanceObjectGoal(final_image)
    relative_distance = getDistnaceBeginningEnd(simulation_result)
    impuls = getImpuls(simulation_result)
    #print("distance:", distance)
    if distance <= 1.8*10**(-4):
        distance_score = 100
    elif distance > 1.8*10**(-4) and distance < 0.32:
        distance_score = 100*np.exp(-distance/2)
    else:
        distance_score = 0

    time_score = getTemporalScore(simulation_result)
    impuls = getImpuls(simulation_result)


    if(distance_score < 0 ):
        impuls_score = 0
    else:
        impuls_score = impuls
    score = distance_score + time_score + impuls_score*20
    return score

def getTemporalScore(simulation_result):
    images = simulation_result.images
    count = 0

    #for i in simulation_result:
    for i in range(len(images)):
        distance = getDistanceObjectGoal(simulation_result, i)

        if distance <= 1.8*10**(-4):
           count = count + 1

    tempo_score = count * 10
    return tempo_score

def getDistanceObjectGoal(simulation_result, i=-1):

    distance = 0

    diameter_object = simulation_result.featurized_objects.diameters[1]
    x_object = simulation_result.featurized_objects.xs[i][1]
    y_object = simulation_result.featurized_objects.ys[i][1]

    vector_object = np.array([x_object, y_object])


    diameter_goal = simulation_result.featurized_objects.diameters[0]
    x_goal = simulation_result.featurized_objects.xs[i][0]
    y_goal = simulation_result.featurized_objects.ys[i][0]
    vector_goal = np.array([x_goal, y_goal])


    distance = np.linalg.norm(vector_goal-vector_object)
    distance = distance - diameter_goal/2 - diameter_object/2

    return distance

# path =  "/home/kyra/Desktop/phyre/agents/samples/SimulationLog20210118-184723.npy"
# sample = np.load(path, allow_pickle=True)
#
# value = ScoreFunctionValue(sample[1])
# print("value:",value)
# images = sample[1].images
# a.animateSimulatedTask(images)

