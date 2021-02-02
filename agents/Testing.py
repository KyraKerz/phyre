import os
import time

import PrepareSamples as ps
import ScoreFunction as sf
import keras
import numpy as np
# random.seed(time)
import phyre


def simulate_result(chosen_action, chosen_score, model_number, generation_number):
    eval_setup = 'ball_cross_template'
    fold_id = 0  # For simplicity, we will just use one fold for evaluation.
    train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, 0)
    action_tier = phyre.eval_setup_to_action_tier(eval_setup)
    tasks = dev_tasks[0:1]
    simulator = phyre.initialize_simulator(tasks, action_tier)
    evaluator = phyre.Evaluator(tasks)
    # Simulate the given action and add the status from taking the action to the evaluator.
    simulation_result = simulator.simulate_action(0, chosen_action, need_images=True, need_featurized_objects=True)
    simulation_score = sf.ScoreFunctionValue(simulation_result)
    pair = np.array([chosen_action, simulation_score])
    timestr = time.strftime("%Y%m%d-%H%M%S")
    score_pair = [chosen_score, simulation_score, model_number, generation_number]
    score_string = "ScoreLog" + timestr
    path = "/home/kyra/Desktop/phyre/agents/Scores"
    np.save(os.path.join(path, score_string), score_pair)
    return pair, simulation_result


def add_samples_to_trained_model(generation_number):
    test_action_list = []
    test_action_list_with_with_sim_score = []
    # loadModels
    loaded_model1 = keras.models.load_model('/home/kyra/Desktop/phyre/agents/model1')
    loaded_model2 = keras.models.load_model('/home/kyra/Desktop/phyre/agents/model2')
    loaded_model3 = keras.models.load_model('/home/kyra/Desktop/phyre/agents/model3')
    loaded_model4 = keras.models.load_model('/home/kyra/Desktop/phyre/agents/model4')
    # loaded_model5 = keras.models.load_model('/tmp/model5')
    models = [loaded_model1, loaded_model2, loaded_model3, loaded_model4]
    # load TestImages and actions
    test_actions, test_images = ps.prepareTestSamples()
    # print(len(test_actions))

    # Feed Testimages & action in Network
    predicted_score_list = []
    model_number = 0
    for model in models:
        model_number = model_number + 1
        for i in range(test_actions.shape[0]):

            predicted_score = model.predict(x=[test_images[i], test_actions[i]])

            predicted_score_list.append((np.asarray(test_actions[i][0]), predicted_score[0][0]))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    # predicted_score_list = np.asarray(predicted_score_list)
    max_value = max(predicted_score_list, key=lambda item: item[1])
    chosen_action = max_value[0]
    chose_score = max_value[1]




    pair, simulation_result = simulate_result(chosen_action, chose_score, model_number, generation_number)
    test_action_list_with_with_sim_score.append([simulation_result, pair[1]])

    simulation_string = "SimulationLog" + timestr
    path = "/home/kyra/Desktop/phyre/agents/UpdatedSamples"
    np.save(os.path.join(path, simulation_string), test_action_list_with_with_sim_score)


# add_samples_to_trained_model()
