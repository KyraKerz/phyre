import random

import numpy as np
import phyre
from tqdm import tqdm_notebook

import animations

random.seed(0)

# Evaluation Setup
eval_setup = 'ball_cross_template'
fold_id = 0  # For simplicity, we will just use one fold for evaluation.
train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup,0)
action_tier = phyre.eval_setup_to_action_tier(eval_setup)
tasks = dev_tasks[0:1]
print((tasks))
simulator = phyre.initialize_simulator(tasks, action_tier)
actions = simulator.build_discrete_action_space(max_actions=1000)


def evaluate_random_agent(tasks, tier):
    # Create a simulator for the task and tier.
    simulator = phyre.initialize_simulator(tasks, tier)
    evaluator = phyre.Evaluator(tasks)
    assert tuple(tasks) == simulator.task_ids
    images = []
    actions = []
    for task_index in tqdm_notebook(range(len(tasks)), desc='Evaluate tasks'):
        while evaluator.get_attempts_for_task(
                task_index) < phyre.MAX_TEST_ATTEMPTS:
            # Sample a random valid action from the simulator for the given action space.
            action = simulator.sample()
            # Simulate the given action and add the status from taking the action to the evaluator.
            status = simulator.simulate_action(task_index,
                                               action,
                                               need_images=True)

            stati = status.status
            actions.append(action)
            images.append(status.images)
            evaluator.maybe_log_attempt(task_index, stati)
    return evaluator, images, actions
print("hello")

finish, images, actionlog = evaluate_random_agent(tasks, action_tier)
finishlog = finish._log

indices1 = [i for i, x in enumerate(images) if x is None]


for i in reversed(indices1):
    del images[i]
    del actionlog[i]
    #del finishlog[i]
#print(images)
indices2 = [i for i, x in enumerate(images) if images[i].shape[0] != 17]
for i in reversed(indices2):
    del images[i]
    del actionlog[i]
    del finishlog[i]


new_list = [ seq[1] for seq in finishlog ]

print(len(images))
print("actionlog:",len(actionlog))
print(len(new_list))
imagearray = np.asarray(images)
print(imagearray.shape)
np.save('ImagesLog1.npy', images)
np.save('EvaluationsLog1.npy', new_list)
np.save('ActionLog1.npy1', actionlog)



animations.animateSimulatedTask(images)
