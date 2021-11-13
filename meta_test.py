import metaworld
import time
import random

ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks
TRAIN_CLASS = 'drawer-close-v1'
env = ml10.train_classes[TRAIN_CLASS]()
task = [task for task in ml10.train_tasks
                        if task.env_name == TRAIN_CLASS]
env.set_task(task[0])
print(env)
for _ in range(1):
  print(f'before reset: {env._target_pos}')
  # env.reset()
  print(f'after reset: {env._target_pos}')
  print(f'OBJECT POS ===> {env._get_pos_objects}')
  for _ in range(500):
    env.render()
    env.step(env.action_space.sample())
    # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
    time.sleep(0.1)