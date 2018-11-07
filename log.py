import cv2
import random
from gym_duckietown.envs import DuckietownEnv
from teacher import PurePursuitExpert
from _loggers import Logger

# Log configuration, you can pick your own values here
# the more the better? or the smarter the better?
EPISODES = 50
STEPS = 512

DEBUG = True
TOP_CROP_VALUE = 17
ERROR_GENERATION_CHANCE = 3
STEPS_PER_ERROR_GENERATION = 20


env = DuckietownEnv(
    map_name='udem1',  # check the Duckietown Gym documentation, there are many maps of different complexity
    max_steps=EPISODES * STEPS
)

# this is an imperfect demonstrator... I'm sure you can construct a better one.
expert = PurePursuitExpert(env=env)

# please notice
logger = Logger(env, log_file='train.log')

error_being_induced = False
error_step_count = STEPS_PER_ERROR_GENERATION

# let's collect our samples
for episode in range(0, EPISODES):
    for steps in range(0, STEPS):
    
        
        if error_being_induced == False and random.randint(1,101)<=ERROR_GENERATION_CHANCE:
            error_being_induced = True
            error_step_count = STEPS_PER_ERROR_GENERATION
        elif error_being_induced == True and error_step_count == 0:
            error_being_induced = False

        if error_being_induced == True:
            action = (0.8,2)
        else:
            # we use our 'expert' to predict the next action.
            action = expert.predict(None)

        
        observation, reward, done, info = env.step(action)
        # we can resize the image here
        observation = cv2.resize(observation, (80, 60))
        observation = observation[TOP_CROP_VALUE:60, 0:80]
        # NOTICE: OpenCV changes the order of the channels !!!
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

        # we may use this to debug our expert.
        if DEBUG:
            cv2.imshow('debug', observation)
            cv2.waitKey(1)

        #If we're not inducing an error, record the data
        if error_being_induced == False:
            logger.log(observation, action, reward, done, info)
        else:
            error_step_count -= 1
        # [optional] env.render() to watch the expert interaction with the environment
        # we log here
    logger.on_episode_done()  # speed up logging by flushing the file
    env.reset()

# we flush everything and close the file, it should be ~ 120mb
# NOTICE: we make the log file read-only, this prevent us from erasing all collected data by mistake
# believe me, this is an important issue... can you imagine loosing 2 GB of data? No? We do...
logger.close()

env.close()
