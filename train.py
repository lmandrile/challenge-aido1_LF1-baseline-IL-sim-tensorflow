import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from _loggers import Reader
from model import TensorflowModel

# configuration zone
BATCH_SIZE = 78
EPOCHS = 20
TOP_CROP_VALUE = 17
loss_table = []
# here we assume the observations have been resized to 60x80
OBSERVATIONS_SHAPE = (None, 60-TOP_CROP_VALUE, 80, 3)
#OBSERVATIONS_SHAPE = (None, 60, 80, 3)
ACTIONS_SHAPE = (None, 1)
SEED = 1234
STORAGE_LOCATION = "trained_models/behavioral_cloning"

reader = Reader('train.log')

observations, actions = reader.read()
actions = np.array(actions)
actions = actions[:,[1]]
print(actions.shape)
observations = np.array(observations)

model = TensorflowModel(
    observation_shape=OBSERVATIONS_SHAPE,  # from the logs we've got
    action_shape=ACTIONS_SHAPE,  # same
    graph_location=STORAGE_LOCATION,  # where do we want to store our trained models
    seed=SEED  # to seed all random operations in the model (e.g., dropout)
)

min_loss = 1000

# we trained for EPOCHS epochs
epochs_bar = tqdm(range(EPOCHS))
for i in epochs_bar:
    # we defined the batch size, this can be adjusted according to your computing resources...
    loss = None
    for batch in range(0, len(observations), BATCH_SIZE):
        loss = model.train(
            observations=observations[batch:batch + BATCH_SIZE],
            actions=actions[batch:batch + BATCH_SIZE]
        )

    epochs_bar.set_postfix({'loss': loss})

    loss_table.append(loss)
    plt.plot(loss_table)
    
    # every 10 epochs, we store the model we have
    # but I'm sure that you're smarter than that, what if this model is worse than the one we had before
    if i % 10 == 0 and loss < min_loss:
        model.commit()
        epochs_bar.set_description('Model saved...')
    else:
        epochs_bar.set_description('')

# the loss at this point should be on the order of 2e-2, which is far for great, right?

# we release the resources...
model.close()
reader.close()

