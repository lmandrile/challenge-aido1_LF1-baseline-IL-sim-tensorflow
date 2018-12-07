import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
from _loggers import Reader
from torchModel import TorchModel

# configuration zone
BATCH_SIZE = 78
EPOCHS = 20
TOP_CROP_VALUE = 17
loss_table = []
# here we assume the observations have been resized to 60x80
OBSERVATIONS_SHAPE = (None, 60, 80, 3)
ACTIONS_SHAPE = (None, 2)
SEED = 1234
STORAGE_LOCATION = "Trained Models/Pytorch_Model"

reader = Reader('Train.log')

model = TorchModel(
    observation_shape=OBSERVATIONS_SHAPE,  # from the logs we've got
    action_shape=ACTIONS_SHAPE,  # same
    save_path=STORAGE_LOCATION  # where do we want to store our trained models
)

observations, actions = reader.read()
actions = np.array(actions)
observations = np.array(observations)
print("Obs shape:" + observations)
print("Act shape:" + actions)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

min_loss = 1000
epochs_bar = tqdm(range(EPOCHS))

for i in epochs_bar:
    # we defined the batch size, this can be adjusted according to your computing resources...
    
    running_loss = 0.0
    for batch in range(0, len(observations), BATCH_SIZE):
        observations=observations[batch:batch + BATCH_SIZE]
        actions=actions[batch:batch + BATCH_SIZE]

        tensor = torch.tensor([BATCH_SIZE, 3, observations])

        optimizer.zero_grad()
        outputs = net(observations)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()

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

