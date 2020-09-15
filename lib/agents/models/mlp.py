from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def mlp(n_obs, n_action, n_hidden_layer=3, n_neuron_per_layer=512,
        activation='relu', loss='mse'):

  model = Sequential()
  
  model.add(Dense(n_neuron_per_layer, input_dim=n_obs, activation=activation))
  model.add(Dropout(0.2)) 
  model.add(BatchNormalization())

  model.add(Dense(2 * n_neuron_per_layer, activation=activation))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())

  model.add(Dense(2 * n_neuron_per_layer, activation=activation))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())

  model.add(Dense(n_neuron_per_layer, input_dim=n_obs, activation=activation))
  model.add(Dropout(0.2)) 
  model.add(BatchNormalization())

  model.add(Dense(n_action, activation='linear'))
  model.compile(loss=loss, optimizer=Adam(lr=0.001), metrics=['accuracy'])
  return model
