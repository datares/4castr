from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


def mlp(n_obs, n_action, n_hidden_layer=3, n_neuron_per_layer=512,
        activation='relu', loss='mse'):
    """
    Constructs a simple fully-connected neural network with Keras with 
        3 hidden layers. Outputs a vector of length n_action. 

    Args:
        n_obs (int): State vector size.
        n_action (int): Action vector size.
        n_hidden_layer (int): Number hidden layers. Defaults to 3.
        n_neuron_per_layer (int): Number of neurons per layer. Defaults to 512.
        activation (string): Activation function to use. Defaults to 'relu'.
        loss (string): Loss to use. Defaults to 'mse'.
    """
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
