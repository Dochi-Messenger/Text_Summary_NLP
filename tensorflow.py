from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

inputs = tf.keras.Input(shape=(32,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)

class CustomLayer(layers.Layer):
    def __init__(self, hidden_dimension, hidden_dimension2, output_dimension):
        self.hidden_dimension = hidden_dimension
        self.hidden_dimension2 = hidden_dimension2
        self.output_dimension = output_dimension
        super(CustomLayer, self).__init__()
        
    def build(self, input_shape):
        self.dense_layer1 = layers.Dense(self.hidden_dimension, activation= 'relu')
        self.dense_layer2 = layers.Dense(self.hidden_dimension2, activation= 'relu')
        self.dense_layer3 = layers.Dense(self.output_dimension, activation= 'softmax')
        
    def call(self, inputs):
        x = self.dense_layer1(inputs)
        x = self.dense_layer2(x)
        
        return self.dense_layer3(x) 