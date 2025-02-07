from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import json

path = 'C:/Users/ujjwa/OneDrive/Documents/E yantra sample/annotations_trainval2017/annotations/instances_val2017.json'
file = open(path)
anns = json.load(file)
print(anns['images'])








# Load the ResNet model, excluding the final dense layer
#base_model = ResNet50(weights='imagenet', include_top=False)

# Add a global average pooling layer and a dense layer with 2 neurons (for binary classification)
#x = base_model.output
#x = GlobalAveragePooling2D()(x)   # Reduces dimensions before the final layer
#predictions = Dense(2, activation='softmax')(x)

# Finalize the model
#model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base layers to retain pre-trained features
#for layer in base_model.layers:
#    layer.trainable = False
