from tensorflow.keras.models import load_model

# Try loading the model with compile=False to avoid deserialization issues
model = load_model("keras_model.h5", compile=False)

# Print model architecture
model.summary()
