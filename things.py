
import tensorflow as tf #type:ignore
import time

print("TensorFlow:", tf.__version__)

# Download 60,000 colorful images
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = y_train.flatten()
y_test = y_test.flatten()

# Make extra training variations
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

# Build a convolutional neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    augmentation,

    tf.keras.layers.Conv2D(
        64, 3, padding="same", use_bias=False
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),

    tf.keras.layers.Conv2D(
        64, 3, padding="same", use_bias=False
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(
        128, 3, padding="same", use_bias=False
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),

    tf.keras.layers.Conv2D(
        128, 3, padding="same", use_bias=False
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(
        256, 3, padding="same", use_bias=False
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

start = time.time()

model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
)

print("\nTesting the boss...")
loss, accuracy = model.evaluate(
    x_test, y_test, verbose=0
)

print(f"\nTest accuracy: {accuracy * 100:.2f}%")
print(f"Elapsed time: {time.time() - start:.1f}s")

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

predictions = model.predict(x_test[:20], verbose=0)

for i, prediction in enumerate(predictions):
    guess = prediction.argmax()
    actual = y_test[i]

    print(
        f"Image {i}: {classes[guess]} "
        f"(actual: {classes[actual]}, "
        f"confidence: {prediction[guess] * 100:.1f}%)"
    )

model.save("h.keras")
