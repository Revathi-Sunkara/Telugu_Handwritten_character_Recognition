import numpy as np
print("Loading imports...")
import joblib
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from model import build_model  # make sure this imports the new transformer version
print("✓ All imports loaded")

# ---------------- Load Processed Data ---------------- #
print("Loading processed data...")
data = np.load("processed_data.npz")
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']
print(f"✓ Data loaded: {x_train.shape[0]} training samples, {x_test.shape[0]} test samples")

# ---------------- Build Model ---------------- #
print("Building model...")
input_shape = x_train.shape[1:]
num_classes = y_train.shape[1]

model = build_model(input_shape, num_classes)
print("✓ Model built")
print("\nModel Summary:")
model.summary()

# ---------------- Compile Model ---------------- #
print("Compiling model...")
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("✓ Model compiled")

# ---------------- Callbacks ---------------- #
checkpoint = ModelCheckpoint(
    'telugu_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)

# ---------------- Train Model ---------------- #
print("Starting training...")
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# ---------------- Save Training History ---------------- #
print("Saving training history...")
joblib.dump(history.history, 'training_history.pkl')
print("✓ Training complete! Files saved: telugu_model.h5 and training_history.pkl")