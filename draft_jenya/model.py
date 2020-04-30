from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Dropout
from feature_selection_ratings.features_selection_general_rating import X_train_selected_features, X_test_selected_features, y_train_selected_features, y_test_selected_features

EMBEDDING_DIM = 100
num_labels = 3
input_size = 5

model = Sequential()
model.add(Dense(512, input_shape=(input_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

num_epochs = 10
batch_size = 128
history = model.fit(X_train_selected_features, y_train_selected_features,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=2,
                    validation_split=0.2)

score, acc = model.evaluate(X_test_selected_features, y_test_selected_features,
                       batch_size=batch_size, verbose=2)

print('Test accuracy:', acc)