import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from pycocotools.coco import COCO
import os
import pickle


os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# Задані шляхи до даних
train_data_dir = '123//train'
valid_data_dir = '123//valid'
test_data_dir = '123//test'

# Параметри моделі
num_classes = 1
input_shape = (224, 224, 3)

# Побудова моделі
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(num_classes * 4, activation='linear')(x)  # Вихідний шар з 4 * num_classes нейронами для передбачення координат
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Компіляція моделі
#model.compile(optimizer='adam', loss='mse')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Зчитування даних та анотацій
def load_data(data_dir):
    coco = COCO(f'{data_dir}/_annotations.coco.json')
    image_ids = coco.getImgIds()
    image_paths = []
    labels = []
    for id in image_ids:
        file_name = coco.loadImgs(ids=[id])[0]["file_name"]
        image_path = f'{data_dir}/{file_name}'
        if os.path.exists(image_path):
            annotations = coco.loadAnns(coco.getAnnIds(imgIds=[id]))
            if annotations:
                labels.append(annotations[0]['bbox'])
                image_paths.append(image_path)
            else:
                print(f"No annotations found for image: {image_path}")
        else:
            print(f"File not found: {image_path}")

    print(len(image_paths), len(labels))
    return image_paths, labels

# Передбачення функції для отримання координат
def predict_bounding_box(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=input_shape[:2])
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predictions = tf.reshape(predictions, [-1, num_classes, 4])
    predictions = predictions.numpy()  # Екстракція значень тензорів до масиву NumPy
    x, y, w, h = predictions[0, :, 0], predictions[0, :, 1], predictions[0, :, 2], predictions[0, :, 3]
    return x, y, w, h

# Завантаження даних
train_image_paths, train_labels = load_data(train_data_dir)
valid_image_paths, valid_labels = load_data(valid_data_dir)

# Створення tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_image_paths, valid_labels))

# Підготовка даних
def preprocess_image(image_path, labels):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # або tf.image.decode_png для зображень у форматі PNG
    image = tf.image.resize(image, input_shape[:2])
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, labels

train_dataset = train_dataset.map(preprocess_image).batch(32)
valid_dataset = valid_dataset.map(preprocess_image).batch(32)

# Навчання моделі
model.fit(train_dataset, validation_data=valid_dataset, epochs=1, batch_size=1)

# Збереження моделі
print("Збереження моделі")
with open('find_objects.pkl', 'wb') as f:
    pickle.dump(model, f)
# model.save('object_detection_model', save_format='tf')
# model.save('classificatiion.h5', include_optimizer=True)
