import os
import cv2
import numpy as np
import pickle
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def extract_lmebp_features_median(image, radius=1, n_points=8, bit_planes=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    lmebp = np.zeros((height, width), dtype=np.int32)
    
    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            neighbors = []
            for k in range(n_points):
                angle = 2 * np.pi * k / n_points
                x = int(i + radius * np.sin(angle))
                y = int(j + radius * np.cos(angle))
                neighbors.append(gray[x, y])
            
            # Calculate the median of the neighbors
            local_median = np.median(neighbors)
            center = gray[i, j]
            
            code = 0
            for k, neighbor_value in enumerate(neighbors):
                if neighbor_value >= local_median:
                    code |= (1 << k)
            
            # Keep only the 4 MSB planes by right-shifting the lower 4 bits
            lmebp[i, j] = code >> (n_points - bit_planes)
    
    # Histogram calculation for the 4 MSB bit planes
    max_val = 1 << bit_planes  # 2^bit_planes
    (hist, _) = np.histogram(lmebp.ravel(), bins=np.arange(0, max_val + 1), range=(0, max_val))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
    return hist


def build_dataset(image_folder, feature_csv, max_images=150):
    features = []
    image_paths = []
    image_count = 0

    for class_folder in os.listdir(image_folder):
        class_path = os.path.join(image_folder, class_folder)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                image = cv2.imread(img_path)
                if image is not None:
                    feature = extract_lmebp_features_median(image)
                    features.append(feature)
                    image_paths.append(img_path)
                    image_count += 1
                    if image_count >= max_images:
                        break
                    if image_count % 10 == 0:
                        print(f"Processed {image_count} images.")
            if image_count >= max_images:
                break

    df = pd.DataFrame(features)
    df.insert(0, "image_path", image_paths)
    df.to_csv(feature_csv, index=False)
    print(f"Processed {image_count} images. Feature descriptor saved successfully in CSV!")

def load_features(feature_csv):
    df = pd.read_csv(feature_csv)
    image_paths = df["image_path"].values
    features = df.drop(columns=["image_path"]).values
    return features, image_paths

def retrieve_similar_images(query_image_path, feature_csv, num_results=5):
    features, image_paths = load_features(feature_csv)
    query_image = cv2.imread(query_image_path)
    query_feature = extract_lmebp_features_median(query_image).reshape(1, -1)
   
    knn = NearestNeighbors(n_neighbors=num_results, metric='euclidean')
    knn.fit(features)
    distances, indices = knn.kneighbors(query_feature)
   
    retrieved_images = [image_paths[i] for i in indices[0]]
    return retrieved_images

def display_images(image_paths, query_image_path):
    query_image = cv2.imread(query_image_path)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
   
    fig, axes = plt.subplots(1, len(image_paths) + 1, figsize=(15, 5))
   
    axes[0].imshow(query_image)
    axes[0].set_title('Query Image')
    axes[0].axis('off')
   
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i+1].imshow(img)
        axes[i+1].set_title(f'Result {i+1}')
        axes[i+1].axis('off')
   
    plt.show()

def get_random_query_image(test_folder):
    class_folders = [f for f in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, f))]
    if not class_folders:
        raise ValueError("Test folder is empty or incorrectly structured.")
   
    random_class = random.choice(class_folders)
    class_path = os.path.join(test_folder, random_class)
   
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        raise ValueError(f"No images found in {class_path}")
   
    return os.path.join(class_path, random.choice(images))

if __name__ == "__main__":
    train_folder = "dataset/test_set"  # Training set folder path
    test_folder = "dataset/test_set"  # Test set folder path
    feature_csv = "lmebp_median_features.csv"
   
    # Step 1: Build dataset from training images and store features in CSV
    # build_dataset(train_folder, feature_csv)
   
    # Step 2: Ask user for the number of images to display
    num_results = int(input("Enter the number of similar images to display: "))
   
    # Step 3: Retrieve similar images for a random query image from test set
    # query_image_path="dataset/training_set/bus/317.jpg"
    # query_image_path="dataset/test_set/dinosaurs/404.jpg"
    # query_image_path="dataset/training_set/elephants/512.jpg"
    # query_image_path="dataset/training_set/flowers/628.jpg"
    query_image_path="noisy_image.jpg"
    # query_image_path = get_random_query_image(test_folder)
    print(f"Query image: {query_image_path}")
   
    retrieved_images = retrieve_similar_images(query_image_path, feature_csv, num_results)
   
    # Step 4: Display the retrieved images
    display_images(retrieved_images, query_image_path)
