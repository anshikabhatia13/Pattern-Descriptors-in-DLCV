import os
import cv2
import numpy as np
import pickle
import random
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def extract_lbp_features(image, radius=1, n_points=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate LBP using skimage's local_binary_pattern
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    
    # Calculate the histogram of LBP values
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    
    # Normalize the histogram
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)  # Add a small value to avoid division by zero
    
    return hist

def build_dataset(image_folder, feature_csv, max_images=300):
    features = []
    image_paths = []
    image_count = 0  # Initialize a counter

    for class_folder in os.listdir(image_folder):
        class_path = os.path.join(image_folder, class_folder)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                image = cv2.imread(img_path)
                if image is not None:
                    feature = extract_lbp_features(image)  # Use LBP feature extraction
                    features.append(feature)
                    image_paths.append(img_path)
                    image_count += 1  # Increment the counter
                    if image_count >= max_images:  # Stop when 100 images are processed
                        break
            if image_count >= max_images:  # Stop if the limit is reached
                break

    # Save the features to the CSV file
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
    query_feature = extract_lbp_features(query_image).reshape(1, -1)
    
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
    feature_csv = "lbp_features.csv"
    
    # Step 1: Build dataset from training images and store features in CSV
    # build_dataset(train_folder, feature_csv)
    
    # Step 2: Ask user for the number of images to display
    num_results = int(input("Enter the number of similar images to display: "))
    
    # Step 3: Retrieve similar images for a random query image from test set
    # query_image_path = get_random_query_image(test_folder)
    # query_image_path="dataset/training_set/bus/317.jpg"
    # query_image_path="dataset/training_set/dinosaurs/415.jpg"
    # query_image_path="dataset/training_set/elephants/512.jpg"
    # query_image_path="dataset/training_set/flowers/628.jpg"
    # query_image_path="dataset/training_set/mountains_and_snow/834.jpg"
    query_image_path="noisy_image.jpg"
    
    print(f"Query image: {query_image_path}")
    
    retrieved_images = retrieve_similar_images(query_image_path, feature_csv, num_results)
    
    # Step 4: Display the retrieved images
    display_images(retrieved_images, query_image_path)
