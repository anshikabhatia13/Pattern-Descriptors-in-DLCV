import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

# Precompute uniform patterns lookup table
def precompute_uniform_patterns():
    uniform_patterns = []
    for i in range(256):
        binary_string = format(i, '08b')
        transitions = sum((binary_string[i] != binary_string[i+1]) for i in range(7)) + (binary_string[0] != binary_string[-1])
        if transitions <= 2:
            uniform_patterns.append(i)
    
    uniform_pattern_to_index = {p: idx for idx, p in enumerate(uniform_patterns)}
    return uniform_pattern_to_index

uniform_pattern_to_index = precompute_uniform_patterns()

def get_uniform_pattern_index(pattern):
    """ Map the pattern to the precomputed uniform pattern index. """
    return uniform_pattern_to_index.get(pattern, len(uniform_pattern_to_index))

def extract_uniform_lbp_features(image, radius=1, n_points=4):
    """ 
    Optimized version of LBP feature extraction. 
    - Reduced number of neighbors from 8 to 4 for faster processing. 
    - Downsamples image for faster computation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Downsample image to reduce processing
    gray = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))
    height, width = gray.shape
    
    lbp = np.zeros((height, width), dtype=np.int32)
    
    # Create histogram with bins for uniform patterns and one for non-uniform patterns
    uniform_hist = np.zeros(len(uniform_pattern_to_index) + 1, dtype=int)  # One additional bin for non-uniform patterns
    
    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            neighbors = []
            for k in range(n_points):
                angle = 2 * np.pi * k / n_points
                x = int(i + radius * np.sin(angle))
                y = int(j + radius * np.cos(angle))
                neighbors.append(gray[x, y])
            
            center = gray[i, j]
            binary_pattern = 0
            
            for k, neighbor_value in enumerate(neighbors):
                binary_pattern |= (1 << k) if neighbor_value >= center else 0
            
            # Map the binary pattern to the correct uniform bin
            uniform_bin = get_uniform_pattern_index(binary_pattern)
            uniform_hist[uniform_bin] += 1
    
    # Normalize the histogram
    uniform_hist = uniform_hist.astype('float')
    uniform_hist /= (uniform_hist.sum() + 1e-6)
    return uniform_hist

# Functions to build dataset, retrieve similar images, and display results (unchanged from the original code)

def build_dataset(image_folder, feature_csv, max_images=500):
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
                    feature = extract_uniform_lbp_features(image)
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
    query_feature = extract_uniform_lbp_features(query_image).reshape(1, -1)
   
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
    feature_csv = "proposed_optimal_u2.csv"
   
    # Step 1: Build dataset from training images and store features in CSV
    # build_dataset(train_folder, feature_csv)
   
    # Step 2: Ask user for the number of images to display
    num_results = int(input("Enter the number of similar images to display: "))
   
    # Step 3: Retrieve similar images for a random query image from test set
    # query_image_path="dataset/training_set/bus/317.jpg"
    # query_image_path="dataset/training_set/dinosaurs/415.jpg"
    # query_image_path="dataset/training_set/elephants/512.jpg"
    # query_image_path="dataset/training_set/flowers/628.jpg"
    # query_image_path="dataset/training_set/mountains_and_snow/834.jpg"
    query_image_path="noisy_image.jpg"
    # query_image_path = get_random_query_image(test_folder)
    # query_image_path = "235a3354df30adb58a04ead70b3c357b.jpg"
    print(f"Query image: {query_image_path}")
   
    retrieved_images = retrieve_similar_images(query_image_path, feature_csv, num_results)
   
    # Step 4: Display the retrieved images
    display_images(retrieved_images, query_image_path)
