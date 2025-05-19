import cv2
import numpy as np
import pandas as pd
import os
import random
from skimage.feature import local_binary_pattern, hog
from skimage import color
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.neighbors import NearestNeighbors

def extract_lbp_hog_features(image, lbp_radius_list=[1, 2, 3], n_points=8, hog_orientations=9):
    # Extract LBP features from all 3 channels (BGR)
    lbp_features = []
    for radius in lbp_radius_list:
        for channel in cv2.split(image):  # Split image into 3 channels (B, G, R)
            lbp = local_binary_pattern(channel, n_points, radius, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype('float')
            hist /= (hist.sum() + 1e-6)  # Normalize
            lbp_features.append(hist)

    # Concatenate LBP features for all scales and channels
    lbp_features = np.concatenate(lbp_features)
    
    # Convert the image to grayscale for HOG feature extraction
    gray_image = color.rgb2gray(image)
    
    # Extract HOG features
    hog_features = hog(gray_image, orientations=hog_orientations, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys')
    
    # Combine LBP and HOG features
    combined_features = np.concatenate([lbp_features, hog_features])
    
    return combined_features

# Modify build_dataset to use the new feature extraction method
def build_dataset_with_lbp_hog(image_folder, feature_csv, max_images=300):
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
                    feature = extract_lbp_hog_features(image)  # Use combined LBP + HOG feature extraction
                    features.append(feature)
                    image_paths.append(img_path)
                    image_count += 1  # Increment the counter
                    if image_count >= max_images:  # Stop when max images are processed
                        break
                    if image_count % 10 == 0:
                        print(f"Processed {image_count} images")
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
    query_feature = extract_lbp_hog_features(query_image).reshape(1, -1)
    
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
    feature_csv = "3channel_lbp_features.csv"
    
    # Step 1: Build dataset from training images and store features in CSV
    # build_dataset_with_lbp_hog(train_folder, feature_csv)
    
    # Step 2: Ask user for the number of images to display
    num_results = int(input("Enter the number of similar images to display: "))
    
    # Step 3: Retrieve similar images for a random query image from test set
    # query_image_path = get_random_query_image(test_folder)
    query_image_path="noisy_image.jpg"
    # query_image_path="dataset/training_set/dinosaurs/415.jpg"
    # query_image_path="dataset/training_set/elephants/512.jpg"
    # query_image_path="dataset/training_set/flowers/628.jpg"
    # query_image_path="dataset/training_set/mountains_and_snow/834.jpg"
    # query_image_path="c:/Users/anshi/Downloads/f.jpg"
    
    print(f"Query image: {query_image_path}")
    
    retrieved_images = retrieve_similar_images(query_image_path, feature_csv, num_results)
    
    # Step 4: Display the retrieved images
    display_images(retrieved_images, query_image_path)
