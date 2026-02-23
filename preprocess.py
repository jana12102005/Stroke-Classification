import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

input_folders = {
    "Normal": r"A:\bloodclot\Brain_Data_Organised\Normal",
    "Stroke": r"A:\bloodclot\Brain_Data_Organised\Stroke"
}
output_folders = {
    "Normal": r"A:\bloodclot\Processed_Data\Normal",
    "Stroke": r"A:\bloodclot\Processed_Data\Stroke"
}

for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

for category, folder in input_folders.items():
    if not os.path.exists(folder):
        print(f"❌ Error: Folder not found -> {folder}")
        exit(1)  

def process_image(args):
    img_path, category = args
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        if img is None:
            return None  

        img = cv2.resize(img, (224, 224))
        img = img / 255.0 

        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folders[category], filename)
        cv2.imwrite(save_path, (img * 255).astype(np.uint8)) 

        return save_path  

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

image_paths = []
for category, folder in input_folders.items():
    if os.path.exists(folder): 
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            image_paths.append((img_path, category))

if __name__ == "__main__":
    print(f"🚀 Processing {len(image_paths)} images using {cpu_count()} CPU cores...")
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(process_image, image_paths), total=len(image_paths)))

    print("✅ Preprocessing complete!")
