import os
import shutil
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf

def setup_directories(base_dir, splits, classes):
    """Initializes the directory tree for the split dataset."""
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

def main():
    print("[INFO] Initializing Automated Data Engineering Pipeline...")
    
    # Load parameters from config
    cfg = OmegaConf.load("config.yaml")
    
    source_cavity = cfg.paths.raw_cavity
    source_intact = cfg.paths.raw_intact
    base_dir = cfg.paths.processed_data
    
    if not os.path.exists(source_cavity) or not os.path.exists(source_intact):
        raise FileNotFoundError("[ERROR] Source directories not found. Please check config.yaml paths.")

    classes = ['cavity_leak', 'intact_soil']
    splits = ['train', 'val', 'test']
    setup_directories(base_dir, splits, classes)

    # Gather file paths and assign binary labels for stratified splitting
    dataset = []
    
    cavity_files = [f for f in os.listdir(source_cavity) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    for f in cavity_files:
        dataset.append({'path': os.path.join(source_cavity, f), 'label': 0, 'filename': f, 'class': 'cavity_leak'})
        
    intact_files = [f for f in os.listdir(source_intact) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    for f in intact_files:
        dataset.append({'path': os.path.join(source_intact, f), 'label': 1, 'filename': f, 'class': 'intact_soil'})

    if not dataset:
        raise ValueError("[ERROR] No valid images found in the specified source directories.")

    paths = [item['path'] for item in dataset]
    labels = [item['label'] for item in dataset]

    # Stratified Split: Train (70%), Temp (30%)
    val_test_ratio = cfg.data.val_split + cfg.data.test_split
    paths_train, paths_temp, labels_train, labels_temp = train_test_split(
        paths, labels, test_size=val_test_ratio, stratify=labels, random_state=42
    )
    
    # Stratified Split: Temp -> Val (15%), Test (15%)
    test_ratio_adjusted = cfg.data.test_split / val_test_ratio
    paths_val, paths_test, _, _ = train_test_split(
        paths_temp, labels_temp, test_size=test_ratio_adjusted, stratify=labels_temp, random_state=42
    )

    # Route files to designated directories
    def copy_files(file_paths, split_name):
        for src_path in file_paths:
            filename = os.path.basename(src_path)
            # Identify class from source directory
            cls_name = 'cavity_leak' if source_cavity in src_path else 'intact_soil'
            dest_path = os.path.join(base_dir, split_name, cls_name, filename)
            shutil.copy(src_path, dest_path)

    copy_files(paths_train, 'train')
    copy_files(paths_val, 'val')
    copy_files(paths_test, 'test')

    print("[SUCCESS] Data split complete with strict stratification.")
    print(f"Train: {len(paths_train)} | Val: {len(paths_val)} | Test: {len(paths_test)}")

if __name__ == "__main__":
    main()