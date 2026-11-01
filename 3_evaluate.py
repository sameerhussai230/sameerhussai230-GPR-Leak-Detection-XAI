import os
import torch
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from omegaconf import OmegaConf
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Import model architecture dynamically
import importlib
train_module = importlib.import_module("2_train")
GPRClassifier = train_module.GPRClassifier

def main():
    print("[INFO] Executing Model Evaluation and Diagnostics Pipeline...")
    cfg = OmegaConf.load("config.yaml")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(cfg.paths.checkpoint_dir, 'best_gpr_model.ckpt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[ERROR] Checkpoint not found at {checkpoint_path}")
    
    print("[INFO] Restoring model weights from checkpoint...")
    lit_model = GPRClassifier(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    lit_model.load_state_dict(checkpoint['state_dict'])
    lit_model.to(device)

    # Initialize dedicated output directory for evaluation sets
    output_dir = "evaluation_diagnostics"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Diagnostic outputs will be saved to: ./{output_dir}/")

    # Gather Unseen Test Images from all classes
    test_dir = os.path.join(cfg.paths.processed_data, 'test')
    classes = ['cavity_leak', 'intact_soil']
    
    all_test_files = []
    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        if os.path.exists(cls_dir):
            files = [os.path.join(cls, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            all_test_files.extend(files)

    if not all_test_files:
         raise FileNotFoundError(f"[ERROR] No test images found in {test_dir}")
        
    # Standardize to 5 samples for comprehensive reporting
    num_samples = min(5, len(all_test_files))
    random.seed(42)  # Maintain reproducibility for academic reporting
    selected_files = random.sample(all_test_files, num_samples)

    transform = transforms.Compose([
        transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for idx, img_rel_path in enumerate(selected_files):
        true_class = os.path.dirname(img_rel_path)
        base_filename = os.path.splitext(os.path.basename(img_rel_path))[0]
        img_path = os.path.join(test_dir, img_rel_path)
        
        # Format the class name for professional display in plot titles
        display_class = true_class.replace('_', ' ').title()
        if display_class == "Cavity Leak": 
            display_class = "Cavity / Leak"
            
        print(f"\n[INFO] Processing Inference Set {idx+1}/{num_samples} | Class: {true_class}")

        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
        rgb_img_resized = cv2.resize(rgb_img, (cfg.data.img_size, cfg.data.img_size))
        rgb_img_float = np.float32(rgb_img_resized) / 255
        
        pil_img = Image.fromarray(rgb_img_resized)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # ==========================================
        # 1. Epistemic Uncertainty (Monte Carlo Dropout)
        # ==========================================
        lit_model.eval()
        
        # Robust check: explicitly activate nn.Dropout modules for variance approximation
        dropout_found = False
        for m in lit_model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
                dropout_found = True
                
        if not dropout_found:
            print("[WARNING] No nn.Dropout layers found. Variance will be 0.0%.")

        predictions = []
        with torch.no_grad():
            for _ in range(cfg.evaluation.mc_iterations):
                output = lit_model(input_tensor)
                prob = torch.nn.functional.softmax(output, dim=1)[0][0].item() # Probability of Cavity/Leak
                predictions.append(prob)

        mean_prob = np.mean(predictions)
        std_dev = np.std(predictions)
        print(f"[METRICS] Expected Confidence: {mean_prob*100:.2f}% | Variance (Std Dev): ±{std_dev*100:.2f}%")

        # ==========================================
        # 2. Explainable AI (Grad-CAM)
        # ==========================================
        lit_model.eval() # Disable dropout for deterministic spatial mapping
        
        # Target final residual block for timm resnet
        target_layers = [lit_model.model.layer4[-1]]
        
        cam = GradCAM(model=lit_model.model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

        # ==========================================
        # Save Standalone Original vs Heatmap Comparison
        # ==========================================
        fig_xai, axes_xai = plt.subplots(1, 2, figsize=(10, 5))
        
        axes_xai[0].imshow(rgb_img_resized)
        axes_xai[0].set_title(f"Original B-Scan (GT: {display_class})", fontsize=14)
        axes_xai[0].axis('off')
        
        axes_xai[1].imshow(visualization)
        axes_xai[1].set_title("Grad-CAM Spatial Activation", fontsize=14)
        axes_xai[1].axis('off')
        
        plt.tight_layout()
        xai_filename = os.path.join(output_dir, f'Set{idx+1}_Isolated_XAI_{true_class}_{base_filename}.png')
        fig_xai.savefig(xai_filename, bbox_inches='tight', dpi=300)
        plt.close(fig_xai) 

        # ==========================================
        # 3. Render and Save Complete Output Profile
        # ==========================================
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title(f"Original GPR B-Scan\n(Ground Truth: {display_class})")
        plt.imshow(rgb_img_resized)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Grad-CAM Spatial Activation")
        plt.imshow(visualization)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title(f"Epistemic Uncertainty Distribution (N={cfg.evaluation.mc_iterations})")
        
        min_val, max_val = min(predictions), max(predictions)
        margin = max(0.02, (max_val - min_val) * 0.5)
        bins = np.linspace(max(0.0, min_val - margin), min(1.0, max_val + margin), 15)

        plt.hist(predictions, bins=bins, color='royalblue', alpha=0.75, edgecolor='black')
        plt.axvline(mean_prob, color='crimson', linestyle='dashed', linewidth=2, 
                    label=f'Mean Pr(Cavity): {mean_prob*100:.1f}%\nUncertainty: ±{std_dev*100:.1f}%')
        plt.xlabel("Predictive Confidence (Class: Cavity/Leak)")
        plt.ylabel("Frequency (MC Iterations)")
        plt.legend(loc="upper left")
        
        plt.suptitle(f"GPR Inference Diagnostics {idx+1}: Spatial Activation & Epistemic Uncertainty", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_filename = os.path.join(output_dir, f'Set{idx+1}_Diagnostic_Profile_{true_class}_{base_filename}.png')
        plt.savefig(output_filename, bbox_inches='tight', dpi=300)
        plt.close()
        
    print(f"\n[SUCCESS] Pipeline completed successfully. 5 evaluation sets saved in './{output_dir}/'.")

if __name__ == "__main__":
    main()