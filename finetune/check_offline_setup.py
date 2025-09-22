
import os
import sys
from pathlib import Path


def check_file_exists(path: str, description: str) -> bool:
    """Check if a file or directory exists"""
    if Path(path).exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: {path} (NOT FOUND)")
        return False


def check_offline_setup():
    """Check all requirements for offline MIL training"""
    print("🔍 Checking Offline MIL Training Setup")
    print("=" * 50)
    
    all_ready = True
    
    # Check data directory
    data_path = "../anne_data/512px_uni-vit-l-16_0.5mpp_0xdown_normal"
    if check_file_exists(data_path, "Data directory"):
        # Check for H5 files
        h5_files = list(Path(data_path).rglob("*.h5"))
        if h5_files:
            print(f"  📊 Found {len(h5_files)} H5 files")
            
            # Check filename patterns for label extraction
            bcc_files = [f for f in h5_files if any(x in f.stem.lower() for x in ['sbbc', 'ibcc'])]
            scc_files = [f for f in h5_files if 'pek' in f.stem.lower()]
            unknown_files = [f for f in h5_files if f not in bcc_files and f not in scc_files]
            
            print(f"  🔬 BCC files (sbbc/ibcc): {len(bcc_files)}")
            print(f"  🔬 SCC files (pek): {len(scc_files)}")
            if unknown_files:
                print(f"  ⚠️  Unknown pattern files: {len(unknown_files)}")
                print(f"     First few: {[f.stem for f in unknown_files[:3]]}")
            
            # Check a sample file structure
            try:
                import h5py
                sample_file = h5_files[0]
                with h5py.File(sample_file, 'r') as f:
                    if 'features' in f:
                        features_shape = f['features'].shape
                        print(f"  📋 Sample features shape: {features_shape}")
                        
                        # Check if features are pickled
                        features = f['features'][()]
                        if hasattr(features, 'tobytes'):
                            print(f"  🥒 Features appear to be pickled (size: {len(features.tobytes())} bytes)")
                        else:
                            print(f"  📊 Features are direct arrays")
                    else:
                        print("  ⚠️  No 'features' dataset in H5 file")
                        all_ready = False
                        
                    if 'coordinates' in f:
                        coords_shape = f['coordinates'].shape
                        print(f"  📍 Sample coordinates shape: {coords_shape}")
                    else:
                        print("  ⚠️  No 'coordinates' dataset in H5 file")
                        
            except ImportError:
                print("  ⚠️  h5py not installed - cannot verify H5 structure")
                all_ready = False
            except Exception as e:
                print(f"  ⚠️  Error reading H5 file: {e}")
        else:
            print("  ❌ No H5 files found in data directory")
            all_ready = False
    else:
        all_ready = False
    
    print()
    
    # Check model files - using local paths for offline training
    model_files = [
        ("../microsoft_biogpt-large", "BioGPT model directory"),
        ("./histogpt-l-6k-pruned.pt", "HistoGPT pretrained weights")
    ]
    
    for path, desc in model_files:
        if not check_file_exists(path, desc):
            all_ready = False
    
    print()
    
    # Check if BioGPT has required files
    biogpt_path = Path("../microsoft_biogpt-large")
    if biogpt_path.exists():
        required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
        for file in required_files:
            file_path = biogpt_path / file
            if file_path.exists():
                print(f"✅ BioGPT {file}: Found")
            else:
                print(f"❌ BioGPT {file}: Missing")
                all_ready = False
    
    print()
    print("🌐 Network Settings")
    print("-" * 20)
    
    # Set offline environment variables
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    print("✅ TRANSFORMERS_OFFLINE=1")
    print("✅ HF_DATASETS_OFFLINE=1")
    
    print()
    print("📦 Import Tests")
    print("-" * 15)
    
    # Test critical imports for MIL training
    imports_to_test = [
        ("torch", "PyTorch"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("transformers", "Transformers"),
        ("h5py", "H5Py"),
        ("sklearn", "Scikit-learn"),
        ("numpy", "NumPy"),
    ]
    
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"✅ {name}: Available")
        except ImportError:
            print(f"❌ {name}: Missing")
            all_ready = False
    
    print()
    print("=" * 50)
    
    if all_ready:
        print("🎉 READY FOR OFFLINE MIL TRAINING!")
        print()
        print("Start training with:")
        print("python train_mil_classifier.py \\")
        print("  --feat_path ../anne_data/512px_uni-vit-l-16_0.5mpp_0xdown_normal \\")
        print("  --histogpt_weights ./histogpt-l-6k-pruned.pt \\")
        print("  --k_folds 5 --max_epochs 50")
        return True
    else:
        print("❌ NOT READY - Fix missing components above")
        print()
        print("To fix missing models:")
        print("1. Download microsoft/biogpt-large to ../microsoft_biogpt-large/")
        print("2. Place histogpt-l-6k-pruned.pt in current directory")
        print("3. Install missing packages: pip install torch pytorch-lightning transformers h5py scikit-learn numpy")
        return False


if __name__ == "__main__":
    ready = check_offline_setup()
    sys.exit(0 if ready else 1)