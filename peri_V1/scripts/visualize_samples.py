import sys
from pathlib import Path
import torch

# Projenin ana dizinini (peri_V1) Python yoluna ekliyoruz
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peri.training import TrainingConfig, build_dataloaders
from peri.models import PERIModel

# Checkpoint yolu (Senin son çalışmandan)
CHECKPOINT_PATH = r"C:\Users\Gokhan\Desktop\PERI\peri_V1\outputs\runs\paper_faithful\peri\20260406_130007_peri\checkpoints\best.pt"
OUTPUT_DIR = Path("pas_goruntuleri")

def main():
    print("Checkpoint yükleniyor...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    config = TrainingConfig.from_dict(checkpoint["config"])
    
    # PAS Debug ayarlarını zorla açıyoruz
    config.pas_debug = True
    config.pas_debug_dir = OUTPUT_DIR
    config.pas_debug_max_samples = 5  # Sadece ilk 5 örneği kaydet
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"PAS görüntüleri '{OUTPUT_DIR}' klasörüne kaydedilecek...")
    dataloaders = build_dataloaders(config)
    
    # (Sadece dataloader'dan bir batch geçirmek yeterli, 
    # çünkü PAS üretimi dataloader içinde gerçekleşiyor)
    for i, batch in enumerate(dataloaders.test_loader):
        print(f"Batch {i+1} işleniyor...")
        if i >= 0: # İlk batch yetti
            break
            
    dataloaders.close()
    print("\nİşlem tamamlandı! 'pas_goruntuleri' klasörüne bakabilirsin.")

if __name__ == "__main__":
    main()
