# config.py
IMAGE_SIZE = 300  # B3 için altın kural
NUM_CLASSES = 8
CHANNELS = 3

# B3 daha fazla VRAM kullanır. Eğer hata alırsan 32'ye çekebilirsin.
BATCH_SIZE = 32

EPOCHS_FROZEN = 8    # B3'ün head kısmını oturtmak için biraz daha süre tanıyalım
EPOCHS_FINETUNE = 15
EPOCHS_WARMUP = 7    # Çok düşük LR ile son bir "cilalama" aşaması

LR_FROZEN = 1e-3
LR_FINETUNE = 2e-5   # B3 hassastır, fine-tune'da LR'yi biraz daha kıstık (B0'da 1e-4 idi)
LR_WARMUP = 5e-6     # Final aşaması için çok çok düşük LR

# B3'te daha fazla blok var. 80 katman yaklaşık son 2-3 bloğu açar.
# Daha derin bir fine-tune için bu sayıyı 120-150 yapabilirsin.
LAYERS_TO_FINE_TUNE = 100
