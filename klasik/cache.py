# cache.py

import os
import numpy as np
import hashlib

CACHE_DIR = "cache"
VERSION = "1.0"   # preprocessing pipeline versiyon numarası


def get_file_hash(path):
    """Dosyanın MD5 hash değerini döndürür."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def file_mtime(path):
    """Dosyanın son değiştirilme zamanını döndürür."""
    return os.path.getmtime(path)


def get_cache_path(img_size):
    """IMG_SIZE'a göre cache dosyasının yolu."""
    return os.path.join(CACHE_DIR, f"features_{img_size}.npz")


def save_cache(X, y, img_size, csv_path, script_path):
    """Çıkarılmış veriyi cache dosyasına kaydeder."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    np.savez_compressed(
        get_cache_path(img_size),
        X=X,
        y=y,
        img_size=img_size,
        csv_modified=file_mtime(csv_path),
        script_hash=get_file_hash(script_path),
        version=VERSION
    )

    print(f"[CACHE] Kaydedildi → {get_cache_path(img_size)}")


def load_cache(img_size, csv_path, script_path):
    """
    Cache varsa ve güncelse yükler.
    Değilse None döndürür.
    """
    path = get_cache_path(img_size)
    if not os.path.exists(path):
        return None, None, None

    data = np.load(path, allow_pickle=True)

    # --- Cache Invalidation Kontrolleri ---
    if data["version"] != VERSION:
        print("[CACHE] Versiyon değişti → cache geçersiz.")
        return None, None, None

    if data["img_size"] != img_size:
        print("[CACHE] IMG_SIZE değişti → cache geçersiz.")
        return None, None, None

    if data["csv_modified"] != file_mtime(csv_path):
        print("[CACHE] CSV dosyası değişti → cache geçersiz.")
        return None, None, None

    if data["script_hash"] != get_file_hash(script_path):
        print("[CACHE] preprocessing.py kodu değişti → cache geçersiz.")
        return None, None, None

    print(f"[CACHE] Yüklendi → {path}")
    return data["X"], data["y"], data
