#!/usr/bin/env python3
"""
Kullanım:
    python run_model.py --input "[5.1, 3.5, 1.4, 0.2]"

Bu script:
- Daha önce eğitilmiş modeli (model.pkl) yükler
- Komut satırından verilen özelliklerle tahmin yapar
- Tahmin sonucunu hem sınıf id hem de çiçek ismi olarak döner
"""

import argparse
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.datasets import load_iris  # Çiçek isimlerini (label mapping) almak için

# Eğitilmiş modelin dosya yolu
MODEL_PATH = Path("artifacts/model.pkl")

def load_model():
    """
    Model dosyasını diskten yükler.
    Model bulunamazsa hata fırlatır.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # Model binary formatta yüklenir
    return joblib.load(MODEL_PATH)

def main():
    # Komut satırı argümanlarını tanımlayan parser
    parser = argparse.ArgumentParser()

    # --input parametresi: JSON formatında feature listesi beklenir
    parser.add_argument(
        "--input",
        required=True,
        help="JSON formatında özellik listesi. Örnek: \"[5.1,3.5,1.4,0.2]\""
    )

    # Argümanları parse et
    args = parser.parse_args()

    # Girilen input'u JSON olarak parse etmeye çalış
    try:
        features = json.loads(args.input)
    except json.JSONDecodeError:
        # JSON formatı bozuksa hata ver
        raise ValueError(
            "Geçersiz input formatı. JSON liste kullanın, örn: --input \"[5.1,3.5,1.4,0.2]\""
        )

    # Feature listesini numpy array'e çevir
    # reshape(1, -1): tek bir örnek olacak şekilde boyutlandır
    X = np.array(features).reshape(1, -1)

    # Eğitilmiş modeli diskten yükle
    model = load_model()

    # Model ile tahmin yap
    # Çıktı: numpy array içinde sınıf id (0, 1, 2)
    pred = model.predict(X)

    # Tahmin edilen sınıf id (int)
    pred_id = int(pred[0])

    # === LABEL → ÇİÇEK ADI EŞLEMESİ ===
    iris = load_iris()

    # Sınıf isimlerini al (['setosa', 'versicolor', 'virginica'])
    class_names = iris.target_names.tolist()

    # Tahmin edilen id'yi çiçek ismine çevir
    pred_name = class_names[pred_id]

    # Sonucu JSON formatında ekrana yazdır
    print(json.dumps({
        "prediction_id": pred_id,
        "prediction_name": pred_name
    }))

# Script doğrudan çalıştırılırsa main() fonksiyonu çağrılır
if __name__ == "__main__":
    main()
