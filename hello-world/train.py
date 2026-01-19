"""
Basit bir model eğitme (training) scripti:
- sklearn içinden iris veri setini yükler
- Logistic Regression modeli eğitir
- modeli model.pkl dosyasına kaydeder
"""

# Gerekli kütüphanelerin import edilmesi
from sklearn.datasets import load_iris               # Iris veri setini yüklemek için
from sklearn.linear_model import LogisticRegression  # Logistic Regression modeli
from sklearn.model_selection import train_test_split # Eğitim / test ayırma
import joblib                                        # Modeli dosyaya kaydetmek için
import os                                            # Dosya & klasör işlemleri
import json                                          # Metrikleri JSON formatında yazmak için

def main():
    # Iris veri setini yükle
    iris = load_iris()

    # Özellikler (X) ve hedef değişken (y)
    # X: sepal length, sepal width, petal length, petal width
    # y: çiçek türü (0, 1, 2)
    X, y = iris.data, iris.target

    # Veriyi eğitim (%80) ve test (%20) olarak ayır
    # random_state aynı sonucu tekrar üretmek için
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Logistic Regression modelini oluştur
    # max_iter: maksimum iterasyon sayısı (yakınsama için)
    model = LogisticRegression(max_iter=200)

    # Modeli eğitim verisiyle eğit
    model.fit(X_train, y_train)

    # Modeli kaydetmek için "artifacts" klasörü oluştur (yoksa)
    os.makedirs("artifacts", exist_ok=True)

    # Modelin kaydedileceği dosya yolu
    model_path = os.path.join("artifacts", "model.pkl")

    # Eğitilen modeli diske kaydet
    joblib.dump(model, model_path)

    # Test verisi üzerinde modelin doğruluğunu (accuracy) hesapla
    acc = model.score(X_test, y_test)

    # Metrikleri sözlük (dict) formatında hazırla
    metrics = {
        "accuracy": float(acc)  # JSON uyumlu olması için float'a çeviriyoruz
    }

    # Metrikleri JSON dosyasına kaydet
    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(metrics, f)

    # Konsola bilgi mesajları yazdır
    print(f"Model kaydedildi: {model_path}")
    print(f"Test doğruluğu (accuracy): {acc:.4f}")

# Bu dosya doğrudan çalıştırılırsa main() fonksiyonunu çağır
if __name__ == "__main__":
    main()
