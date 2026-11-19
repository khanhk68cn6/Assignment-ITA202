"""Giai đoạn 1: EDA -> Làm sạch -> Tiền xử lý & chia dữ liệu -> Baseline Linear Regression."""
import shutil
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.evaluate import danh_gia, ve_thuc_te_vs_du_doan
from src.preprocess import TARGET, clean_data, load_data, split_and_preprocess, thong_ke_thieu

ROOT = Path(__file__).resolve().parent
RAW, PROC, OUT, MODELS = ROOT / "data/raw/house_prices.csv", ROOT / "data/processed", ROOT / "outputs", ROOT / "models"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def buoc(tieu_de):
    print(f"\n{'=' * 70}\n{tieu_de}\n{'=' * 70}")


def chuan_bi_thu_muc():
    """Tạo thư mục và chuyển ./house_prices.csv vào data/raw/ nếu chưa chuyển."""
    for d in (RAW.parent, PROC, OUT, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    goc = ROOT / "house_prices.csv"
    if goc.exists() and not RAW.exists():
        shutil.move(str(goc), RAW)
        print(f"Đã chuyển {goc.name} -> {RAW.relative_to(ROOT)}")
    if not RAW.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu. Hãy đặt file vào {RAW}")


def ve_eda(df):
    """Lưu 2 biểu đồ EDA: phân phối SalePrice và top cột thiếu dữ liệu."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(df[TARGET], bins=50, color="#4C72B0")
    ax[0].set(title="Phân phối SalePrice (lệch phải)", xlabel="Giá bán (USD)", ylabel="Số lượng")
    ax[1].hist(np.log1p(df[TARGET]), bins=50, color="#55A868")
    ax[1].set(title="Phân phối log(1 + SalePrice) (gần chuẩn)", xlabel="log(Giá bán)")
    fig.tight_layout()
    fig.savefig(OUT / "01_saleprice_distribution.png", dpi=150)
    plt.close(fig)

    thieu = thong_ke_thieu(df, top=15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(thieu.index[::-1], thieu["ti_le_%"][::-1], color="#C44E52")
    ax.set(title="Top 15 cột thiếu dữ liệu nhiều nhất", xlabel="Tỉ lệ thiếu (%)")
    fig.tight_layout()
    fig.savefig(OUT / "02_missing_values_top15.png", dpi=150)
    plt.close(fig)


def main():
    chuan_bi_thu_muc()

    # ---------- Bước 1: EDA ----------
    buoc("BƯỚC 1: KHÁM PHÁ DỮ LIỆU (EDA)")
    df = load_data(RAW)
    print(f"Kích thước (shape): {df.shape[0]} dòng x {df.shape[1]} cột\n")
    print("--- info() ---")
    df.info()
    print("\n--- describe() (cột số) ---")
    print(df.describe().T.round(2))
    print("\n--- Top 15 cột thiếu dữ liệu ---")
    print(thong_ke_thieu(df))
    gia = df[TARGET]
    print(f"\n--- Phân phối {TARGET} ---")
    print(gia.describe().round(2))
    print(f"Độ lệch (skewness): {gia.skew():.3f} | Độ nhọn (kurtosis): {gia.kurt():.3f}")
    ve_eda(df)
    print("Đã lưu biểu đồ EDA vào outputs/")

    # ---------- Bước 2: Làm sạch ----------
    buoc("BƯỚC 2: LÀM SẠCH DỮ LIỆU")
    so_dong_truoc = len(df)
    df_sach = clean_data(df)
    print("Đã bỏ cột Id. Đã điền NaN = 'None'/0 cho các cột 'không có'.")
    print(f"Đã loại outlier (GrLivArea > 4000 & SalePrice < 300000): {so_dong_truoc - len(df_sach)} dòng")
    print(f"Shape sau làm sạch: {df_sach.shape}")
    print(f"Cột còn NaN (số -> điền median, phân loại -> điền mode trong bộ tiền xử lý):\n{thong_ke_thieu(df_sach, 5)}")

    # ---------- Bước 3: Tiền xử lý & chia dữ liệu ----------
    buoc("BƯỚC 3: MÃ HÓA, CHUẨN HÓA & CHIA TRAIN/TEST (80/20, random_state=42)")
    X_train, X_test, y_train, y_test, pre = split_and_preprocess(df_sach, test_size=0.2, seed=42)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape} | y_test: {y_test.shape}")
    print(f"Số đặc trưng sau One-Hot Encoding: {X_train.shape[1]}")
    print(f"NaN còn lại trong X_train/X_test: {int(X_train.isna().sum().sum())}/{int(X_test.isna().sum().sum())}")
    for ten, d in {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}.items():
        d.to_csv(PROC / f"{ten}.csv", index=False)
    joblib.dump(pre, MODELS / "preprocessor.joblib")
    print("Đã lưu 4 file CSV vào data/processed/ và bộ tiền xử lý vào models/preprocessor.joblib")

    # ---------- Bước 4: Baseline ----------
    buoc("BƯỚC 4: MÔ HÌNH BASELINE - LINEAR REGRESSION")
    mo_hinh = LinearRegression().fit(X_train, y_train)
    print(f"Đã huấn luyện LinearRegression trên {len(X_train)} mẫu.")

    # ---------- Bước 5: Đánh giá ----------
    buoc("BƯỚC 5: ĐÁNH GIÁ TRÊN TẬP TEST")
    y_pred = mo_hinh.predict(X_test)
    kq = danh_gia(y_test, y_pred)
    print(f"MAE  : {kq['MAE']:>14,.2f} USD")
    print(f"MSE  : {kq['MSE']:>14,.2f}")
    print(f"RMSE : {kq['RMSE']:>14,.2f} USD")
    print(f"R2   : {kq['R2']:>14.4f}")
    pd.DataFrame([kq], index=["LinearRegression"]).to_csv(OUT / "baseline_metrics.csv")
    ve_thuc_te_vs_du_doan(y_test, y_pred, OUT / "03_actual_vs_predicted.png")
    print("Đã lưu biểu đồ và bảng chỉ số vào outputs/")
    print("\nHOÀN THÀNH GIAI ĐOẠN 1.")


if __name__ == "__main__":
    main()
