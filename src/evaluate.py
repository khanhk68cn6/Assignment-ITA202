"""Hàm đánh giá & vẽ biểu đồ dùng chung cho mọi mô hình (Phase 1 và Phase 2)."""
import matplotlib
matplotlib.use("Agg")  # lưu ảnh không cần mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def danh_gia(y_true, y_pred) -> dict:
    """Trả về MAE, MSE, RMSE, R2."""
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def ve_thuc_te_vs_du_doan(y_true, y_pred, path, ten_mo_hinh="Linear Regression"):
    """Biểu đồ phân tán Giá thực tế vs Giá dự đoán + đường chéo hoàn hảo."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=22, color="#4C72B0", label="Mẫu test")
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", label="Dự đoán hoàn hảo (y = x)")
    ax.set(xlabel="Giá thực tế (USD)", ylabel="Giá dự đoán (USD)",
           title=f"{ten_mo_hinh}: Giá thực tế vs Giá dự đoán")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
