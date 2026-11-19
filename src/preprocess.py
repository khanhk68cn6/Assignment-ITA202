"""Các hàm tiền xử lý dùng lại được cho Giai đoạn 1 và Giai đoạn 2 (Pipeline/GridSearchCV)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "SalePrice"

# Trong Ames Housing, NaN ở các cột này nghĩa là "không có" (không có hồ bơi, hẻm, gara...)
COT_NAN_LA_NONE = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]
# Cột số mà NaN nghĩa là "không có" -> điền 0
COT_NAN_LA_0 = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]


def load_data(path) -> pd.DataFrame:
    """Đọc file CSV thô."""
    return pd.read_csv(path)


def thong_ke_thieu(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Bảng top cột thiếu dữ liệu nhiều nhất (số lượng và tỉ lệ %)."""
    thieu = df.isna().sum()
    thieu = thieu[thieu > 0].sort_values(ascending=False)
    return pd.DataFrame({"so_luong": thieu, "ti_le_%": (thieu / len(df) * 100).round(2)}).head(top)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch: bỏ Id, điền NaN 'không có', xử lý outlier, ép MSSubClass về dạng phân loại.
    Chỉ dùng quy tắc cố định (không tính thống kê) nên an toàn, không gây rò rỉ dữ liệu."""
    df = df.drop(columns="Id", errors="ignore").copy()

    # NaN mang nghĩa "không có" -> nhãn "None" (cột phân loại) hoặc 0 (cột số)
    for c in COT_NAN_LA_NONE:
        if c in df:
            df[c] = df[c].fillna("None")
    for c in COT_NAN_LA_0:
        if c in df:
            df[c] = df[c].fillna(0)
    # Không có gara -> lấy năm gara = năm xây nhà để tránh giá trị 0 làm lệch thang đo
    if {"GarageYrBlt", "YearBuilt"} <= set(df.columns):
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])

    # MSSubClass là mã loại nhà (nhãn), không phải số có thứ tự -> chuyển sang chuỗi
    if "MSSubClass" in df:
        df["MSSubClass"] = df["MSSubClass"].astype(str)

    # Outlier cực đoan: diện tích rất lớn nhưng giá rất thấp (nhà bán bất thường)
    outlier = (df["GrLivArea"] > 4000) & (df[TARGET] < 300_000)
    return df[~outlier].reset_index(drop=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Cột số: điền median + StandardScaler. Cột phân loại: điền mode + OneHotEncoder.
    Danh mục hiếm (<10 mẫu) được gộp chung để Linear Regression không bị bất ổn."""
    cot_so = X.select_dtypes(include="number").columns.tolist()
    cot_phan_loai = X.select_dtypes(exclude="number").columns.tolist()

    so = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    pl = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                                 sparse_output=False)),
    ])
    pre = ColumnTransformer([("so", so, cot_so), ("phan_loai", pl, cot_phan_loai)],
                            verbose_feature_names_out=False)
    return pre.set_output(transform="pandas")


def split_and_preprocess(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Chia train/test TRƯỚC, rồi fit bộ tiền xử lý CHỈ trên train (tránh data leakage)."""
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    pre = build_preprocessor(X_train)
    X_train = pre.fit_transform(X_train)
    X_test = pre.transform(X_test)
    return X_train, X_test, y_train, y_test, pre
