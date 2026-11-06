
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ==============================
# 1. TIÊU ĐỀ VÀ TẢI DỮ LIỆU
# ==============================

st.title("Phân tích các yếu tố ảnh hưởng đến giá vàng")
st.write("Phân tích dữ liệu bằng tương quan Pearson và hồi quy tuyến tính đa biến.")

uploaded_file = st.file_uploader("Tải file CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "Date" in df.columns and "Quarter" not in df.columns:
        df = df.rename(columns={"Date": "Quarter"})

    # ==============================
    # 2. LÀM SẠCH DỮ LIỆU
    # ==============================

    required_cols = [
        "Quarter",
        "Gold_Price_USD_oz",
        "CPIAUCSL",
        "DFII10"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Thiếu các cột dữ liệu: {missing_cols}")
        st.stop()

    numeric_cols = [
        "Gold_Price_USD_oz",
        "CPIAUCSL",
        "DFII10"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sắp xếp dữ liệu theo thời gian
    df = df.dropna(subset=["Quarter"]).copy()

    df["Gold_Price_Lag1"] = df["Gold_Price_USD_oz"].shift(1)

    df_clean = df.dropna(
        subset=[
            "Gold_Price_USD_oz",
            "CPIAUCSL",
            "DFII10",
            "Gold_Price_Lag1"
        ]
    ).copy()

    if len(df_clean) < 5:
        st.error("Không đủ dữ liệu để huấn luyện và kiểm tra mô hình.")
        st.stop()

    st.subheader("1. Dữ liệu sau khi làm sạch")
    st.dataframe(df_clean, use_container_width=True)

    # ==============================
    # 3. MA TRẬN TƯƠNG QUAN PEARSON
    # ==============================

    st.subheader("2. Ma trận tương quan Pearson")

    features = [
        "Gold_Price_Lag1",
        "CPIAUCSL",
        "DFII10"
    ]

    target = "Gold_Price_USD_oz"

    correlation_cols = [target] + features

    corr_matrix = df_clean[correlation_cols].corr(method="pearson")

    st.dataframe(
        corr_matrix.style.format("{:.4f}").background_gradient(
            cmap="RdBu_r",
            vmin=-1,
            vmax=1
        ),
        use_container_width=True
    )

    # Biểu đồ ma trận tương quan
    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(
        corr_matrix,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(correlation_cols)))
    ax.set_yticks(range(len(correlation_cols)))

    ax.set_xticklabels(correlation_cols, rotation=45, ha="right")
    ax.set_yticklabels(correlation_cols)

    for i in range(len(correlation_cols)):
        for j in range(len(correlation_cols)):
            ax.text(
                j, i,
                f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black"
            )

    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Pearson Correlation Matrix")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ==============================
    # 4. BIỂU ĐỒ PHÂN TÁN
    # ==============================

    st.subheader("3. Phân tích mối quan hệ giữa các biến")

    for feature in features:

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(
            df_clean[feature],
            df_clean[target],
            alpha=0.7,
            label="Dữ liệu thực tế"
        )

        # Đường xu hướng tuyến tính
        z = np.polyfit(
            df_clean[feature],
            df_clean[target],
            1
        )

        p = np.poly1d(z)

        x_line = np.linspace(
            df_clean[feature].min(),
            df_clean[feature].max(),
            100
        )

        ax.plot(
            x_line,
            p(x_line),
            color="red",
            linewidth=2,
            label="Đường hồi quy"
        )

        ax.set_xlabel(feature)
        ax.set_ylabel("Gold Price (USD/oz)")
        ax.set_title(f"{feature} và giá vàng")
        ax.legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ==============================
    # 5. HỒI QUY TUYẾN TÍNH ĐA BIẾN
    # ==============================

    st.subheader("4. Mô hình hồi quy tuyến tính đa biến")

    X = df_clean[features]
    y = df_clean[target]

    split_index = int(len(df_clean) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # ==============================
    # 6. ĐÁNH GIÁ MÔ HÌNH
    # ==============================

    st.subheader("5. Đánh giá mô hình")

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    col1, col2 = st.columns(2)

    with col1:
        st.metric("R² - Train", f"{train_r2:.4f}")
        st.metric("RMSE - Train", f"{train_rmse:.2f}")

    with col2:
        st.metric("R² - Test", f"{test_r2:.4f}")
        st.metric("RMSE - Test", f"{test_rmse:.2f}")

    # ==============================
    # 7. HỆ SỐ HỒI QUY
    # ==============================

    st.subheader("6. Hệ số hồi quy")

    coefficients = pd.DataFrame({
        "Biến": features,
        "Hệ số hồi quy": model.coef_
    })

    st.dataframe(
        coefficients.style.format({
            "Hệ số hồi quy": "{:.4f}"
        }),
        use_container_width=True
    )

    st.write(f"**Intercept:** {model.intercept_:.4f}")

    # ==============================
    # 8. BIỂU ĐỒ GIÁ THỰC TẾ VÀ DỰ ĐOÁN
    # ==============================

    st.subheader("7. Giá vàng thực tế và giá vàng dự đoán")

    all_pred = np.concatenate([y_train_pred, y_test_pred])

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        df_clean["Quarter"],
        y,
        color="black",
        label="Thực tế"
    )

    ax.plot(
        df_clean["Quarter"],
        all_pred,
        color="red",
        linestyle="--",
        label="Dự đoán"
    )

    ax.set_title("Multiple Linear Regression: Train 80% - Test 20%")
    ax.set_xlabel("Quý")
    ax.set_ylabel("Giá vàng (USD/oz)")
    ax.legend()
    ax.grid()

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("Vui lòng tải file CSV để bắt đầu phân tích.")
