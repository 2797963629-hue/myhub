import os
import json
import pandas as pd
import numpy as np
import pymysql
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# ======================
# 你的 MySQL 信息
# ======================
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "123456"  # 改成你自己的
DB_NAME = "idas_userif"       # 你自己建的库名

def get_conn():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4"
    )

# 配置文件上传路径
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 限制上传文件大小为 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 存储当前会话的数据状态（单用户模式用全局变量，多用户需改用 session/file）
current_state = {
    "file_path": None,   # 当前数据文件路径
    "columns": [],       # 列名
    "dtypes": {}         # 列类型
}

# ---------------------- 路由部分 ----------------------

# 首页
@app.route("/")
def index():
    return redirect(url_for("login_page"))

# 登录页面
@app.route("/login_page")
def login_page():
    return render_template("login.html")

# 注册页面
@app.route("/register_page")
def register_page():
    return render_template("register.html")

# 注册逻辑
@app.route("/register", methods=["POST"])
def register():
    user_id = request.form.get("id")
    pwd = request.form.get("pwd")
    pwd2 = request.form.get("pwd2")

    if not user_id or not pwd or not pwd2:
        return jsonify({"status": "error", "msg": "内容不能为空"})
    if pwd != pwd2:
        return jsonify({"status": "error", "msg": "两次密码不一致"})

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
    if cur.fetchone():
        conn.close()
        return jsonify({"status": "error", "msg": "账号已存在"})

    cur.execute("INSERT INTO users (id, pwd) VALUES (%s, %s)", (user_id, pwd))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "msg": "注册成功！即将跳转..."})

# 登录逻辑
@app.route("/login", methods=["POST"])
def login():
    user_id = request.form.get("id")
    pwd = request.form.get("pwd")

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=%s AND pwd=%s", (user_id, pwd))
    user = cur.fetchone()
    conn.close()

    if user:
        return jsonify({"status": "success", "msg": "登录成功！"})
    else:
        return jsonify({"status": "error", "msg": "账号或密码错误"})

# 文件上传接口 - 已修复嵌套错误
@app.route("/api/upload", methods=["POST"])
def upload_api():
    if 'file' not in request.files:
        return jsonify({"status": "error", "msg": "未找到文件部分"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "msg": "未选择文件"})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # 读取文件
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                # 注意：读取 excel 需要安装 pip install openpyxl
                df = pd.read_excel(file_path)

            columns = df.columns.tolist()
            total_rows = len(df)

            # 保存当前数据状态
            current_state["file_path"] = file_path
            current_state["columns"] = columns
            current_state["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

            # 限制预览行数
            limit = 1000
            display_df = df.iloc[:limit]

            # 转换为列表并处理 NaN 值 (JSON 不支持 NaN)
            data_to_send = display_df.where(pd.notnull(display_df), None).values.tolist()

            return jsonify({
                "status": "success",
                "columns": columns,
                "data": data_to_send,
                "total_rows": total_rows,
                "is_truncated": total_rows > limit,
                "msg": f"解析成功，共 {total_rows} 行"
            })
        except Exception as e:
            return jsonify({"status": "error", "msg": f"解析失败: {str(e)}"})

# 数据清洗接口
@app.route("/api/clean", methods=["POST"])
def clean_api():
    req = request.get_json()
    action = req.get("action", "")

    file_path = current_state.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"status": "error", "msg": "请先上传数据文件"})

    try:
        # 读取当前数据
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        rows_before = len(df)
        cols_before = len(df.columns)

        # 执行清洗操作
        if action == "dropna":
            df = df.dropna()
            msg = f"已去除缺失值：{rows_before} 行 → {len(df)} 行（删除 {rows_before - len(df)} 行）"
        elif action == "drop_duplicates":
            df = df.drop_duplicates()
            msg = f"已去除重复项：{rows_before} 行 → {len(df)} 行（删除 {rows_before - len(df)} 行）"
        else:
            return jsonify({"status": "error", "msg": f"未知的清洗操作: {action}"})

        # 保存清洗后的数据（覆盖原文件）
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
        else:
            df.to_excel(file_path, index=False)

        # 更新状态
        current_state["columns"] = df.columns.tolist()

        # 返回预览数据
        limit = 1000
        display_df = df.iloc[:limit]
        data_to_send = display_df.where(pd.notnull(display_df), None).values.tolist()

        return jsonify({
            "status": "success",
            "msg": msg,
            "columns": df.columns.tolist(),
            "data": data_to_send,
            "total_rows": len(df),
            "is_truncated": len(df) > limit,
            "removed_rows": rows_before - len(df)
        })
    except Exception as e:
        return jsonify({"status": "error", "msg": f"清洗失败: {str(e)}"})

# 导出数据接口
@app.route("/api/export")
def export_api():
    file_path = current_state.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"status": "error", "msg": "请先上传数据"})

    from flask import send_file
    return send_file(file_path, as_attachment=True)

# 获取列信息（用于图表参数配置）
@app.route("/api/columns")
def columns_api():
    file_path = current_state.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"status": "error", "msg": "请先上传数据"})

    cols = current_state.get("columns", [])
    dtypes = current_state.get("dtypes", {})
    numeric_cols = [c for c in cols if 'int' in dtypes.get(c, '') or 'float' in dtypes.get(c, '')]
    categorical_cols = [c for c in cols if c not in numeric_cols]

    return jsonify({
        "status": "success",
        "columns": cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols
    })

# 分析接口（聚类/预测/降维/分类 + 效果评估）
@app.route("/api/analyze", methods=["POST"])
def analyze_api():
    file_path = current_state.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"status": "error", "msg": "请先上传数据文件"})

    req = request.get_json() or {}
    algorithm = req.get("algorithm", "kmeans")
    params = req.get("params", {})

    # 图表相关参数
    x_col = req.get("x_col", "")
    y_col = req.get("y_col", "")

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # 取数值列用于建模
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return jsonify({"status": "error", "msg": "至少需要 2 个数值列才能进行分析"})

        # 只对前 5000 行做分析（避免数据过大导致超时）
        if len(df) > 5000:
            df_sample = df.sample(5000, random_state=42)
        else:
            df_sample = df.copy()

        # 对数值列做标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(df_sample[numeric_cols].dropna())

        result = {"status": "success"}

        # ============================================================
        # K-Means 聚类
        # ============================================================
        if algorithm == "kmeans":
            k = int(params.get("n_clusters", 3))
            k = max(2, min(k, 10))

            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            df_sample["_cluster"] = labels.tolist()

            # 效果评估：轮廓系数
            sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0
            # 簇内平方和（inertia）
            inertia = float(model.inertia_)

            # 聚类中心（逆标准化以便理解）
            centers = scaler.inverse_transform(model.cluster_centers_)
            centers_data = []
            for i, c in enumerate(centers):
                centers_data.append({f"{numeric_cols[j]}": round(float(c[j]), 4) for j in range(len(numeric_cols))})

            # 散点图数据（用前两个数值列/主成分做可视化）
            pca_viz = PCA(n_components=2)
            coords = pca_viz.fit_transform(X)
            scatter_data = [{"x": float(coords[i][0]), "y": float(coords[i][1]), "cluster": int(labels[i])}
                            for i in range(len(coords))]

            # 分类统计
            cluster_counts = {}
            for lb in labels.tolist():
                ck = f"聚类 {int(lb)+1}"
                cluster_counts[ck] = cluster_counts.get(ck, 0) + 1

            result.update({
                "algorithm": "K-Means 聚类",
                "evaluation": {
                    "轮廓系数 (Silhouette Score)": round(sil, 4),
                    "簇内平方和 (Inertia)": round(inertia, 2)
                },
                "cluster_centers": centers_data,
                "cluster_distribution": cluster_counts,
                "n_clusters": k,
                "chart_default": {
                    "type": "scatter",
                    "data": scatter_data,
                    "title": f"K-Means 聚类结果 (k={k})",
                    "x_label": "主成分 1",
                    "y_label": "主成分 2",
                    "legend": [f"聚类 {i+1}" for i in range(k)]
                }
            })

        # ============================================================
        # 线性回归
        # ============================================================
        elif algorithm == "regression":
            if not y_col or y_col not in df_sample.columns:
                y_col = numeric_cols[-1]
            feature_cols = params.get("feature_cols", [c for c in numeric_cols if c != y_col])
            if not feature_cols:
                feature_cols = [c for c in numeric_cols if c != y_col]

            X_reg = df_sample[feature_cols].dropna()
            y_reg = df_sample.loc[X_reg.index, y_col].dropna()
            X_reg = X_reg.loc[y_reg.index]
            y_reg = y_reg.loc[X_reg.index]

            if len(X_reg) < 10:
                return jsonify({"status": "error", "msg": "有效样本太少，无法做回归分析"})

            X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            # 效果评估
            r2 = r2_score(y_te, y_pred)
            mse = mean_squared_error(y_te, y_pred)
            mae = mean_absolute_error(y_te, y_pred)
            rmse = np.sqrt(mse)

            # 系数
            coef_dict = {feature_cols[i]: round(float(model.coef_[i]) if len(feature_cols) > 1 else float(model.coef_), 4)
                         for i in range(len(feature_cols))}
            intercept = round(float(model.intercept_), 4)

            # 图表数据：预测值 vs 真实值散点
            scatter_data = [{"x": float(y_te.iloc[i]), "y": float(y_pred[i])}
                            for i in range(len(y_te))]

            result.update({
                "algorithm": "线性回归",
                "target_column": y_col,
                "feature_columns": feature_cols,
                "coefficients": coef_dict,
                "intercept": intercept,
                "evaluation": {
                    "R² 决定系数": round(r2, 4),
                    "MSE 均方误差": round(mse, 4),
                    "RMSE 均方根误差": round(rmse, 4),
                    "MAE 平均绝对误差": round(mae, 4)
                },
                "chart_default": {
                    "type": "scatter",
                    "data_type": "pred_vs_true",
                    "data": scatter_data,
                    "title": f"线性回归: 预测值 vs 真实值 (R²={round(r2,3)})",
                    "x_label": f"真实值 ({y_col})",
                    "y_label": f"预测值 ({y_col})"
                }
            })

        # ============================================================
        # PCA 降维
        # ============================================================
        elif algorithm == "pca":
            n = int(params.get("n_components", 2))
            n = max(1, min(n, min(len(numeric_cols), len(df_sample))))

            pca_model = PCA(n_components=n)
            reduced = pca_model.fit_transform(X)

            # 解释方差比
            evr = [round(float(v), 4) for v in pca_model.explained_variance_ratio_]
            cumsum = [round(float(v), 4) for v in np.cumsum(evr)]

            # 降维后的散点数据
            if n >= 2:
                scatter_data = [{"x": float(reduced[i][0]), "y": float(reduced[i][1])}
                                for i in range(len(reduced))]
            else:
                scatter_data = [{"x": float(reduced[i][0]), "y": 0}
                                for i in range(len(reduced))]

            # 主成分载荷
            loadings = {}
            for i in range(n):
                loadings[f"PC{i+1}"] = {numeric_cols[j]: round(float(pca_model.components_[i][j]), 4)
                                         for j in range(len(numeric_cols))}

            result.update({
                "algorithm": "PCA 主成分分析",
                "n_components": n,
                "explained_variance_ratio": evr,
                "cumulative_variance": cumsum,
                "component_loadings": loadings,
                "evaluation": {
                    "累计解释方差": round(cumsum[-1], 4),
                    f"PC1 解释方差": evr[0],
                },
                "chart_default": {
                    "type": "scatter",
                    "data": scatter_data,
                    "title": f"PCA 降维 ({n} 维) — 前 2 主成分投影",
                    "x_label": f"PC1 ({round(evr[0]*100,1)}%)",
                    "y_label": f"PC2 ({round(evr[1]*100,1)}%)" if n >= 2 else "—"
                }
            })

        # ============================================================
        # 随机森林分类
        # ============================================================
        elif algorithm == "random_forest":
            # 需要目标列（分类标签）
            categorical_cols = df_sample.select_dtypes(include=['object', 'category']).columns.tolist()
            if not categorical_cols and len(numeric_cols) > 2:
                # 用最后一个数值列做分箱，当作分类目标
                target_col = numeric_cols[-1]
                feature_cols = [c for c in numeric_cols if c != target_col]
                y_raw = pd.cut(df_sample[target_col].dropna(), bins=3, labels=["低", "中", "高"])
                y_val = y_raw.dropna()
            elif categorical_cols:
                target_col = categorical_cols[0]
                feature_cols = [c for c in numeric_cols if c != target_col]
                y_val = df_sample[target_col].dropna()
            else:
                return jsonify({"status": "error", "msg": "需要分类标签列或足够的数值列"})

            X_rf = df_sample.loc[y_val.index, feature_cols].dropna()
            y_val = y_val.loc[X_rf.index]

            if len(set(y_val)) < 2:
                return jsonify({"status": "error", "msg": "分类标签种类不足，无法分类"})

            # 编码标签
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_val)

            X_tr, X_te, y_tr, y_te = train_test_split(X_rf, y_encoded, test_size=0.2, random_state=42)

            n_est = int(params.get("n_estimators", 100))
            rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
            rf.fit(X_tr, y_tr)
            y_pred = rf.predict(X_te)

            acc = accuracy_score(y_te, y_pred)
            class_names = [str(c) for c in le.classes_]

            # 特征重要性
            importance = {feature_cols[i]: round(float(rf.feature_importances_[i]), 4)
                          for i in range(len(feature_cols))}
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

            # 分类报告
            try:
                report = classification_report(y_te, y_pred, target_names=class_names, output_dict=True)
                report_clean = {k: {kk: round(float(vv), 4) if isinstance(vv, (float, np.floating)) else vv
                                    for kk, vv in v.items()}
                                for k, v in report.items() if k not in ("accuracy",)}
            except Exception:
                report_clean = {}

            # 混淆矩阵样式的热力图
            confusion = {}
            for i, true_lbl in enumerate(class_names):
                row_data = {}
                for j, pred_lbl in enumerate(class_names):
                    row_data[pred_lbl] = int(sum((y_te == i) & (y_pred == j)))
                confusion[true_lbl] = row_data

            result.update({
                "algorithm": "随机森林分类",
                "target_column": target_col if categorical_cols or len(numeric_cols) > 2 else y_col,
                "feature_columns": feature_cols,
                "n_classes": len(class_names),
                "class_names": class_names,
                "feature_importance": dict(sorted_imp),
                "evaluation": {
                    "准确率 (Accuracy)": round(acc, 4),
                },
                "classification_report": report_clean,
                "confusion_matrix": confusion,
                "chart_default": {
                    "type": "bar",
                    "data_type": "importance",
                    "data": [{"name": k, "value": v} for k, v in sorted_imp],
                    "title": "随机森林 — 特征重要性排序",
                    "x_label": "特征",
                    "y_label": "重要性"
                }
            })

        else:
            return jsonify({"status": "error", "msg": f"未知算法: {algorithm}"})

        # 附带数值列的分布统计（供前端生成更多图表）
        result["numeric_columns"] = numeric_cols
        result["columns"] = df_sample.columns.tolist()
        result["summary"] = {
            col: {
                "mean": round(float(df_sample[col].dropna().mean()), 4),
                "std": round(float(df_sample[col].dropna().std()), 4),
                "min": round(float(df_sample[col].dropna().min()), 4),
                "max": round(float(df_sample[col].dropna().max()), 4),
                "median": round(float(df_sample[col].dropna().median()), 4)
            }
            for col in numeric_cols[:10]  # 限制前 10 列
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "msg": f"分析失败: {str(e)}"})

# 工作台主页
@app.route("/home")
def home():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)