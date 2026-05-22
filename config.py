"""
配置文件 - 集中管理应用配置
"""
import os

# ======================
# 数据库配置
# ======================
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "123456"
DB_NAME = "idas_userif"
DB_CHARSET = "utf8mb4"

# ======================
# Flask 应用配置
# ======================
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
DEBUG = True

# ======================
# 分析配置
# ======================
MAX_SAMPLE_SIZE = 5000  # 分析时最大样本数
PREVIEW_LIMIT = 1000    # 预览最大行数
CHART_SAMPLE_SIZE = 2000  # 图表数据最大采样数
