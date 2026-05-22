"""
数据库连接管理模块
负责：MySQL 连接、连接池管理
分工：Team B - 数据库管理
"""
import pymysql
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_CHARSET


class DatabaseConnection:
    """数据库连接管理类"""
    
    @staticmethod
    def get_conn():
        """获取数据库连接"""
        try:
            conn = pymysql.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                charset=DB_CHARSET
            )
            return conn
        except Exception as e:
            raise Exception(f"数据库连接失败: {str(e)}")
    
    @staticmethod
    def close_conn(conn):
        """关闭数据库连接"""
        if conn:
            conn.close()
    
    @staticmethod
    def execute_query(query, params=None):
        """执行查询"""
        conn = DatabaseConnection.get_conn()
        try:
            cur = conn.cursor()
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            result = cur.fetchall()
            conn.close()
            return result
        except Exception as e:
            conn.close()
            raise Exception(f"查询执行失败: {str(e)}")
    
    @staticmethod
    def execute_update(query, params=None):
        """执行更新/插入/删除"""
        conn = DatabaseConnection.get_conn()
        try:
            cur = conn.cursor()
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            conn.commit()
            affected_rows = cur.rowcount
            conn.close()
            return affected_rows
        except Exception as e:
            conn.rollback()
            conn.close()
            raise Exception(f"更新执行失败: {str(e)}")
