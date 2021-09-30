from flask_sqlalchemy import SQLAlchemy
# import mysql
import MySQLdb
from flask import Flask

# app = Flask(__name__)

# db = SQLAlchemy(app)
db = SQLAlchemy()


class User(db.Model):
    __tablename__ = 'users2'  # 定义数据库中的表名
    id = db.Column(db.Integer, primary_key=True)  # 主键
    username = db.Column(db.String(16), unique=True)  # 用户名，不允许重复
    password = db.Column(db.String(20), nullable=False)  # 密码，不允许为空

# 创建表前，需保证172.17.0.3:3306/dts，对应是database存在；

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@172.17.0.3:3306/dts?charset=utf8mb4'
    app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
    db.init_app(app)
    db.create_all(app=app)
    return app

if __name__ == '__main__':
    app = create_app()
    app.run()



