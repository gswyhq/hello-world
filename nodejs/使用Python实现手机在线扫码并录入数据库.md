以下是使用Python实现手机在线扫码并录入数据库的步骤详解：

一、技术选型
1、前端框架：
    Barcode Scanner：使用QuaggaJS或Html5-QRCode实现网页扫码
    移动端适配：Bootstrap 5响应式布局
2、后端框架：
    FastAPI（轻量级异步框架）
    SQLAlchemy（ORM工具）
3、数据库：
    SQLite（开发环境）
    MySQL/PostgreSQL（生产环境）
4、部署方案：
    Nginx反向代理
    Gunicorn应用服务器

二、实现步骤
1. 创建数据库模型(database.py)
```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

#生产环境建议更换为MySQL/PostgreSQL：
# database.py中修改为
#DATABASE_URL = "mysql+pymysql://user:password@localhost/dbname"

DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
```

2. 创建数据库模型（models.py）
```python
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from database import Base  # 改为从database导入

class Cargo(Base):
    __tablename__ = 'cargos'

    id = Column(Integer, primary_key=True)
    box_number = Column(String(50), unique=True)
    container_number = Column(String(50))
    scan_time = Column(DateTime, default=datetime.now)
    operator = Column(String(50))

```

3. 实现FastAPI后端（main.py）
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import SessionLocal, engine  # 改为从database导入
import models


app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})

class ScanData(BaseModel):
    box_number: str
    container_number: str
    operator: str

@app.post("/scan")
async def record_scan(data: ScanData):
    db = SessionLocal()
    try:
        # 检查是否重复扫描
        existing = db.query(models.Cargo).filter(
            models.Cargo.box_number == data.box_number
        ).first()

        if existing:
            return {"status": "error", "message": "该箱号已扫描登记"}

        new_scan = models.Cargo(
            box_number=data.box_number,
            container_number=data.container_number,
            operator=data.operator
        )

        db.add(new_scan)
        db.commit()
        return {"status": "success", "data": data.dict()}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
```

4. 前端扫码页面（templates/index.html）
```html
<!DOCTYPE html>
<html>
<head>
    <title>扫码入库系统</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h2 class="mb-4">货物扫码入库</h2>

        <div class="row">
            <div class="col-md-6">
                <div id="scanner-container"></div>
                <button class="btn btn-primary mt-3" onclick="startScan()">开始扫描</button>
            </div>

            <div class="col-md-6">
                <form id="scanForm">
                    <div class="mb-3">
                        <label class="form-label">集装箱号</label>
                        <input type="text" class="form-control" id="containerNumber" required>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">操作员</label>
                        <input type="text" class="form-control" id="operator" required>
                    </div>

                    <button type="submit" class="btn btn-success">提交数据</button>
                </form>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/html5-qrcode/2.3.4/html5-qrcode.min.js"></script>
    <script>
        let html5QrcodeScanner = null;

        function startScan() {
            if(html5QrcodeScanner) return;

            html5QrcodeScanner = new Html5QrcodeScanner(
                'scanner-container',
                {
                    fps: 10,
                    qrbox: 250,
                    formatsToSupport: [
                        Html5QrcodeSupportedFormats.UPC_A,
                        Html5QrcodeSupportedFormats.EAN_13,
                        Html5QrcodeSupportedFormets.CODE_128
                    ]
                });

            html5QrcodeScanner.render(onScanSuccess);
        }

        async function onScanSuccess(decodedText) {
            const containerNumber = document.getElementById('containerNumber').value;
            const operator = document.getElementById('operator').value;

            try {
                const response = await fetch('/scan', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        box_number: decodedText,
                        container_number: containerNumber,
                        operator: operator
                    })
                });

                const result = await response.json();
                if(result.status === 'success') {
                    alert('登记成功！');
                } else {
                    alert('错误：' + result.message);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('提交失败');
            }
        }

        document.getElementById('scanForm').onsubmit = function(e) {
            e.preventDefault();
            startScan();
            return false;
        };
    </script>
</body>
</html>
```

三、运行与部署
1、初始化数据库：
`python -c "from models import Base; from database import engine; Base.metadata.create_all(bind=engine)"`
2、启动服务：
`uvicorn main:app --reload`
3、访问系统：
打开手机浏览器访问 http://<服务器IP>:8000 即可使用

四、扩展功能建议
1、用户认证：
    添加JWT认证
    用户角色管理（管理员/操作员）
2、实时监控：
    使用WebSocket实现实时数据更新
    大屏展示装柜进度
3、移动端优化：
    使用PWA技术实现类App体验
    添加离线存储功能
4、数据校验增强：
    箱号校验算法（校验位验证）
    集装箱号国际标准校验
5、系统集成：
    对接企业微信/钉钉通知
    生成电子装柜单（PDF）
该方案利用现代Web技术实现了跨平台的扫码解决方案，通过响应式设计适配手机屏幕，结合FastAPI的高性能特性，能够满足中小型物流企业的装柜管理需求。实际部署时建议添加HTTPS加密和安全审计功能。


