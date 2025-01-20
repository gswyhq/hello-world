
# 实现类似SSO（Single Sign-On，单点登录）的功能
####################################################################################################################
# 应用A 
# 浏览器访问应用a, http://localhost:5000/login_form
# http://localhost:5000/ 跳转到 http://localhost:5001/protected 访问应用B

from flask import Flask, request, jsonify, make_response
import jwt
import datetime
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# `your_secret_key` 是一个用于加密和解密 JWT 的密钥，它应该是一个随机生成的字符串，用于保证 JWT 的安全性和唯一性。这个密钥需要在生成 JWT 的服务（应用A）和验证 JWT 的服务（应用B）之间共享，但对外部世界保持秘密。
#你可以通过多种方式生成这个密钥，例如使用 Python 的 `secrets` 模块来生成一个安全的随机字符串：

#```python
#import secrets
#secret_key = secrets.token_hex(16)  # 生成一个32字符长的十六进制随机字符串
#print(secret_key)
#```

#运行这段代码，你会得到一个类似这样的输出：
# f7d9e8c0f5b14e23b2e4a7c7d5e6a8f2

#这个字符串就是你的 `your_secret_key`，请确保将其保存在安全的地方，并在应用A和应用B中都使用相同的密钥值。在实际生产环境中，这个密钥通常会存储在环境变量或配置文件中，而不是硬编码在代码里，以增加安全性。

@app.route('/')
def index():
    return '''
    <html>
        <body>
            <h1>Welcome to Application A</h1>
            <p>Click the link below to access Application B:</p>
            <a href="http://localhost:5001/protected">访问应用B</a>
        </body>
    </html>
    '''

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(data, *args, **kwargs)
    return decorated

@app.route('/login_form', endpoint='login_form')
def index():
    return '''
    <form action="/login" method="post">
        <label for="username">Username:</label><br>
        <input type="text" id="username" name="username"><br>
        <label for="password">Password:</label><br>
        <input type="password" id="password" name="password"><br>
        <input type="submit" value="Submit">
    </form>
    '''

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    if username == 'your_username' and password == 'password':
        token = jwt.encode({
            'user': username,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
        }, app.config['SECRET_KEY'])
        if isinstance(token, bytes):
            token = token.decode('UTF-8')
        response = make_response(jsonify({'token': token}), 201)
        response.set_cookie('jwt', token)
        return response
    auth = request.authorization
    if auth and auth.password == 'password':
        token = jwt.encode({
            'user': auth.username,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
        }, app.config['SECRET_KEY'])
        response = make_response(jsonify({'token': token}), 201)
        response.set_cookie('jwt', token)
        return response
    return make_response('Could not verify!', 401)

if __name__ == '__main__':
    app.run(debug=True)


####################################################################################################################
# 应用B
from flask import Flask, request, jsonify
import jwt
import datetime
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'jwt' in request.cookies:
            token = request.cookies.get('jwt')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(data, *args, **kwargs)
    return decorated


# 基于JWT（JSON Web Tokens）的简单身份验证机制。然而，这个机制仅应用于装饰有`@token_required`的路由。在你的例子中，只有`'/protected'`这个路由被保护起来了。
# 这意味着，如果你的应用中有其他未使用`@token_required`装饰器的路由，它们将不会进行任何身份验证检查，因此可以直接访问

@app.route('/protected', methods=['GET'])
@token_required
def protected(data):
    return jsonify({'message': 'This is a protected route.'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)

