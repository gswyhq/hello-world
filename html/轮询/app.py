from flask import Flask,render_template,request,jsonify
 
 
app = Flask(__name__, static_url_path='')
 
 
USERS = {
    '1':{'name':'贝贝','count':1},
    '2':{'name':'小东北','count':0},
    '3':{'name':'何伟明','count':0},
}
 
@app.route('/user/list')
def user_list():
    import time
    return render_template('user_list.html',users=USERS)
 
@app.route('/vote',methods=['POST'])
def vote():
    uid = request.form.get('uid')
    USERS[uid]['count'] += 1
    return "投票成功"
 
@app.route('/get/vote',methods=['GET'])
def get_vote():
    USERS['1']['count'] += 1
    return jsonify(USERS)
 
@app.route('/')
def index():
    return app.send_static_file('index.html')
 
if __name__ == '__main__':
    # app.run(host='192.168.13.253',threaded=True)
    app.run(threaded=True) #多线程


