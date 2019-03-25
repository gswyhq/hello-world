#!/usr/bin/python3
# coding=utf-8

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os


app = Flask(__name__)

@app.route('/upload', methods=['POST','GET'])
def upload():

    if request.method == 'POST':
        # try:
        pid = request.form["pid"]
        file_synonym = request.files["file1"]
        file_knowledge = request.files["file2"]

        print(file_synonym.filename)
        print(file_knowledge.filename)
        print(">>>>>>>")
        basepath = os.path.dirname(__file__)
        upload_path_synonyms = os.path.join(basepath, "load_file", "synonyms",
                                            secure_filename(file_synonym.filename))
        upload_path_knowledge = os.path.join(basepath, "load_file", "knowledge",
                                            secure_filename(file_knowledge.filename))
        upload_path_knowledge = os.path.join('/home/xp/work/upload_file/load_file/knowledge', file_knowledge.filename)
        print('------------', upload_path_synonyms)
        print('------------', upload_path_knowledge)
        file_synonym.save(upload_path_synonyms)
        file_knowledge.save(upload_path_knowledge)
        redirect(url_for('upload',file_name=file_synonym.filename))
        redirect(url_for('upload', file_name=file_knowledge.filename))
        print("保存成功")

        # except Exception as e:
        #     print("失败",e)
    # print(">>>MMMM")
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=7805, use_reloader=False)

# curl -XPOST 192.168.3.176:7805/upload -H "Content-Type:multipart/form-data" -F "pid=abcd" -F "file1=@kill.sh" -F "file2=@中文你好.txt"

def main():
    pass


if __name__ == '__main__':
    main()