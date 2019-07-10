#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/2 17:49
#@Author: ykp
#@File  : web_show.py

from flask import Flask,render_template,Response
from face_recognition_web import Face_recognition
from datetime import timedelta
import tensorflow as tf
global graph
a=Face_recognition()

graph = tf.get_default_graph()

app= Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=0)
def gen(cam):
    while True:
        with graph.as_default():
            frame = cam.actual_time_recognition()
        if frame is None:
            break
        yield(b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/test/')
def show():
    return render_template('test.html')
@app.route('/video_feed/')
def video_feed():
  return Response(gen(a),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/top/')
def top():
    return render_template('top.html')
@app.route('/left/')
def left():
    return render_template('left.html')
@app.route('/main/')
def main():
    return render_template('main.html')

if __name__ == '__main__':
    app.run(debug = True)