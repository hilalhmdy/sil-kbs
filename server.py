from cgitb import html
from crypt import methods
from flask import Flask, render_template, request
import model

app = Flask(__name__)

# @app.route("/")
def hello_world():
    pred = model.predict([2,0,0,2,1,5,41,0,1,0,1,2,2,1,1,0,False,False,False,False,False])
    return "<p>Result: "+ str(pred) +"</p>"

@app.route('/', methods=['POST', 'GET'])
def homepage():
    print(request.method)
    if (request.method == 'POST'):
        return "<p> Hello world </p>"

    if (request.method == 'GET'):
        return render_template('homepage.html')
    

if __name__ == '__main__':
    model.init()
    app.run(debug=True)