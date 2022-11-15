from flask import Flask
import model

app = Flask(__name__)

@app.route("/")
def hello_world():
    pred = model.predict([2,0,0,2,1,5,41,0,1,0,1,2,2,1,1,0,False,False,False,False,False])
    return "<p>Result: "+ str(pred) +"</p>"


if __name__ == '__main__':
    model.init()
    app.run(debug=True)