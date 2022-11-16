from flask import Flask, render_template, request
import model

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def homepage():
    prediction = "GET"

    if (request.method == 'POST'):
        payload = request.form.to_dict()
        payload = list(payload.values())
        payload = list(map(int, payload))
        print(payload)
        result = model.predict(payload)
        if (result == 0):
            prediction = False
        else:
            prediction= True
  
    return render_template('homepage.html', prediction=prediction)
    

if __name__ == '__main__':
    model.init()
    app.run(debug=True)