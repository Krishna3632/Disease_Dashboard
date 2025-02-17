from flask import Flask,render_template

app=Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes')
def diab():
    return render_template('diabetes.html')
if __name__=="__main__":
    app.run(debug=True)