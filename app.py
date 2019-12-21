import os
from flask import Flask, request, render_template
import sys
sys.path.append("src")
from lime_explainer import explainer, tokenizer, METHODS
import time
app = Flask(__name__)
SECRET_KEY = os.urandom(24)

@app.route('/')
@app.route('/result', methods=['POST'])
def index():
    exp = ""
    if request.method == 'POST':
        text = tokenizer(request.form['entry'])
        method = request.form['classifier']
        n_samples = request.form['n_samples']
        if any(not v for v in [text, n_samples]):
            raise ValueError("Please do not leave text fields blank.")
        s = time.time()
        exp = explainer(method,
                        text=text,
                        num_samples=int(n_samples),
                        num_classes=["negative","neutral","positive"]
                        )
        e = time.time()
        print ((e-s)/60.0)
        exp = exp.as_html()

        return render_template('index.html', exp=exp, entry=text, n_samples=n_samples, classifier=method)
    return render_template('index.html', exp=exp)


if __name__ == '__main__':
    app.secret_key = SECRET_KEY
    app.run(debug=True)
