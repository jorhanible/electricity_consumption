from flask import Flask, render_template
from backend import stacc
import io

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    result = stacc.main()
    return render_template('index.html', message=result)

@app.route('/electricity_consumption_plot')
def electricity_consumption_plot():
    image_path = stacc.original_electricity_consumption(consumption_df)  

    return render_template('index.html', image_path=image_path)

@app.route('/comparison_plot')
def comparison_consumption_plot():
    image_path_vs = stacc.original_vs_predicted_electricity_consumption(combined_df)  

    return render_template('index.html', image_path=image_path_vs)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
