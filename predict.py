import pickle
from flask import Flask
from flask import request
from flask import jsonify

filename = 'house_price.pkl'

app = Flask('house_price')
loaded_model = pickle.load(open(filename,'rb'))
@app.route('/predict', methods=['POST'])

def predict():
    house_data = request.get_json()

    y_pred = best_reg_model.score(x_test, y_test)

    result = {
        'house_price': float(y_pred),
        'Accuracy': bool(accuracy)        
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
