import pickle
import json
from flask import Flask, request
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/prophetv3', methods=['POST'])
def prophet3():
    producto = request.json['producto']
    mes = int(request.json['mes'])
    pickle_to_merge =[]

    if producto == 'Todo':
        pickle_to_merge.append('Prophet_Cisternas de GLP.pckl')
        pickle_to_merge.append('Prophet_Baranda de madera.pckl')
        pickle_to_merge.append('Prophet_Cisternas de Agua.pckl')
        pickle_to_merge.append('Prophet_Tolvas Volquetes.pckl')
        pickle_to_merge.append('Prophet_Cisternas de Ácidos.pckl')
        pickle_to_merge.append('Prophet_Semirremolques.pckl')
        pickle_to_merge.append('Prophet_Cisternas de lacteos.pckl')
        pickle_to_merge.append('Prophet_Cisternas de Combustible.pckl')
        pickle_to_merge.append('Prophet_Remolques.pckl')
        
        forecasts = []
        for filename in pickle_to_merge:
            model = pickle.load(open(filename, 'rb'))
            future = model.make_future_dataframe(periods=mes, freq='M')
            forecast = model.predict(future)
            forecast['Producto'] = filename
            forecasts.append(forecast)
        
        result = pd.concat(forecasts)
        data = result[['Producto', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        last_rows = data.groupby('Producto').last()

        #response = last_rows.to_json(orient='records', date_format='iso')
        response = last_rows.to_json(date_format='iso')
        parsed = json.loads(response)
        return parsed
    else:
        filename = f"Prophet_{producto}.pckl"
        m2 = pickle.load(open(filename, 'rb'))

        future2 = m2.make_future_dataframe(periods=mes, freq='M')
        forecast2 = m2.predict(future2)

        data = forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-1:]
        print("data")
        print(data)

        response = data.to_json(orient='records', date_format='iso')
        parsed = json.loads(response)
        return parsed


if __name__ == '__main__':
    app.run(debug=False, port=4100)
