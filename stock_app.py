import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = dash.Dash(__name__)
server = app.server

scaler = MinMaxScaler(feature_range=(0, 1))

items = [
    {
        'name': 'BTC_USD',
        'df_nse': pd.read_csv("./BTC-USD.csv")
    },
    {
        'name': 'ETH_USD',
        'df_nse': pd.read_csv("./ETH-USD.csv")
    },
    {
        'name': 'ADA_USD',
        'df_nse': pd.read_csv("./ADA-USD.csv")
    },
]

for item in items:
    item['scaler'] = MinMaxScaler(feature_range=(0, 1))
    item['df_nse']["Date"] = pd.to_datetime(item['df_nse'].Date, format="%Y-%m-%d")
    item['df_nse'].index = item['df_nse']['Date']
    data = item['df_nse'].sort_index(ascending=True, axis=0)
    item['new_data'] = pd.DataFrame(index=range(0, len(item['df_nse'])), columns=['Date', 'Close'])
    for i in range(0, len(data)):
        item['new_data']["Date"][i] = data['Date'][i]
        item['new_data']["Close"][i] = data["Close"][i]
    item['new_data'].index = item['new_data'].Date
    item['new_data'].drop("Date", axis=1, inplace=True)
    item['dataset'] = item['new_data'].values
    breakpoint = int(0.7 * len(item['dataset']))
    item['train'] = item['dataset'][0:breakpoint, :]
    item['valid'] = item['dataset'][breakpoint:, :]
    item['scaled_data'] = item['scaler'].fit_transform(item['dataset'])
    item['x_train'], item['y_train'] = [], []
    number_of_days = 100
    for i in range(number_of_days, item['train'].shape[0]):
        item['x_train'].append(item['scaled_data'][i - number_of_days:i, 0])
        item['y_train'].append(item['scaled_data'][i, 0])
    item['x_train'], item['y_train'] = np.array(item['x_train']), np.array(item['y_train'])
    item['x_train'] = np.reshape(item['x_train'], (item['x_train'].shape[0], item['x_train'].shape[1], 1))
    model = load_model(item['name'] + '_model.h5')
    item['inputs'] = item['new_data'][len(item['new_data']) - len(item['valid']) - number_of_days:].values
    item['inputs'] = item['inputs'].reshape(-1, 1)
    item['inputs'] = item['scaler'].transform(item['inputs'])
    item['X_test'] = []
    for i in range(number_of_days, item['inputs'].shape[0]):
        item['X_test'].append(item['inputs'][i - number_of_days:i, 0])
    item['X_test'] = np.array(item['X_test'])
    item['X_test'] = np.reshape(item['X_test'], (item['X_test'].shape[0], item['X_test'].shape[1], 1))
    item['closing_price'] = model.predict(item['X_test'])
    item['closing_price'] = item['scaler'].inverse_transform(item['closing_price'])

    item['train'] = item['new_data'][:breakpoint]
    item['valid'] = item['new_data'][breakpoint:]
    item['valid'] = item['valid'].assign(Predictions=item['closing_price'])



def get_figure(data):
    figure = {
        "data": [
            go.Scatter(
                x=data.index,
                y=data["Close"],
                mode='markers',
                name="Actual closing price"
            ),
            go.Scatter(
                x=data.index,
                y=data["Predictions"],
                mode='markers',
                name="LSTM Predicted closing price"
            )
        ],
        "layout": go.Layout(
            title='Crypto Price Analysis',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Closing Price'}
        )
    }
    return figure


app.layout = html.Div([
    html.H1("Crypto Price Analysis Dashboard", style={"textAlign": "center"}),

    html.Div(id='tabs-header', children=[
        html.Div(id='tabs-header-1', children=[
            html.H3("Select Crypto: ", style={"textAlign": "center"}),
        ]),

        dcc.Dropdown(
            id='my-dropdown',
            options=[
                {'label': 'Bitcoin', 'value': 'BTC_USD'},
                {'label': 'Ethereum', 'value': 'ETH_USD'},
                {'label': 'Cardano', 'value': 'ADA_USD'}
            ],
            value='BTC_USD',
            multi=False,
            clearable=False,
            searchable=False,
            style={"width": "100px", "color": "#000000"}
        ),

    ], style={"display": "flex", "justifyContent": "center", "alignItems": "center", "marginBottom": "50px",
              "margin": "auto", "textAlign": "center"}),

    dcc.Tabs(id="tabs-graph", children=[
        dcc.Tab(label='Crypto Data', children=[
            html.Div(id='graph-content')
        ]),
    ])
])


@app.callback(Output('graph-content', 'children'), [Input('my-dropdown', 'value')])
def update_graph(selected_crypto):
    selected_item = next((item for item in items if item['name'] == selected_crypto), None)
    print("selected_item: ", selected_item)
    if selected_item:
        return [
            html.H2(f"Predicted closing price - {selected_item['name']}", style={"textAlign": "center"}),
            dcc.Graph(
                id="Actual Data",
                figure=get_figure(selected_item['valid'])
            )
        ]
    return []


if __name__ == '__main__':
    app.run_server(debug=True)
