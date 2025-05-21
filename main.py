import dash
from dash import html, dcc, Input, Output, State
import pandas as pd

# Load the dataset
real_estate_data = pd.read_csv("Real_Estate.csv") 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Selecting features and target variable
features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
target = 'House price of unit area'

X = real_estate_data[features]
y = real_estate_data[target]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Real Estate Price Predictor"

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.H1("Real Estate Price Prediction", style={'textAlign': 'center'}),
        
        html.Div([
            dcc.Input(id='distance_to_mrt', type='number', placeholder='Distance to MRT Station (meters)',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='num_convenience_stores', type='number', placeholder='Number of Convenience Stores',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='latitude', type='number', placeholder='Latitude',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='longitude', type='number', placeholder='Longitude',
                      style={'margin': '10px', 'padding': '10px'}),
            html.Button('Predict Price', id='predict_button', n_clicks=0,
                        style={'margin': '10px', 'padding': '10px', 'backgroundColor': '#007BFF', 'color': 'white'}),
        ], style={'textAlign': 'center'}),

        html.Div(id='prediction_output', style={'textAlign': 'center', 'fontSize': '20px', 'marginTop': '20px'})
    ], style={
        'width': '50%',
        'margin': '0 auto',
        'border': '2px solid #007BFF',
        'padding': '20px',
        'borderRadius': '10px',
        'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)'
    }),
    html.Footer("© Laavanjan | Faculty of IT @B22", 
        style={
            'textAlign': 'center',
            'padding': '20px',
            'marginTop': '40px',
            'color': '#333',
            'fontSize': '14px',
            'borderTop': '1px solid #ccc'
        }
    )
])


# Define callback to update output
@app.callback(
    Output('prediction_output', 'children'),
    [Input('predict_button', 'n_clicks')],
    [State('distance_to_mrt', 'value'), 
     State('num_convenience_stores', 'value'),
     State('latitude', 'value'),
     State('longitude', 'value')]
)
def update_output(n_clicks, distance_to_mrt, num_convenience_stores, latitude, longitude):
    if n_clicks > 0 and all(v is not None for v in [distance_to_mrt, num_convenience_stores, latitude, longitude]):
        # Prepare the feature vector
        features = pd.DataFrame([[distance_to_mrt, num_convenience_stores, latitude, longitude]], 
                        columns=['Distance to the nearest MRT station', 
                                 'Number of convenience stores', 
                                 'Latitude', 'Longitude'])

        # Predict
        prediction = model.predict(features)[0]
        return f'Predicted House Price of Unit Area: {prediction:.2f}'
    elif n_clicks > 0:
        return 'Please enter all values to get a prediction'
    return ''

# Run the app
if __name__ == "__main__":
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)
    app.run(debug=True)