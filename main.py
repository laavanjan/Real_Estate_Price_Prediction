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
    # Header Section
    html.Div([
        html.H1("Real Estate Price Prediction", 
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'fontSize': '2rem',
                    'fontWeight': '300',
                    'marginBottom': '1.5rem',
                    'letterSpacing': '1px'
                }),

    ], style={'marginBottom': '1rem'}),
    
    # Main Container
    html.Div([
        # Input Form Section
        html.Div([
            html.H3("Property Details", 
                    style={
                        'color': '#34495e',
                        'marginBottom': '1.5rem',
                        'fontSize': '1.4rem',
                        'fontWeight': '400'
                    }),
            
            # Input Grid
            html.Div([
                html.Div([
                    html.Label("Distance to MRT Station", 
                              style={'color': '#555', 'fontWeight': '500', 'marginBottom': '0.5rem', 'display': 'block'}),
                    dcc.Input(
                        id='distance_to_mrt',
                        type='number',
                        placeholder='Enter distance in meters',
                        style={
                            'width': '100%',
                            'padding': '12px 16px',
                            'border': '2px solid #e0e6ed',
                            'borderRadius': '8px',
                            'fontSize': '1rem',
                            'transition': 'all 0.3s ease',
                            'backgroundColor': '#ffffff',
                            'boxSizing': 'border-box'
                        }
                    )
                ], style={'marginBottom': '1rem'}),
                
                html.Div([
                    html.Label("Number of Convenience Stores", 
                              style={'color': '#555', 'fontWeight': '500', 'marginBottom': '0.5rem', 'display': 'block'}),
                    dcc.Input(
                        id='num_convenience_stores',
                        type='number',
                        placeholder='Count of nearby stores',
                        style={
                            'width': '100%',
                            'padding': '12px 16px',
                            'border': '2px solid #e0e6ed',
                            'borderRadius': '8px',
                            'fontSize': '1rem',
                            'transition': 'all 0.3s ease',
                            'backgroundColor': '#ffffff',
                            'boxSizing': 'border-box'
                        }
                    )
                ], style={'marginBottom': '1rem'}),
                
                html.Div([
                    html.Div([
                        html.Label("Latitude", 
                                  style={'color': '#555', 'fontWeight': '500', 'marginBottom': '0.5rem', 'display': 'block'}),
                        dcc.Input(
                            id='latitude',
                            type='number',
                            placeholder='e.g., 25.0330',
                            style={
                                'width': '100%',
                                'padding': '12px 16px',
                                'border': '2px solid #e0e6ed',
                                'borderRadius': '8px',
                                'fontSize': '1rem',
                                'transition': 'all 0.3s ease',
                                'backgroundColor': '#ffffff',
                                'boxSizing': 'border-box'
                            }
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label("Longitude", 
                                  style={'color': '#555', 'fontWeight': '500', 'marginBottom': '0.5rem', 'display': 'block'}),
                        dcc.Input(
                            id='longitude',
                            type='number',
                            placeholder='e.g., 121.5654',
                            style={
                                'width': '100%',
                                'padding': '12px 16px',
                                'border': '2px solid #e0e6ed',
                                'borderRadius': '8px',
                                'fontSize': '1rem',
                                'transition': 'all 0.3s ease',
                                'backgroundColor': '#ffffff',
                                'boxSizing': 'border-box'
                            }
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
                ], style={'marginBottom': '1.5rem'}),
                
                # Predict Button
                html.Div([
                    html.Button(
                        'Predict Property Value',
                        id='predict_button',
                        n_clicks=0,
                        style={
                            'width': '100%',
                            'padding': '16px 24px',
                            'backgroundColor': '#3498db',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '8px',
                            'fontSize': '1.1rem',
                            'fontWeight': '600',
                            'cursor': 'pointer',
                            'transition': 'all 0.3s ease',
                            'textTransform': 'uppercase',
                            'letterSpacing': '0.5px',
                            'boxShadow': '0 4px 15px rgba(52, 152, 219, 0.3)'
                        }
                    )
                ], style={'textAlign': 'center'})
            ])
        ], style={
            'backgroundColor': '#ffffff',
            'padding': '1.5rem',
            'borderRadius': '16px',
            'boxShadow': '0 10px 30px rgba(0, 0, 0, 0.1)',
            'border': '1px solid #e8ecf4',
            'marginBottom': '2px'
        }),
        
        # Results Section
        html.Div([
            html.Div(
                id='prediction_output',
                style={
                    'textAlign': 'center',
                    'fontSize': '1.4rem',
                    'fontWeight': '600',
                    'color': '#27ae60',
                    'padding': '2rem',
                    'padding-bottom':'5px',
                    'backgroundColor': '#f8f9fa',
                    'borderRadius': '12px',
                    'border': '1px solid #e9ecef',
                    'minHeight': '80px',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center'
                }
            )
        ])
        
    ], style={
        'maxWidth': '600px',
        'margin': '0 auto',
        'padding': '0 20px'
    }),
    
    # Footer
    html.Footer([
        html.Div([
            html.P("© 2025 Laavanjan | Faculty of IT @B22", 
                   style={
                       'margin': '0',
                       'color': '#7f8c8d',
                       'fontSize': '0.9rem'
                   }),
            html.P("Powered by machine learning algorithms", 
                   style={
                       'margin': '0',
                       'color': '#95a5a6',
                       'fontSize': '0.8rem',
                       'marginTop': '0.25rem'
                   })
        ])
    ], style={
        'position': 'fixed',
        'bottom': '0',
        'left': '0',
        'right': '0',
        'textAlign': 'center',
        'padding': '1rem 20px',
        'backgroundColor': '#fafbfc',
        'borderTop': '1px solid #ecf0f1',
        'zIndex': '1000'
    })
    
], style={
    'fontFamily': '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif',
    'backgroundColor': '#f8f9fa',
    'minHeight': '100vh',
    'padding': '1rem 0',
    'paddingBottom': '80px'  # Space for fixed footer
})


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