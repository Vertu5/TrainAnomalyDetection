import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px

# Charger les données depuis les CSV
df = pd.read_csv('ar41_for_ulb_mini.csv', delimiter=';')
df_anomalies = pd.read_csv('anomaly.csv', delimiter=';')

# Créer une application Dash
app = dash.Dash(__name__)

# Créer la mise en page
app.layout = html.Div([
    dcc.Graph(
        id='map-graph'
    )
])

# Callback pour afficher la carte avec les points normaux en bleu et les anomalies en rouge
@app.callback(
    dash.dependencies.Output('map-graph', 'figure'),
    [dash.dependencies.Input('map-graph', 'id')]
)
def update_map(_):
    # Créer une carte avec Plotly Express pour les données normales (en bleu)
    fig = px.scatter_mapbox(
        df[~df.index.isin(df_anomalies.index)],  # Filtrer les données normales
        lat='lat',
        lon='lon',
        hover_data=['timestamps_UTC', 'RS_E_InAirTemp_PC1', 'RS_E_InAirTemp_PC2', 'RS_E_OilPress_PC1', 'RS_E_OilPress_PC2', 'RS_E_RPM_PC1', 'RS_E_RPM_PC2', 'RS_E_WatTemp_PC1', 'RS_E_WatTemp_PC2', 'RS_T_OilTemp_PC1', 'RS_T_OilTemp_PC2'],
        zoom=8,
        color_discrete_sequence=['blue']  # Définir la couleur des points normaux en bleu
    )

    # Ajouter les anomalies (en rouge)
    fig.add_trace(px.scatter_mapbox(
        df_anomalies,
        lat='lat',
        lon='lon',
        hover_data=['timestamps_UTC', 'RS_E_InAirTemp_PC1', 'RS_E_InAirTemp_PC2', 'RS_E_OilPress_PC1', 'RS_E_OilPress_PC2', 'RS_E_RPM_PC1', 'RS_E_RPM_PC2', 'RS_E_WatTemp_PC1', 'RS_E_WatTemp_PC2', 'RS_T_OilTemp_PC1', 'RS_T_OilTemp_PC2'],
        zoom=8,
        color_discrete_sequence=['red']  # Définir la couleur des anomalies en rouge
    ).data[0])

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=5,
        mapbox_center={"lat": df['lat'].mean(), "lon": df['lon'].mean()},
    )

    return fig

# Exécuter l'application
if __name__ == '__main__':
    app.run_server(debug=True)
