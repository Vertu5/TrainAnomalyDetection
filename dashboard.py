import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px

# Charger les données depuis le fichier CSV
df = pd.read_csv('ar41_for_ulb_mini.csv', delimiter=';')

# Charger les données depuis le fichier des anomalies
df_anomalies = pd.read_csv('anomaly.csv', delimiter=';')

# Identifier les trains ayant des anomalies
trains_with_anomalies = df_anomalies['mapped_veh_id'].unique()

# Filtrer les données pour les trains ayant des anomalies
df_filtered = df[df['mapped_veh_id'].isin(trains_with_anomalies)].copy()
df_filtered[['lat', 'lon']] = df_filtered[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')

# Initialiser l'application Dash
app = Dash(__name__)

# Définir les options d'onglets
tabs = dcc.Tabs(
    id='tabs', value='oil', children=[
    dcc.Tab(label='Oil', value='oil'),
    dcc.Tab(label='Water', value='water'),
    dcc.Tab(label='Air', value='air'),
    dcc.Tab(label='Anomaly Map', value='map'),
    dcc.Tab(label='RPM', value='rpm'),
    dcc.Tab(label='PC1vsPC2', value='pc1vspc2'),
    dcc.Tab(label='Water vs Oil', value='watvsoil'),
])

# Définir la mise en page du tableau de bord
app.layout = html.Div(children=[
    html.H1(children='Temperature and Pressure Dashboard'),

    # Sélecteur d'onglet
    html.Div(tabs),

    # Contenu de l'onglet sélectionné
    html.Div(id='tabs-content')
])

# Callback pour mettre à jour le contenu de l'onglet en fonction de la sélection de l'utilisateur
@app.callback(Output('tabs-content', 'children'), [Input('tabs', 'value')])
def update_tab(selected_tab):
    if selected_tab == 'oil':
        return [
            dcc.Graph(
                id='oil-pressure-chart',
                figure=px.line(df_filtered, x='timestamps_UTC', y=['RS_E_OilPress_PC1', 'RS_E_OilPress_PC2'],
                               title='Oil Pressure Over Time')
            ),
            dcc.Graph(
                id='oil-temperature-chart',
                figure=px.line(df_filtered, x='timestamps_UTC', y=['RS_T_OilTemp_PC1', 'RS_T_OilTemp_PC2'],
                               title='Oil Temperature Over Time')
            )
        ]
    elif selected_tab == 'water':
        return [
            dcc.Graph(
                id='water-temperature-chart',
                figure=px.line(df_filtered, x='timestamps_UTC', y=['RS_E_WatTemp_PC1', 'RS_E_WatTemp_PC2'],
                               title='Water Temperature Over Time')
            )
        ]
    elif selected_tab == 'air':
        return [
            dcc.Graph(
                id='air-temperature-chart',
                figure=px.line(df_filtered, x='timestamps_UTC', y=['RS_E_InAirTemp_PC1', 'RS_E_InAirTemp_PC2'],
                               title='Air Temperature Over Time')
            ),
        ]
    elif selected_tab == 'map':
        # Ajouter les points normaux pour les trains avec anomalies (en bleu)
        fig = px.scatter_mapbox(
            df_filtered,
            lat='lat',
            lon='lon',
            hover_data=['timestamps_UTC', 'RS_E_InAirTemp_PC1', 'RS_E_InAirTemp_PC2', 'RS_E_OilPress_PC1', 'RS_E_OilPress_PC2', 'RS_E_RPM_PC1', 'RS_E_RPM_PC2', 'RS_E_WatTemp_PC1', 'RS_E_WatTemp_PC2', 'RS_T_OilTemp_PC1', 'RS_T_OilTemp_PC2'],
            zoom=5,
            color_discrete_sequence=['blue']  # Définir la couleur des points normaux en bleu
        )

        # Créer une carte avec Plotly Express pour les anomalies (en rouge)
        fig.add_trace(px.scatter_mapbox(
            df_anomalies,
            lat='lat',
            lon='lon',
            hover_data=['timestamps_UTC', 'RS_E_InAirTemp_PC1', 'RS_E_InAirTemp_PC2', 'RS_E_OilPress_PC1', 'RS_E_OilPress_PC2', 'RS_E_RPM_PC1', 'RS_E_RPM_PC2', 'RS_E_WatTemp_PC1', 'RS_E_WatTemp_PC2', 'RS_T_OilTemp_PC1', 'RS_T_OilTemp_PC2'],
            zoom=5,
            color_discrete_sequence=['red']  # Définir la couleur des anomalies en rouge
        ).data[0])

        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=5,
            mapbox_center={"lat": df_filtered['lat'].mean(), "lon": df_filtered['lon'].mean()},
        )

        return [dcc.Graph(id='anomaly-map', figure=fig)]  
    elif selected_tab == 'rpm':
        return [
            dcc.Graph(
                id='rpm-chart',
                figure=px.line(df_filtered, x='timestamps_UTC', y=['RS_E_RPM_PC1', 'RS_E_RPM_PC2'],
                               title='RPM Over Time')
            ),
        ]
    elif selected_tab == 'pc1vspc2':
        fig = px.scatter(df_filtered, x='RS_E_RPM_PC1', y='RS_E_RPM_PC2', title='RPM vs RPM', color_discrete_sequence=['blue'])
        fig.add_trace(px.scatter(df_anomalies, x='RS_E_RPM_PC1', y='RS_E_RPM_PC2', title='RPM vs RPM', color_discrete_sequence=['red']).data[0])

        fig2 = px.scatter(df_filtered, x='RS_E_InAirTemp_PC1', y='RS_E_InAirTemp_PC2', title='Air 1 vs Air 2', color_discrete_sequence=['blue'])
        fig2.add_trace(px.scatter(df_anomalies, x='RS_E_InAirTemp_PC1', y='RS_E_InAirTemp_PC2', title='Air 1 vs Air 2', color_discrete_sequence=['red']).data[0])

        fig3 = px.scatter(df_filtered, x='RS_E_WatTemp_PC1', y='RS_E_WatTemp_PC2', title='Water 1 vs Water 2', color_discrete_sequence=['blue'])
        fig3.add_trace(px.scatter(df_anomalies, x='RS_E_WatTemp_PC1', y='RS_E_WatTemp_PC2', title='Water 1 vs Water 2', color_discrete_sequence=['red']).data[0])

        fig4 = px.scatter(df_filtered, x='RS_T_OilTemp_PC1', y='RS_T_OilTemp_PC2', title='Oil 1 vs Oil 2', color_discrete_sequence=['blue'])
        fig4.add_trace(px.scatter(df_anomalies, x='RS_T_OilTemp_PC1', y='RS_T_OilTemp_PC2', title='Oil 1 vs Oil 2', color_discrete_sequence=['red']).data[0])

        fig5 = px.scatter(df_filtered, x='RS_E_OilPress_PC1', y='RS_E_OilPress_PC2', title='Oil 1 vs Oil 2', color_discrete_sequence=['blue'])
        fig5.add_trace(px.scatter(df_anomalies, x='RS_E_OilPress_PC1', y='RS_E_OilPress_PC2', title='Oil 1 vs Oil 2', color_discrete_sequence=['red']).data[0])

        return [dcc.Graph(id='rpmvsrpm-chart', figure=fig), dcc.Graph(id='airvsair-chart', figure=fig2), dcc.Graph(id='watvswat-chart', figure=fig3), dcc.Graph(id='oilvsoiltemp-chart', figure=fig4), dcc.Graph(id='oilvsoilpress-chart', figure=fig5)]
    elif selected_tab == 'watvsoil':
        fig = px.scatter(df_filtered, x='RS_E_WatTemp_PC1', y='RS_T_OilTemp_PC1', title='Water vs Oil', color_discrete_sequence=['blue'])
        fig.add_trace(px.scatter(df_anomalies, x='RS_E_WatTemp_PC1', y='RS_T_OilTemp_PC1', title='Water vs Oil', color_discrete_sequence=['red']).data[0])

        fig2 = px.scatter(df_filtered, x='RS_E_WatTemp_PC2', y='RS_T_OilTemp_PC2', title='Water vs Oil', color_discrete_sequence=['blue'])
        fig2.add_trace(px.scatter(df_anomalies, x='RS_E_WatTemp_PC2', y='RS_T_OilTemp_PC2', title='Water vs Oil', color_discrete_sequence=['red']).data[0])

        return [dcc.Graph(id='watvsoil-chart', figure=fig), dcc.Graph(id='watvsoil2-chart', figure=fig2)]

# Exécuter l'application Dash
if __name__ == '__main__':
    app.run_server(debug=True)
