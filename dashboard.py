import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px

def count_anomalies_per_train(df_anomalies):
    anomalies_count = df_anomalies['mapped_veh_id'].value_counts().to_dict()
    return anomalies_count

# Charger les données depuis le fichier CSV
df = pd.read_csv('morceau_1.csv', delimiter=';')

# Charger les données depuis le fichier des anomalies
df_anomalies = pd.read_csv('anomalies_if.csv', delimiter=',')

#convertir les temperatures de Kelvin en Celsius
df_anomalies['RS_E_InAirTemp_PC1'] = df_anomalies['RS_E_InAirTemp_PC1'] - 273.15
df_anomalies['RS_E_InAirTemp_PC2'] = df_anomalies['RS_E_InAirTemp_PC2'] - 273.15
df_anomalies['RS_E_WatTemp_PC1'] = df_anomalies['RS_E_WatTemp_PC1'] - 273.15
df_anomalies['RS_E_WatTemp_PC2'] = df_anomalies['RS_E_WatTemp_PC2'] - 273.15
df_anomalies['RS_T_OilTemp_PC1'] = df_anomalies['RS_T_OilTemp_PC1'] - 273.15
df_anomalies['RS_T_OilTemp_PC2'] = df_anomalies['RS_T_OilTemp_PC2'] - 273.15



# Identifier les trains ayant des anomalies
trains_with_anomalies = df_anomalies['mapped_veh_id'].unique()

# Filtrer les données pour ne garder que le train avec l'id 194
df_anomalies_filtered = df_anomalies[df_anomalies['mapped_veh_id'] == 194]
df_filtered = df[df['mapped_veh_id'] == 194]

# Utilisez la fonction pour générer les options du menu déroulant avec le nombre d'anomalies
anomalies_count = count_anomalies_per_train(df_anomalies)
dropdown_options = [
    {'label': f"Train ID: {train_id} - Anomalies: {count}", 'value': train_id}
    for train_id, count in anomalies_count.items()
]


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



# Ajoutez ceci à la mise en page du tableau de bord, avant les onglets
train_dropdown = dcc.Dropdown(
    id='train-dropdown',
    options=dropdown_options,
    value=list(anomalies_count.keys())[0],  # Sélectionnez par défaut le premier ID de train
    clearable=False,  # Empêchez la suppression de la sélection
)

# Ajouter le menu déroulant au layout
app.layout = html.Div(children=[
    html.H1(children='Temperature and Pressure Dashboard'),

    # Sélecteur de train
    html.Label('Select Train:'),
    train_dropdown,

    # Sélecteur d'onglet
    html.Div(tabs),

    # Contenu de l'onglet sélectionné
    html.Div(id='tabs-content')
])

# Callback pour mettre à jour les données en fonction du train sélectionné
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value'), Input('train-dropdown', 'value')]
)

def update_tab(selected_tab, selected_train):
    # Filtrer les données en fonction du train sélectionné
    df_filtered = df[df['mapped_veh_id'] == selected_train]
    df_anomalies_filtered = df_anomalies[df_anomalies['mapped_veh_id'] == selected_train]

    if selected_tab == 'oil':
        fig_1 = px.scatter( df_filtered, x='timestamps_UTC', y=['RS_T_OilTemp_PC1'], title='Oil Temperature 1', color_discrete_sequence=['blue'])
        fig_1.add_trace(px.scatter(df_anomalies_filtered, x='timestamps_UTC', y=['RS_T_OilTemp_PC1'], title='Oil Temperature 1', color_discrete_sequence=['red']).data[0])

        fig_2 = px.scatter( df_filtered, x='timestamps_UTC', y=['RS_T_OilTemp_PC2'], title='Oil Temperature 2', color_discrete_sequence=['blue'])
        fig_2.add_trace(px.scatter(df_anomalies_filtered, x='timestamps_UTC', y=['RS_T_OilTemp_PC2'], title='Oil Temperature 2', color_discrete_sequence=['red']).data[0])

        fig_3 = px.scatter( df_filtered, x='timestamps_UTC', y=['RS_E_OilPress_PC1'], title='Oil Pressure 1', color_discrete_sequence=['blue'])
        fig_3.add_trace(px.scatter(df_anomalies_filtered, x='timestamps_UTC', y=['RS_E_OilPress_PC1'], title='Oil Pressure 1', color_discrete_sequence=['red']).data[0])

        fig_4 = px.scatter( df_filtered, x='timestamps_UTC', y=['RS_E_OilPress_PC2'], title='Oil Pressure 2', color_discrete_sequence=['blue'])
        fig_4.add_trace(px.scatter(df_anomalies_filtered, x='timestamps_UTC', y=['RS_E_OilPress_PC2'], title='Oil Pressure 2', color_discrete_sequence=['red']).data[0])

        return [dcc.Graph(id='oil-temperature1-chart', figure=fig_1), dcc.Graph(id='oil-temperature2-chart', figure=fig_2), dcc.Graph(id='oil-pressure1-chart', figure=fig_3), dcc.Graph(id='oil-pressure2-chart', figure=fig_4)]
    elif selected_tab == 'water':
        fig_1 = px.scatter( df_filtered, x='timestamps_UTC', y=['RS_E_WatTemp_PC1'], title='Water Temperature 1', color_discrete_sequence=['blue'])
        fig_1.add_trace(px.scatter(df_anomalies_filtered, x='timestamps_UTC', y=['RS_E_WatTemp_PC1'], title='Water Temperature 1', color_discrete_sequence=['red']).data[0])

        fig_2 = px.scatter( df_filtered, x='timestamps_UTC', y=['RS_E_WatTemp_PC2'], title='Water Temperature 2', color_discrete_sequence=['blue'])
        fig_2.add_trace(px.scatter(df_anomalies_filtered, x='timestamps_UTC', y=['RS_E_WatTemp_PC2'], title='Water Temperature 2', color_discrete_sequence=['red']).data[0])

        return [dcc.Graph(id='water-temperature1-chart', figure=fig_1), dcc.Graph(id='water-temperature2-chart', figure=fig_2)]
    elif selected_tab == 'air':
        fig_1 = px.scatter( df_filtered, x='timestamps_UTC', y=['RS_E_InAirTemp_PC1'], title='Air Temperature 1', color_discrete_sequence=['blue'])
        fig_1.add_trace(px.scatter(df_anomalies_filtered, x='timestamps_UTC', y=['RS_E_InAirTemp_PC1'], title='Air Temperature 1', color_discrete_sequence=['red']).data[0])

        fig_2 = px.scatter( df_filtered, x='timestamps_UTC', y=['RS_E_InAirTemp_PC2'], title='Air Temperature 2', color_discrete_sequence=['blue'])
        fig_2.add_trace(px.scatter(df_anomalies_filtered, x='timestamps_UTC', y=['RS_E_InAirTemp_PC2'], title='Air Temperature 2', color_discrete_sequence=['red']).data[0])

        return [dcc.Graph(id='air-temperature1-chart', figure=fig_1), dcc.Graph(id='air-temperature2-chart', figure=fig_2)]
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
            df_anomalies_filtered,
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
        fig1 = px.scatter(df_filtered, x='timestamps_UTC', y=['RS_E_RPM_PC1'], title='RPM 1', color_discrete_sequence=['blue'])
        fig1.add_trace(px.scatter(df_anomalies_filtered, x='timestamps_UTC', y=['RS_E_RPM_PC1'], title='RPM 1', color_discrete_sequence=['red']).data[0])

        fig2 = px.scatter(df_filtered, x='timestamps_UTC', y=['RS_E_RPM_PC2'], title='RPM 2', color_discrete_sequence=['blue'])
        fig2.add_trace(px.scatter(df_anomalies_filtered, x='timestamps_UTC', y=['RS_E_RPM_PC2'], title='RPM 2', color_discrete_sequence=['red']).data[0])

        return [dcc.Graph(id='rpm1-chart', figure=fig1), dcc.Graph(id='rpm2-chart', figure=fig2)]
    elif selected_tab == 'pc1vspc2':
        fig = px.scatter(df_filtered, x='RS_E_RPM_PC1', y='RS_E_RPM_PC2', title='RPM vs RPM', color_discrete_sequence=['blue'])
        fig.add_trace(px.scatter(df_anomalies_filtered, x='RS_E_RPM_PC1', y='RS_E_RPM_PC2', title='RPM vs RPM', color_discrete_sequence=['red']).data[0])

        fig2 = px.scatter(df_filtered, x='RS_E_InAirTemp_PC1', y='RS_E_InAirTemp_PC2', title='Air 1 vs Air 2', color_discrete_sequence=['blue'])
        fig2.add_trace(px.scatter(df_anomalies_filtered, x='RS_E_InAirTemp_PC1', y='RS_E_InAirTemp_PC2', title='Air 1 vs Air 2', color_discrete_sequence=['red']).data[0])

        fig3 = px.scatter(df_filtered, x='RS_E_WatTemp_PC1', y='RS_E_WatTemp_PC2', title='Water 1 vs Water 2', color_discrete_sequence=['blue'])
        fig3.add_trace(px.scatter(df_anomalies_filtered, x='RS_E_WatTemp_PC1', y='RS_E_WatTemp_PC2', title='Water 1 vs Water 2', color_discrete_sequence=['red']).data[0])

        fig4 = px.scatter(df_filtered, x='RS_T_OilTemp_PC1', y='RS_T_OilTemp_PC2', title='Oil 1 vs Oil 2', color_discrete_sequence=['blue'])
        fig4.add_trace(px.scatter(df_anomalies_filtered, x='RS_T_OilTemp_PC1', y='RS_T_OilTemp_PC2', title='Oil 1 vs Oil 2', color_discrete_sequence=['red']).data[0])

        fig5 = px.scatter(df_filtered, x='RS_E_OilPress_PC1', y='RS_E_OilPress_PC2', title='Oil 1 vs Oil 2', color_discrete_sequence=['blue'])
        fig5.add_trace(px.scatter(df_anomalies_filtered, x='RS_E_OilPress_PC1', y='RS_E_OilPress_PC2', title='Oil 1 vs Oil 2', color_discrete_sequence=['red']).data[0])

        return [dcc.Graph(id='rpmvsrpm-chart', figure=fig), dcc.Graph(id='airvsair-chart', figure=fig2), dcc.Graph(id='watvswat-chart', figure=fig3), dcc.Graph(id='oilvsoiltemp-chart', figure=fig4), dcc.Graph(id='oilvsoilpress-chart', figure=fig5)]
    elif selected_tab == 'watvsoil':
        fig = px.scatter(df_filtered, x='RS_E_WatTemp_PC1', y='RS_T_OilTemp_PC1', title='Water vs Oil', color_discrete_sequence=['blue'])
        fig.add_trace(px.scatter(df_anomalies_filtered, x='RS_E_WatTemp_PC1', y='RS_T_OilTemp_PC1', title='Water vs Oil', color_discrete_sequence=['red']).data[0])

        fig2 = px.scatter(df_filtered, x='RS_E_WatTemp_PC2', y='RS_T_OilTemp_PC2', title='Water vs Oil', color_discrete_sequence=['blue'])
        fig2.add_trace(px.scatter(df_anomalies_filtered, x='RS_E_WatTemp_PC2', y='RS_T_OilTemp_PC2', title='Water vs Oil', color_discrete_sequence=['red']).data[0])

        return [dcc.Graph(id='watvsoil-chart', figure=fig), dcc.Graph(id='watvsoil2-chart', figure=fig2)]

# Exécuter l'application Dash
if __name__ == '__main__':
    app.run_server(debug=True)
