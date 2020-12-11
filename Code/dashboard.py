def instantiate_dash(server):
    import dash  
    import dash_core_components as dcc
    import dash_html_components as html
    from datetime import date
    import pickle
    import os
    import glob
    print('')
    print('Instantiating DASH Server')
    print('')

    path = r"C:\Users\sands\OneDrive\Desktop\II_MSDS_Data_Practicum\Data"
    filename = '\\' + str(date.today().strftime("%m_%d_%Y"))

    files = os.listdir(path + filename)

    app = dash.Dash(__name__, server=server, url_base_pathname='/dash_dashboard/')

    # I still need to figure out how to dynamically create the graphs here... I initially attempted to include multiple graphs, but this did not work well.
    # So in order to include multiple graphs saved from an individual file, there will need to be further research conducted
    print('Loading Dashboard...')
    print('')
    with open(path + filename + '\\' + files[0], 'rb') as f:
        graph1 = pickle.load(f)

    with open(path + filename + '\\' + files[1], 'rb') as f:
        graph2 = pickle.load(f)

    app.layout = html.Div(children=[
        html.H1(
            children='Claims Fraud Model Independent Feature Comparison Dashboard',
            style={'textAlign':'center'}
        ),
        dcc.Graph(figure=graph1),
        dcc.Graph(figure=graph2)
    ])
    return