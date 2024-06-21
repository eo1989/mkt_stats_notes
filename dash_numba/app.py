# noqa:PD902,NPY002
import dash
from dash import html, dcc, callback, Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
from app_funcs import transform_data

app = dash.Dash(__name__)


df = pd.DataFrame(
    {
        "c1": np.random.normal(size=1_000_000),
        "c2": np.random.normal(size=1_000_000),
        "c3": np.random.normal(size=1_000_000),
    }
)

app.layout = html.Div(
    [
        dcc.Store(id="data", data=df.to_json(orient="split")),
        html.H1("My app with numba"),
        html.Div(
            [
                html.Label("Enter number of rows to remove:"),
                dcc.Input(id="row-input", type="number", min=0, max=len(df), step=1),
                html.Button("Update Data", id="update-button"),
            ]
        ),
        dcc.Graph(id="myplot"),
    ]
)


@app.callback(
    Output("myplot", "figure"),
    Input("update-button", "n_clicks"),
    State("row-input", "value"),
    State("data", "data"),
)
def update_graph(n_clicks, n_rows, json_data):
    if n_clicks is None:
        return dash.exceptions.PreventUpdate

    dff = pd.read_json(json_data, orient="split")

    if n_rows is not None:
        dff = dff.iloc[:-n_rows]

    # transform the data
    x = dff.c1.to_numpy()
    y = dff.c2.to_numpy()
    z = dff.c3.to_numpy()
    result = transform_data(x, y, z)

    # create the figure
    x_axis = list(range(len(result)))
    fig = px.line(
        x=x_axis, y=result, labels={"x": "x", "y": "res"}, title="Numba Ex Data"
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)