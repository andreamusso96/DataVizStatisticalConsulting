from dash import Dash, dcc, html
import dash_mantine_components as dmc

from Scripts.Plotter import DetectionResultPlot, DetectionResult

external_stylesheets = [dmc.theme.DEFAULT_COLORS]
app = Dash(__name__, external_stylesheets=external_stylesheets)


if __name__ == '__main__':
    detection_result = DetectionResult.load_example()
    detection_result_plot = DetectionResultPlot(detection_result=detection_result)
    figures = detection_result_plot.plot()

    app.layout = html.Div(
        style={'overflowY': 'scroll', 'height': '1200px'},  # This enables scrolling
        children=[dcc.Graph(figure=fig) for fig in figures]
    )

    app.run_server(debug=True)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
