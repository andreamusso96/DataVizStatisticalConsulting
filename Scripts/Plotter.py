import os
from datetime import datetime

import plotly.graph_objects as go
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class DetectionResult:
    def __init__(self, traffic_time_series_data, daily_component_traffic_time_series_data, sleep_change_points, sleep_scores):
        self.traffic_time_series_data = traffic_time_series_data
        self.daily_component_traffic_time_series_data = daily_component_traffic_time_series_data
        self.sleep_change_points = sleep_change_points
        self.sleep_scores = sleep_scores

    @staticmethod
    def load_example():
        path_to_current_folder = os.path.dirname(os.path.abspath(__file__))
        path_to_data_folder = f'{os.path.dirname(path_to_current_folder)}/Data'
        traffic_time_series_data = pd.read_csv(f'{path_to_data_folder}/traffic_data.csv', index_col=0, parse_dates=True)
        daily_component_traffic_time_series_data = pd.read_csv(f'{path_to_data_folder}/daily_component_traffic.csv', index_col=0, parse_dates=True)
        sleep_change_points = pd.read_csv(f'{path_to_data_folder}/sleep_change_points.csv')
        tuples = [(datetime.fromisoformat(d).date(), i) for d, i in zip(sleep_change_points['date'], sleep_change_points['sleep_state'])]
        sleep_change_points.index = pd.MultiIndex.from_tuples(tuples, names=['date', 'sleep_state'])
        sleep_change_points.drop(columns=['date', 'sleep_state'], inplace=True)
        sleep_scores = pd.read_csv(f'{path_to_data_folder}/sleep_score.csv', index_col=0, parse_dates=True)
        return DetectionResult(traffic_time_series_data, daily_component_traffic_time_series_data, sleep_change_points, sleep_scores)


class DetectionResultPlot:
    def __init__(self, detection_result: DetectionResult):
        self.traffic_time_series_data = detection_result.traffic_time_series_data
        self.daily_component_traffic_time_series_data = detection_result.daily_component_traffic_time_series_data
        self.sleep_change_points = detection_result.sleep_change_points
        self.sleep_scores = detection_result.sleep_scores
        self.location_ids = [c for c in self.sleep_change_points.columns if not c.startswith('unc_')]

    def plot(self):
        figures = [self._make_plot_location(location_id=location_id) for location_id in self.location_ids]
        return figures

    def _make_plot_location(self, location_id):
        fig = go.Figure()
        fig.add_trace(self._get_trace_traffic_location(location_id=location_id))
        fig.add_trace(self._get_trace_traffic_daily_component_location(location_id=location_id))
        fig.add_trace(self._get_trace_sleep_scores(location_id=location_id))
        fig.add_trace(self._get_trace_sleep_change_points_location(location_id=location_id, sleep_state='asleep'))
        fig.add_trace(self._get_trace_sleep_change_points_location(location_id=location_id, sleep_state='awake'))
        fig = DetectionResultPlot._set_layout(fig=fig, location_id=location_id)
        return fig

    @staticmethod
    def _set_layout(fig, location_id):
        font = dict(size=18)
        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='Traffic')
        fig.update_layout(title_text=f'Sleep patterns for location {location_id}', xaxis_rangeslider_visible=True,
                          height=700, font=font)
        return fig

    def _get_trace_traffic_location(self, location_id):
        scaled_data = DetectionResultPlot._scale_data(data=self.traffic_time_series_data)
        trace_traffic_location = go.Scatter(x=scaled_data.index, y=scaled_data[location_id], name='Traffic')
        return trace_traffic_location

    def _get_trace_traffic_daily_component_location(self, location_id):
        scaled_data = DetectionResultPlot._scale_data(data=self.daily_component_traffic_time_series_data)
        trace_traffic_daily_component_location = go.Scatter(x=scaled_data.index, y=scaled_data[location_id], name='Daily Component')
        return trace_traffic_daily_component_location

    def _get_trace_sleep_scores(self, location_id):
        scaled_data = DetectionResultPlot._scale_data(data=self.sleep_scores)
        trace_sleep_scores = go.Scatter(x=scaled_data.index, y=scaled_data[location_id], name='Sleep Scores')
        return trace_sleep_scores

    def _get_trace_sleep_change_points_location(self, location_id, sleep_state):
        sleep_change_points_x = self.sleep_change_points[location_id].xs(sleep_state, level="sleep_state").values
        sleep_change_points_y = 0.4 * np.ones(len(sleep_change_points_x))
        sleep_change_points_uncertainty = 1 + 3*self.sleep_change_points[f'unc_{location_id}'].xs(sleep_state, level="sleep_state").values
        trace_sleep_change_points_location = go.Scatter(x=sleep_change_points_x, y=sleep_change_points_y, name=f'{sleep_state}', mode='markers', marker=dict(color='Yellow', symbol='line-ns', size=250, line=dict(width=sleep_change_points_uncertainty)))
        return trace_sleep_change_points_location

    @staticmethod
    def _scale_data(data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns, index=data.index)