"""Tools the agent can call."""


def registry_tools() -> list:
    from adsci.tools.chart_tool import CreateChartTool
    from adsci.tools.model_tool import TrainModelTool
    from adsci.tools.profile_tool import ProfileTool
    from adsci.tools.python_tool import RunPythonTool
    from adsci.tools.sql_tool import RunSQLTool
    from adsci.tools.stats_tool import StatsTestTool
    from adsci.tools.timeseries_tools import AnomalyTool, ForecastTool, SegmentTool

    return [ProfileTool(), RunSQLTool(), RunPythonTool(), CreateChartTool(), StatsTestTool(), TrainModelTool(), ForecastTool(), AnomalyTool(), SegmentTool()]
