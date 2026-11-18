"""Tools the agent can call."""


def registry_tools() -> list:
    from adsci.tools.chart_tool import CreateChartTool
    from adsci.tools.profile_tool import ProfileTool
    from adsci.tools.python_tool import RunPythonTool
    from adsci.tools.sql_tool import RunSQLTool

    return [ProfileTool(), RunSQLTool(), RunPythonTool(), CreateChartTool()]
