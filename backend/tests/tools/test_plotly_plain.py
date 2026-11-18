import plotly.express as px

from adsci.kernel.plotly_json import figure_to_plain


def test_bdata_arrays_become_lists():
    fig = px.bar(x=["a", "b"], y=[1.5, 2.0])
    plain = figure_to_plain(fig)
    assert plain["data"][0]["y"] == [1.5, 2.0]
    assert plain["data"][0]["x"] == ["a", "b"]


def test_dict_input_with_shape():
    import base64

    import numpy as np

    arr = np.arange(6, dtype="i4").reshape(2, 3)
    d = {"data": [{"z": {"dtype": "i4", "bdata": base64.b64encode(arr.tobytes()).decode(), "shape": "2, 3"}}]}
    assert figure_to_plain(d)["data"][0]["z"] == [[0, 1, 2], [3, 4, 5]]
