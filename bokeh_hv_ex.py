import numpy as np
import pandas as pd
import hvplot.streamz
import holoviews as hv

from streamz import Stream
from streamz.dataframe import DataFrame


renderer = hv.renderer("bokeh")

# Sets up a streaming dataframe
stream = Stream()
index = pd.DatetimeIndex([])
ex = pd.DataFrame({"x": [], "y": [], "z": []}, columns=["x", "y", "z"], index=[])

df = DataFrame(stream, example=ex)
cumulative = df.cumsum()[["x", "z"]]

# Declaring the plots to be used
line = cumulative.hvplot.line(width=400)
scatter = cumulative.hvplot.scatter(width=400)
bars = df.groupby("y").sum().hvplot.bar(width=400)
box = df.hvplot.box(width=400)
kde = df.x.hvplot.kde(width=400)


# composing the plots
layout = (line * scatter + bars + box + kde).cols(2)


# set the callback function w/ streaming data
def emit():
    now = pd.datetime.now()
    delta = np.timedelta64(500, "ms")
    index = pd.date_range(np.datetime64(now) - delta, now, freq="100ms")
    df = pd.DataFrame(
        {
            "x": np.random.randn(len(index)),
            "y": np.random.randint(0, 10, len(index)),
            "z": np.random.randn(len(index)),
        },
        columns=["x", "y", "z"],
        index=index,
    )
    stream.emit(df)


# render the layout w/ Bokeh server Document and attach the callback function
doc = renderer.server_doc(layout)
doc.title = "Streamz HoloViews based Plotting API Bokeh App Demo"
doc.add_periodic_callback(emit, 500)
