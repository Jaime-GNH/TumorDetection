import plotly.graph_objects as go

from TumorDetection.utils.dict_classes import BaseUpdateLayout


def base_update_layout(fig: go.Figure) -> go.Figure:
    """
    Applies basic update layout to figure
    :param fig:
    :return:
    """
    fig = fig.update_layout(
        title=BaseUpdateLayout.get('title'),
        xaxis=BaseUpdateLayout.get('xaxis'),
        yaxis=BaseUpdateLayout.get('yaxis'),
        margin = BaseUpdateLayout.get('margin'),
        paper_bgcolor=BaseUpdateLayout.get('paper_bgcolor'),
        plot_bgcolor=BaseUpdateLayout.get('plot_bgcolor')
    )

    return fig
