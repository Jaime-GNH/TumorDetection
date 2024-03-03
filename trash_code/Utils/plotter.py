
def plot_graph(graph, path=None, show=True, **kwargs):
    """

    :param graph:
    :param path:
    :param show:
    :param kwargs:
    :return:
    """
    pos = graph.pos.to('cpu').numpy()
    edges = graph.edge_index.to('cpu')
    edges_show = kwargs.get('edges_show', [0, pos.max()//2, pos.max()-1,
                                           len(pos)//2, len(pos)//2 + (pos.max()//2),
                                           len(pos)//2 + pos.max(),
                                           len(pos)-pos.max(), len(pos) - (pos.max()//2), len(pos)])
    edge_filter = torch.isin(edges[0, :], torch.tensor(edges_show))
    fig = go.Figure(
        data=[go.Scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            mode='markers',
            marker=dict(
                color=graph.x[:, 0].to(torch.int32).to('cpu').numpy()/255.,
                colorscale='gray',
                showscale=False,
                size=15,
                line=dict(width=0)
            ),
            hoverinfo='skip',
            showlegend=False
        )],
        layout=go.Layout(
            annotations=[dict(
                x=pos[es][0],
                y=pos[es][1],
                ax=pos[ed][0],
                ay=pos[ed][1],
                axref='x',
                ayref='y',
                arrowwidth=2,
                arrowcolor='red',
                visible=True
            ) for es, ed in zip(edges[0, :][edge_filter], edges[1, :][edge_filter])]
        )
    )

    fig = base_update_layout(fig)
    fig = fig.update_layout(
        title=dict(text=kwargs.get('title', f'Graph Plot.')),
        yaxis=dict(scaleanchor='x'), #,  autorange='reversed', visible=False),
        xaxis=dict(visible=False)
    )
    if path is not None:
        if kwargs.get('format', 'svg') == 'html':
            pio.write_html(fig, path + '.html', auto_open=kwargs.get('auto_open', False))
        else:
            fig.write_image(path + '.' + kwargs.get('format', 'png'),
                            format=kwargs.get('format', 'png'))
    if show:
        fig.show(renderer=kwargs.get('renderer', 'browser'))
    return fig
