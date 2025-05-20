import plotly.graph_objects as go
import wandb

def plot_connectivity(influence, x_chars, y_chars, run=None, title="Connectivity"):
    fig = go.Figure(data=go.Heatmap(
        z=influence,
        x=x_chars,
        y=y_chars,
        colorscale="Viridis",
        showscale=True,
        hovertemplate="<b>%{y}</b> depends on <b>%{x}</b><extra></extra>"
    ))
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        title=title
    )
    fig.show()
    if run:
        run.log({"connectivity": wandb.Html(fig.to_html(include_plotlyjs='cdn'))})
