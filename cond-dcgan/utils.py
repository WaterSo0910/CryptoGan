import plotly.graph_objects as go
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px


def plot_dis(args, fake: list, real: list, **kwargs):
    epoch = kwargs.get("epoch")
    iters = kwargs.get("iters")
    df = pd.DataFrame(
        {
            f"Generate {args.seq_len} hrs interval Prices": fake,
            f"Original {args.seq_len} hrs interval Price)": real,
        }
    )
    fig = ff.create_distplot(
        [df[c] for c in df.columns],
        df.columns,
        colors=["#A56CC1", "#A6ACEC", "#63F5EF"],
        bin_size=0.2,
    )
    fig.write_image(f"{args.dist_path}/E{epoch}_I{iters}.png")


def plot_res(args, fake: list, real: list, mask: list, **kwargs):
    epoch = kwargs.get("epoch")
    iters = kwargs.get("iters")
    # plot_df = pd.DataFrame(data={
    #     "x": list(range(timeseries_size)),
    #     'fake': fake,
    #     'real': real,
    #     "mask": mask,
    # })
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(args.seq_len)),
            y=real,
            mode="lines",
            name="Real",
            marker_color="#A6ACEC",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(args.seq_len)),
            y=fake,
            mode="lines",
            name="Fake",
            marker_color="#9263c2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(args.seq_len)),
            y=[f if mask[i] == 0 else None for i, f in enumerate(fake)],
            mode="markers",
            name="Generate",
            marker_color="#66428a",
        )
    )
    # fig = px.line(plot_df, x='x', y=plot_df.columns,
    #             title='custom tick labels')
    fig.write_image(f"{args.image_path}/E{epoch}_I{iters}.png")


def plot_dist(G_losses, D_losses):
    df = pd.DataFrame(
        data={
            "G": G_losses,
            "D": D_losses,
        }
    )
    fig = px.line(
        df, y=df.columns, title="Generator and Discriminator Loss During Training"
    )
    fig.write_image(f"loss.png")
