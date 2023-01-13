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


def plot_dist(G_losses, D_losses, **kwargs):
    epoch = kwargs.get("epoch")
    iters = kwargs.get("iters")
    df = pd.DataFrame(
        data={
            "G": G_losses,
            "D": D_losses,
        }
    )
    fig = px.line(
        df, y=df.columns, title="Generator and Discriminator Loss During Training"
    )
    fig.write_image(f"loss_E{epoch}_I{iters}.png")


def plot_2_seqs(args, trues: list, preds: list, **kwargs):
    fill_type = kwargs.get("fill_type")
    colors = [
        "#D95D39",
        "#F0A202",
        "#3587A4",
        "#7B5C91",
        "#579B4E",
        "#725E54",
        "#2A2B2A",
        "#CE7B91",
    ]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=len(trues),
            y=trues,
            mode="lines",
            name="Trues",
            marker_color="#7B5C91",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=len(preds),
            y=preds,
            mode="lines",
            name="Preds",
            marker_color="#A6ACEC",
        )
    )
    fig.write_image(f"{args.image_path}/eval_{fill_type}.png")


def plot_evaluation(args, fakes_dict: dict[list], real: list, mask: list, **kwargs):
    iters = kwargs.get("iters")
    path = kwargs.get("path")
    colors = [
        "#D95D39",
        "#F0A202",
        "#3587A4",
        "#7B5C91",
        "#579B4E",
        "#725E54",
        "#2A2B2A",
        "#CE7B91",
    ]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(args.seq_len)),
            y=real,
            mode="lines+markers",
            name="Real",
            marker_color="#A6ACEC",
        )
    )
    for i, (name, fake) in enumerate(fakes_dict.items()):
        # fig.add_trace(
        #     go.Scatter(
        #         x=list(range(args.seq_len)),
        #         y=fake,
        #         mode="lines",
        #         name=name,
        #         marker_color=colors[i],
        #     )
        # )
        fig.add_trace(
            go.Scatter(
                x=list(range(args.seq_len)),
                y=[f if mask[i] == 0 else None for i, f in enumerate(fake)],
                mode="markers",
                name=f"{name}(generate)",
                marker_color=colors[i],
            )
        )
    fig.write_image(f"{args.image_path}/eval_{iters}.png")
