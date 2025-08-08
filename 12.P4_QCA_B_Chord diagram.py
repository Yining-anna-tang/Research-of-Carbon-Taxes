import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

def plot_chord(csv_path, cmap, edge_cmap, edge_line_scale=0.1):
    data = pd.read_csv(csv_path, encoding="GBK")
    df = pd.DataFrame(data, columns=['source', 'target', 'value'])
    chord = hv.Chord(df)
    chord.opts(
        opts.Chord(
            cmap=cmap,
            edge_cmap=edge_cmap,
            edge_color=hv.dim('source').str(),
            labels='source',
            node_color=hv.dim('index').str(),
            edge_line_width=hv.dim('value') * edge_line_scale
        )
    )
    return chord

# Examples of usage:

# Chord plot 1
chord1 = plot_chord(
    r'/Users/yiningtang/PycharmProjects/pythonProject1/和弦图1.csv',
    cmap='RdBu',
    edge_cmap='RdBu',
    edge_line_scale=0.1
)

# Chord plot 2 (lighter blue)
chord2 = plot_chord(
    r'/Users/yiningtang/PycharmProjects/pythonProject1/和弦图2.csv',
    cmap='Blues',
    edge_cmap='GnBu',
    edge_line_scale=0.00001
)

# Chord plot 3
chord3 = plot_chord(
    r'/Users/yiningtang/PycharmProjects/pythonProject1/和弦图3.csv',
    cmap='BuGn',
    edge_cmap='BuGn',
    edge_line_scale=0.1
)

# Chord plot 4 (color test)
chord4 = plot_chord(
    r'/Users/yiningtang/PycharmProjects/pythonProject1/和弦图4.csv',
    cmap='GnBu',
    edge_cmap='viridis',
    edge_line_scale=0.1
)

# Chord plot 5 (reds)
chord5 = plot_chord(
    r'/Users/yiningtang/PycharmProjects/pythonProject1/和弦图4.csv',
    cmap='Reds',
    edge_cmap='BrBG',
    edge_line_scale=0.1
)

# Chord plot 6 (pink)
chord6 = plot_chord(
    r'/Users/yiningtang/PycharmProjects/pythonProject1_城市韧性/和弦图4.csv',
    cmap='pink',
    edge_cmap='pink',
    edge_line_scale=0.1
)

# Chord plot 7 (copper-summer)
chord7 = plot_chord(
    r'/Users/yiningtang/PycharmProjects/pythonProject1/和弦图4.csv',
    cmap='copper',
    edge_cmap='summer',
    edge_line_scale=0.1
)

# To display one chord diagram (example):
# hv.show(chord1)
