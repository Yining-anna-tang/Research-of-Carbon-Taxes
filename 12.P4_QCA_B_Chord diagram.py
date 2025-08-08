#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 示例数据 第一个和弦图（红蓝配色）

import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

# 示例数据
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/和弦图1.csv',encoding = "GBK")
# 转换为DataFrame
df = pd.DataFrame(data, columns=['source', 'target', 'value'])

# 创建和弦图
chord = hv.Chord(df)

# RdBu、GnBu、Winter
# 设置图表选项
chord.opts(
    opts.Chord(
         cmap='RdBu',
        edge_cmap='RdBu',
        edge_color=hv.dim('source').str(),
        labels='source',
        node_color=hv.dim('index').str(),
        edge_line_width=hv.dim('value')*0.1
    )
)


# In[59]:


# 示例数据 第二个和弦图（淡粉色）

import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

# 示例数据
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/和弦图2.csv',encoding = "GBK")
# 转换为DataFrame
df = pd.DataFrame(data, columns=['source', 'target', 'value'])

# 创建和弦图
chord = hv.Chord(df)

# RdBu、GnBu、Winter
# 设置图表选项
# 设置淡红粉色系配色方案
# 设置更柔和的淡粉色方案
chord.opts(
    opts.Chord(
        cmap='Blues',
        edge_cmap='GnBu', 
        edge_color=hv.dim('source').str(),
        labels='source',
        node_color=hv.dim('index').str(),
        edge_line_width=hv.dim('value')*0.00001,
    )
)


# In[5]:


import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

# 示例数据
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/和弦图3.csv',encoding = "GBK")
# 转换为DataFrame
df = pd.DataFrame(data, columns=['source', 'target', 'value'])

# 创建和弦图
chord = hv.Chord(df)

# 1.RdBu、GnBu、3.BuGn
# 设置图表选项
chord.opts(
    opts.Chord(
        cmap='BuGn',
        edge_cmap='BuGn',
        edge_color=hv.dim('source').str(),
        labels='source',
        node_color=hv.dim('index').str(),
        edge_line_width=hv.dim('value')*0.1
    )
)


# In[8]:


import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

# 示例数据
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/和弦图4.csv',encoding = "GBK")
# 转换为DataFrame
df = pd.DataFrame(data, columns=['source', 'target', 'value'])

# 创建和弦图
chord = hv.Chord(df)

# 颜色测试：GnBu
# 设置图表选项
chord.opts(
    opts.Chord(
        cmap='GnBu',
        edge_cmap='viridis',
        edge_color=hv.dim('source').str(),
        labels='source',
        node_color=hv.dim('index').str(),
        edge_line_width=hv.dim('value')*0.1
    )
)


# In[9]:


import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

# 示例数据
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/和弦图4.csv',encoding = "GBK")
# 转换为DataFrame
df = pd.DataFrame(data, columns=['source', 'target', 'value'])

# 创建和弦图
chord = hv.Chord(df)

# 1.RdBu、GnBu、3.BuGn
# 设置图表选项
chord.opts(
    opts.Chord(
        cmap='Reds',
        edge_cmap='BrBG',
        edge_color=hv.dim('source').str(),
        labels='source',
        node_color=hv.dim('index').str(),
        edge_line_width=hv.dim('value')*0.1
    )
)


# In[2]:


import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

# 示例数据
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1_城市韧性/和弦图4.csv',encoding = "GBK")
# 转换为DataFrame
df = pd.DataFrame(data, columns=['source', 'target', 'value'])

# 创建和弦图
chord = hv.Chord(df)

# 1.RdBu、GnBu、3.BuGn
# 设置图表选项
chord.opts(
    opts.Chord(
        cmap='pink',
        edge_cmap='pink',
        edge_color=hv.dim('source').str(),
        labels='source',
        node_color=hv.dim('index').str(),
        edge_line_width=hv.dim('value')*0.1
    )
)


# In[57]:


import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

# 示例数据
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/和弦图4.csv',encoding = "GBK")
# 转换为DataFrame
df = pd.DataFrame(data, columns=['source', 'target', 'value'])

# 创建和弦图
chord = hv.Chord(df)

# 1.RdBu、GnBu、3.BuGn
# 设置图表选项
chord.opts(
    opts.Chord(
        cmap='copper',
        edge_cmap='summer',
        edge_color=hv.dim('source').str(),
        labels='source',
        node_color=hv.dim('index').str(),
        edge_line_width=hv.dim('value')*0.1
    )
)


# In[6]:





# In[ ]:




