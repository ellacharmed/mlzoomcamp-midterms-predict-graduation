import time
from urllib.error import URLError

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📊")

st.markdown("# Exploratory Data Analysis")
st.sidebar.header("Exploratory Data Analysis")
st.write(
    """Just some basic EDA.
(Data courtesy [Royce Kimmons](http://roycekimmons.com/tools/generated_data/graduation_rate).)"""
)

# Load the data from the URL
DATA_URL = ('https://raw.githubusercontent.com/ellacharmed/mlzoomcamp-midterms-predict-graduation/main/data/graduation_rate.csv')


@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    def lowercase(x): return str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache_data)")

GRADUATE_THRESHOLD = 5
# create a new column called 'target' that is set to 0 if years_to_graduate is below the graduate_threshold, so that I can use mean() on the negative label
data['target'] = [0 if years <=
                  GRADUATE_THRESHOLD else 1 for years in data['years to graduate']]

if st.checkbox('Show raw data'):
    st.subheader('Raw data, first 5 records')
    data = load_data(5)
    st.write(data)


tab1, tab2, tab3 = st.tabs(["SAT histogram", "GPAs histogram", "target pie"])

with tab1:

    st.header("Distribution of students' SAT total score")
    # Create a histogram of the "sat total score" column
    fig = px.histogram(data, x='sat total score',
                       title='Histogram of SAT Total Scores')

    # Display the histogram in the Streamlit app
    st.plotly_chart(fig)

with tab2:

    st.header("Comparing GPAs")
    # Create the histogram traces
    high_school_trace = go.Histogram(
        x=data['high school gpa'],
        name='High School GPA',
        opacity=0.7,
    )

    college_trace = go.Histogram(
        x=data['college gpa'],
        name='College GPA',
        opacity=0.5,
    )

    # Create the layout
    layout = go.Layout(
        title='Comparison of High School GPA and College GPA',
        xaxis_title='high school vs college',
        yaxis_title='Frequency',
    )

    # Create the subplots
    fig = make_subplots(1, 1)
    fig.add_trace(high_school_trace, row=1, col=1)
    fig.add_trace(college_trace, row=1, col=1)
    fig.update_layout(layout)

    # Display the plot
    st.plotly_chart(fig)

with tab3:

    st.header("Pie charts for the 'target'")
    # Create a pie chart of the target variable
    fig = px.pie(
        data_frame=data,
        names='target',
        values='years to graduate',
        title="Breakdown of 'target' positive vs negative labels",
    )
    st.plotly_chart(fig)

    # Create a donut chart of the years_to_graduate distribution for each target value
    labels = ["3", "4", "5", "6", "7", "8", "9", "10"]
    values0 = [1361, 1115, 372, 0, 0, 0, 0, 0]
    values1 = [0, 0, 0, 551, 340, 183, 66, 12]

    fig = make_subplots(1, 2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])

    fig.add_trace(go.Pie(
        labels=labels,
        values=values0,
        name="years to graduate <= 5"
    ),
        row=1, col=1
    )

    fig.add_trace(go.Pie(
        labels=labels,
        values=values1,
        name="years to graduate > 5"
    ),
        row=1, col=2
    )

    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    fig.update_layout(
        height=600, width=800,
        title_text="Breakdown of 'years_to_graduate' as percentage of the whole<br>for target==0 (<=5) vs target==1 (>5)",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='<=5', x=0.2, y=0.5, font_size=20, showarrow=False),
                     dict(text='>5', x=0.8, y=0.5, font_size=20, showarrow=False)]
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')

    st.plotly_chart(fig)
