#!/usr/bin/env python
# coding: utf-8

# # Project 1: Improvements

# ## Research Question:
# How have the temperatures across states in the US changed overtime and what patterns can be observed in different regions? Is there a correlation between temperature patterns and economic indicators in different regions? What factors may contribute to these changes?

# ## Intro:
# I will use the "Earth's Surface Termperature Data" dataset to conduct my research. 
# 
# For this project, I want to find the historical temperature changes for the US overtime, with data starting from the second industrial revolution (1900), since this marks a larger change in temperature. I want to be able to measure which states had the largest temperature changes overtime and which ones has the lowest temperature changes overtime. I also want to be able to measure the variability temperature changes overtime in a given US state to see if there is a temperature pattern.
# 
# Once finding these changes, I will use other outside variables for certain indicators in a state such as the population/state to determine if these outside economic factors are significant in the temperature findings overtime. My reason for the research question is that I want to be able to determine how/why certain states in the US have had larger/smaller temperature increases/decreases overtime and the economic indicators that might point to the answer. If the findings are economically important, these indicators can be good determinants of predicting a state temperature overtime and giverning policies can be centered around climate issues that may affect the long term temperature of a given state. Geographic regions are also important to consider, so I will consider those as well when interpreting the importance of my findings. 
# 
# I will use variables such as US state temp and find the percent change, time, population percent change and economic indicators. (In this section I only used population percent change, but I will continue to build up this report until I have a few economic indictors that may aid my discoveries). 

# ## Summary Findings:
# For the temperature change in the US states overtime, so far, I have discovered that there is a larger change in temperature increases for US states that are goegraphically more in  the north and western region. Furthermore, I have found that using my summary statistics data and mapping over the US states groupbed by the seasons, that the winter months in any given state are a lot more prone to temperature variability and have changed a lot more in temperature compared to the baseline years for the data.To make an accurate comparison overtime, I have used the mean temperature from years 1900-1950 as the "baseline years temperature", and compared this temperature to the temperatures against other years as benchmark. I have used these years 
# 
# This is a seasonal trend that I have uncovered, and have noticed that this trend is more prominent in norther and southern states. Meaning that in the winter time, nothern states have had a larger and more severe trend towards a decrease in temperature than the southern states. These temperatures have been more variable than the southern states. Also, in the spring time, we can see that there is a larger temperature decrase trend in nothern states that are closer to bodies of water. This may point to the fact large bodies of water may be related to larger temperature variation in the springtime in neighbouring states. 
# 
# For an economic interpretation, I have also included population percent changes per decade for each state and compared that against the percent temperature change in each decade per state. 

# ## Data Cleaning/Loading:

# In[2]:


# Lets clean and load the US temperature state data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
plt.rcParams["figure.figsize"]=(20,20)
# Uncomment following line to install on colab
get_ipython().system(' pip install qeds fiona geopandas xgboost gensim folium pyLDAvis descartes')
import geopandas as gpd
from shapely.geometry import Point
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objs as go


# In[5]:


# get the us data from the state data. 


# In[3]:


# look at all the countries that are available in the state data.
state = pd.read_csv('GlobalLandTemperaturesByState.csv')
state = state[state['AverageTemperature'].notnull()]
state.head()


# In[4]:


state = state.loc[state['Country'] == 'United States']
state = state.loc[state['dt'] >= '1900-01-01']
state.head()


# In[5]:


# let's find the temp of each state overtime. first set datetime object.  
state["dt"]=pd.to_datetime(state["dt"],format="%Y-%m-%d")
state.set_index("dt", inplace=True)
state.head()


# In[6]:


# now plot a new month column. 
state['Month'] = state.index.month
state['Year'] = state.index.year
state.head()
state.tail()


# In[7]:


# import decade percent change data of the population - I will try to make a comparison of population change 
# against the temperature change later on. 
pop = pd.read_csv('apportionment.csv') # imported from government census website. 
# link: https://www.census.gov/data/tables/time-series/dec/popchange-data-text.html
pop = pop.loc[pop['Geography Type'] == 'State']
pop = pop.loc[pop['Name'] != 'Puerto Rico']
pop.rename(columns={'Name': 'State'}, inplace=True)
pop


# # Summary Statistics
# Here we will find the summary statistics here to determine which variables we should use and which strategy will be best to answer the research question. 

# In[67]:


# Find the temp from the earliest year and 2013 for all the states -> see how the temperature changes. 
state_stats = state.groupby('State')['AverageTemperature'].agg('mean')
df = pd.DataFrame(state_stats)
# this is the average temperature per state starting from year 1900. 
state_stats2 = state.groupby('State')['AverageTemperature'].agg('std')
df2 = pd.DataFrame(state_stats2)
# merge. 


# In[68]:


merged_stats = df.merge(df2, on='State')
merged_stats.rename(columns={'AverageTemperature_y': 'std'}, inplace=True)
merged_stats.rename(columns={'AverageTemperature_x': 'mean_temp'}, inplace=True)
merged_stats


# Ok, now we have state average temperature and standard deviation summary statistics. Now I want to see if there is a regional pattern among states for the temp. However, geographically speaking, this is going to be a biased since cetain states in certain regions are more 'hot' than others because of their location (closer/further from the equator). Therefore, the classification of states in terms of the average temperature may not be that meaningful for the research question. To solve this problem, I can take the average temperature percent change of each state overtime and see some prominent changes in certain regions better to further aid my hypothesis. 
# 
# However, from this dataframe, we can analyze the standard deviation (variation) of the average temperature of each state, and maybe this will tell us something about the variation of the temperature for each state. After all climate change can not only be defined through the overtime increase in temperature, but also higher variability in temperature overtime. Maybe there are certain states in certain regions that experience more variability in temperature changes overtime. 
# 
# I will also look at seasonality, since for certain states (such as Alaska or Hawaii), the temperature changes might be more variable or systematically higher/lower on average due to its  location, so taking the aggregate temp. may not be the more efficient way to analyze long term trends. 
# 
# So overall, for a better look that the research question, I think its more meaningful to analyze the temperature change, standard devaation trends and any other potential variables overtime, instead of getting the overall average. We can get the summary stats for this next. 

# In[8]:


# add season variable to take this into consideration.
state["Season"] = state["Month"].apply(lambda x: "Winter" if x in [12, 1, 2] else ("Spring" if x in [3, 4, 5] else ("Summer" if x in [6, 7, 8] else "Fall")))
state


# In[9]:


baseline_years = range(1900, 1951) # baseline years.
baseline_data = state[state['Year'].isin(baseline_years)]
baseline_data 


# In[10]:


# Group data by state and season
grouped_data = baseline_data.groupby(['State', 'Season'])['AverageTemperature'].mean().reset_index()
grouped_data.columns = ['State', 'Season', 'BaselineAvgTemperature']

# Merge baseline data with original data to get temperature change
merged_data = pd.merge(state, grouped_data, on=['State', 'Season'])
merged_data['TemperatureChange'] = ((merged_data['AverageTemperature'] - merged_data['BaselineAvgTemperature'])/merged_data['BaselineAvgTemperature'])*100

# Group data by state and season and calculate average temperature change and std deviation
grouped_data = merged_data.groupby(['State', 'Season']).agg({'TemperatureChange': ['mean', 'std']}).reset_index()
grouped_data.columns = ['State', 'Season', 'AvgTemperatureChange', 'StdTemperatureChange']

# Pivot the data to get the desired output format
output_data = grouped_data.pivot(index='State', columns='Season')
output_data.columns = [col[0] + '_' + col[1] for col in output_data.columns.values]
output_data.reset_index(inplace=True)

# Rename the columns to remove 'TemperatureChange' from the column names
output_data.columns = ['State'] + [col.replace('TemperatureChange', '') + 'Percent Change' for col in output_data.columns[1:]]

# Display the output data
series = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN','MS', 'MO', 'MT', 'NE', 'NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']
output_data['State Abbrevation'] = series
output_data


# Ok, so now we have a dataframe that depicts the seasonal percent change of temperature and its standard deviation in each state.
# 
# I used the measure of percent change against the baseline years because it takes into account the size of temp. magnitude and it normalizes it according to the analysis of change instead of size. 
# 
# Instead of comparing the temperature change in a certain year to the base year, 1900, I decided that it would be more accurate for my research question to have baseline years from 1900-1950 instead, so I can find the average temperature between the baseline years and then find the percent change between this baseline temperature and a given temperature in a given state in a given season. I did this because climate in a given year can be variable and many different reasons can alter the climate in a given year, that may not carry over to the years after that (such as forest fires, unseasonably cold/warm ocean temperature or land temperature or other weather anomalies etc). Averaging out the temperature over a few years can provide a more accurate representation of long-term temperature trends. By using a baseline period that spans over multiple years, these short-term fluctuations are averaged out, providing a more stable and representative picture of the long-term trends in temperature. This baseline can then be used to assess changes in temperature over time and help inform decisions related to climate policy and adaptation.
# 
# I also grouped by seasons for each state to find the percent change. It is important to take into account seasons because temperature patterns vary depending on the season. For example, during the winter months, the temperature in a particular state or region may be significantly colder than during the summer months. By grouping the temperature data by season and calculating the average temperature change and standard deviation for each season, we can get a better understanding of how temperatures are changing over time for each season, rather than just looking at the overall temperature change. This can be important for understanding how climate change is affecting different regions in different seasons, and why this might be the case. This dataframe is packed with information so let's dig in!

# First, we can see without doing any type of formal analysis, that in the winters, there is A LOT more temperature variability compared to the other seasons. This is an interesting find, although not too surprising. One reason why it may be easier to see climate trends in states during winter months is that the temperature variations during this season tend to be larger than during other seasons, especially in northern regions of the US. This means that any changes or anomalies in temperature would be more apparent during the winter months.
# 
# However, it is also important to note that climate change impacts are not limited to changes in temperature alone, and other factors such as precipitation, extreme weather events, and sea level rise can also have significant impacts on different regions. Therefore, it would be important to analyze and consider these factors as well in order to fully understand the climate patterns and their potential impacts on different regions.

# ### Let's use the 'pop' data to complement our findings and further our analysis. We will use population percent change per decade in US states starting from 1910 to complement our data. My hypothesis is that the higher population density in states will be related to the temperature changes in that given state. Since higher population can lead to higher temperature changes, I predict a positive relation between the economic variable and the geographic variable. Lets test this!

# In[11]:


# merge with ouput_data for comparison in the later section
pop = pop.merge(output_data, on='State')
pop


# So as you may have noticed, the data in pop is time series data with an observation being the percent change in a given decade in a given state. To make a meaningful comparison against the population change for temperature change, I think it would be unwise to compare cross-sectional temperature percent change for each season against the given decade percent change in population. So, in the next part, I will basically group the 'state' data again for decade and state, instead of season and state to find the percrent change per decade for each state. This can be better compared against the population change data. In the mapping and figures section, I will attempt to discover a relationship amongst these variables for an economic interpretation for the my research question. 

# In[12]:


# First, convert the 'Year' column to a datetime object in state
state['Year'] = pd.to_datetime(state['Year'], format='%Y')

# Then, create a new column for the decade using the 'Year' column
state['Decade'] = state['Year'].apply(lambda x: int(x.year/10)*10)
state


# In[13]:


# Group the data by state and decade, and calculate the average temperature for each group
grouped_data = state.groupby(['State', pd.Grouper(key='Year', freq='10Y')])['AverageTemperature', 'Decade'].mean().reset_index()

grouped_data


# In[14]:


baseline_years = range(1900, 1951)
baseline_data = grouped_data[grouped_data['Decade'].isin(baseline_years)]
baseline_data


# In[15]:


merged_data = pd.merge(grouped_data, baseline_data, on=['State'])
merged_data['TemperatureChange'] = ((merged_data['AverageTemperature_x'] - merged_data['AverageTemperature_y'])/merged_data['AverageTemperature_y'])*100
merged_data


# In[17]:


# Group the data by state and year_x again, and calculate the average temperature change and std deviation for each group
final_data = merged_data.groupby(['State', 'Year_x']).agg({'TemperatureChange': ['mean', 'std']}).reset_index()
final_data.columns = ['State', 'Decade', 'AvgTemperatureChange', 'StdTemperatureChange']

# Pivot the data to get the desired output format
output_data2 = final_data.pivot(index='State', columns='Decade')
output_data2.columns = [col[0] + '_' + str(col[1].year) for col in output_data2.columns.values]
output_data2.reset_index(inplace=True)

# Rename the columns to remove 'TemperatureChange' from the column names
output_data2.columns = ['State'] + [col.replace('TemperatureChange', '') + 'Percent Change' for col in output_data2.columns[1:]]

# Display the output data
output_data2


# Ok, we can see now that we have columns that correpond to the temperature change in a given given decade in a state. I basically used the same strategy as 'ouput_data' to find the baseline years average temperature per decade and get the change of that against all other decades in a given state. I did this for all the baseline decades, comparing against each decade until 2010. I can plot figured to find patterns among these variables. 

# # Plots/Histograms/ Figures:
# ### For part 2) Visualization Section, I have included that in this part instead because I think it would be more meaningful for my data plots to be side by side. 
# This is one of the most important parts of discovering patterns overtime to figure out season trends. Let's continue.  

# First, let's create hovermaps to visualize our {output_data} dataframe, and check for any seasonable trends/outlier states. I will create four maps per season to visualize these trends for the average temperature percent change columns, as well as standard deviation of the percent change columns per season.  

# In[32]:


# just temp
for season in ['Fall', 'Winter', 'Summer', 'Spring']:
    fig = px.scatter(output_data, x='State', y=f'Avg_{season}Percent Change', color=f'Avg_{season}Percent Change',
                     title=f'Percent Change in Temperature for {season}', hover_name='State', hover_data=[f'Avg_{season}Percent Change'])
    fig.show()


# In[33]:


# just std
for season in ['Fall', 'Winter', 'Summer', 'Spring']:
    fig = px.scatter(output_data, x='State', y=f'Std_{season}Percent Change', color=f'Std_{season}Percent Change',
                     title=f' Std. of Percent Change in Temperature for {season}', hover_name='State', hover_data=[f'Std_{season}Percent Change'])
    fig.show()


# In[ ]:


# Further data analysis on population change and temperature change. 


# In[39]:


import plotly.express as px
for season in ['Fall', 'Spring', 'Winter', 'Summer']:

    fig = px.scatter(pop, x='Percent Change in Resident Population', y=f'Avg_{season}Percent Change', color='State Abbreviation', hover_name='State')
    fig.update_layout(title=f'Population Change vs. {season }Temperature Change')
    fig.show()


# As we can see its hard to see a relationship between time series population change data and cross sectional output_data. So we can create figures using output_data2. 

# # Part 2: Mapping of Data
# We will now map the data of the summary statistics tables that we have created earlier. As helpful as the scatterplots are, it will be slightly easier to visualize regional changes through a map. 

# ## THE MESSAGE: 
# The message or purpose of my data is to answer my research question, stated above. I want to find patterns in temperature changes overtime for the US states and somehow find economic variables that may be a good detemrinant of these temperature changes over certain regions. I think this can be economically significant since this research can aid policymakers to take into account climate when they are implementing economic policies.

# ## Maps and Interpretations:

# In[34]:


for season in ['Fall', 'Winter', 'Summer', 'Spring']:
    fig = px.choropleth(output_data, locations='State Abbreviation', locationmode="USA-states",
                        color=f'Avg_{season}Percent Change', scope="usa",
                        color_continuous_scale="RdBu", range_color=(-10, 10),
                        title=f'Percent Change in Temperature for {season}')
    fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))
    fig.show()


# In[35]:


# do the same for the standard deviation. 
for season in ['Fall', 'Winter', 'Summer', 'Spring']:
    fig = px.choropleth(output_data, locations='State Abbreviation', locationmode="USA-states",
                        color=f'Std_{season}Percent Change', scope="usa",
                        color_continuous_scale="RdBu", range_color=(0, output_data[f'Std_{season}Percent Change'].max()),
                        title=f'Standard Deviation of Temperature Change for {season}')
    fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))
    fig.show()


# In[38]:


# map the population change in pop df overall first, to see any overall trends. 
fig = px.choropleth(pop, locations='State Abbreviation', locationmode="USA-states",
                    color='Percent Change in Resident Population', scope="usa",
                    color_continuous_scale="RdBu", range_color=(-10, 10),
                    title=f'Percent Change in Population')
fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))
fig.show()


# #### I have a small aside here. I am going to map the AverageTemperatureUncertainty for each state here. The reason for this is because I want to check whether all my findings so far are accurate in terms of historic temperature measurements and information availability. I want to check that the AverageTemperatureUncertainty - which measures the temperature measurement accuracy overtime. I want to check that the accuracy of the measuring temperature overtime has indeed increased, due to better technology, information availabilty and climate change awareness overtime. 

# In[3]:


temp_df = pd.read_csv("GlobalLandTemperaturesByState.csv", parse_dates=["dt"])

# Filter to mainland US states
mainland_states = ["Alabama", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
mainland_df = temp_df[temp_df.State.isin(mainland_states)].reset_index(drop=True)

# Calculate the baseline period average temperatures for each state
baseline_start = "1900-01-01"
baseline_end = "1951-01-01"
baseline_temps = mainland_df[(mainland_df.dt >= baseline_start) & (mainland_df.dt <= baseline_end)].groupby("State").mean().reset_index()[["State", "AverageTemperatureUncertainty"]]
baseline_temps.columns = ["State", "BaselineTemp"]

# Merge the baseline temperatures with the temperature data
mainland_df = pd.merge(mainland_df, baseline_temps, on="State")

# Calculate the temperature change from the baseline
mainland_df["TempChange"] = mainland_df["AverageTemperatureUncertainty"] - mainland_df["BaselineTemp"]

# Group by state and calculate the mean temperature change per year
mainland_grouped = mainland_df.groupby(["State", pd.Grouper(key="dt", freq="Y")])["TempChange"].mean().reset_index()
mainland_grouped["Year"] = mainland_grouped["dt"].dt.year

# Pivot the data to have states as rows and years as columns
mainland_pivot = mainland_grouped.pivot(index="State", columns="Year", values="TempChange")

# Load the state shapefile
us_states_df = gpd.read_file("http://www2.census.gov/geo/tiger/GENZ2016/shp/cb_2016_us_state_5m.zip")
us_states_df = us_states_df.rename(columns={"NAME": "State"})

# Merge the temperature data with the state shapefile
merged = us_states_df.merge(mainland_pivot, on="State")

# Create the map
fig, ax = plt.subplots(figsize=(20, 15))

merged.plot(column=2013, cmap="coolwarm", linewidth=0.8, ax=ax, edgecolor="0.8")

ax.set_title("Mainland US Average Temperature Uncertainty Change 1900-2013", fontdict={"fontsize": "16", "fontweight" : "bold"})
ax.set_axis_off()

# Create the legend
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=merged[2013].min(), vmax=merged[2013].max()))
sm._A = []
cbar = fig.colorbar(sm)

plt.show()


# Bluer colored states represent a decrease in the average temperature uncertainty from the baseline period to the year specified in the merged.plot() function. This means that the range of possible temperature values for these states became smaller over time, indicating more reliable temperature measurements.
# 
# However, it is important to note that the range of possible temperature values is just one aspect of uncertainty in temperature measurements, and it does not necessarily mean that the actual temperature change was smaller in these states. There could be other sources of uncertainty in the temperature measurements that are not captured by this analysis.
# 
# As we can see there is a larger cluster of states in the midwest that have a lower temperature measurement uncertaintly compared to some of the north-eastern states.
# 
# There could be several factors that contribute to the pattern of lower temperature uncertainty in the Midwestern states and higher temperature uncertainty in the Northeastern region.
# 
# One possible explanation is that the Midwestern states have a relatively stable climate with less variation in temperature over time compared to other regions, which could lead to less uncertainty in temperature measurements. However this differs from the map generated above. However, just because the temperature change of the northwest states has increased more overtime does not mean that that the temperature itself is stable. It could just mean that for these states, the temperature measurements may have immproved frrom the baseline year for a more accurate measurement. Additionally, the Midwestern states may have had more consistent and reliable temperature monitoring stations over time, leading to more consistent measurements and less uncertainty.
# 
# On the other hand, the Northeastern region has more complex topography, with mountains and coastlines that can influence local weather patterns, and may have had more variation in temperature over time, leading to higher uncertainty in temperature measurements. Additionally, the Northeastern region is more densely populated and urbanized, which can create heat islands and other local climate effects that can further complicate temperature measurements.
# 
# It is important to note that these are just potential explanations and other factors could also be contributing to the observed patterns. This map only shows the temperature measurement changes not the temperature changes itself, so it is important to keep that into account. My reasoning for including this map is to show that temperature monitoring has drastically improved overtime in states that may not have been previously less populated. This is important to understand since we need to know to what extent the temperature measurements are certain for our findings. 

# 

# 

# # Conclusion:

# While there are many factors that can contribute to temperature changes in individual states, there are some economic indicators that may help explain why certain states have experienced more dramatic temperature changes over time than others. Here are a few possibilities:
# 
# Industrial activity: States with high levels of industrial activity may experience more significant temperature changes due to increased greenhouse gas emissions. This is because industrial activity often involves the burning of fossil fuels, which release large amounts of carbon dioxide and other greenhouse gases into the atmosphere. States with large manufacturing sectors, such as Ohio and Michigan, may therefore experience more significant temperature changes than states with less industrial activity.
# 
# Energy consumption: States with high levels of energy consumption may also experience more significant temperature changes over time. This is because the production and consumption of energy often involves the burning of fossil fuels, which contribute to greenhouse gas emissions. States with high levels of energy consumption per capita, such as Wyoming and North Dakota, may therefore experience more significant temperature changes than states with lower levels of energy consumption.
# 
# Agriculture: Changes in agricultural practices over time may also contribute to temperature changes in certain states. For example, increased irrigation and the use of fertilizers can lead to changes in land surface temperature. States with large agricultural sectors, such as California and Texas, may therefore experience more significant temperature changes than states with smaller agricultural sectors.
# 
# Of course, these are just a few examples of economic factors that may contribute to temperature changes in individual states. Other factors, such as geography and climate, may also play a role. Ultimately, the causes of temperature changes are complex and multifaceted, and are influenced by a wide range of economic, social, and environmental factors.
# 
# For the next part, I will use some of these variables in my data to find more trends like I did with population percent change data in this section. 
# 
# I hope to uncover more trends and answer my reserach question by the next time. 

# 
