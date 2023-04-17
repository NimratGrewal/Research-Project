#!/usr/bin/env python
# coding: utf-8

# # PROJECT 1

# ## Research Question:
# How have the temperatures across states in the US changed overtime and what patterns can be observed in different regions? Is there a correlation between temperature patterns and economic indicators in different regions? What factors may contribute to these changes?

# ##### Google Drive link (to view interactive maps) : https://drive.google.com/file/d/1cGmB5OXDHyk6p8I7rvhwGk-ReAr9z5jk/view?usp=sharing Note: Please download the file in the google drive link and then view it to see the interactive maps and graphs.  

# ## 1.1 Introduction
# I will use the "Earth's Surface Termperature Data" dataset to conduct my research. 
# 
# For this project, I want to find the historical temperature changes for the US overtime, with data starting from the second industrial revolution (1900), since this marks a larger change in temperature. I want to be able to measure which states had the largest temperature changes overtime and which ones has the lowest temperature changes overtime. I also want to be able to measure the variability temperature changes overtime in a given US state to see if there is a temperature pattern.
# 
# Once finding these changes, I will use other outside variables for certain indicators in a state such as the population/state to determine if these outside economic factors are significant in the temperature findings overtime. My reason for the research question is that I want to be able to determine how/why certain states in the US have had larger/smaller temperature increases/decreases overtime and the economic indicators that might point to the answer. If the findings are economically important, these indicators can be good determinants of predicting a state temperature overtime and giverning policies can be centered around climate issues that may affect the long term temperature of a given state. Geographic regions are also important to consider, so I will consider those as well when interpreting the importance of my findings. 
# 
# I will use variables such as US state temp and find the percent change, time, population percent change and economic indicators.

# ### Summary Findings
# 
# For the temperature change in the US states overtime, so far, I have discovered that there is a larger change in temperature increases for US states that are geographically in the north and western region. Furthermore, I have found that using my summary statistics data and mapping over the US states groupbed by the seasons, that the winter months in any given state are a lot more prone to temperature variability and have changed a lot more in temperature compared to the baseline years for the data. To make an accurate comparison overtime, I have used the mean temperature from years 1900-1950 as the "baseline years temperature", and compared this temperature to the temperatures against other years as benchmark.
# 
# This is a seasonal trend that I have uncovered, and have noticed that this trend is more prominent in northern and southern states. Meaning that in the winter time, nothern states have had a larger and more severe trend towards a decrease in temperature than the southern states. Also, in the spring and fall time, we can see that there is a larger temperature decrase trend in nothern states that are closer to bodies of water. This may point to the fact large bodies of water may be related to larger temperature variation in the spring and fall in neighbouring states. 
# For an economic interpretation, I have also included population percent changes per decade for each state and compared that against the percent temperature change in each decade per state. While I could not uncover the linear trends because I had too little data for population percentage change, I noticed that some states, while having a large population overtime, do not have large temperature changes, indicating that population may not play a large part in the states that have less temperature variability than geographic location. 

# ## 1.2 Data Cleaning/Loading

# In[ ]:


# Lets clean and load the US temperature state data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
plt.rcParams["figure.figsize"]=(20,20)
# Uncomment following line to install on colab
get_ipython().system(' pip install qeds fiona geopandas xgboost gensim folium pyLDAvis descartes')
get_ipython().system(' pip install linearmodels')
get_ipython().system(' pip install sklearn')
import geopandas as gpd
from shapely.geometry import Point
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objs as go
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from linearmodels.iv import IV2SLS
from sklearn import linear_model
get_ipython().system(' pip install stargazer')
from stargazer.stargazer import Stargazer
from IPython.core.display import HTML


# In[2]:


# get the us data from the state data. 


# In[65]:


# look at all the countries that are available in the state data.
state = pd.read_csv('GlobalLandTemperaturesByState.csv')
state = state[state['AverageTemperature'].notnull()]
state.head()


# In[66]:


state = state.loc[state['Country'] == 'United States']
state = state.loc[state['dt'] >= '1900-01-01']
state.head()


# In[67]:


# let's find the temp of each state overtime. first set datetime object.  
state["dt"]=pd.to_datetime(state["dt"],format="%Y-%m-%d")
state.set_index("dt", inplace=True)
state.head()


# In[68]:


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


# ## 1.3 Summary Statistics
# Here we will find the summary statistics here to determine which variables we should use and which strategy will be best to answer the research question. 

# In[8]:


# Find the temp from the earliest year and 2013 for all the states -> see how the temperature changes. 
state_stats = state.groupby('State')['AverageTemperature'].agg('mean')
df = pd.DataFrame(state_stats)
# this is the average temperature per state starting from year 1900. 
state_stats2 = state.groupby('State')['AverageTemperature'].agg('std')
df2 = pd.DataFrame(state_stats2)
# merge. 


# In[9]:


merged_stats = df.merge(df2, on='State')
merged_stats.rename(columns={'AverageTemperature_y': 'std'}, inplace=True)
merged_stats.rename(columns={'AverageTemperature_x': 'mean_temp'}, inplace=True)
merged_stats


# In[10]:


merged_stats.reset_index(inplace=True)


# In[11]:


plot = merged_stats.plot(x = 'State', y =['mean_temp', 'std'], kind='bar', 
                         figsize=(15,10), ylabel= 'Temperature (in Degrees Celcius)')


# This plot displays the average temperature and standard deviation of the average temperature overtime using 1900-2013 historical monthly averages. From this we can determine how certain states may behave in terms of temeperature changes over the course of our analysis. For example, we can see that certain states have a lower standard deviation, hence we may observe these states have lower temperature variablity in any given year, while other states that have high standard devation may be have more variation in temperature changes. 
# 
# Using this analysis, we can try to determine if these results from the graph are a byproduct of the geographic location of the state or whether there are other factors that can help predict temperature changes/variability.

# Ok, now we have state average temperature and standard deviation summary statistics. Now I want to see if there is a regional pattern among states for the temp. However, geographically speaking, this is going to be a biased since cetain states in certain regions are more 'hot' than others because of their location (closer/further from the equator). Therefore, the classification of states in terms of the average temperature may not be that meaningful for the research question. To solve this problem, I can take the average temperature percent change of each state overtime and see some prominent changes in certain regions better to further aid my hypothesis. 
# 
# However, from this dataframe, we can analyze the standard deviation (variation) of the average temperature of each state, and maybe this will tell us something about the variation of the temperature for each state. After all climate change can not only be defined through the overtime increase in temperature, but also higher variability in temperature overtime. Maybe there are certain states in certain regions that experience more variability in temperature changes overtime. 
# 
# I will also look at seasonality, since for certain states (such as Alaska or Hawaii), the temperature changes might be more variable or systematically higher/lower on average due to its  location, so taking the aggregate temp. may not be the more efficient way to analyze long term trends. 
# 
# So overall, for a better look that the research question, I think its more meaningful to analyze the temperature change, standard devaation trends and any other potential variables overtime, instead of getting the overall average. We can get the summary stats for this next. 

# ##### In this section, we will create a dataframe that can look at seasonal trends of the state temperate overtime. 

# In[69]:


# add season variable to take this into consideration.
state["Season"] = state["Month"].apply(lambda x: "Winter" if x in [12, 1, 2] else ("Spring" if x in [3, 4, 5] else ("Summer" if x in [6, 7, 8] else "Fall")))
state


# In[13]:


baseline_years = range(1900, 1951) # baseline years.
baseline_data = state[state['Year'].isin(baseline_years)]


# In[14]:


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
output_data['State Abbreviation'] = series
output_data.head()


# Ok, so now we have a dataframe that depicts the seasonal percent change of temperature and its standard deviation in each state.
# 
# The measure of percent change is used against the baseline years because it takes into account the size of temperature magnitude and it normalizes it according to the analysis of change instead of size. 
# 
# Instead of comparing the temperature change in a certain year to the base year, 1900, it would be more accurate for the research question to have baseline years from 1900-1950 instead, so we can find the average temperature between the baseline years and then find the percent change between this baseline temperature and a given temperature in a given state in a given season. This approach makes sense because climate in a given year can be variable and many different reasons can alter the climate in a given year, that may not carry over to the years after that (such as forest fires, unseasonably cold/warm ocean temperature or land temperature or other weather anomalies etc). Averaging out the temperature over a few years can provide a more accurate representation of long-term temperature trends. By using a baseline period that spans over multiple years, these short-term fluctuations are averaged out, providing a more stable and representative picture of the long-term trends in temperature. This baseline can then be used to assess changes in temperature over time and help inform decisions related to climate policy and adaptation.
# 
# Each state was also grouped by seasons to find the percent change in a given season. It is important to take into account seasons because temperature patterns vary depending on the season. For example, during the winter months, the temperature in a particular state or region may be significantly colder than during the summer months. By grouping the temperature data by season and calculating the average temperature change and standard deviation for each season, we can get a better understanding of how temperatures are changing over time for each season, rather than just looking at the overall temperature change. This can be important for understanding how climate change is affecting different regions in different seasons, and why this might be the case. This dataframe is packed with information so let's dig in!

# In the winters, there is a lot more temperature variability compared to the other seasons. This is an interesting find, although not too surprising. One reason why it may be easier to see climate trends in states during winter months is that the temperature variations during this season tend to be larger than during other seasons, especially in northern regions of the US. This means that any changes or anomalies in temperature would be more apparent during the winter months.
# 
# However, it is also important to note that climate change impacts are not limited to changes in temperature alone, and other factors such as precipitation, extreme weather events, and sea level rise can also have significant impacts on different regions. Therefore, it would be important to analyze and consider these factors as well in order to fully understand the climate patterns and their potential impacts on different regions.

# #### Creating multiple regressions per state to analyze seasonal/annual components

# I predict there is a linear trend relationship between my independent and depdendent variable. According to this graph:

# In[15]:


plt.style.use('seaborn')
state.plot(x='Year', y='AverageTemperature', kind='scatter')
plt.show()


# We can see there is a linear relationship between year and temperature change overtime. This makes sense since temperature increments very slowly, and can be easily prdicted in the near future using an OLS estimete. To continue, in order to denote how important seasonal trends are in predicting temperature for a given state, a multiple regression model can be helpful, for each given state. 
# 
# The independent variable will be each season, using dummy variables for each season in order to create categorical data. Next, the dependent variable be the temperature in degrees celcius. We want to find out if there are certain seasons that are better predictors of the temperature in a given state, and why this might be the case. 
# This analysis will give some concrete statisical analysis to shed some light on the questoin that we are trying to answer.

# In[16]:


merged_data # use this data, instead of <output_data> for linear regression because of more observations 
# our <output_data> column is for mapping analysis only. 


# In[17]:


dummy_merged = merged_data.copy(deep=True)


# In[18]:


dummy_merged['Winter'] = dummy_merged['Season'].apply(lambda x: 1 if x == 'Winter' else 0) 
dummy_merged['Spring'] = dummy_merged['Season'].apply(lambda x: 1 if x == 'Spring' else 0) 
dummy_merged['Fall'] = dummy_merged['Season'].apply(lambda x: 1 if x == 'Fall' else 0)
dummy_merged['Summer'] = dummy_merged['Season'].apply(lambda x: 1 if x == 'Summer' else 0)


# In[19]:


grouped3 = dummy_merged.groupby('State')


# In[20]:



lst1 = []
for groups in grouped3.groups:
    state2 = grouped3.get_group(groups)
    X = state2[['Winter', 'Fall', 'Spring', 'Summer', 'Year']]
    Y = state2[['AverageTemperature']]
 
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    lst1.append(model)
    predictions = model.predict(X) 


# In[21]:


stargazer_2 = Stargazer(lst1)
lst = []
state_lst = []
for _ in range(51):
    lst.append(1)
    
for group in grouped3.groups:
    state_lst.append(group)

stargazer_2.custom_columns(state_lst, lst)
stargazer_2.show_model_numbers(False)
stargazer_2.show_degrees_of_freedom(False)


HTML(stargazer_2.render_html())


# Since we used regression analysis on each state, our model can be introduced like this: 
# 
# $$
# {AverageTemperature}_i = \beta_0 + \beta_1 {year}_i + \beta_2 {summer}_i + \beta_3 {spring}_i + \beta_4 {winter}_i + \beta_5 {fall}_i + u_i
# $$
# 
# where:
# 
# - $ \beta_0 $ is the intercept of the linear trend line on the
#   y-axis  
# - $ \beta_1 $ is the year variable explaining how much temperature increases by (in Celcius) annually, on average in that state
# - $ \beta_{2..5} $ are the dummy variables that represent the average temperature that year in that state, or 0 if    it is not that season. So, on average, how much the temperature changes in Celcius during *that season* compared to the baseline years' average temperature in *that season*, taking into account yearly changes and controlling for other seasons. 
# - $ u_i $ is a random error term (deviations of observations from
#   the linear trend due to factors not included in the model)  
# 

# Here, each model corresponds to a regression run for a state. Grouping the regressions by seasons and state is the most reasonable way to go given the question we are trying to answer. We want to isolate the effects of seasonality for any given state, and see how the results of the states differ, i.e. which state may not have any or all statistically significant dummy variables, and how their geographic location might be the case.  
# 
# I will do an interpretation of the results for Alaska for example, and then point out some general trends: 
# 
# The multiple regression model shows that the season variables (Winter, Fall, Spring, and Summer) and the year variable have a significant impact on the average temperature in Alaska.
# 
# The R-squared value of 0.768 indicates that the model explains 76.8% of the variance in the dependent variable (Average Temperature). This means that the model is a reasonably good fit for the data.
# 
# The coefficients for the season variables show how much the average temperature in Alaska changes for each season, controlling for the other seasons, and year. The Winter coefficient of 19.416 indicates that, on average, after controlling for other seasonal trends and yearly temperature changes, the change in temperature in Alaska during winter is 19.4 degrees Celsius lower than the average temperature. Similarly, the Fall coefficient of 5.655 indicates that the temperature during the fall season is 5.655 degrees Celsius lower than the average temperature. 
# 
# The coefficient for the Year variable is 0.012, which means that for every one-year increase in the Year variable, the temperature in Alaska increases by an average of 0.012 degrees Celsius, controlling for seasonality.
# 
# The p-values for all of the seasonal coefficients is significant at 1%, the p-value for the annual coeffienct is significant at 5%. which means that they are statistically significant at the 95% confidence level. This suggests that the explanatory variables are significant predictors of the average temperature in Alaska overtime.
# 
# Intuitively the coefficients make sense. The other seasons (Winter, Fall, and Spring) seem to have a negative impact on the average temperature in Alaska, while the Summer season has a positive impact (can be seen by the signs of the coefficients). This could be because Alaska is a colder state and experiences relatively mild summers, so the lower and more drastic temperatures in the other seasons has a greater impact on the temperature changes overtime. It could also be due to other factors like precipitation, cloud cover, and wind patterns, which can vary between seasons and affect the average temperature differently.
# 
# To assess my regression results, I will point out some trends that can be generalized among all the states. There is actually an interesting trend in the results for this regression, that may help us in furthering our understanding for the research question. 
# 
# It's worth noting that the coefficient for the year variable is statistically significant for all states, indicating that temperature has been increasing over time, on average, across all states, controlling for seasonality! Ok so, even when we take into account seasonal temperature changes in our model, time plays a significant role on prediciting temperature in each state. 
# 
# We can see most states have a p-value that is not statistically significant for the spring or fall month dummies. 
# 
# Overall, most states have either the winter or summer dummy as a statistically significant predictor of the average temperature. It makes sense intuitively that the winter and summer dummies are more likely to be significant predictors of average temperature compared to spring and fall dummies. Winter and summer are the two most extreme seasons in terms of temperature. Therefore, it is more likely that the winter and summer dummies will have a stronger relationship with the average temperature compared to the more mild seasons of spring and fall. Additionally, the winter and summer seasons tend to have more distinct and consistent weather patterns, which can also contribute to their significance as predictors.

# Next, I will also do a regression for each season! This aggregates the states results together so we only get to look at the seasonal changges overall. In this regression I will have the independent variables: Baseline Temperature for that season in that given year in that state, Temperature Change against the baseline in that month, and Year. The dependent variable will be Average Temperature in that given month in that season. 
# 
# My reasoning for running this regression is that I want to isolate seasonal temperature effects by 1) first, controlling for that season's average temperature as a whole and then 2) look at the temperature change (fluctuations from that average) from that baseline average and in the end, looking that how significant these two might be in determining that season's overall temperature. This regression will make clear if the temperature fluctuations are a good indicator of average temperature - meaning that they are somewhat repetitive and have a clear pattern in any given state on a seasonal basis when regressed against average temperature. 
# 
# The model is as such: 
# 
# $$
# {AverageTemperature}_i = \beta_0 + \beta_1 {year}_i + \beta_2 {BaselineTemp}_i + \beta_3 {TemperatureChange}_i + u_i
# $$
# 
# where:
# 
# - $ \beta_0 $ is the intercept of the linear trend line on the
#   y-axis  
# - $ \beta_1 $ is the year variable explaining how much temperature increases by (in Celcius) annually, on average in that season
# - $ \beta_2 $ is the baseline average temperature in degrees Celsius for that season in a given year in a given state, aggregated for each state
# - $ \beta_3 $ is the temperature percent change in that given season for a given month for a certain state, which was calculated by the formula (averagetemp monthly temp in a state in a given year for a given season - baseline average temp for that season in that year for that state / baseline average temp for that season in that year for that state), in percent. 
# - $ u_i $ is a random error term (deviations of observations from
#   the linear trend due to factors not included in the model) 

# In[22]:


season_group = dummy_merged.groupby('Season')


# In[23]:


lst_3 = []
for group in season_group.groups:
    season = season_group.get_group(group)
    X = season[['Year', 'BaselineAvgTemperature', 'TemperatureChange']]
    Y = season[['AverageTemperature']]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    lst_3.append(model)
    predictions = model.predict(X)
    


# In[24]:


stargazer_3 = Stargazer(lst_3)
season_lst = []
    
for group in season_group.groups:
    season_lst.append(group)

stargazer_3.custom_columns(season_lst, [1,1,1,1])
stargazer_3.show_model_numbers(False)
stargazer_3.show_degrees_of_freedom(False)


HTML(stargazer_3.render_html())


# As we can observe, most of the coefficients are statistically significant at the 99 percent confidence level. This means that for each season, the baseline temperature is a strong predictor of average temperature. This is not a surprising result since the baseline is calculated from the average temperature aggregations. The main reason for having this variable in this regression is to interpret the temperature change variable. As we can see, temperature change is statistically significant for all seasons except winter. So, for the interpretation, we can say that the overall fluctuations around a given mean temperature (baseline value) for summer, spring and fall are still statistically significant predictors of temperature activity. Meaning that the fluctuations need not be random, and there can be a somewhat predictability around the magnitude of these changes around the baseline values. However, for winter, the temperature fluctuations are not significant predictors of temperature at all. The coefficient is 0, meaning that these fluctuations do not contribute to finding the average temperature overall, while baseline values are still statistically significant. Intuitively, this means that winter fluctuations are too volatile to be good predictors of temperature. Alternatively, a downside of this regression is that we cannot see this breakdown on a state-by-state basis (the regressing would be too large and we would have to create dummies for each state), so we don't know if there may be a few states that are skewing the results (e.g. Alaska or North Dakota).
# 

# #### Next, use the population data to complement our findings and further our economic analysis. We will use population percent change per decade in US states starting from 1910 to complement our data. The hypothesis is that the higher population density in states will be related to the temperature changes in that given state. Since higher population can lead to higher temperature changes due to various economic reasons, there might be positive relation between the economic variable and the geographic variable. Lets test this!

# In[25]:


# merge with ouput_data for comparison
pop2 = pop.merge(output_data, on='State')
pop2.head()


# So as you may have noticed, the data in pop is time series data with an observation being the percent change in a given decade in a given state. To make a meaningful comparison against the population change for temperature change, it would be unwise to compare cross-sectional temperature percent change for each season against the given decade percent change in population. So, in the next part, I will basically group the 'state' data again for decade and state, instead of season and state to find the percent change per decade for each state using all observations. This can be better compared against the population change data. In the mapping and figures section, I will attempt to discover a relationship amongst these variables for an economic interpretation for the my research question. 

# In[26]:


# First, convert the 'Year' column to a datetime object in state
state['Year'] = pd.to_datetime(state['Year'], format='%Y')

# Then, create a new column for the decade using the 'Year' column
state['Decade'] = state['Year'].apply(lambda x: int(x.year/10)*10)
state


# In[ ]:


# Group the data by state and decade, and calculate the average temperature for each group
grouped_data = state.groupby(['State', pd.Grouper(key='Year', freq='10Y')])['AverageTemperature', 'Decade'].mean().reset_index()


# In[28]:


baseline_years = range(1900, 1951)
baseline_data = grouped_data[grouped_data['Decade'].isin(baseline_years)]
baseline_data


# In[29]:


merged_data = pd.merge(grouped_data, baseline_data, on=['State'])
merged_data['TemperatureChange'] = ((merged_data['AverageTemperature_x'] - merged_data['AverageTemperature_y'])/merged_data['AverageTemperature_y'])*100


# In[30]:


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
output_data2.head()


# In[31]:


output_data2['State Abbreviation'] = series


# In[32]:


output_data2.head()


# Ok, we can see now that we have columns that correpond to the temperature change in a given given decade in a state. I basically used the same strategy as 'ouput_data' to find the baseline years average temperature per decade and get the change of that against all other decades in a given state. I did this for all the baseline decades, comparing against each decade until 2010. I can plot figured to find patterns among these variables. 

# ## 1.4 Plots/Histograms/Figures
# ### For the part 2) visualization section, I have included that in this part instead because I think it would be more meaningful for my data plots to be side by side. 
# This is one of the most important parts of discovering patterns overtime to figure out season trends. Let's continue.  

# First, let's create hovermaps to visualize our {output_data} dataframe, and check for any seasonable trends/outlier states. I will create four maps per season to visualize these trends for the average temperature percent change columns, as well as standard deviation of the percent change columns per season.  

# In[33]:


# just temp
for season in ['Fall', 'Winter', 'Summer', 'Spring']:
    fig = px.scatter(output_data, x='State', y=f'Avg_{season}Percent Change', color=f'Avg_{season}Percent Change',
                     title=f'Percent Change in Temperature for {season}', hover_name='State', hover_data=[f'Avg_{season}Percent Change'])
    fig.show()


# We can see that temperature changes the most during the winter months as compared to any other season. 
# Also note an interesting trend: The percent change of the temperature during the summer months has historically remained almost above 0%. Meaning that, compared to the baseline years, the temperature change has increased. This means that during the summer months, temperature is actually increasing more than it was compared to the baseline year summer temperatures. The winters are the most volatile and have the highest standard deviation (graphs below), and overall, the temperature changes in fall and spring have also remained above 0% (except Alaska due it its geographic location). 

# In[34]:


# just std
for season in ['Fall', 'Winter', 'Summer', 'Spring']:
    fig = px.scatter(output_data, x='State', y=f'Std_{season}Percent Change', color=f'Std_{season}Percent Change',
                     title=f' Std. of Percent Change in Temperature for {season}', hover_name='State', hover_data=[f'Std_{season}Percent Change'])
    fig.show()


# The standard deviation is lowest during the summer months and highest during winter. States such as Maine, Minnesota, Alaska, New Hamphire, Vermont and North Dakota have been the most volatile during any given seasons. Note that these are also the states that had a higher standard deviation than its mean temp in the summary statistics section. 

# In[35]:


pop.drop(['Geography Type', 'Number of Representatives', 'Change in Number of Representatives', 'Average Apportionment Population Per Representative'], axis=1, inplace = True)


# In[36]:


pop.tail() # we basically want to merge this data with the output_data2 to see trends among the temperature change 
# variable per decade and the percent change in population per decade. However, here, merging is a bit difficult
# because we have columns of decade data per state in the output_data2 df, and we want to merge on state and decade


# In[37]:


output_data2.head()


# In[38]:


pivot_population_data = pd.pivot_table(pop, values='Percent Change in Resident Population', index='State', columns='Year')

# Rename the columns to match the temperature data
new_columns = ['Avg_' + str(decade) + ' Pop Percent Change' for decade in pivot_population_data.columns]
pivot_population_data.columns = new_columns

# Merge the temperature data and the population data
merged_data = pd.merge(output_data2, pivot_population_data, on='State')


# In[39]:


merged_data.head()


# In[40]:


for decade in range(1910, 2020, 10):
    fig = px.scatter(merged_data, x=f'Avg_{decade} Pop Percent Change', y=f'Avg_{decade}Percent Change', color='State')
    fig.update_layout(title=f'{decade}s', xaxis_title='Population Percent Change', yaxis_title='Temperature Percent Change')
    fig.show()


# Here, let's check trends with the population data and temperature data. I did not perform a linear regression here beacause I did not have enough variables for my population dataframe, so I decided to do a scatterplot instead. I will most likely conduct a proper linear regression for another economic variable. 
# 
# Here I graphed the temperature percent change in the y axis and the population percent change in the x axis.
# 
# In 1910, There is a general upward trend, indicating that there is a postiive relationships between population and temperature in this decade. However, there are states that may not have had a large temperature change but the population percent change is a lot larger. This initially makes sense since this decade denotes the industrial revolution, so the effect of increasing population would not be evident in the temperature yet, since we are looking at the baseline years to comapre temperature. There are also outliers, such as Alaska which has a large temp. change but little to no population change. This makes sense since this map doesn't take into account seasonailty to Alaska temp is variable. Therefore, Alaska will more likely be an outlier in most graphs due to seasonality. 
# 
# Overall, there is a general trend: We can see the much hotter states such as California, Idaho, Florida, Nevada and Arizona have a consistently higher population percent change overtime, while their temperature change overtime remains consistently closer to 0 than most other states. This overall trend is consistent with our seasonal findings (which you will see in the mapping section): The hotter states exhibit an overall LESS volatile change in temperature as compared to the northern and nothereastern states, or 'cooler' states. Geographically speaking, this trend means that for mid-southern states, population changes may not be the most accurate determinant of temeprature change overtime; that the change can be denoted to the  geographic location of the state. Again I have to reiterate, correlation does not equal causation, so we can't make any causal inferences regarding this observation. This can merely be ONE of the many variables that effect temperature overtime. 
# 
# There is another very significant trend. As we can see, starting the 1990's for most states (again, Alaska being an outlier due to its geographical location), the temperature change veers more towards a positive increase, and less states have a mean temperature change (compared to the baseline years) that has REDUCED from the baseline year mean. Visually, this can be seen by the 1990, 2000, and 2010 scatterplots, where the temp change does not go into the negatives. This is so interesting to see this visual trend because it confirms something about our data, whether its relevent to the population variable or not (which in itself, is a good indicator of whether or not this is an economically significant variable in determining climate trends). We can see from these scatterplots that overtime, the temp change has gone from negative change to a positive change overtime, so there is definitely some form of climate warmth in the US in each state, which is especially seen from 1990 onwards.
# 
# Regarding population change factor, we note that there are some patterns that may be important such as historically warmer states being less sensitive to population changes overtime.
# 
# Overall, we really can't make a strong connection using a linear regression because we don't have enough variables for the population change overtime. If we had, for example, monthly data for percent change like we did for the temperature, then we could have run those as x and y variables side by side, so this is a pretty limited conclusion, but one still worth pursuing. 

# # PROJECT TWO
# We will now map the data of the summary statistics tables that we have generated earlier. As helpful as the scatterplots are, it will be slightly easier to visualize regional changes through a map. 

# ## 2.1 THE MESSAGE
# The message or purpose of my data is to answer my research question, stated above. This is, I want to find patterns in temperature changes overtime for the US states and somehow find economic variables that may be a good determinant of these temperature changes over certain regions. I also want to decipher if economic variables can potentially be more or less significant indicators compared to the geographic locations of any given state in the US. I think this can be economically significant since this research can aid policymakers to take into account climate when they are implementing economic policies.

# Recall the {merged_stats} dataset in the Summary Statistics section, that gave us the average temperature and standard deviation of temperature per state, overall. The first visualization I want to create is a map to add meaning to my "message". Using colour to indicate which states have a higher standard deviation that its mean temperature using {merged_stats}, I can specifically focus on these states, as they may have more distinct temperature patterns. I want to do this to show, without taking into account seasonality or annual variability, how certain states may behave as benchmark historically. 

# In[41]:


merged_stats['State Abbreviation'] = series


# In[42]:


us_states = gpd.read_file('cb_2018_us_state_500k.shp')
merged_df = us_states.merge(merged_stats, left_on='STUSPS', right_on='State Abbreviation')
subset_df = merged_df[merged_df['std'] > merged_df['mean_temp']]

fig, ax = plt.subplots(1, figsize=(60, 40), subplot_kw={'aspect': 'equal'})
#ax.axis('off')
ax.set_xlim([-130, -65])
ax.set_ylim([23, 50])
ax.set_title('US States - STD > Mean Temp', fontdict={'fontsize': '25', 'fontweight' : '3'})

merged_df.plot(column='std', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', 
               legend=True,legend_kwds={'label': "Standard Deviation (degrees Celcius)", 'shrink': 0.3})

subset_df.plot(column='std', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8',
               alpha=0.7, legend=False)

# Set color for all other states
other_df = merged_df[~merged_df.isin(subset_df)].dropna()
other_df.plot(color='white', linewidth=0.8, ax=ax, edgecolor='0.8')

plt.show()


# In[43]:


fig2, ax2 = plt.subplots(1, figsize=(10, 20), subplot_kw={'aspect': 'equal'})
#ax.axis('off')
ax2.set_xlim([-180, -130])
ax2.set_ylim([50, 72])
ax2.set_title('US States - STD > Mean Temp', fontdict={'fontsize': '25', 'fontweight' : '3'})

merged_df.plot(column='std', cmap='OrRd', linewidth=0.8, ax=ax2, edgecolor='0.8', 
               legend=True,legend_kwds={'label': "Mean Temp (degrees Celcius)", 'shrink': 0.3})

subset_df.plot(column='std', cmap='OrRd', linewidth=0.8, ax=ax2, edgecolor='0.8',
               alpha=0.7, legend=False)

# Set color for all other states
other_df = merged_df[~merged_df.isin(subset_df)].dropna()
other_df.plot(color='white', linewidth=0.8, ax=ax2, edgecolor='0.8')

plt.show()


# In[44]:


# plot the overall standard devation in the states overtime - benchmark. 
fig, ax = plt.subplots(1, figsize=(80, 60), subplot_kw={'aspect': 'equal'})
ax.set_xlim([-130, -65])
ax.set_ylim([23, 50])

# Plot the map
merged_df.plot(column='std', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', 
               legend=True, legend_kwds={'label': "Standard Deviation (degrees Celcius)", 'shrink': 0.5})

# Create a second map that includes Alaska
fig2, ax2 = plt.subplots(1, figsize=(20, 20), subplot_kw={'aspect': 'equal'})
ax2.set_xlim([-180, -130])
ax2.set_ylim([50, 72])

# Plot the map with Alaska
merged_df.plot(column='std', cmap='OrRd', linewidth=0.8, ax=ax2, edgecolor='0.8', 
               legend=True, legend_kwds={'label': "Standard Deviation (degrees Celcius)", 'shrink': 0.5})

# Add Alaska to the second map
alaska_df = merged_df[merged_df['STUSPS'] == 'AK']
alaska_df.plot(column='std', cmap='OrRd', linewidth=0.8, ax=ax2, edgecolor='0.8', alpha=0.7, legend=False)

plt.show()


# In[45]:


# plot the overall temperature in the states overtime - benchmark. 
fig, ax = plt.subplots(1, figsize=(80, 60), subplot_kw={'aspect': 'equal'})
ax.set_xlim([-130, -65])
ax.set_ylim([23, 50])

# Plot the map
merged_df.plot(column='mean_temp', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', 
               legend=True, legend_kwds={'label': "Average Temperature (degrees Celcius)", 'shrink': 0.5})

# Create a second map that includes Alaska
fig2, ax2 = plt.subplots(1, figsize=(20, 20), subplot_kw={'aspect': 'equal'})
ax2.set_xlim([-180, -130])
ax2.set_ylim([50, 72])

# Plot the map with Alaska
merged_df.plot(column='mean_temp', cmap='OrRd', linewidth=0.8, ax=ax2, edgecolor='0.8', 
               legend=True, legend_kwds={'label': "Average Temperature (degrees Celcius)", 'shrink': 0.5})

# Add Alaska to the second map
alaska_df = merged_df[merged_df['STUSPS'] == 'AK']
alaska_df.plot(column='mean_temp', cmap='OrRd', linewidth=0.8, ax=ax2, edgecolor='0.8', alpha=0.7, legend=False)

plt.show()


# In[46]:


print('States that have higher temperature variability in terms of their mean temp overtime:')
for state in subset_df['State'].values:
    print(state)


# The map depicts all the states, on average (exluding seasonality trends), that have st. dev. larger than its state mean temp overtime using averges from 1900-2013. We can use this map to roughly estimate that these states (listed above) may be the most sensitive in terms of temperature change and variability, in terms of geographic location.
# 
# However, this is just a rough estimate, since we do not take into account seasonality. Also using the mean_temp and standard deviation maps, we can see that colder states have higher standard deviation of temp over average. We will look out for the same patterns when mapping in the next section.
# 
# We also note that in our previous graph analysis, we noted that most of these states states above had more temperature variability in any given month. From this we can infer that northern states, indeed, have more volatile changes in temperature overtime compared to the baselien temperatures. 

# ## 2.2 Maps and Interpretations

# ### Seasonal Trends:

# In[47]:


# here we will use our <output_data> dataframe that we created for seasonal trend. 
output_data.head()


# In[48]:


for season in ['Fall', 'Winter', 'Summer', 'Spring']:
    fig = px.choropleth(output_data, locations='State Abbreviation', locationmode="USA-states",
                        color=f'Avg_{season}Percent Change', scope="usa",
                        color_continuous_scale="RdBu",range_color=(-5, 5),
                        title=f'Percent Change in Temperature for {season}')
    fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))
    fig.show()


# As we can see with our maps, the temperature change from the baseline years until 2013 is most visible in the winter seasons. This is consistent with our regression findings earlier. It is even more interesting to see that the northearn states have a temperature change that is lower than the winter baseline average for that given state. This means that compared to the baseline historical average of these states in the winter months until 1950, the winter average temperature in the winters has decreased more on average. Some of these states are also the ones we mapped earlier that have a large standard deviation in comparison to its mean. Since that map did not consider seasonality, the variability in the seasonal changes may have caused the standard deviation of these states to be high in the preiovus map. That can be one potential explanation. 
# 
# The temperature change in spring is also quite different. The northern states, again, exhibit a larger temperature change. Compared to the baselinea years average temperature in the spring months, the average temperature has increased more on average, in these northern states. We can note that these states are cooler over average, and are closer to colder bodies of water (The Great Lakes and the Atlantic Ocean) which may get cooler/warmer during the spring due to a drastic shift in temperature. In these northern states, the changing of season's from cold to hot can take longer, therefore, the shift in temperature of the bodies of water and the states may be nore variable, reflecting that. 

# In[49]:


for season in ['Fall', 'Winter', 'Summer', 'Spring']:
    fig = px.choropleth(output_data, locations='State Abbreviation', locationmode="USA-states",
                        color=f'Std_{season}Percent Change', scope="usa",
                        color_continuous_scale="RdBu",
                        title=f'Percent Change in Standard Deviation of Temperature for {season}')
    fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))
    fig.show()


# The standard deviations for the states per season differ greatly overtime. During the winter, all the states have larger temperature variability, compared to all other seasons. In the summer, fall, and spring months, the northern states have more variability in temperature, as well. 
# 
# These results can indicate that geographic location has a huge part to play in how temperature may come to behave overtime. While southern, more warmer states may experience less temperature variability through any given season, they also have systemtically higher temperatures. Also, while the northern states have lower temperatures, they experience a lot more variation during any given season. While using economic indicators such as C02 emissions or precipitation may aid our understanding of these temp changes, it is important to take notice of the geographic factors as well.  

# ##### Aside: I am going to map the AverageTemperatureUncertainty for each state . The reason for this is because I want to check whether all my findings so far are accurate in terms of historic temperature measurements and information availability. I want to check that the AverageTemperatureUncertainty - which measures the temperature measurement accuracy overtime. I want to check that the accuracy of the measuring temperature overtime has indeed increased, due to better technology, information availabilty and climate change awareness overtime.

# In[50]:


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


# Bluer colored states represent a lower increase in the average temperature uncertainty from the baseline period to the year specified in the merged.plot() function. This means that the range of possible temperature values for these states became smaller over time, indicating more reliable temperature measurements.
# 
# However, it is important to note that the range of possible temperature values is just one aspect of uncertainty in temperature measurements, and it does not necessarily mean that the actual temperature change was smaller in these states. There could be other sources of uncertainty in the temperature measurements that are not captured by this analysis.
# 
# As we can see there is a larger cluster of states in the midwest that have a higher temperature measurement uncertaintly compared to some of the north-eastern states, indicating that the states in the north regions have more reliable temperature measurements.
# 
# One possible explanation is that the Midwestern states have a relatively stable climate with less variation in temperature over time compared to other regions, which could lead to less uncertainty in temperature measurements. However this differs from the map generated above. However, just because the temperature change of the northwest states has increased more overtime does not mean that that the temperature itself is stable. It could just mean that for these states, the temperature measurements may have immproved frrom the baseline year for a more accurate measurement. Additionally, the Midwestern states may have had more consistent and reliable temperature monitoring stations over time, leading to more consistent measurements and less uncertainty.
# 
# On the other hand, the Northeastern region has more complex topography, with mountains and coastlines that can influence local weather patterns, and may have had more variation in temperature over time, leading to higher uncertainty in temperature measurements. Additionally, the Northeastern region is more densely populated and urbanized, which can create heat islands and other local climate effects that can further complicate temperature measurements.
# 
# It is important to note that these are just potential explanations and other factors could also be contributing to the observed patterns. This map only shows the temperature measurement changes not the temperature changes itself, so it is important to keep that into account. My reasoning for including this map is to show that temperature monitoring has drastically improved overtime in states that may not have been previously less populated. This is important to understand since we need to know to what extent the temperature measurements are certain for our findings. 

# ### Population and Temperature Mapping Analysis:

# In[51]:


# let's map the <output_data2> and <merged_data> to map the temperature change and popualtion change.
for year in range(1910, 2020, 10):
    fig = px.choropleth(merged_data, locations='State Abbreviation', locationmode="USA-states",
                        color=f'Avg_{year}Percent Change', scope="usa",
                        color_continuous_scale="Reds",
                        title=f'Average of Temperature Percent Change for Decade: {year}')
    #fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))
    fig.show()
# I used red here as a scale because i wanted to show the dramatic increase in temp change overtime. 


# As we can see, decade by decade the percent change in temp from the baseline years increases, especially during the last 3 decades. We can see this because the mapping scale increases into the double digits those 3 decades for most states as compared to the previous decades where the temp percent change may be only increasing/decreasing a lot more for only a few states. The magnifude of the changes has increased, while in most other decades, the percent change in temperature hits the double digits and we can see this in the dramatic colour change. There is a large increase in temp in the 1980-1990 decade. We can also see that the northern states experience more of a positive temperature change overtime in each decade. Climate change has definitely been at more of a forefront since 1990 and onward, compared with the decades before. 
# 
# Some states may be an exception due to their geographic location - for example. Here, we can see that since the northern states exhibit a larger change overtime in each decade because 1) They are cooler on average and may be more variable since the <merged_data> df does not take into account seasonality (Alaska, the Dakota's, Michigan), 2) They experience more changes in precipitation due to their proximity to the ocean so this could also effect temp. analysis (Maine, Vermont, New Hampshire, etc). These states are also more variable on average (we also saw this in the standard deviation graphs previously, so our analysis is consistent with that one - and our population scatter analysis which states that warmer states are less volatile and less sensitive to changes in population). To confirm this, we can also just take the our original state data and groupby the year and plot it. 

# # PROJECT THREE

# ## 3.1 Potential Data to Scrape

# Potential data I would like to scrape would be monthly C02 emissions per state or industrialization percentage change per state overtime. Preferrable if the dataset was large and dated back a long time, it would be very meaningful for my research question. Since both are important factors in determining temperature overtime, with a large dataset, regressions could be run that could add more meaning to my question. Monthly datasets would be more meaningful than aggregating outwards and getting annual values because my temperature dataset is in monthly frequency. Although, it might be difficult to find a dataset that perfectly complements my research question. Some datasets that I found for the US are useful but they are not in monthly frequency and some of them don't have specific state data, just country data. 

# Here is a potential source that complements my research:
# We will use EIA's website - US Energy Information Administration to find the dataset. 
# https://www.eia.gov/environment/emissions/state/ - for co2 energy related carbon emissions by state dating back 1970 -2020. 
# This is the best dataset I could find that fits my research question that I can potentially web scrape. This website uses API key access and an API url for web scraping, so I can web scrape this data. 

# I can creata a dataframe out of this dataset, and merge it with my aggregated data on an annual basis. That way, they are both of the same frequency. I can look at the c02 emissions of each state and compare it with the temperature of that state and how they both change overtime, and how the temperature might reflect the change in the c02 emissions. I am hoping that there is a correlation amongst the two. I wish to uncover that overtime, c02 emissions can be a function of the temperature overtime, that they are positively correlated. This is important to my research question because I can discover whether or not c02 emissions most effect a state's temperature overtime, compared to geographic location or other economic variables. 
# 
# Alternatively, I could use the two datasets separately to conduct parallel analyses and then compare the results. For instance, I could analyze the temperature data to look for trends or anomalies and then see if these correspond to any patterns in the carbon dioxide emissions data.
# 
# Overall, combining the temperature and carbon dioxide emissions datasets could help me to gain a more comprehensive understanding of the environmental factors that influence carbon dioxide emissions and climate change.

# ## 3.2 Potential Challenges

# The data source above is one of the only few that I found that shows c02 emissions by state and dates back quite a while ago to make for a longer dataset. The reason I ended up choosing this dataset, even though it does not have monthly data is because it allows me to use an API key to conduct API web scraping. Other websites that have similar data would either not allow me to conduct web-scraping or did not have data as large as this dataset. Also scraping this dataset takes a very short amount of time, which makes it more convenient to use for a project such as this which has time limitations. 
# 
# Potential Challenges that I face with scraping these sources are: 
# 
# - The data is not monthly, so if I outer merge with my current dataset, it will not be as meaningful if  had monthly observations I could track overtime. But every dataset I found has certain trade-off's. Some datasets have longer time frames, but don't have state data, some have state data, but the time frame is not as large, and for most of them, web-scraping is tricky. This is the best dataset that I found under these circumstances. 
# 
# - I need to look out for the c02 emissions values in the dataset, meaning that i need to make a meaningful comparison against temperature so it would be difficult to maybe get the change in c02 emissoins from one year to another. I think I would potentially need to group the data by state, then the fuel_name and then get the change per year for each type of fuel_name and then see if there are certain fuel emissions that are better predictors of temperature. This can be technically challenging to think about and code up, especially for running regressions. 
# 
# - The EIA's API does not allow me to scrape more than 5000 rows, so I will have to use the 'offset' parameter to specify the starting row for the next API request. I can start the next request at row 5000 (the maximum number of rows returned by the API in a single request) and then increment the offset by 5000 for each subsequent request until I have all the data. I will most likely create a while loop for this part and keep iterating until I don't get anything from the API. 
# 
# Let's scrape! 

# ## 3.3 Scraping Data from a Website

# Here, we will scrape the data using API-web scraping, using the EIA's API personal access key. 
# Here is the API link that I used to get the data: https://www.eia.gov/opendata/index.php

# In[52]:


import requests
import json

offset = 0
# we will first set offset to 0, indicating we want the first 5000 rows, then keep updating it in each loop 
# iteration and adding it in our dataframe to get the next 5000 rows -> [0, 5000] -> [5000, 10,000] etc. 

rows = []
# rows is an empty list at first 

while True:
    # here is the API url. I had to sign up and get the access key and input it into the 
    # url in a very specific manner. Then near the end of the url, I am using the f"" function to update the 
    # offset parameter per iteration so the API knows which rows to give me. 
    
    url = f'https://api.eia.gov/v2/co2-emissions/co2-emissions-aggregates/data/?api_key=qo4UGF5ygDuaa28f53pLo4QD6v2rgPGOMMJZ6mc5&frequency=annual&data[0]=value&facets[stateId][]=AK&facets[stateId][]=AL&facets[stateId][]=AR&facets[stateId][]=AZ&facets[stateId][]=CA&facets[stateId][]=CO&facets[stateId][]=CT&facets[stateId][]=DC&facets[stateId][]=DE&facets[stateId][]=FL&facets[stateId][]=GA&facets[stateId][]=HI&facets[stateId][]=IA&facets[stateId][]=ID&facets[stateId][]=IL&facets[stateId][]=IN&facets[stateId][]=KS&facets[stateId][]=KY&facets[stateId][]=LA&facets[stateId][]=MA&facets[stateId][]=MD&facets[stateId][]=ME&facets[stateId][]=MI&facets[stateId][]=MN&facets[stateId][]=MO&facets[stateId][]=MS&facets[stateId][]=MT&facets[stateId][]=NC&facets[stateId][]=ND&facets[stateId][]=NE&facets[stateId][]=NH&facets[stateId][]=NJ&facets[stateId][]=NM&facets[stateId][]=NV&facets[stateId][]=NY&facets[stateId][]=OH&facets[stateId][]=OK&facets[stateId][]=OR&facets[stateId][]=PA&facets[stateId][]=RI&facets[stateId][]=SC&facets[stateId][]=SD&facets[stateId][]=TN&facets[stateId][]=TX&facets[stateId][]=US&facets[stateId][]=UT&facets[stateId][]=VA&facets[stateId][]=VT&facets[stateId][]=WA&facets[stateId][]=WI&facets[stateId][]=WV&facets[stateId][]=WY&start=1970&end=2020&sort[0][column]=period&sort[0][direction]=asc&offset={offset}&length=5000'
    
    response = requests.get(url)
    
    print(response.status_code)
    # here, i am making sure that each iteration gets me the data i asked for. I want it to give me 
    # 200 as a response, telling me that I can successfully fetch the data I asked for in this iteration. 
    
    # now i'm using the json library to give it the data in a dictionary format using the data text.
    api_response = json.loads(response.text)

    # Extract the relevant data from the dictionary
    
    data = api_response['response']['data']
    
    # if there is no more data, we are done, so end the loop.
    if not data:
        break

    # Loop through the data and extract the required columns for each row, where each item is a small dict object. 
    for item in data:
        row = {'period': item['period'],
               'sector_name': item['sector-name'],
               'fuel_name': item['fuel-name'],
               'state': item['state-name'],
               'State Abbreviation': item['stateId'],
               'value': item['value'],
               'value_units': item['value-units']}

        # Add the row to the list of rows if it's not a duplicate (as a precaution)
        if row not in rows:
            rows.append(row)

    # Increment the offset by 5000 for the next API request
    offset += 5000

api_df = pd.DataFrame(rows)


# In[53]:


api_df


# Now we have our dataset that we have successfully scraped into our project. 
# 
# This data provides information on carbon dioxide emissions for each state in the United States from 1970 to 2020. The dataset includes information on the amount of emissions produced by various sectors, such as residential and transportation, and from different types of fuels, such as coal and natural gas. The emissions are measured in million metric tons of CO2, and the dataset also includes information on the units of measurement for each value. This data can be used to analyze trends in carbon dioxide emissions over time and to compare emissions across different states and sectors.

# ## 3.4 Merging the Scraped Data

# Since this data annual I will use one of the datasets that I created when I was creating the <output_data2>, since I aggregated the data annually for that section. 

# In[54]:


api_df
api_df.rename(columns={'period': 'year'}, inplace=True)
api_df.rename(columns={'state': 'State'}, inplace=True)


# In[55]:


api_df.head(30) # this is what the dataframe looks like. 


# In[70]:


state.rename(columns={'Year': 'year'}, inplace=True)


# In[71]:


state


# In[ ]:


state_data = state.groupby(['State', 'year'])[['AverageTemperature']].mean().reset_index()
state_data


# In[73]:


state_df = state_data.loc[(state_data['year'] >= 1970) & (state_data['year'] <= 2013)]


# merge the two dataframes on the common column 'State'
merged_df = pd.merge(api_df, state_df, on=['year', 'State'])

merged_df


# Ok we have successfully merged the {state} dataset with the {api_df} dataset. As we can see here, we have 51,332 rows and 8 columns. When I grouped the data by state (which is what I need to do to get relevent information for my research question), the number of rows per state are around 1056 observations. So, to run regressions the number of observations, this number seems suitable since there are enough observations for state analysis and regional analysis - where I can potentially group certain states together and analyze it against emissions. This way, I can get more observations if I group by region instead of state. 

# ## 3.5 Visualizing the Scraped Data

# In[74]:


visual = merged_df.groupby('sector_name')


# In[75]:


for group in visual.groups:
    thing = visual.get_group(group)
    plt.style.use('seaborn')
    thing.plot(x='year', y='value', kind='scatter')
    plt.title(f"{group}")
    plt.show()


# Since I want to create regressions, it is important to note how each sector's CO2 emissions behaves overtime. I want to see if I can potentially fit linear regression models against temperature to get a linear trend. As we can see, there are a lot of spikes in these charts, so there may be some states that use more energy for a given sector. More generally, it seems that these graphs can be skewed, but most have a near linear trend. Since this data is smaller, it may be hard to denote non linear patterns. Let's move onto finding some regression results. 

# #### Regressions for CO2 emissions dataset and temperature dataset 

# In[76]:


merged_df['Residential carbon dioxide emissions'] = merged_df['sector_name'].apply(lambda x: 1 if x == 'Residential carbon dioxide emissions' else 0) 
merged_df['Industrial carbon dioxide emissions'] = merged_df['sector_name'].apply(lambda x: 1 if x == 'Industrial carbon dioxide emissions' else 0) 
merged_df['Electric Power carbon dioxide emissions'] = merged_df['sector_name'].apply(lambda x: 1 if x == 'Electric Power carbon dioxide emissions' else 0)
merged_df['Transportation carbon dioxide emissions'] = merged_df['sector_name'].apply(lambda x: 1 if x == 'Transportation carbon dioxide emissions' else 0)
merged_df['Commercial carbon dioxide emissions'] = merged_df['sector_name'].apply(lambda x: 1 if x == 'Commercial carbon dioxide emissions' else 0)
merged_df['Total carbon dioxide emissions from all sectors'] = merged_df['sector_name'].apply(lambda x: 1 if x == 'Total carbon dioxide emissions from all sectors' else 0)


# Here we can't groupby season since the CO2 data is on an annual frequency. 
# The first regression will simply be on a per state basis and we will use the CO2 emission values along with the year variable to predict the temperature to see how if CO2 emissions can predict temperatures is any given state. 
# 
# Here is the model:
# 
# $$
# {AverageTemperature}_i = \beta_0 + \beta_1 {year}_i + \beta_2 {CO2emissions}_i + u_i
# $$
# 
# where:
# 
# - $ \beta_0 $ is the intercept of the linear trend line on the
#   y-axis  
# - $ \beta_1 $ is the year variable explaining how much temperature increases by (in Celcius) annually, on average in that state
# - $ \beta_{2} $ is the CO2 emission value in that state overall. 
# - $ u_i $ is a random error term (deviations of observations from
#   the linear trend due to factors not included in the model)  
# 

# In[77]:


grouped_state = merged_df.groupby('State')


# In[78]:


lst10 = []
for group in grouped_state.groups:
    lst10.append(group)


# In[81]:


lst4 = []
for groups in grouped_state.groups:
    state3 = grouped_state.get_group(groups)
    X = state3[['year', 'value']]
    Y = state3[['AverageTemperature']]
 
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    lst4.append(model)
    predictions = model.predict(X) 


# In[82]:


stargazer10 = Stargazer(lst4)
stargazer10.custom_columns(lst10, lst[:49])
stargazer10.show_model_numbers(False)
stargazer10.show_degrees_of_freedom(False)

HTML(stargazer10.render_html())


# As we can see here, this regression is not statistically significant since none of the CO2 emission values are statisically or economically significant predictors for temperature overtime. This may be because we are aggregating all of the CO2 values together instead of picking and chosing which ones may be more important. However, I think the bigger picture may be that CO2 emissions may have become more prominent in later years, and therefore, regressing them against temperature for earlier years on all sectors may not be that meanigful. Instead, let's pick and chose more important vairable to groupby.

# In this next model, we can isolate yearly values, since in the previous regression, the only significant variables was the year variables. Therefore, there must be a correlation between the year variable, and CO2 emissions as well as temperature changes. In this next regression we can also control for the different types of sectors by creating dummy variables, to see which sectors may be more/less significant overtime. Note that we don't just use the total emissions secotr in this model since we want to find out which secotr may produce the most statistically significant results overtime. In this model, the y variables is the average temperature in a given year from 1970-2013. 
# 
# Here is this model:
# 
# $$
# {AverageTemperature}_i = \beta_0 + \beta_1 {CO2emissions}_i + \beta_2 {ResidentialCO2}_i + + \beta_3 {IndustrialCO2}_i + \beta_4 {ElectricCO2}_i + \beta_5 {TransportationCO2}_i + \beta_6 {CommercialCO2}_i + iu_i
# $$
# 
# where:
# 
# - $ \beta_0 $ is the intercept of the linear trend line on the
#   y-axis  
# - $ \beta_1 $ is the CO2 emissions in total for that given year in million metric tons of CO2
# - $ \beta_{2...6} $ are the CO2 emissions dummy variables per sector type overtime for that given year in million metric tons of CO2
# - $ u_i $ is a random error term (deviations of observations from
#   the linear trend due to factors not included in the model)  
# 

# In[83]:


grouped_years = merged_df.groupby('year')


# In[84]:


lst5 = []
for groups in grouped_years.groups:
    year_df = grouped_years.get_group(groups)
    X = year_df[['value', 'Residential carbon dioxide emissions', 'Industrial carbon dioxide emissions', 'Electric Power carbon dioxide emissions', 'Transportation carbon dioxide emissions', 'Commercial carbon dioxide emissions']]
    Y = year_df[['AverageTemperature']]
 
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    lst5.append(model)
    predictions = model.predict(X) 


# In[85]:


years_lst = []
for year in range(1970,2014):
    years_lst.append(str(year))
anotherlst = []
for _ in range(2014-1970):
    anotherlst.append(1)
stargazer11 = Stargazer(lst5)
stargazer11.custom_columns(years_lst, anotherlst)
stargazer11.show_model_numbers(False)
stargazer11.show_degrees_of_freedom(False)

HTML(stargazer11.render_html())


# Here we can see that all of the values for the CO2 emissiona are statistically significant at the 99 percent confidence level. Meaning that, when grouping by year, CO2 emissions are strong predictors of annual average temperature. However, we must note that the R^2 value is very low, meaning that model does not explain most of variance surrounding the average temperature, but its still significant. This could indicate that factors other than CO2 emissions are more relevent in predicting temperature overtime. The model however, is still statisically significant and therefore can be used for forther analysis. All of the coefficients are positive in this model, meaning that CO2 emissions and yearly temperature are positively correlated, which is not a big surprise in this case. We can also note that since this data was annual inctead of monehtly, a lot of valuable observations were aggregated, therefore, the model may lost some precision.
# 
# When breaking down the sector analysis, we can see that Industrial, Residential and Commerical CO2 emissions are the most statisically significant, and therefore, contribute most to the temperature changes overtime. Also, the coefficients for these three have an upward trend, meaning that they contribute more overtime to CO2 emissions, and therefore, to the changes in temperature. For an interpretation for the year 1970, we can say that, after controlling for the different types of sector emissions, a 0.021 degree Celsius increase in temperature can be observed if we increase overall CO2 emissions in 1 million metric tons of CO2 in the year 1970 for all the states in the US. 
# 
# So, from these findings we can see that there are other variables that may be able to predict how temperature may change overtime. CO2 emissions are slow acting and for this very small dataset, even though the R^2 is very small, we can still make inferences on how changing variables can effect our results. 

# # Conclusion:

# While there are many factors that can contribute to temperature changes in individual states, there are some economic indicators that may help explain why certain states have experienced more dramatic temperature changes over time than others. Here are a few possibilities that we have uncovered through our findings:
# 
# Sector based CO2 activity: Certain sectors such as residential and industrial sectors play a more important role in determining how a state's temperature may behave. Overtime, sector based CO2 activities can become better determinants of predicting state temperature. Intuitively, this makes sense, since larger industrial activity may point to higher CO2 emissions and larger population changes may cause larger residential CO2 emissions. 
# 
# We have also noted that geographically, northern states exhibit more volatile temperature changes overtime, no matter what the season is, and southern states (or the "hotter" states) are less effected by economic variables such as population changes overtime, due to thier systemtically high temperatures. 
# 
# Overtime policymakers should take into account these economic findings to make decisions that effect state level output activity caused by more industrialization and immigration policies to assess how climate change can be effected by population changes. 
# 

# 
