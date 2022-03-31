#!/usr/bin/env python
# coding: utf-8

# In[1]:


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


# In[2]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import plotnine
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
from kmodes.kprototypes import KPrototypes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_regression
from matplotlib import pyplot
pd.set_option('display.max_rows', 500)
import warnings
warnings.filterwarnings("ignore")


# ### Part One: EDA

# #### Read in data
# Ok, let's go ahead and load in the data using a read file function.

# In[3]:


def read_file(filename):
    try:
        df = pd.read_csv(filename, low_memory=False)
        return df
    except:
        print('Please type an appropriate file path')


# In[4]:


gps_df = read_file('vehicle_gps_Data_Test_CSV.csv')
display(gps_df.head(20))

kin_df = read_file('vehicle_kinematics_Data_Test_CSV.csv')
display(kin_df.head(20))


# #### Info about the columns:
# 
# _gps_df:_
# 
# 1. vehicle_id: The name of the vehicle.
# 2. time: Seconds since the vehicle started tracking the metrics.
# 3. latitude: self-explanatory
# 4. longitude: also self-explanatory
# 
# _kin_df:_
# 1. vehicle_id: same
# 2. time: same
# 3. acceleration: acceleration in m/s^2
# 4. velocity: velocity in m/s

# #### Basics
# Let's start with the basics. The dataframes (thankfully) have no null vlaues, they both have 3 float columns and 1 object (string) column, the GPS dataset has 27,241 observations, and the Kinematics dataset has 140,655 observations. Good to know.

# In[5]:


print(gps_df.info())
print()
print(kin_df.info())


# Now let's take a look at the first 5 observations for each car type. I see that the velocities for three of the cars (Moon, Marble, and Motto) start at 0 and stay that way for the first 5 observations. What's going on there? The accelerations are fluctuating, but the velocities aren't. Interesting.
# 
# Also, Motto starts off with a negative acceleration. Is it slowing down? Interesting.

# In[6]:


kin_df.groupby('vehicle_id').head()


# #### Kinematics dataframe descriptions (with grouped vehicle IDs)
# Let's also take a look at a description of the data. Time is useless since we know that it will constantly be going up, but the acceleration and velocity show some interesting information. The average acceleration is negative (slowing down) and the average velocity is 3 m/s with a standard deviation of 10. That's some pretty significant variance.

# In[7]:


kin_df[['acceleration','velocity']].describe()


# Let's take a look at these numbers on a more granular level (by vehicle_id). The first thing I see is that there aren't a lot of Motto observations. And this time around, the 'Time' column is much more useful. We can see from the max time that the Motto wasn't driving very long (309 seconds).
# 
# We can gather a ton of other information from this grouped description, but I'll just point out a few quick things that stuck out to me below:
# 
# 1. The Marble and Mette cars went really fast at one point. Is this an error in the data?? Yes, undoubtedly. It's almost 5000 mph; Teslas don't even go that fast! I'll leave them in the dataset for now, but that is alarming.
# 
# 2. It seems like the Marble, Mette, and Motto cars spent most of their rides slowing down. Their average accelerations are negative.
# 
# 3. It looks like the Marble and Mette cars got turned around. They both have negative minimum velocities, so that tells me that they retraced their steps at some point.

# In[8]:


kin_df.groupby(['vehicle_id']).describe()


# #### Value Counts (for both dataframes)
# This isn't really necessary, but I just wanted to take a quick look at the value counts for both of the datasets. I wanted to get a better idea of the actual numbers that I was dealing with. It's not incredibly useful information, but it helps me formulate a better picture of what's going on in my head.

# In[9]:


[kin_df[i].value_counts() for i in kin_df]


# In[10]:


[gps_df[i].value_counts() for i in gps_df]


# #### Conclusion: 
# As you can probably guess from my blurbs above, the datasets had some interesting quirks that caught me off guard. For example, I didn't expect to see any negative velocities. However, when you think about it, this really shouldn't be that surprising. Cars turn around and retrace their steps all the time (ex. going to and from work), so it's not outrageous to see a negative velocity in a dataset. I think I had an idea of how the cars were going to drive in my mind before I started analyzing the data, and it took a second to shake that assumption. I initially thought that the cars were going to go from point A to point B with no detours, but that was not the case at all. 
# 
# So yes, there were some surprising things about the data because I made some baseless initial assumptions about how the cars were going to drive, but after some quick analysis, I have a much better idea of what I'm working with.

# ## Finding the absolute distance driven by each vehicle.
# 
# #### My methodology:
# I looked up the distance formula, and this is the equation I found:  D = v*t + 1/2*a*t^2
# 
# Since it seems like some of the cars may have retraced their steps, it seems unwise to just calculate the distance between the start point and the end point for each vehicle type. I think that it would be a better idea to calculate the distance between each observation in the dataset for each vehicle and then find the sum of those mini distances. This way, we can take all types of scenarios into account. If a car drove to a NASCAR track and drove in circles, we'll still be able to calculate the distance it traveled. If a car started at my house, then drove to the mall, then stopped, and then drove back to the crib, we will be able to calculate that. And of course, if a car drive from point A to point B and stopped, we'll definitely be able to caluclate that. All we have to do is use the distance formula mentioned above and use that for every row in the dataset.  
# 
# 1. First, I will need to create new dataframes for each vehicle type. We need the distances of each vehicle, and since we will be performing complex calculations on each vehicle, we need to separate them now so our calculations don't get messed up. You will see what I mean in the next step.
# 
# 2. Second, I will calculate the time elapsed between each row so we can have a number to plug in for the "t" in the distance formula. You can do this by using df.shift. However, this doesn't take into account the time elapsed in the first row of each car's dataset. None of the cars start off with a 'time' of 0.00 seconds, so there is some initial time that has elapsed. I will need to make sure to fill in the first row of each car's 'time elapsed' column with the initial time. This 'time elapsed' column also shows why we needed to create new dataframes for each vehicle type. Since we had to perform a shift and since we need every first row of each car's dataframe to be the initial time, we needed to make sure to split the original dataset first so we didn't end up with incorrect initial times for each vehicle.
# 
# 3. Finally, we can use our distance formula to calculate the distance and make a new 'distance' column for each row's distance. And once we have our distances for each vehicle dataframe, we can find the sum of each dataframe's 'distance' column and report the absolute distance for each vehicle. I also made sure to change the distances to integers by rounding my results. I also included the float results for reference.
# 
# Easy, right? Please take a look at my code below to see how I did everything.

# #### Here are the functions:

# In[11]:


kin_df_copy = kin_df.copy()


# In[12]:


def return_compiled_kin_dfs(df):
     """
    This function will split the dataframe (df) into different dataframes (by vehicle_id) and then create new time_elapsed, absolute velocity, and distance columns in each of those new dataframes.
    """
    d = {}
    for name, group in df.groupby('vehicle_id'):
        d['group_' + str(name)] = group
        d['group_' + str(name)]['AbsVel'] = d['group_' + str(name)]['velocity'].abs()
        d['group_' + str(name)]['time_elapsed'] = d['group_' + str(name)]['time'].shift(1).sub(d['group_' + str(name)]['time']).abs()
        d['group_' + str(name)]['time_elapsed'].iloc[0] = d['group_' + str(name)]['time'].iloc[0]
        d['group_' + str(name)]['distance'] = (d['group_' + str(name)]['AbsVel']*d['group_' + str(name)]['time_elapsed'])+(0.5*d['group_' + str(name)]['acceleration'])*(d['group_' + str(name)]['time_elapsed']**2)
    return d


# In[13]:


d = return_compiled_kin_dfs(kin_df_copy)


# In[14]:


def print_distances(vehicle_df):
    print('The distance for', vehicle_df.reset_index().vehicle_id[0].capitalize(), 'is: ', vehicle_df.reset_index().groupby('vehicle_id')['vehicle_id','distance'].sum().sum().sum(), 'meters')
    print('Which rounds to: ', round(vehicle_df.reset_index().groupby('vehicle_id')['vehicle_id','distance'].sum().sum().sum()), 'meters')


# In[15]:


for i in d:
    print_distances(d[i])
    print()


# #### Now, let's break down what we just did.

# First, I copied the kin_df dataframe. No real reason for doing this. I'm just paranoid, so I always make a copy df to work with when I can.

# In[16]:


kin_df_copy = kin_df.copy()


# Note: Since some of the velocities are negative, we should probably go ahead and make some a column that contains the absolute values now so we don't mess up the calculations later. Even if a car was moving in the opposite direction, it was still moving, so that needs to be taken into account.

# In[17]:


kin_df['AbsVel'] = kin_df['velocity'].abs()


# This is where I made new dataframes for each vehicle ID, calculated the time elapsed, added the first time to each dataframe's first row, and calculated the distance with the distance equation. I used a for loop to do create the dataframes and do all the calculations because it would've been INCREDIBLY repetitive to do all of this one by one. Gotta simplify where you can, right?

# In[18]:


d = {}
for name, group in kin_df.groupby('vehicle_id'):
    d['group_' + str(name)] = group
    d['group_' + str(name)]['time_elapsed'] = d['group_' + str(name)]['time'].shift(1).sub(d['group_' + str(name)]['time']).abs()
    d['group_' + str(name)]['time_elapsed'].iloc[0] = d['group_' + str(name)]['time'].iloc[0]
    d['group_' + str(name)]['distance'] = (d['group_' + str(name)]['AbsVel']*d['group_' + str(name)]['time_elapsed'])+(0.5*d['group_' + str(name)]['acceleration'])*(d['group_' + str(name)]['time_elapsed']**2)


# Let's take a look at the moon dataframe. It looks good to me, so I will go ahead and calculate the distances for each vehicle type.

# In[19]:


d['group_moon']


# And here are each of the distances.

# In[20]:


print('The distance for Moon is: ', d['group_moon'].reset_index().groupby('vehicle_id')['vehicle_id','distance'].sum().sum().sum(), 'meters')

print('Which rounds to: ', round(d['group_moon'].reset_index().groupby('vehicle_id')['vehicle_id','distance'].sum().sum().sum()), 'meters')


# In[21]:


print('The distance for Marble is: ', d['group_marble'].reset_index().groupby('vehicle_id')['vehicle_id','distance'].sum().sum().sum(), 'meters')

print('Which rounds to: ', round(d['group_marble'].reset_index().groupby('vehicle_id')['vehicle_id','distance'].sum().sum().sum()), 'meters')


# In[22]:


print('The distance for Motto is: ', d['group_motto'].reset_index().groupby('vehicle_id')['vehicle_id','distance'].sum().sum().sum(), 'meters')

print('Which rounds to: ', round(d['group_motto'].reset_index().groupby('vehicle_id')['vehicle_id','distance'].sum().sum().sum()), 'meters')


# In[23]:


print('The distance for Mette is: ', d['group_mette'].reset_index().groupby('vehicle_id')['vehicle_id','distance'].sum().sum().sum(), 'meters')

print('Which rounds to: ', round(d['group_mette'].reset_index().groupby('vehicle_id')['vehicle_id','distance'].sum().sum().sum()), 'meters')


# ### Now let's double check our answer with _Geopy Distance_
# 
# Here's why I'm doing this:
# 
# 1. It's always good to double check your work if you can.
# 2. Those alarming outliers are still on my mind. I want to see if they throw off the distance measurements in any significant ways.
# 
# Here's what I did:
# 
# It's really similar to what I did with the Kinematics dataset (finding the sum of all distances in a newly created 'distance' column), but instead of using the distance formula and time elapsed, I'm using the geopy distance package to find the distance between each row using latitude and longitude. It's also important to note that I don't need the first row this time around because that literally is the starting point for each vehicle. Even if there is some time that has elapsed in the first row, I don't know where the car was before that first time, so there's not much I can do about that.
# 
# Hopefully these results will be similar to what I got with the previous dataset!

# So, here's what I did in the code block below. My hands are locking up from typing all of this, so please forgive me if I start to do a little shorthand:
# 
# 1. I had to start off by making a geopy distance calculation function so I could apply it to each of the vehicle dataframes.
# 
# 2. I split the DFs up by vehicle type.
# 
# 3. Add 'next_lat' and 'next_long' columns to the DFs that contain the lats and longs from the row below the current row. This way, I can apply my distance(r) function on each row to easily create a 'distance' column that calculates the distance in each row. I also got rid of all the first rows in each of the dataframes because of the reasons mentioned above.

# In[24]:


gps_df_copy = gps_df.copy()


# In[25]:


import geopy.distance

def distancer(row):
    coords_1 = (row['latitude'], row['longitude'])
    coords_2 = (row['next_latitude'], row['next_longitude'])
    return geopy.distance.distance(coords_1, coords_2).meters

def return_compiled_gps_dfs(df):
    gps = {}
    for name, group in df.groupby('vehicle_id'):
        gps['group_' + str(name)] = group
        gps['group_' + str(name)]['next_latitude'] = gps['group_' + str(name)]['latitude'].shift(1)
        gps['group_' + str(name)]['next_longitude'] = gps['group_' + str(name)]['longitude'].shift(1)
        gps['group_' + str(name)] = gps['group_' + str(name)].dropna()
        gps['group_' + str(name)]['distance'] = gps['group_' + str(name)].apply(distancer, axis=1)
    return gps


# Let's take a look at it. It's good!

# In[26]:


gps = return_compiled_gps_dfs(gps_df_copy)

gps['group_moon']


# In[27]:


for i in gps:
    print_distances(gps[i])
    print()


# Now let's print the kinematic distances again for reference:

# In[28]:


for i in d:
    print_distances(d[i])
    print()


# And let's print the differences:

# In[29]:


def print_differences(df1, df2):
    print('The difference between df1 and df2 for', df1.reset_index().vehicle_id[0].capitalize(), 'is: ', df1.reset_index().groupby('vehicle_id')['vehicle_id','distance'].sum().sum().sum() - df2.reset_index().groupby('vehicle_id')['vehicle_id','distance'].sum().sum().sum(), 'meters')
    print('Which rounds to: ', round(df1.reset_index().groupby('vehicle_id')['vehicle_id','distance'].sum().sum().sum() - df2.reset_index().groupby('vehicle_id')['vehicle_id','distance'].sum().sum().sum()), 'meters')


# In[30]:


for k,g in zip(d,gps):
    print_differences(d[k],gps[g])
    print()


# #### Conclusion: 
# These geopy distances seem pretty close to the distances calculated with the distance formula. The only difference that worries me is the Mette difference. Could this be because of the outlier?

# ## Finding the total distance driven by all the vehicles.

# This is pretty simple. All I have to do is sum the distances from the last question. I did one for the Kinematics DF and one for the GPS DF. Thankfully, they're pretty close. 

# #### Answer:

# In[31]:


def sum_distances(compiled_df):
    
    summed_distance = 0
    for i in compiled_df:
        summed_distance += compiled_df[i].reset_index().groupby('vehicle_id')['vehicle_id','distance'].sum().sum().sum()
    return summed_distance


# In[32]:


print('Kinematics Distance Sum: ', round(sum_distances(d)), 'meters')

print()

print('GPS Distance Sum (using Geopy distance): ', round(sum_distances(gps)), 'meters')


# ### Bonus Round:
# Okay, this is cool and everything, but I'm just not satisfied. I'm still curious about those weird velocity outliers. Let's go ahead and replace those outliers and see if we get different results.
# 
# 1. I will remove the velocity outliers that have a z-score higher than 3 and replace them with the median velocity. Then, I will calculate the distances using the same distance formula we used before. This is also where the kin_df_copy came in handy. I should've done the original distance calculations on the copy dataframe, but it's not a big deal since we can use the copy df here.

# In[33]:


from scipy import stats
kin_df_outliers_removed = kin_df.copy()
median = kin_df_outliers_removed.loc[np.abs(stats.zscore(kin_df_outliers_removed['velocity'])) < 3, 'velocity'].median()
kin_df_outliers_removed['velocity'] = np.where((np.abs(stats.zscore(kin_df_outliers_removed['velocity'])) > 3), median,kin_df_outliers_removed['velocity'])
kin_df_outliers_removed


# The outliers have been removed. Yay.

# In[34]:


print(color.BOLD + color.BLUE + 'Original Description' + color.END)
display(kin_df.groupby(['vehicle_id']).describe())

print()

print(color.BOLD + color.RED + 'Outliers Removed' + color.END)
display(kin_df_outliers_removed.groupby('vehicle_id').describe())


# In[35]:


d_outs = return_compiled_kin_dfs(kin_df_outliers_removed)
d_outs['group_moon']


# In[36]:


for i in d_outs:
    print_distances(d_outs[i])
    print()


# In[37]:


for k,o in zip(d,d_outs):
    print_differences(d[k],d_outs[o])
    print()


# In[38]:


for o,g in zip(d_outs, gps):
    print_differences(d_outs[o], gps[g])
    print()


# Uh oh. Marble got better, but Mette got worse. Now we can say that it's not because of velocity outliers, but something is definitely throwing this distance off.

# In[39]:


print('Kinematics Distance Sum (Outliers Removed): ', round(sum_distances(d_outs)), 'meters')


# #### Conclusion: 
# So, our new sum is a bit different (the last one was 17,009 meters), but we still have some issues that we need to deal with in future experiments. I also wonder if those weird outliers affected the GPS dataset in any way. For example, did the car misreport the gps data as well when it misreported the velocity? Did it say that it was somewhere that it really wasn't during that time? That may be worth looking into in a future experiment.

# ## Now let's try something different: 
# **I am going to show you the starting and ending location of each vehicle, then I am going to tell you the distance between these two locations, and then I'm going to provide a Google Map link to each of these locations so you can see them in better detail.**

# #### Here's the function:

# In[40]:


import ast
import geopy.distance
from IPython.display import HTML

def start_to_end_distance(df):
    """
    This function will calculate the start to end distance for each vehicle and print Google Maps links for each starting and ending location.
    """
    g = df.reset_index().groupby('vehicle_id')
    gps_df_grouped = (pd.concat([g.head(1), g.tail(1)]).drop_duplicates().sort_values('index').reset_index(drop=True))
    gps_grouped_list_names = gps_df_grouped['vehicle_id'].unique().tolist()
    gps_grouped_list = gps_df_grouped[['latitude','longitude']].values.tolist()
    print("Start to End Distances for Each Vehicle:")
    for i,k,n in zip(gps_grouped_list[0::2], gps_grouped_list[1::2], gps_grouped_list_names):
        print (geopy.distance.distance(i, k).meters, "-", n, "distance")
    
    gps_df_grouped2 = gps_df_grouped.copy()
    gps_df_grouped2["lat_long"] = gps_df_grouped2["latitude"].astype(str) + ',+' + gps_df_grouped2["longitude"].astype(str)
    
    def google_maps(req): 

        req.replace(' ','+')
        uri = 'https://www.google.com/maps/place/'
        url = uri+req
        return url

    gps_df_grouped2['google_maps_link'] = google_maps(gps_df_grouped2['lat_long'])
    gps_df_grouped2 = HTML(gps_df_grouped2.to_html(render_links=True, escape=False))
    
    print()
    print('Google Map Dataframe:')
    display(gps_df_grouped2)
    return gps_df_grouped2


# In[41]:


google_maps_df = start_to_end_distance(gps_df)


# #### Let's talk about everything in more detail.

# Let's start this endeavor by grouping the Kinematics dataframe by vehicle. Really, it's already grouped like that, but with a dataset this large, I feel more comfortable doing this just to make sure that all the rows are ordered.

# In[42]:


g = gps_df.reset_index().groupby('vehicle_id')
g.describe()


# Now, let's find the first and last values of each row. We can do this with a simple concatenation and the pandas head and tail functions.

# In[43]:


gps_df_grouped = (pd.concat([g.head(1), g.tail(1)])
   .drop_duplicates()
   .sort_values('index')
   .reset_index(drop=True))

gps_df_grouped


# Let's get the names of each vehicle so we can identify which distance is which when we calculate the geopy distance.

# In[44]:


gps_grouped_list_names = gps_df_grouped['vehicle_id'].unique().tolist()
gps_grouped_list_names


# And let's get the lats and longs so we can calculate the geopy distances.

# In[45]:


gps_grouped_list = gps_df_grouped[['latitude','longitude']].values.tolist()
gps_grouped_list


# We can find the distances using a for loop that zips each first and second set of lats and longs in our gps_grouped_list and prints the distance (along with the name of each vehicle).

# In[46]:


import geopy.distance

for i,k,n in zip(gps_grouped_list[0::2], gps_grouped_list[1::2], gps_grouped_list_names):
    print (geopy.distance.distance(i, k).meters, "-", n, "distance")
    


# And now we have our distances! We can see that the Mette traveled the longest distance away from the starting location, and Moon and Marble ended up almost right where they started. They must have been driving in circles!

# #### Find each starting and ending location on Google Maps
# Now let's see where these locations actually are in the world. We can do this by creating links to Google Maps that include our lats and longs. However, we have to make sure that our lats and longs are formatted correctly since they're going to be used in website links, so let's turn them into strings, get rid of the spaces, and add plus signs where the spaces used to be. And let's do that in a new column called 'lat_long'.

# In[47]:


gps_df_grouped2 = gps_df_grouped.copy()
gps_df_grouped2["lat_long"] = gps_df_grouped2["latitude"].astype(str) + ',+' + gps_df_grouped2["longitude"].astype(str)

gps_df_grouped2


# Now let's create a function where we add our newly formatted 'lat_long' to the Google Maps url and return the full, concatenated url.

# In[48]:


import ast

def google_maps(req): 

    req.replace(' ','+')
    uri = 'https://www.google.com/maps/place/'
    url = uri+req
    return url

gps_df_grouped2['google_maps_link'] = google_maps(gps_df_grouped2['lat_long'])
gps_df_grouped2


# Finally, let's turn our urls into clickable links in our dataframe and display the dataframe. Now we can see where the cars started and completed their rides with the click of a button!

# In[49]:


from IPython.display import HTML

gps_df_grouped2 = HTML(gps_df_grouped2.to_html(render_links=True, escape=False))

gps_df_grouped2


# ## Now that that's done, let's try something else:
# 
# **I am going to find the closest gas station to the last known location of each vehicle and tell you the price of the gas at that station.**
# 
# Methodology:
# 
# 1. I am going scrape the web to find gas stations in or near Ann Arbor (that's where all these cars drove) and add the information about each of these gas stations to a dataframe (this was easily the most challenging part).
# 
# 2. Once I have this information in a dataframe, I am going to compare each of the latitudes and longitudes in this dataframe to the lats and longs of each vehicle's ending location and find which one is the closest (a.k.a. which distance is the smallest).

# #### Here's the function:

# In[50]:


import requests
from bs4 import BeautifulSoup
import re
from ast import literal_eval

def closest_gas(url, gps_df):
     """
    This function will scrape any autoblog.com gas station search web page, find information about each gas station on that page, and then find the closest gas station to each vehicle's ending location.
    """
    htmldata = requests.get(url).text
    soup = BeautifulSoup(htmldata, 'html5lib')
    script_text = soup.find("script", type = 'text/javascript').contents[0]
    model_data = re.compile(r'globalArray\.push(\({.*?}\));', flags=re.S)
    matches = model_data.finditer(script_text)
    match_list = []
    for match in matches:
        match_list.append(match.group(1).replace('\n', ' ').replace('\r', '').replace('\t', '').replace("'\\'", '').replace('({','').replace('})','').replace('"',"").replace('",','').replace("'","").strip().split(','))
    
    flat_match_list = []
    for l in match_list:
        for item in l:
            flat_match_list.append(item)
            
    cars = [c.split(": ", 1) for c in flat_match_list if c]        
            
    def split_at(lst, f):
        inds = [i for i, x in enumerate(lst) if f(x)]
        for i, j in zip(inds, inds[1:]):
            yield lst[i:j]
    
    gas_df = pd.DataFrame([dict(c) for c in split_at(cars, lambda x: "id" in x[0])])
    
    gas_df.columns = [c.replace(' ', '_') for c in gas_df.columns]
    gas_df.columns = [c.replace('_', '') for c in gas_df.columns]
    
    result2 = gas_df.groupby(np.arange(len(gas_df))//2).sum()
    
    result2 = result2.iloc[:-1]
    
    gps_df_coors = gps_df.groupby('vehicle_id')[['vehicle_id','time','latitude','longitude']].tail(1).values.tolist()
    
    result2_info = result2[['name','phone','address','city','zip','regularPrice','lat','long']].values.tolist()
    
    gas_distance_list = []

    for i in gps_df_coors:
        for x in result2_info:
            gas_distance_list.append([i[0], geopy.distance.distance(i[-2:], x[-2:]).meters, x[:-2]])
            
    gas_distance_df = pd.DataFrame(gas_distance_list, columns=['vehicle_id','distance_from_last_location','name'])

    gas_distance_df[['name','phone','address','city', 'zip','price']] = pd.DataFrame(gas_distance_df.name.tolist(), index= gas_distance_df.index)
    
    gas_distance_df = gas_distance_df.loc[gas_distance_df.groupby('vehicle_id').distance_from_last_location.idxmin()].reset_index(drop=True)
    
    return gas_distance_df


# In[51]:


closest_gas('https://www.autoblog.com/ann+arbor+mi-gas-prices/', gps_df_copy)


# #### Let's see what's going on with all this stuff.

# Let's import my least favorite, but consistently most useful package: BeautifulSoup. Sigh.

# In[52]:


import requests
from bs4 import BeautifulSoup


# In[53]:


def getdata(url):
    r = requests.get(url)
    return r.text


# I scraped this website for my gas prices: https://www.autoblog.com/ann+arbor+mi-gas-prices/
# 
# And here's why this part was so challenging. Typically, when you scrape data, the data you're looking for is located within the HTML code on the website. That allows for a clean and easy scrape. However, this data was located within some JavaScript code because it was confined within a map. So instead of a clean and easy scrape, I had to use my second least favorite thing: Regular Expressions. But thankfully, I was able to find the right combination of regular expressions that extracted the text I was looking for (it was trapped within a lot of whitespace, quotation marks, and JavaScript code) and add it to a list.

# In[54]:


import re
from ast import literal_eval

htmldata = getdata("https://www.autoblog.com/ann+arbor+mi-gas-prices/")
soup = BeautifulSoup(htmldata, 'html5lib')
script_text = soup.find("script", type = 'text/javascript').contents[0]
model_data = re.compile(r'globalArray\.push(\({.*?}\));', flags=re.S)
matches = model_data.finditer(script_text)
match_list = []
for match in matches:
    match_list.append(match.group(1).replace('\n', ' ').replace('\r', '').replace('\t', '').replace("'\\'", '').replace('({','').replace('})','').replace('"',"").replace('",','').replace("'","").strip().split(','))
match_list


# Now that I have my list, I need to flatten it because it is a list within a list (a.k.a. a nested list).

# In[55]:


flat_match_list = []
for l in match_list:
    for item in l:
        flat_match_list.append(item)


# In[56]:


flat_match_list


# Now, we have another problem. The data we need is in a list, but each key and value is separated by a colon instead of a comma. So unfortunately, we can't just turn the list into a dictionary and turn the dictionary into a dataframe. Instead, we need to create a function that counts the indexes, splits each index into a new row whenever it sees a new 'id', and prints all that data into our dataframe. I almost got it to work, but it wouldn't split the data on the correct index. 'id' shows up on every 18th index, but the function counted every single key and every single value as 1 index. So I was able to get a dataframe that had all my information, but half of the information was on one row, and the other half was on another row.
# 
# Not great, but also not a life shattering problem. It just meant that I had to do some data cleaning.

# In[57]:


def split_at(lst, f):
    inds = [i for i, x in enumerate(lst) if f(x)]
    for i, j in zip(inds, inds[1:]):
        yield lst[i:j]
    #yield lst[j:]


# In[58]:


cars = [c.split(": ", 1) for c in flat_match_list if c]
cars


# In[59]:


gas_df = pd.DataFrame([dict(c) for c in split_at(cars, lambda x: "id" in x[0])])
gas_df


# **So now let's do some data cleaning.** There are a few things wrong with the dataframe above. The obvious one is that the data is split between rows. However, a problem that I didn't initially notice was the fact that all of the column names have an obscene amount of spaces. I replaced the spaces with underscores so you can see what I mean.

# In[60]:


gas_df.columns = [c.replace(' ', '_') for c in gas_df.columns]


# In[61]:


gas_df


# Not great, right? So let's go ahead and get rid of those underscores so we can call our columns by name.

# In[62]:


gas_df.columns = [c.replace('_', '') for c in gas_df.columns]


# In[63]:


gas_df


# Now, let's smush our rows together so we can finally work with this dataframe.

# In[64]:


result2 = gas_df.groupby(np.arange(len(gas_df))//2).sum()
result2


# Also, let's get rid of the last row because it doesn't contain much useful information.

# In[65]:


result2 = result2.iloc[:-1]
result2


# Now, let's create a list that contains the ending lat and long for each vehicle. I also included the time because that could be useful in future experiments (ex. "if a car drove for less than 10 minutes, it doesn't need to look for a gas station").

# In[66]:


gps_df_coors = gps_df.groupby('vehicle_id')[['vehicle_id','time','latitude','longitude']].tail(1).values.tolist()

for i in gps_df_coors:
    print(i)


# Let's also go ahead and create a list that has information about our scraped gas stations.

# In[67]:


result2_info = result2[['name','phone','address','city','zip','regularPrice','lat','long']].values.tolist()
result2_info


# Now, let's create a nested for loop that calculates the distances for each car and each gas station and adds each distance to a list. It would also be wise to include all of the other information about the cars and the gas stations as well so we can stuff it all in a dataframe.

# In[68]:


gas_distance_list = []

for i in gps_df_coors:
    for x in result2_info:
        gas_distance_list.append([i[0], geopy.distance.distance(i[-2:], x[-2:]).meters, x[:-2]])
gas_distance_list


# Let's stuff everything in a dataframe, clean up the data a little bit (I had the gas station's name, phone number, address, etc. all in a list that only took up one column, so I needed to split that list), and take a look at the first few values of this dataframe. Remember this dataframe includes EVERY calculation from our nested for loop, so it's a little large.

# In[69]:


gas_distance_df = pd.DataFrame(gas_distance_list, columns=['vehicle_id','distance_from_last_location','name'])

gas_distance_df[['name','phone','address','city', 'zip','price']] = pd.DataFrame(gas_distance_df.name.tolist(), index= gas_distance_df.index)

gas_distance_df.head()


# Finally, let's narrow our large, nested for loop dataset down to just the minimum distance for each car (we do this by using a groupby with idxmin) and take a look at the results.

# In[70]:


gas_distance_df.loc[gas_distance_df.groupby('vehicle_id').distance_from_last_location.idxmin()].reset_index(drop=True)


# And there you have it! The gas prices for the closest gas stations to the last location of each car.

# ## And let's end this with a cool dashboard. I am going to plot the paths of each vehicle, provide interactive (and easy to understand) charts that go into more detail about each path, and make it all look nice in a clean Tableau dashboard.
# 
# I did this in Tableau because it's much quicker and easier for all types of audiences to understand. So here's the link to the dashboard I made: https://public.tableau.com/app/profile/jonathan.evans/viz/Cars_16481634364010/AutonomousCarPaths?publish=yes

# #### Thank you for taking the time to look through this notebook. Peace ✌🏾.
