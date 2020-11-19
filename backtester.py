import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
import numpy
import numpy as np
import math
import pandas as pd  
import math as m
import gc
from plotly.graph_objs import *
import plotly.graph_objects as go
import plotly as py

import matplotlib.pyplot as plt
import datetime







def percent_change(starting_point, current_point):
    """
    Computes the percentage difference between two points
    :return: The percentage change between starting_point and current_point
    """
    default_change = 0.00001
    try:
        change = ((float(current_point) - starting_point) / abs(starting_point)) * 100.00
        if change == 0.0:
            return default_change
        else:
            return change
    except:
        return default_change
    



    
def pattern_Recognitor(df):

    gc.collect()
        
    accuracy_array = []
        
    pattern_images_folder = "patterns/"
        
    end_point = 50000 
                                            
    plot_data = False
        
    samples = 0
        
    pattern_similarity_value = 70
        
    pattern_array = []
        
    performance_array = []
        
    pattern_for_recognition = []
        
    dots_for_pattern = 30
        
    all_data = df.close.values
        
    average_line = all_data[:end_point]
        
    x = len(average_line) - (30 + dots_for_pattern)
    
    y = 1 + dots_for_pattern
        
    for index in reversed(range(1, dots_for_pattern + 1)):
        pattern = percent_change(all_data[- dots_for_pattern - 1],all_data[- index])
        pattern_for_recognition.append(pattern)
            
        #print(all_data[-1] , all_data[-31])
        
    while y < x:
            
        pattern = []
    
        for index in reversed(range(dots_for_pattern)):
            point = percent_change(average_line[y - dots_for_pattern] , average_line[y - index])
            pattern.append(point)
    
            # Create the pattern array and store it
        pattern_array.append(pattern)
    
            # Take the range of the outcome using 10 values from the 20th after the current point
        outcome_range = average_line[y+4:y+14]
    
            # Take the current point
        current_point = average_line[y]
    
            # Get the average value of the outcome
        try:
            average_outcome = np.average(outcome_range)
        except Exception as e:
            print(e)
            average_outcome = 0
    
            # Get the future outcome for the pattern based on the average outcome value
        future_outcome = percent_change(current_point, average_outcome)
    
            # Store the outcome value
        performance_array.append(future_outcome)
    
        y += 1
        
    found_pattern = False
    
        # Contains the array of patterns to be plotted
    plot_pattern_array = []
    
        # Contains the array of outcomes predicted by the identified patterns
    predicted_outcomes_array = []
    
    for pattern in pattern_array[:-5]:
            # Tells if enough similarities have been found in order to consider the pattern similar to the one currently
            #samples considered
        similarities_are_found = True
    
            # Compute the percent changes for each point of the two patterns, the one that we are considering and the
            # current one obtained from the last 10 entries of the data
        similarities_array = []
        for index in range(dots_for_pattern):
                # Compute the values of similarity only if it's the first value to be computed, or if the previous one was
                # at least 50% similar
            if index == 0 or similarities_array[index - 1] > 0:
                similarities_array.append(100.00 - abs(percent_change(pattern[index], pattern_for_recognition[index])))
    
                # Otherwise just break the for loop and stop computing similarities
            else:
                similarities_are_found = False
                break
    
            # If sufficient similarities were found continue on
        if similarities_are_found:
                # Compute how similar are the two patterns
            how_similar = np.sum(similarities_array) / dots_for_pattern

            if how_similar > pattern_similarity_value:
                    # If a pattern satisfies the similarity value, remember that a pattern was found and append that
                    # pattern to the list of patterns to plot
                found_pattern = True
                plot_pattern_array.append(pattern)
    
    prediction_array = []

    gc.collect()
        
    if found_pattern:
            
            # If at least one similar pattern was found then print all the pattern that are in the array of patterns to be
            # plotted
        xp = np.arange(0, dots_for_pattern, 1)
        if plot_data:
            plt.figure(figsize=(10, 6))
    
        for pattern in plot_pattern_array:
            pattern_index = pattern_array.index(pattern)
    
                # Determine the color based on the prediction of the pattern
            if performance_array[pattern_index] > pattern_for_recognition[dots_for_pattern - 1]:
                    # If the prediction of the pattern is greater than the value of the pattern use the green
                plot_color = '#24BC00'
                prediction_array.append(1.00)
    
            else:
                    # Otherwise use the red
                plot_color = '#D40000'
                prediction_array.append(-1.00)
            predicted_outcomes_array.append(performance_array[pattern_index])
            if plot_data:
                plt.plot(xp, pattern)
                predicted_outcomes_array.append(performance_array[pattern_index])
    
                    # Plot the dot representing the value predicted by the pattern
                    # The color of the dot will be red if the outcome is good, and red otherwise
                plt.scatter(dots_for_pattern + 5, performance_array[pattern_index], c=plot_color, alpha=.3)
    
            # Get the average of 10 future values to determine the chart gait and plot the dot as a reference of what is
            # going to happen
# =============================================================================
##             real_outcome_range = all_data[end_point+20:end_point+30]
##             real_average_outcome = np.average(real_outcome_range)
##             real_movement = percent_change(all_data[end_point], real_average_outcome)
#     
#             if plot_data:
#                 plt.scatter(40, real_movement, s=25, c='#54FFF7')
# =============================================================================
    
            # Get the average of the predicted values and plot a dot representing what the system has predicted will happen
        predicted_average_outcome = np.average(predicted_outcomes_array)
    
        if plot_data:
            plt.scatter(40, predicted_average_outcome, s=25, c='b')
    
            # Also plot the patter that has been recognized with a different line width and color to make it stand out on
            # the graph with all the similar patterns
        if plot_data:
            plt.plot(xp, pattern_for_recognition, '#54FFF7', linewidth=3)
    
            plt.grid(True)
            plt.title("Pattern recognition")
            plt.suptitle("Patterns recognized after {} samples".format(samples))
            plt.savefig(pattern_images_folder + "patter_recognition_{}_samples.png".format(samples))
    
            #print(prediction_array)
    
        prediction_average = np.average(prediction_array)
            #print(prediction_average)
    
        if prediction_average < 0:
            return "sell" , predicted_average_outcome , prediction_array , pattern_for_recognition
            pass
                #print("Drop predicted")
                #print(pattern_for_recognition[29])
# =============================================================================
#                 print(real_movement)
#                 if real_movement < pattern_for_recognition[29]:
#                     accuracy_array.append(100)
#                 else:
#                     accuracy_array.append(0)
# =============================================================================
    
        if prediction_average > 0:
                #print("Rise predicted ")
                #print(pattern_for_recognition[29])
            return "buy" , predicted_average_outcome , prediction_array , pattern_for_recognition
# =============================================================================
#                 print(real_movement)
#                 if real_movement > pattern_for_recognition[29]:
#                     accuracy_array.append(100)
#                 else:
#                     accuracy_array.append(0)
# =============================================================================



import shelve
from datetime import datetime



df = pd.read_csv('BTCUSDT-5m-data.csv')

results = []
for i in range(1,1000):
    print(i)
    try:
        t , predicted_average_outcome , prediction_array , pattern_for_recognition = pattern_Recognitor(df)
        results.append([t , predicted_average_outcome , prediction_array , pattern_for_recognition,df.iloc[-1]])
        df = df[:-i]
    except:
        df = df[:-i]


db = shelve.open("results_5m")
db['data'] = results
db.close()


final = []

for i in range(len(results)):
	if ((results[i][1] - results[i][3][-1] > 0.09) or (results[i][1] - results[i][3][-1] < -0.09)) and len(results[i][2]) > 20:
		final.append([results[i][0],results[i][1] - results[i][3][-1],len(results[i][2]),datetime.fromtimestamp(int(results[i][4]['close_time'])/1000)])




        
