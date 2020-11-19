# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:29:24 2020

@author: Ali
"""


import shelve
import numpy as np
class asset:
    
    def __init__(self , path ):
        
        df1 = pd.read_csv(path)
        try:
            df1 = df1[['open', 'high', 'low', 'close' , 'volume']]
        except:
            df1 = df1[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']]
        df1.columns = ['Open','High','Low', 'Close','Volume']
    
        df1 = df1.dropna()
# =============================================================================
#         df1['RSI_14'] = computeRSI(df1['Close'], 14)
#         #df1 = MFI(df1,14)
#         #df1 = RSI(df1,14)
#         df1 = MACD(df1,12,26)
#         df1 = BBANDS2(df1,21)
#         #df1 = OBV(df1,8)
#         df1 = StochRSI(df1)
# =============================================================================
# =============================================================================
#         Bollingerb_21,BollingerM_21,BollingerB_21 = talib.BBANDS(df1.Close,21)
#         df1['Bollingerb_21'] = Bollingerb_21
#         df1['BollingerM_21'] = BollingerM_21
#         df1['BollingerB_21'] = BollingerB_21
# =============================================================================
        

        df1 = df1.dropna()
        self.asset_name = path[:-12]
        self.df = df1
# =============================================================================
#         self.ma = self.df.BollingerM_21.values
#         self.upper_band = self.df.Bollingerb_21.values
#         self.lower_band = self.df.BollingerB_21.values
#         self.rsi = self.df.RSI_14.values
#         self.macddiff_12_26 = self.df.MACDdiff_12_26.values
#         self.price = self.df.Close.values
#         self.macdsign_12_26 = self.df.MACDsign_12_26.values
#         self.macd_12_26 = self.df.MACD_12_26.values
#         self.volume = self.df.Volume.values
#         self.sok = self.df.sok.values
#         self.sod = self.df.sod.values
# =============================================================================
        self.price_now = 0
        self.buy_price = 0
        self.sell_price = 0
        self.buy_flag = True
        self.sell_flag = False
        self.trade_data = []
        self.buys = []
        self.sells = []
        self.bid_profit = []
        self.log = []
    def update_data(self):

        gc.collect()
        
        data = get_all_binance(self.asset_name,"1m",save = True)
        
        path = self.asset_name + '-1m-data.csv'
            
        df1 = pd.read_csv(path)
        try:
            df1 = df1[['open', 'high', 'low', 'close' , 'volume']]
        except:
            df1 = df1[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']]
        df1.columns = ['Open','High','Low', 'Close','Volume']
    
        df1 = df1.dropna()
# =============================================================================
#         df1['RSI_14'] = computeRSI(df1['Close'], 14)
#         #df1 = MFI(df1,14)
#         #df1 = RSI(df1,14)
#         df1 = MACD(df1,12,26)
#         df1 = BBANDS2(df1,21)
#         #df1 = OBV(df1,8)
#         df1 = StochRSI(df1)
# =============================================================================
# =============================================================================
#         Bollingerb_21,BollingerM_21,BollingerB_21 = talib.BBANDS(df1.Close,21)
#         df1['Bollingerb_21'] = Bollingerb_21
#         df1['BollingerM_21'] = BollingerM_21
#         df1['BollingerB_21'] = BollingerB_21
# =============================================================================
        

        df1 = df1.dropna()
        self.asset_name = path[:-12]
        self.df = df1
# =============================================================================
#         self.ma = self.df.BollingerM_21.values
#         self.upper_band = self.df.Bollingerb_21.values
#         self.lower_band = self.df.BollingerB_21.values
#         self.rsi = self.df.RSI_14.values
#         self.macddiff_12_26 = self.df.MACDdiff_12_26.values
#         self.price = self.df.Close.values
#         self.macdsign_12_26 = self.df.MACDsign_12_26.values
#         self.macd_12_26 = self.df.MACD_12_26.values
#         self.volume = self.df.Volume.values
#         self.sok = self.df.sok.values
#         self.sod = self.df.sod.values
# =============================================================================

    def online_backtest(self):
        
        benefit = []
        
        self.bid_profit = []
        
        i = 0
        
        try:
        
            if self.trade_data[-1][1] == 'buy':
                
                self.trade_data.pop()
        except:
            
            pass
        
        for index,value in enumerate(self.trade_data):
            
            if i > 0 and i%2 != 0:
            
                temp = (value[0] / self.trade_data[index - 1][0]) - 1
                
                self.bid_profit.append(temp)
                
                benefit.append(temp)
                
            i = i + 1
            
        benefit = np.array(benefit).sum() * 100
        
        return benefit            
  
          


    def trend_angle(self , i , period=3):
                    
        m1 = (self.ma[i] - self.ma[i-period]) / period
        
        m2 = 0 
        
        theta = abs((m1 - m2) / (1 + m1 * m2))
        
        theta = math.atan(theta) * 180 / math.pi    
        
        if m1 > 0 :
            
            theta = theta
            
        else:
            
            theta = -theta
            
        return theta
    
    
    def band_angle(self , i , period=3):
                    
        m1 = (self.upper_band[i] - self.upper_band[i-period]) / period
        
        m2 = (self.lower_band[i] - self.lower_band[i-period]) / period
        
        theta = abs((m1 - m2) / (1 + m1 * m2))
        
        theta = math.atan(theta) * 180 / math.pi    
        
        if m1 > m2 :
            
            theta = theta
            
        else:
            
            theta = -theta
            
        return theta
    
    
    def stochrsi_status(self , i , period=1):
        
        m2 = (self.sok[i] - self.sok[i-period]) / period
        
        m1 = (self.sod[i] - self.sod[i-period]) / period
        
        theta = abs((m1 - m2) / (1 + m1 * m2))
        
        theta = math.atan(theta) * 180 / math.pi    
        
        if self.sod[i-1] > self.sok[i-1] and m1 < m2 :
            
            return theta , "buy"
        
        elif self.sod[i-1] < self.sok[i-1] and m1 > m2 :
            
            return theta , "sell"
        
        else:
            
            return 0 , "unknown"
        
        
    def macd_status(self , i , period=1):
        
        m1 = (self.macdsign_12_26[i] - self.macdsign_12_26[i-period]) / period
        
        m2 = (self.macd_12_26[i] - self.macd_12_26[i-period]) / period
        
        theta = abs((m1 - m2) / (1 + m1 * m2))
        
        theta = math.atan(theta) * 180 / math.pi    
        
        if self.macdsign_12_26[i] > self.macd_12_26[i] and m1 < m2 :
            
            return theta , "buy"
        
        elif self.macdsign_12_26[i] < self.macd_12_26[i] and m1 > m2 :
            
            return theta , "sell"
        
        else:
            
            return 0 , "unknown"
        
    
    def plot(self , start , end ):

        self.buys_init()
        self.sells_init()
        self.trade_data_init()
        fig = go.Figure()
        df = self.df[start:end]
        
        plot_end = 0
        for index,value in enumerate(self.trade_data):
            if value[0] > start :
                plot_start = index
                break
        for index,value in enumerate(self.trade_data):
            if value[1] > end :
                plot_end = index - 1
                break
        if plot_end == 0:
            plot_end = end
        X = self.trade_data[plot_start:plot_end + 1]
        l = []
        for item in X:
            l.append([item[0]-start,item[1]-start])
        trade_data = l
        print(len(trade_data))
        
        
        df.index = range(len(df))
        magnify = 50
        trace0 = go.Scatter(x = df.index + start,y = df.Close.values, name = 'Price')
        trace1 = go.Scatter(x = df.index + start,y = df.MACD_12_26.values, name = 'MACD_12_26')
        trace2 = go.Scatter(x = df.index + start,y = df.MACDsign_12_26.values, name = 'MACDsign_12_26')
        trace3 = go.Bar(x = df.index + start,y = df.MACDdiff_12_26.values , name = 'MACDdiff_12_26')
        trace4 = go.Scatter(x = df.index + start,y = df.RSI_14.values , name = 'Rsi')
        trace5 = go.Scatter(x = df.index + start,y = df.BollingerB_21.values , name = 'BollingerB_21')
        trace6 = go.Scatter(x = df.index + start,y = df.Bollingerb_21.values  , name = 'Bollingerb_21')
        trace7 = go.Scatter(x = df.index + start,y = df.BollingerM_21.values , name = 'BollingerM_21')
        trace8 = go.Bar(x = df.index + start,y = df.Volume.values , name = 'MACDdiff_12_26')
        trace9 = go.Scatter(x = df.index + start,y = df.sok.values , name = 'sok')
        trace10 = go.Scatter(x = df.index + start,y = df.sod.values , name = 'sod')
        data = [trace0,trace1,trace2,trace3,trace4]
        fig = py.tools.make_subplots(rows=2,cols=1,shared_xaxes=True)
        fig.add_trace(trace0,1,1)
        fig.add_trace(trace9,2,1)
        fig.add_trace(trace10,2,1)
        
# =============================================================================
#         fig.add_trace(trace4,2,1)
#         fig.add_trace(trace1,3,1)
#         fig.add_trace(trace2,3,1)
#         fig.add_trace(trace3,3,1)
        fig.add_trace(trace5,1,1)
        fig.add_trace(trace6,1,1)
        fig.add_trace(trace7,1,1)
#         fig.add_trace(trace8,4,1)
# =============================================================================
        fig.update_xaxes(range=[start, end])
        fig
        if trade_data != []:
                
            for i in range(len(trade_data)):
                fig.add_annotation(
                                go.layout.Annotation(
                                    x=trade_data[i][0] + start,
                                    y=self.price[X[i][0]],
                                    xref="x",
                                    yref="y",
                                    text="buy",
                                    showarrow=True,
                                    font=dict(
                                        family="Courier New, monospace",
                                        size=12,
                                        color="#ffffff"
                                        ),
                                    align="center",
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=2,
                                    arrowcolor="#636363",
                                    ax=0,
                                    ay=30,
                                    bordercolor="#c7c7c7",
                                    borderwidth=2,
                                    borderpad=4,
                                    bgcolor="green",
                                    opacity=0.8
                                    )
                            )
            for i in range(len(trade_data)):    
                fig.add_annotation(
                                go.layout.Annotation(
                                    x=trade_data[i][1] + start,
                                    y=self.price[X[i][1]],
                                    xref="x",
                                    yref="y",
                                    text="sell",
                                    showarrow=True,
                                    font=dict(
                                        family="Courier New, monospace",
                                        size=12,
                                        color="#ffffff"
                                        ),
                                    align="center",
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=2,
                                    arrowcolor="#636363",
                                    ax=0,
                                    ay=-30,
                                    bordercolor="#c7c7c7",
                                    borderwidth=2,
                                    borderpad=4,
                                    bgcolor="red",
                                    opacity=0.8
                                    )
                            )

            temp = X[-1][1]
            for item in self.buys:
                if temp < item:
                    if item < df.index[-1] + start:
                        temp = item
                        break

            if temp != X[-1][1]:
                    
                fig.add_annotation(
                                go.layout.Annotation(
                                    x=temp,
                                    y=self.price[temp],
                                    xref="x",
                                    yref="y",
                                    text="buy",
                                    showarrow=True,
                                    font=dict(
                                        family="Courier New, monospace",
                                        size=12,
                                        color="#ffffff"
                                        ),
                                    align="center",
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=2,
                                    arrowcolor="#636363",
                                    ax=0,
                                    ay=30,
                                    bordercolor="#c7c7c7",
                                    borderwidth=2,
                                    borderpad=4,
                                    bgcolor="green",
                                    opacity=0.8
                                    )
                            )
            
        plot(fig,show_link = True, filename = self.asset_name + '.html')     
        

        
    
    def buys_init(self):
        
        self.buys = []
     
        l = []
        
        for i in range(len(self.df)):

            if (self.price[i] < self.ma[i] or (self.price[i] < self.ma[i] + .1 * self.ma[i] and self.macd_status(i)[0] > 15)) and (self.macdsign_12_26[i] > self.macd_12_26[i]) and (abs(self.macddiff_12_26[i]) < abs(np.array(self.macddiff_12_26[i-6:i])).sum()/2) and (self.macd_status(i)[0] != 0) and self.trend_angle(i) > -3e-07:
            
                l.append(i)
                
        l = list(set(l))
        
        l.sort()
        
        places = []
        
        for index,item in enumerate(l):
            
            if item == l[index-1] + 1:
                
                places.append(item)
                
        for item in places:
            
            l.remove(item)
        
        self.buys = l



    def sells_init(self):

        self.sells = []
    
        l = []
        
        for i in range(len(self.df)):

            if (self.price[i] > self.ma[i]) and (self.macdsign_12_26[i] < self.macd_12_26[i]) and (abs(self.macddiff_12_26[i]) < abs(np.array(self.macddiff_12_26[i-6:i])).sum()/2) and (self.macd_status(i)[0] != 0) and  (not self.trend_angle(i) > 3e-07):
                
                l.append(i)
            
            if self.price[i] > self.ma[i]:
                
                temp = []
                
                for j in range(0,10):
                    
                    if self.upper_band[i-j] - self.upper_band[i-j-3] < 0 and self.upper_band[i-j-3] - self.upper_band[i-j-6] > 0 :
                        
                        temp.append(True)
                        
                if True in temp and self.rsi[i] > 50:
                    
                    l.append(i)
                    
        l = list(set(l))
        
        l.sort()
        
        places = []
        
        for index,item in enumerate(l):
            
            if item == l[index-1] + 1:
                
                places.append(item)
                
        for item in places:
            
            l.remove(item)
            
        self.sells = l

    
    
    def trade_data_init(self):
        

        self.trade_data = []        
        pre_buy = 0
        
        for i in range(len(self.buys)):
            
            for j in range(len(self.sells)):
                
                if self.sells[j] > self.buys[i] and self.buys[i] > pre_buy:
                    
                    self.trade_data.append([self.buys[i],self.sells[j]])
                    
                    pre_buy = self.sells[j]
                    
                    break 
        
        
    def backtest(self , start , end):
        
        self.buys_init()
        self.sells_init()
        self.trade_data_init()
        backtest_end = 0
        for index,value in enumerate(self.trade_data):
            if value[0] > start :
                backtest_start = index
                break
        for index,value in enumerate(self.trade_data):
            if value[1] > end :
                backtest_end = index - 1
                break
        if backtest_end ==0:
            backtest_end = end
        trade_data = self.trade_data[backtest_start:backtest_end+1]

        benefit_percent = []
        
        for item in trade_data:
            
            percent = (self.price[item[1]]/self.price[item[0]] - 1)*100
            
            if percent < -1 : percent = -1
            
            benefit_percent.append(percent)
        
        benefit = np.array(benefit_percent).sum()
        
        print(benefit_percent)
        
        print(trade_data)
        
        print(benefit)
        
        print(benefit/len(trade_data) , "  percent per deal")
        
        print(len(trade_data) , " total trades")
        
        
    def pattern_Recognitor(self):
        
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
        
        all_data = self.df.Close.values
        
        average_line = all_data[:end_point]
        
        x = len(average_line) - (30 + dots_for_pattern)
    
        y = 1 + dots_for_pattern
        
        for index in reversed(range(1, dots_for_pattern + 1)):
            pattern = percent_change(all_data[- dots_for_pattern - 1],
                                     all_data[- index])
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
            outcome_range = average_line[y+20:y+30]
    
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
    
                if plot_data:
                    plt.plot(xp, pattern)
                    predicted_outcomes_array.append(performance_array[pattern_index])
    
                    # Plot the dot representing the value predicted by the pattern
                    # The color of the dot will be red if the outcome is good, and red otherwise
                    plt.scatter(dots_for_pattern + 5, performance_array[pattern_index], c=plot_color, alpha=.3)
    
            # Get the average of 10 future values to determine the chart gait and plot the dot as a reference of what is
            # going to happen
# =============================================================================
#             real_outcome_range = all_data[end_point+20:end_point+30]
#             real_average_outcome = np.average(real_outcome_range)
#             real_movement = percent_change(all_data[end_point], real_average_outcome)
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
                pass
##                plt.plot(xp, pattern_for_recognition, '#54FFF7', linewidth=3)
##    
##                plt.grid(True)
##                plt.title("Pattern recognition")
##                plt.suptitle("Patterns recognized after {} samples".format(samples))
##                plt.savefig(pattern_images_folder + "patter_recognition_{}_samples.png".format(samples))
    
            #print(prediction_array)
    
            prediction_average = np.average(prediction_array)
            #print(prediction_average)
    
            if prediction_average < 0:
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
                return "buy"
# =============================================================================
#                 print(real_movement)
#                 if real_movement > pattern_for_recognition[29]:
#                     accuracy_array.append(100)
#                 else:
#                     accuracy_array.append(0)
# =============================================================================

        

db = shelve.open('data')

data = db['data']

db.close()



total = []

for i in range(len(data)):
    temp = []
    for j in range(0,len(data[i].log)-1,2):
        if data[i].log[j+1][2] != "buy":
            temp.append((((data[i].log[j+1][0]/data[i].log[j][0]) - 1) * 100) - .1)
    total.append(temp)
        

initial = [10,10,10,10,10,10,10]
j = 0
for item in total:
    
    for i in range(len(item)):
        initial[j] += initial[j] * (item[i]/100)
    j += 1


print(((sum(initial)/70) - 1)*100)

