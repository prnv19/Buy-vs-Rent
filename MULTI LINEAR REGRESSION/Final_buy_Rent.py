import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import tkinter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_white
import statsmodels.regression.linear_model as lm
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# Reading all CSV files #Dropping uncessary columns 
mortgage_rate_30 = pd.read_csv("MORTGAGE30US.csv")
condo_home_value_index = pd.read_csv("Condo_metro_home_value_index.csv")
Heat_index = pd.read_csv("Monthly Heat index.csv")
single_family_home_value_index = pd.read_csv("Single_family_metro_Home_value_index.csv")
single_family_obeserved_rent_index = pd.read_csv("Single_family_metro_Obeserved_rent_index.csv")
single_family_observed_rent_demand = pd.read_csv("single_family_oberserved_rent_demand.csv")
number_of_listing = pd.read_csv("the_count_of_unique_listing_for_sale.csv")
cpi = pd.read_csv("ConsumerPriceIndex.csv")
unemploymentDf = pd.read_csv("Unemployment.csv")
consumer_sentiment = pd.read_csv("ConsumerSentimentIndex.csv")


cpiDf = pd.DataFrame(cpi)
cpiDf["CPILFESL"] = pd.to_numeric(cpiDf["CPILFESL"], errors='coerce')
cpiDf['DATE'] = pd.to_datetime(cpiDf['DATE'],errors = 'coerce') + pd.offsets.MonthEnd(0)
cpiDf["Montly Inflation"] = cpiDf["CPILFESL"].pct_change() * 100
cpiDf = cpiDf.dropna()

unemploymentDf.columns = ["DATE", "Unemployment rate"]
unemploymentDf = unemploymentDf[4:]
unemploymentDf['DATE'] = pd.to_datetime(unemploymentDf['DATE'], errors='coerce') + pd.offsets.MonthEnd(0)

mortgage_rate_30.columns = ["DATE","30 year Mortage rate"]
mortgage_rate_30['DATE'] = pd.to_datetime(mortgage_rate_30['DATE'], errors='coerce') + pd.offsets.MonthEnd(0)

consumer_sentiment.columns = ["DATE", "Consumer Sentiment Index"]
consumer_sentiment['DATE'] = pd.to_datetime(consumer_sentiment["DATE"], errors='coerce') + pd.offsets.MonthEnd(0)

combinedDF = pd.merge(cpiDf, unemploymentDf, on='DATE', how='inner')
theCombinedDF = pd.merge(combinedDF,mortgage_rate_30, on='DATE', how='inner')
theCombinedDF = theCombinedDF.drop(columns=["CPILFESL"], inplace=False, errors='raise')
the_fina_df = pd.merge(consumer_sentiment,theCombinedDF, on="DATE", how='inner')



condo_home_value_index = condo_home_value_index.drop(condo_home_value_index.loc[:, "1/31/2000":"7/31/2018"].columns, axis=1)
condo_home_value_index = condo_home_value_index[condo_home_value_index["StateName"] == "CO"]
condo_home_value_index = condo_home_value_index.drop(columns=["RegionID","SizeRank","RegionType","StateName"], inplace=False, errors="raise")
single_family_home_value_index = single_family_home_value_index[single_family_home_value_index["StateName"] == "CO"]
single_family_home_value_index = single_family_home_value_index.drop(single_family_home_value_index.loc[:, "1/31/2000":"7/31/2018"].columns, axis=1)
single_family_home_value_index = single_family_home_value_index.drop(columns=["RegionID","SizeRank","RegionType","StateName"], inplace=False, errors="raise")

single_family_obeserved_rent_index 
single_family_obeserved_rent_index  = single_family_obeserved_rent_index.drop(single_family_obeserved_rent_index.loc[:, "1/31/2015":"7/31/2018"].columns, axis=1)
single_family_obeserved_rent_index = single_family_obeserved_rent_index[single_family_obeserved_rent_index["StateName"] == "CO"]
single_family_obeserved_rent_index = single_family_obeserved_rent_index.drop(columns=["RegionID","SizeRank","RegionType","StateName"], inplace=False, errors="raise")

single_family_observed_rent_demand = single_family_observed_rent_demand[single_family_observed_rent_demand["StateName"] == "CO"]
single_family_observed_rent_demand = single_family_observed_rent_demand.drop(columns=["RegionID","SizeRank","RegionType","StateName"], inplace=False, errors="raise")

Heat_index  = Heat_index.drop(Heat_index.loc[:, "1/31/2018":"7/31/2018"].columns, axis=1)
Heat_index = Heat_index[Heat_index["StateName"] == "CO"]
Heat_index = Heat_index.drop(columns=["RegionID","SizeRank","RegionType","StateName"], inplace=False, errors="raise")

number_of_listing  = number_of_listing.drop(number_of_listing.loc[:, "3/31/2018":"7/31/2018"].columns, axis=1)
number_of_listing = number_of_listing[number_of_listing["StateName"] == "CO"]
number_of_listing = number_of_listing.drop(columns=["RegionID","SizeRank","RegionType","StateName"], inplace=False, errors="raise")

number_of_listing = number_of_listing.iloc[:, :-2]
Heat_index = Heat_index.iloc[:, :-2]
condo_home_value_index = condo_home_value_index.iloc[:, :-2]
single_family_home_value_index = single_family_home_value_index.iloc[:, :-2]
the_fina_df = the_fina_df.iloc[:-1]
single_family_obeserved_rent_index = single_family_obeserved_rent_index.iloc[:, :-2]

single_family_obeserved_rent_index = single_family_obeserved_rent_index[~single_family_obeserved_rent_index["RegionName"].isin([
    'Edwards, CO', 'Durango, CO', 'Montrose, CO', 'Breckenridge, CO', 'Steamboat Springs, CO', 'Ca-¦on City, CO','Craig, CO','Sterling, CO','Fort Morgan, CO','Glenwood Springs, CO',
'Pueblo, CO'])]
condo_home_value_index = condo_home_value_index[~condo_home_value_index["RegionName"].isin([
    'Edwards, CO', 'Durango, CO', 'Montrose, CO', 'Breckenridge, CO', 'Steamboat Springs, CO', 'Ca-¦on City, CO','Craig, CO','Sterling, CO','Fort Morgan, CO','Glenwood Springs, CO',
'Pueblo, CO'])]
Heat_index = Heat_index[~Heat_index["RegionName"].isin([
    'Edwards, CO', 'Durango, CO', 'Montrose, CO', 'Breckenridge, CO', 'Steamboat Springs, CO', 'Ca-¦on City, CO','Craig, CO','Sterling, CO','Fort Morgan, CO','Glenwood Springs, CO',
'Pueblo, CO'])]
single_family_home_value_index = single_family_home_value_index[~single_family_home_value_index["RegionName"].isin([
    'Edwards, CO', 'Durango, CO', 'Montrose, CO', 'Breckenridge, CO', 'Steamboat Springs, CO', 'Ca-¦on City, CO','Craig, CO','Sterling, CO','Fort Morgan, CO','Glenwood Springs, CO',
'Pueblo, CO'])]
single_family_observed_rent_demand = single_family_observed_rent_demand[~single_family_observed_rent_demand["RegionName"].isin([
    'Edwards, CO', 'Durango, CO', 'Montrose, CO', 'Breckenridge, CO', 'Steamboat Springs, CO', 'Ca-¦on City, CO','Craig, CO','Sterling, CO','Fort Morgan, CO','Glenwood Springs, CO',
'Pueblo, CO'])]
number_of_listing = number_of_listing[~number_of_listing["RegionName"].isin([
    'Edwards, CO', 'Durango, CO', 'Montrose, CO', 'Breckenridge, CO', 'Steamboat Springs, CO', 'Ca-¦on City, CO','Craig, CO','Sterling, CO','Fort Morgan, CO','Glenwood Springs, CO',
'Pueblo, CO'])]

#Reshaping 
number_of_listing_long = number_of_listing.melt(id_vars="RegionName", var_name="Time", value_name="NumberPfListings")
condo_home_value_index_long = condo_home_value_index.melt(id_vars="RegionName", var_name="Time", value_name="CondoHomeValue")
single_family_home_value_index_long = single_family_home_value_index.melt(id_vars="RegionName", var_name="Time", value_name="SingleFamilyHomeValueIndex")
single_family_obeserved_rent_index_long = single_family_obeserved_rent_index.melt(id_vars="RegionName", var_name="Time", value_name="SingleFamilyRentIndex")
single_family_observed_rent_demand_long = single_family_observed_rent_demand.melt(id_vars="RegionName", var_name="Time", value_name="SingleFamilyRentDemand")
Heat_index_long = Heat_index.melt(id_vars="RegionName", var_name="Time", value_name="HeatIndex")


#Mergin all datasets 
combined_Zillow_long = pd.merge(number_of_listing_long,condo_home_value_index_long, on=["RegionName", "Time"], how="inner")
combined_Zillow_long = pd.merge(combined_Zillow_long,single_family_home_value_index_long, on=["RegionName", "Time"], how="inner")
combined_Zillow_long = pd.merge(combined_Zillow_long,single_family_obeserved_rent_index_long, on=["RegionName", "Time"], how="inner")
combined_Zillow_long = pd.merge(combined_Zillow_long,Heat_index_long, on=["RegionName", "Time"], how="inner")

#combined_Zillow_long = pd.merge(combined_Zillow_long,single_family_observed_rent_demand_long, on=["RegionName", "Time"], how="inner")
the_fina_df.rename(columns={"DATE":"Time"}, inplace=True)

#making sure that both Dfs are in datetime formate 
the_fina_df["Time"] = pd.to_datetime(the_fina_df["Time"]) 
combined_Zillow_long["Time"] = pd.to_datetime(combined_Zillow_long["Time"])
the_final_combined_df = pd.merge(the_fina_df,combined_Zillow_long, on="Time", how="left")

column_order = [
    "RegionName",               
    "Time",                     
    "NumberPfListings",
    "HeatIndex",
    "SingleFamilyHomeValueIndex",
    "SingleFamilyRentIndex",
    "Consumer Sentiment Index", 
    "Montly Inflation",
    "Unemployment rate",
    "30 year Mortage rate"
]


new_order_df = the_final_combined_df[column_order]

#print(new_order_df.columns)
#new_order_df.to_excel("data_excel.xlsx", index=False)





####Liner Regression 

new_order_df["Time Index"] = np.arange(len(new_order_df))
new_order_df = pd.get_dummies(new_order_df, columns=["RegionName"], drop_first=True)

#print(len(new_order_df))
X_sales = new_order_df.drop(columns=["Time","SingleFamilyHomeValueIndex","SingleFamilyRentIndex"])
Y_sales = new_order_df["SingleFamilyHomeValueIndex"]

print(X_sales.head())
X_sales_train, X_sales_test, Y_sales_train, Y_sales_test = train_test_split(X_sales,Y_sales, test_size=0.2, shuffle=False)
salesModel = LinearRegression()
salesModel.fit(X_sales_train,Y_sales_train)
y_sales_pred = salesModel.predict(X_sales_test)

#Evaluation of Sales 
sales_mse = mean_squared_error(Y_sales_test,y_sales_pred)
sales_r2 = r2_score(Y_sales_test,y_sales_pred)
sales_mae = mean_absolute_error(Y_sales_test, y_sales_pred)
sales_mape = np.mean(np.abs((Y_sales_test - y_sales_pred) / np.where(Y_sales_test != 0, Y_sales_test, np.nan))) * 100
sales_n = X_sales_test.shape[0]
sales_p = X_sales_test.shape[1]

adjusted_R2_sales = 1 - ((1-sales_r2) * (sales_n -1)) / (sales_n - sales_p -1)

print(f"Mean Squered Error (MSE): {sales_mse}")
print(f"R-Squered (R^2): {sales_r2}")
print(f"Adjusted R-Squared: {adjusted_R2_sales}")
print(f"Mean Absolute Error (MAE): {sales_mae}")
print(f"Mean Absolute Percentage Error (MAPE): {sales_mape}%")

sales_n = X_sales_test.shape[0]
sales_p = X_sales_test.shape[1]

adjusted_R2_sales = 1 - ((1-sales_r2) * (sales_n -1)) / (sales_n - sales_p -1)



# Linear regression for rent
X_rent = new_order_df.drop(columns=["Time","SingleFamilyHomeValueIndex","SingleFamilyRentIndex"])
Y_rent = new_order_df["SingleFamilyRentIndex"]

X_rent_train, X_rent_test, Y_rent_train, Y_rent_test = train_test_split(X_rent,Y_rent, test_size=0.2, shuffle=False)
rentModel = LinearRegression()
rentModel.fit(X_rent_train,Y_rent_train)
y_rent_predict = rentModel.predict(X_rent_test)

rent_mse = mean_squared_error(Y_rent_test,y_rent_predict)
r2_rent = r2_score(Y_rent_test,y_rent_predict)
rent_mae = mean_absolute_error(Y_rent_test, y_rent_predict)
rent_mape = np.mean(np.abs((Y_rent_test - y_rent_predict) / np.where(Y_rent_test != 0, Y_rent_test, np.nan))) * 100

rent_n = X_rent_test.shape[0]
rent_p = X_rent_test.shape[1]

adjusted_R2_rent = 1 - ((1-r2_rent) * (rent_n -1)) / (rent_n - rent_p -1)
print(f"Mean Squered Error (MSE): {rent_mse}")
print(f"R-Squered (R^2): {r2_rent}")
print(f"Adjusted R-Squared: {adjusted_R2_rent}")
print(f"Mean Absolute Error (MAE): {rent_mae}")
print(f"Mean Absolute Percentage Error (MAPE): {rent_mape}%")


def modelevaluator(m):
    sensitivity = ((m[0][0])/sum([m[0][0],m[0][1]]))*100
    specificity = ((m[1][1])/sum([m[1][1],m[1][0]]))*100
    accuracy = (sum([m[0][0],m[1][1]])/np.sum(m))*100
    error_rate = (sum([m[1][0],m[0][1]])/np.sum(m))*100
    precision = ((m[0][0])/sum([m[0][0],m[1][0]]))*100
    recall = ((m[0][0])/sum([m[0][0],m[0][1]]))*100
    F1 = 2 * (precision * recall)/(precision+recall)
    return {
    "Sensitivity": sensitivity,
    "specifity": specificity,
    "accuracy": accuracy,
    "error_rate": error_rate,
    "precision": precision,
    "recall": recall,
    "F1": F1}

# plotting both Models 

plt.figure(figsize=(10,6))
plt.scatter(Y_sales_test,y_sales_pred, alpha=0.6, color='blue')
plt.plot([min(Y_sales_test),max(Y_sales_test)], [min(Y_sales_test), max(Y_sales_test)],
         color = "red", linestyle ='--', linewidth=2)
plt.xlabel("Actual Sales Value Index")
plt.ylabel("Predicated Sales Value Index")
plt.title("Single Family Value Index: Actual vs Predicted")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(Y_rent_test, y_rent_predict, alpha=0.6, color='blue', label="Predictions")
plt.plot([min(Y_rent_test), max(Y_rent_test)], 
         [min(Y_rent_test), max(Y_rent_test)], 
         color="red", linestyle="--", linewidth=2, label="Ideal Fit")
plt.xlabel("Actual Rent Value")
plt.ylabel("Predicted Rent Value")
plt.title("Single Family Rent Index: Actual vs Predicted")
plt.legend()
plt.show()

# Cheking for non-constant vaiance

salesResidual = Y_sales_test - y_sales_pred
rentResidual = Y_rent_test - y_rent_predict
#plt.scatter(Y_sales_test,salesResidual)
#plt.axhline(0,color ="red", linestyle = '--')
#plt.xlabel("Fitted Sale Value")
#plt.ylabel("Sales Residuals")
#plt.title("Sales Residual Vs Fitted Sales Value")
#plt.show()


#plt.scatter(Y_rent_test,rentResidual)
#plt.axhline(0,color ="red", linestyle = '--')
#plt.xlabel("Fitted Rent Value")
#plt.ylabel("Rent Residuals")
#plt.title("Rent Residual Vs Fitted Rent Value")
#plt.show()


#Checking for colliniarity
durbinWatson_sales = durbin_watson(salesResidual)
durbinWatson_rent = durbin_watson(rentResidual)

#OLS and GLS separate altered data (go means OLS)

go_x_sales = new_order_df.drop(columns=["Time","SingleFamilyHomeValueIndex","SingleFamilyRentIndex"])
go_y_sales = new_order_df["SingleFamilyHomeValueIndex"]

go_x_sales= go_x_sales.applymap(lambda x: int(x) if isinstance(x, bool) else x)
#go_x_sales = go_x_sales.apply(pd.to_numeric, errors = 'coerce').dropna()
#go_y_sales = go_y_sales.apply(pd.to_numeric, errors= 'coerce').dropna()
go_x_sales, go_y_sales = go_x_sales.align(go_y_sales, join='inner', axis=0)

#Splitting
go_x_sales_train, go_x_sales_test, go_y_sales_train, go_y_sales_test = train_test_split(go_x_sales,go_y_sales, test_size=0.2, shuffle=False)

go_x_sales_train_const = sm.add_constant(go_x_sales_train)
go_y_sales_train = go_y_sales_train.astype(float)

sales_ols_model = lm.OLS(go_y_sales_train, go_x_sales_train_const).fit()

weight_gls = 1 / (np.abs(sales_ols_model.fittedvalues) + 1e-5)
sales_Gls_model = sm.GLS(go_y_sales_train, go_x_sales_train_const, sigma=weight_gls).fit()
#print(go_x_sales.dtypes)
#print(go_x_sales.isnull().sum())
go_x_sales_test_const = sm.add_constant(go_x_sales_test)
go_y_sales_pred = sales_Gls_model.predict(go_x_sales_test_const)






