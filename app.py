import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



st.set_page_config(layout="wide",page_title='Customer Segmentation')
 

data = pd.read_csv('Online Retail.csv',encoding='latin1')
data.dropna(inplace=True)
data['CustomerID'] = data['CustomerID'].astype('str')
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['Amount'] = data['Quantity'] * data['UnitPrice']
data = data[data['UnitPrice'] > 0]
data = data[data['Amount'] > 0]

# amount
amount_spent = data.groupby('CustomerID')['Amount'].sum().reset_index()

# frequency
customer_frequency= data.groupby('CustomerID')['InvoiceNo'].count().reset_index()
customer_frequency = customer_frequency.rename(columns={'InvoiceNo':'Frequency'})

# merge amount and frequency
merge_df = pd.merge(amount_spent,customer_frequency,on='CustomerID',how='inner')

# recency
max_date = max(data['InvoiceDate'])
data['diff'] = max_date - data['InvoiceDate']
customer_recency = data.groupby('CustomerID')['diff'].min().reset_index()
customer_recency['diff'] = customer_recency['diff'].dt.days

final_df = pd.merge(merge_df,customer_recency,on='CustomerID',how='inner')
final_df = final_df.rename(columns={'diff':'Recency'})


Q1 = final_df['Amount'].quantile(0.05)
Q3 = final_df['Amount'].quantile(0.95)
IQR = Q3-Q1
final_df = final_df[(final_df['Amount'] >= Q1 - 1.5*IQR) & (final_df['Amount'] <= Q1 + 1.5*IQR)]

Q1 = final_df['Frequency'].quantile(0.05)
Q3 = final_df['Frequency'].quantile(0.95)
IQR = Q3-Q1
final_df = final_df[(final_df['Frequency'] >= Q1 - 1.5*IQR) & (final_df['Frequency'] <= Q1 + 1.5*IQR)]

Q1 = final_df['Recency'].quantile(0.05)
Q3 = final_df['Recency'].quantile(0.95)
IQR = Q3-Q1
final_df = final_df[(final_df['Recency'] >= Q1 - 1.5*IQR) & (final_df['Recency'] <= Q1 + 1.5*IQR)]


scale_features = final_df[['Amount','Frequency','Recency']]

scaling = StandardScaler()
scaled_df = pd.DataFrame(scaling.fit_transform(scale_features))
scaled_df.columns=['Amount','Frequency','Recency']

kmeans = KMeans(n_clusters=3,max_iter=300)
kmeans.fit(scaled_df)
final_df['Clsuter_ID'] = kmeans.predict(scaled_df)



with st.sidebar:
    st.title(":red[Custoner Segmentation]")
    st.link_button(url='https://github.com/gauravbosamiya/Ignite-ML-Intern-Customer-Segmentation-Analysis-with-RFM-and-KMeans-Clustering/blob/main/online-retail-customer-segmentation.ipynb',label='Jupyter notebook')
    add_radio = st.radio(
        "Choose a option",
        ("Dataset Overview", "Exploratory Data Analysis","Customers clusters with RFM")
    )
if add_radio == "Dataset Overview":
    st.title(":red[Customer Dataset]")
    st.dataframe(data,use_container_width=100,hide_index=True)
    st.write("Rows     : ",data.shape[0])
    st.write("Columns  : ",data.shape[1])
    
    st.title(":red[Total Invoice count of each customer]")
    data2 = pd.DataFrame(data.groupby('CustomerID')['InvoiceNo'].count().reset_index())
    st.dataframe(data2,use_container_width=100,hide_index=True)

if add_radio == "Exploratory Data Analysis":
    st.title(":red[Top 10 Products]")
    desc_df = data['Description'].value_counts().reset_index()
    desc_df = desc_df.rename(columns={'index':'Description','Description':'Count'})
    # st.dataframe(desc_df)
    st.bar_chart(data=desc_df[:10],x='Description',y='Count')
    
    st.title(":red[Top 10 Country based on most number of Customers]")
    st.bar_chart(data=data['Country'].value_counts()[:10].sort_values(ascending=False))
    
    
    st.title(":red[Top 10 Country based on less number of Customers]")
    st.bar_chart(data= data['Country'].value_counts()[-10:].sort_values(ascending=True))
    
    
    st.title(":red[Amount spent by customer in every month]")
    data['Month'] = data['InvoiceDate'].dt.month_name()
    month_df = data.groupby('Month')['Amount'].sum().reset_index()
    st.bar_chart(data=month_df,x='Month',y='Amount',color='Month')
    
    
    st.title(":red[Amount spent by customer in Morning/Afternoon/Evening]")
    data['Hour'] = data['InvoiceDate'].dt.hour
    hour_df = data['Hour'].value_counts().reset_index()
    hour_df = hour_df.rename(columns={'index':'Hour','Hour':'Count'})
    
    def convert_time(time):
        if (time >=6 and time<=11):
            return 'Morning'
        elif (time>=12 and time<=17):
            return 'Afternoon'
        else:
            return 'Evening'
        
    hour_df['time_type'] = hour_df['Hour'].apply(convert_time)
    st.bar_chart(data=hour_df,x='time_type',y='Count',color='time_type')
    
    

if add_radio == "Customers clusters with RFM":
    st.header(":red[Create the RFM columns (Recency, Frequency, Monetary value)]")
    
    st.subheader("Recency - How recently did the customer purchase?")
    st.subheader("Frequency - How often do they purchase?")
    st.subheader("Monetary - How much do they spend?")
    
    st.header(":red[How to find RFM Values ?]")
    st.subheader(":blue[Recency:]")
    st.write("In order to find the recency value of each customer, we need to determine the last invoice date as the current date and subtract the last purchasing date of each customer from this date.")
    
    
    st.subheader(":blue[Frequency:]")
    st.write("In order to find the frequency value of each customer, we need to determine how many times the customers make purchases.")

    
    st.subheader(":blue[Monetary:]")
    st.write("In order to find the monetary value of each customer, we need to determine how much do the customers spend on purchases.")
    
    st.title(":red[RFM Dataset]")
    st.dataframe(final_df,hide_index=True,use_container_width=100)
    
    

    st.title(":red[Customers cluster based on Recency, Frequency, Monetary]")    
    fig, axes = plt.subplots(3,1, figsize=(6, 10))

    sns.stripplot(ax=axes[0],x=final_df['Clsuter_ID'],y='Amount',data=final_df,palette='icefire')
    axes[0].set_title('Amount')

    sns.stripplot(ax=axes[1],x=final_df['Clsuter_ID'],y='Frequency',data=final_df,palette='cubehelix')
    axes[1].set_title('Frequency')


    sns.stripplot(ax=axes[2],x=final_df['Clsuter_ID'],y='Recency',data=final_df,palette='dark')
    axes[2].set_title('Recency')
    plt.tight_layout()

    st.pyplot(fig)
    
        