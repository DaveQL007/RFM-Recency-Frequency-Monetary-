#!/usr/bin/env python
# coding: utf-8

# # RFM ANALYSIS OF SALES DATA FOR AN ONLINE STORE

# In[3]:


# IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


#IMPORTING THE DATASETS
df1 = pd.read_excel("C:\\Users\\lenovo\\Desktop\\2009_2010.xlsx")
df2 = pd.read_excel("C:\\Users\\lenovo\\Desktop\\2010_2011.xlsx")
print(df1.head())
print(df2.head())


# In[9]:


# COMBINING MY DATASETS
df = pd.concat([df1,df2], ignore_index = True)
df


# In[39]:


# DEFINE RFM METRICS
df['OrderDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalAmount'] = df['Quantity'] * df['Price']
df['TotalAmount']
 


# In[21]:


# RECENCY
recency_df = df.groupby('Customer ID')['OrderDate'].max().reset_index() 
recency_df.columns = ['Customer ID' , 'last_order_date']
recency_df['recency'] = (recency_df['last_order_date'].max() - recency_df['last_order_date']).dt.days
recency_df['recency']                               


# In[43]:


# FREQUENCY
frequency_df = df.groupby('Customer ID')['StockCode'].count().reset_index() 
frequency_df.columns = ['Customer ID' , 'frequency']
frequency_df
frequency_sort = frequency_df.sort_values(by='frequency', ascending = False)
frequency_sort.head()


# In[41]:


# MONETARY
monetary_df =  df.groupby('Customer ID')['TotalAmount'].sum().reset_index()
monetary_df.columns = ['Customer ID' , 'TotalAmount']
monetary_sort = monetary_df.sort_values(by='TotalAmount', ascending = False)
monetary_sort.head()


# In[45]:


# MERGING RFM METRICS
rfm_merge = pd.merge(recency_df,frequency_df,on = 'Customer ID')
rfm_merged = pd.merge(rfm_merge,monetary_df,on = 'Customer ID')
rfm_merged


# In[52]:


#MAKING THE RESULTS SUITABLE FOR CALCULATIONS(NORMALIZATION)
rfm_merged['recency_score'] = pd.qcut(rfm_merged['recency'],5, labels = [5,4,3,2,1])
rfm_merged['frequency_score'] = pd.qcut(rfm_merged['frequency'],5, labels = [1,2,3,4,5])                                  
rfm_merged['monetary_score'] = pd.qcut(rfm_merged['TotalAmount'],5, labels = [1,2,3,4,5])


# In[57]:


# CALCULATING RFM SCORE
rfm_merged['rfm_score'] = rfm_merged['recency_score'].astype(str) + rfm_merged['frequency_score'].astype(str) + rfm_merged['monetary_score'].astype(str)
rfm_merged['rfm_score']


# In[77]:


#CUSTOMER SEGMENTATION
def segment_customers(score):
    if score >= '500':
        return 'Best Customers'
    elif score >= '400':
        return 'Loyal Customers'
    elif score >= '220':
        return 'Potential Upgrades'
    else:
        return'Once in a while'
rfm_merged['segments'] = rfm_merged['rfm_score'].apply(segment_customers)
rfm_merged['segments']


# In[78]:


#CUSTOMER DISTRIBUTION
customer_distribution = rfm_merged['segments'].value_counts(normalize=True) * 100
customer_distribution


# In[83]:


#PRINTING OUR RESULTS
print('Customer Segmentation : ')
print(rfm_merged.head())
print('\ncustomer_distribution')
print(customer_distribution)


# In[91]:


#VISUALIZING OUR RESULTS
rfm_merged = rfm_merged.sort_values(by='segments', ascending = False)
plt.figure(figsize=(10,5))
sns.countplot(x='segments', data=rfm_merged)
plt.title('Customer Distribution')
plt.xlabel('Segment')
plt.ylabel('Count')
plt.show()


# In[100]:


#VISUALIZING RFM METRICS
fig,axs = plt.subplots(1,3,figsize=(15,5))
sns.histplot(rfm_merged['recency'], ax = axs[0],  kde = True)
axs[0].set_title('Recency')
sns.histplot(rfm_merged['frequency'], ax = axs[1], kde = True)
axs[1].set_title('Frequency')
sns.histplot(rfm_merged['TotalAmount'], ax = axs[2],  kde = True)
axs[2].set_title('Monetary')
plt.tight_layout()
plt.show()


# In[106]:


# SAVE RESULTS TO EXCEL
rfm_merged.to_excel('rfm_analysis.xlsx', index=True)


# In[ ]:


metric = ['recency','frequency','TotalAmount']
fig,axs = plt.subplots(1,len(metric),figsize=(15,5))
for i,metrics in enumerate(metric):
    sns.histplot(rfm_merged[metric], ax = axs[i], kde = True)
    axs[i].set_xlim([0,rfm_merged[metric].max() * 1.1])
    
plt.tight_layout()
plt.show()

