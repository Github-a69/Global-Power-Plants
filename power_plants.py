
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
import category_encoders 
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from category_encoders import BinaryEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# setting up streamlit dashboard and loading the data and the model 

st.set_page_config(layout='wide',page_title='Power Plants')
st.title('Global Power Plants')

df=pd.read_csv('global_power_clean.csv')

knn = joblib.load('knn.pkl')



# Dividing dashboard into multiple sections


# overview page

page=st.sidebar.radio('select page',['Overview','Analysis','Prediction'])

if page == 'Overview':
    st.image('https://images.unsplash.com/photo-1578776349090-de61da00ff1a?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8cG93ZXIlMjBwbGFudHxlbnwwfHwwfHx8MA%3D%3D',use_container_width=True)
    st.dataframe(df)

    # explaining features meaning
    cols = {'country' :'the country where the power station is located',
                'capacity in MW':'The maximum electrical power the facility can produce in megawatts' ,
                'primary_fuel' :'the primary fuel/technology used (wind, hydro, solar)',
                'geolocation_source':'The source or method used to obtain the facility’s geographic coordinates—such as an official registry, satellite data, or manual geocoding. It indicates how the location was identified, not the location itself',
                'estimated_generation':'The predicted amount of electricity the facility generates in gigawatts'}

    for col, meaning in cols.items():
        with st.sidebar.expander(col):
            st.success(meaning)

# analysis and visualization page

elif page == 'Analysis' :
    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(['KPIs','Distributions','Energy Resources','fuel vs. capacity & generation','Capacity vs. Generation',
                                        'Top 10'])

    with tab1 :

        total_countries = len(df['country'].unique())
        total_geo_sources = len(df['geolocation_source'].unique())
        avg_capacity = round(df['capacity in MW'].mean(),2)
        avg_generation = round(df['estimated_generation'].mean(),2)

        col1,col2 = st.columns(2,gap = 'large')
        col3,col4 = st.columns(2,gap = 'large') 
        col1.metric('Total Countries',total_countries)
        col2.metric('Total Geolocation Sources',total_geo_sources)
        col3.metric('Average Plant Capacity MW',avg_capacity)
        col4.metric('Average Estimated Generation GW',avg_generation)

    with tab2:

        col5,col6= st.columns(2,gap = 'large')


        with col5 :
            fig1 =px.histogram(df,x='capacity in MW',marginal = 'rug',nbins=5,
                        title="Distribution of Plant Capacity",color_discrete_sequence=['cornflowerblue'])
            st.plotly_chart(fig1,use_container_width=True)

        with col6 :
            fig2 =px.histogram(df,x='estimated_generation',marginal = 'rug',nbins=5,
                                    title='Distribution Eestimated Generation',color_discrete_sequence=['cornflowerblue'])
            st.plotly_chart(fig2,use_container_width=True)


    with tab3 :
        fig3 = px.histogram(df,x='primary_fuel',
                        title='Distribution of Primary Fuel',color_discrete_sequence=['dodgerblue'])
        st.plotly_chart(fig3,use_container_width=True)


    with tab4 :


        bar_1= px.bar(df,x = 'primary_fuel',y='capacity in MW',
            color_discrete_sequence=['dodgerblue'],barmode='group')
        st.plotly_chart(bar_1,use_container_width=True)

        bar_2= px.bar(df,x = 'primary_fuel',y='estimated_generation',barmode='group')
        st.plotly_chart(bar_2,use_container_width=True)

        st.markdown('''Hydro power plants are the top in both power capacity and generation, more efforts and focus should be shifted
                    towards solar and wind energy as they are abundunt and available in most countries and regions
                    ''')

    with tab5 :

        scatter = px.scatter(df,y='capacity in MW',x='estimated_generation',trendline='ols',
                     color_discrete_sequence=['steelblue'])
        st.plotly_chart(scatter,use_container_width=True)

        st.markdown('Capacity of power plant correlates strongly with estimated electricity generation')

    with tab6:


        group_1 = df.groupby('country')['capacity in MW'].mean().reset_index().sort_values(by='capacity in MW',ascending=False).head(10)

        bar_3 = px.bar(group_1,y = 'capacity in MW',x ='country',title="TOP 10 Countries by  Average Plant Capacity",
               color_discrete_sequence=['skyblue'])

        st.plotly_chart(bar_3,use_container_width=True)


        group_2 = df.groupby('country')['estimated_generation'].mean().reset_index().sort_values(by='estimated_generation',ascending=False).head(10)

        bar_4 = px.bar(group_2,y = 'estimated_generation',x ='country',title="TOP 10 Countries by  Average Estimated Generation",
               color_discrete_sequence=['cornflowerblue'])

        st.plotly_chart(bar_4,use_container_width=True)


# prediction page

else:
    st.image("https://plus.unsplash.com/premium_photo-1661898205432-d648667b9c76?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8cG93ZXIlMjBwbGFudHxlbnwwfHwwfHx8MA%3D%3D",use_container_width=True)





    # Filters

    st.sidebar.header('Filters')

    country = st.sidebar.selectbox('country',df['country'].unique())
    filtered_geo = df[df['country']==country]['geolocation_source'].unique()
    geo_location = st.sidebar.selectbox('Geolocation',filtered_geo)

    primary_fuel = st.sidebar.radio('Energy Source',df['primary_fuel'].unique())

    capacity = st.sidebar.slider('Power Plant Capacity MW',min_value=float(df['capacity in MW'].min()),
                                                    max_value= float(df['capacity in MW'].max()))
    # Prediction

    if st.button("Predict Electricity Generation"):
        df_prediction = pd.DataFrame(data= [[country,capacity,primary_fuel,geo_location]],
                    columns=df.drop('estimated_generation',axis=1).columns)
        try:
            st.table(df_prediction)
            result = knn.predict(df_prediction)[0]
            prediction = knn.predict(df_prediction)
            st.success(f" Estimated Energy Generation: {prediction[0]:,.2f} gw")
        except Exception as e:
            st.error(f"Prediction Error: {e}")











