import streamlit as st
import pandas as pd
import numpy as np
import itertools
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import json
from prophet.serialize import model_to_json, model_from_json
import holidays
import altair as alt
import plotly.graph_objs as go
import plotly.figure_factory as ff
import base64
from datetime import datetime
import concurrent.futures
import time
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Forecast App", page_icon="🔮", layout="wide")

# Initialize session state for multipage app
if 'page' not in st.session_state:
    st.session_state.page = "Application"

# Sidebar for navigation
st.sidebar.title("Navigation")
st.session_state.page = st.sidebar.radio("Go to", ["Application", "About"])

# Add custom CSS
st.markdown("""
    <style>
    .reportview-container {
        background: linear-gradient(to right, #f5f7fa, #c3cfe2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #4b6cb7, #182848);
        color: white;
    }
    h1, h2, h3 {
        color: #1e3799;
    }
    .stButton>button {
        color: #4b6cb7;
        border-radius: 50px;
        height: 3em;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_csv(uploaded_file):
    df_input = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8',
                           parse_dates=True,
                           infer_datetime_format=True)
    return df_input

def prep_data(df, date_col, metric_col):
    df_input = df.rename({date_col: "ds", metric_col: "y"}, errors='raise', axis=1)
    st.success("✅ Data prepared successfully:")
    st.info("• The selected date column is now labeled as **ds**\n• The values column is now labeled as **y**")
    df_input = df_input[['ds', 'y']]
    df_input = df_input.sort_values(by='ds', ascending=True)
    return df_input

def show_application_page():
    st.title('📊 Forecast Application')
    st.write('This app enables you to generate time series forecasts using Prophet.')
    st.markdown("""The forecasting library used is **[Prophet](https://facebook.github.io/prophet/)** 🔮.""")
    
    # Application page content will be added here in the next chunks

def show_about_page():
    st.title('About the Forecast App')
    st.write("""
    Welcome to the Forecast App, a powerful tool designed to help you predict future values based on historical time series data. This application was created by Mohit to make time series forecasting accessible and user-friendly.
    
    ## What This App Does
    
    The Forecast App utilizes the Prophet forecasting system, developed by Facebook's Core Data Science team. It allows you to:
    
    1. Upload your time series data
    2. Configure various forecasting parameters
    3. Generate predictions for future time periods
    4. Visualize the results
    5. Evaluate the model's performance
    6. Export the results for further analysis
    
    ## How It Works
    
    ### 1. Data Loading
    - You start by uploading a CSV file containing your time series data.
    - The app expects two main columns: a date column and a value column.
    
    ### 2. Data Preparation
    - The app automatically prepares your data for forecasting by renaming columns to match Prophet's requirements.
    
    ### 3. Model Configuration
    - You can customize various aspects of the forecast model, including:
        - Forecast horizon (how far into the future to predict)
        - Seasonality components (daily, weekly, monthly, yearly)
        - Growth model (linear or logistic)
        - Holiday effects
        - Changepoint and seasonality prior scales
    
    ### 4. Forecasting
    - Once configured, the app fits the Prophet model to your data.
    - It then generates future predictions based on your specifications.
    
    ### 5. Visualization
    - The app provides interactive visualizations of your data and the forecast.
    - You can see the overall trend, seasonal patterns, and forecasted values.
    
    ### 6. Model Validation
    - Cross-validation is performed to assess the model's performance.
    - Various error metrics are calculated and displayed.
    
    ### 7. Hyperparameter Tuning
    - The app can automatically search for the best combination of hyperparameters to improve forecast accuracy.
    
    ### 8. Results Export
    - You can export the forecast, performance metrics, and model configuration for further analysis or reporting.
    
    ## Key Components
    
    1. **Data Upload**: Accepts CSV files with time series data.
    2. **Parameter Configuration**: Allows customization of the forecasting model.
    3. **Forecast Generation**: Produces future predictions based on historical data and selected parameters.
    4. **Interactive Visualizations**: Displays data and forecasts in an easy-to-understand format.
    5. **Cross-Validation**: Evaluates model performance using historical data.
    6. **Hyperparameter Tuning**: Optimizes model parameters for better accuracy.
    7. **Export Functionality**: Enables downloading of results in various formats.
    
    ## How to Use This App
    **[Prophet](https://facebook.github.io/prophet/)**
    1. Start by uploading your time series data in the "Data Loading" section.
    2. Configure the model parameters in the "Parameters Configuration" section.
    3. Generate the forecast using the "Forecast" section.
    4. Explore the visualizations and results.
    5. Optionally, perform cross-validation and hyperparameter tuning to improve the model.
    6. Export your results for further analysis or reporting.
    
    This app is designed to be user-friendly while providing powerful forecasting capabilities. Whether you're a data scientist, business analyst, or just curious about predicting future trends, this tool can help you gain insights from your time series data.
    
    For more detailed information on the Prophet forecasting system, please visit [Prophet's documentation](https://facebook.github.io/prophet/).
    
    If you have any questions or encounter any issues while using this app, please don't hesitate to contact the developer, Mohit.
    """)
def show_application_page():
    st.title('📊 Forecast Application')
    st.write('This app enables you to generate time series forecasts using Prophet.')
    st.markdown("""The forecasting library used is **[Prophet](https://facebook.github.io/prophet/)** 🔮.""")

    st.header('1. Data Loading 📁')
    st.write("Import a time series CSV file.")
    with st.expander("ℹ️ Data format", expanded=False):
        st.info(
            "The dataset can contain multiple columns but you will need to select a column to be used as dates "
            "and a second column containing the metric you wish to forecast. The columns will be renamed as "
            "**ds** and **y** to be compliant with Prophet. Even though we are using the default Pandas date parser, "
            "the ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or "
            "YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric."
        )

    uploaded_file = st.file_uploader('', type='csv')
    
    if uploaded_file is None:
        st.write("Or use sample dataset to try the application")
        sample = st.checkbox("Download sample data from Kaggle")

        if sample:
            st.markdown("""[📥 Download sample data from Kaggle](https://www.kaggle.com/datasets/arashnic/learn-time-series-forecasting-from-gold-price)""")
    else:
        with st.spinner('Loading data... 🕑'):
            df = load_csv(uploaded_file)
    
            st.write("Columns:")
            st.write(list(df.columns))
            columns = list(df.columns)
    
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox("Select date column", index=0, options=columns, key="date")
            with col2:
                metric_col = st.selectbox("Select values column", index=1, options=columns, key="values")

            df = prep_data(df, date_col, metric_col)
    
        if st.checkbox('Chart data', key='show'):
            with st.spinner('Plotting data... 📊'):
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(df.style.highlight_max(axis=0))
                
                with col2:    
                    st.write("Dataframe description:")
                    st.write(df.describe().style.highlight_min(axis=0))

            try:
                line_chart = alt.Chart(df).mark_line().encode(
                    x='ds:T',
                    y="y:Q", 
                    tooltip=['ds:T', 'y']
                ).properties(
                    title="Time series preview"
                ).interactive()
                st.altair_chart(line_chart, use_container_width=True)
            
            except:
                st.line_chart(df.set_index('ds')['y'], use_container_width=True, height=300)

    st.header("2. Parameters Configuration ⚙️")

    with st.container():
        st.write('In this section you can modify the algorithm settings.')
        
        with st.expander("🔭 Forecast Horizon"):
            periods_input = st.number_input('Select how many future periods (days) to forecast.',
            min_value=1, max_value=366, value=90)

        with st.expander("🔄 Seasonality"):
            st.markdown("""The default seasonality used is additive, but the best choice depends on the specific case, therefore specific domain knowledge is required. For more information visit the [documentation](https://facebook.github.io/prophet/docs/multiplicative_seasonality.html)""")
            seasonality = st.radio(label='Seasonality', options=['additive', 'multiplicative'])

        with st.expander("🧩 Trend Components"):
            st.write("Add or remove components:")
            daily = st.checkbox("Daily")
            weekly = st.checkbox("Weekly")
            monthly = st.checkbox("Monthly")
            yearly = st.checkbox("Yearly")

        with st.expander("📈 Growth Model"):
            st.write('Prophet uses by default a linear growth model.')
            st.markdown("""For more information check the [documentation](https://facebook.github.io/prophet/docs/saturating_forecasts.html#forecasting-growth)""")

            growth = st.radio(label='Growth model', options=['linear']) 

            if growth == 'linear':
                growth_settings = {
                    'cap': 1,
                    'floor': 0
                }
                cap = 1
                floor = 0
            else:
                st.info('Configure saturation')

                cap = st.slider('Cap', min_value=0.0, max_value=1.0, step=0.05, value=1.0)
                floor = st.slider('Floor', min_value=0.0, max_value=1.0, step=0.05, value=0.0)
                if floor > cap:
                    st.error('Invalid settings. Cap must be higher than floor.')
                    growth_settings = {}
                elif floor == cap:
                    st.warning('Cap must be higher than floor')
                else:
                    growth_settings = {
                        'cap': cap,
                        'floor': floor
                    }

            if 'df' in locals():
                df['cap'] = cap
                df['floor'] = floor
            
        with st.expander('🏖️ Holidays'):
            countries = ['Country name', 'Italy', 'Spain', 'United States', 'France', 'Germany', 'Ukraine']
            
            with st.container():
                years = [2021]
                selected_country = st.selectbox(label="Select country", options=countries)

                if selected_country != 'Country name':
                    country_holidays = getattr(holidays, selected_country[:2])
                    st.write(f"Holidays for {selected_country}:")
                    for date, name in sorted(country_holidays(years=years).items()):
                        st.write(f"🗓️ {date}: {name}")
                
                holidays = st.checkbox('Add country holidays to the model')
                
        with st.expander('🎛️ Hyperparameters'):
            st.write('In this section it is possible to tune the scaling coefficients.')
            
            seasonality_scale_values = [0.1, 1.0, 5.0, 10.0]    
            changepoint_scale_values = [0.01, 0.1, 0.5, 1.0]

            st.write("The changepoint prior scale determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints.")
            changepoint_scale = st.select_slider(label='Changepoint prior scale', options=changepoint_scale_values)
            
            st.write("The seasonality change point controls the flexibility of the seasonality.")
            seasonality_scale = st.select_slider(label='Seasonality prior scale', options=seasonality_scale_values)    

            st.markdown("""For more information read the [documentation](https://facebook.github.io/prophet/docs/diagnostics.html#parallelizing-cross-validation)""")
    with st.container():
        st.header("3. 🔮 Forecast")
        st.write("Fit the model on the data and generate future predictions.")
        
        if 'df' in locals() and not df.empty:
            col1, col2 = st.columns(2)
            with col1:
                fit_button = st.button("🚀 Initialize and Fit Model", key="fit", help="Fit the Prophet model to your data")
            with col2:
                predict_button = st.button("🔮 Generate Forecast", key="predict", help="Generate future predictions")

            if fit_button:
                if len(growth_settings) == 2:
                    with st.spinner('⏳ Fitting the model...'):
                        try:
                            m = Prophet(seasonality_mode=seasonality, 
                                        daily_seasonality=daily,
                                        weekly_seasonality=weekly,
                                        yearly_seasonality=yearly,
                                        growth=growth,
                                        changepoint_prior_scale=changepoint_scale,
                                        seasonality_prior_scale=seasonality_scale)
                            if holidays:
                                m.add_country_holidays(country_name=selected_country)
                                
                            if monthly:
                                m.add_seasonality(name='monthly', period=30.4375, fourier_order=5)

                            m = m.fit(df)
                            future = m.make_future_dataframe(periods=periods_input, freq='D')
                            future['cap'] = cap
                            future['floor'] = floor
                            
                            # Save the model and future dataframe in session state
                            st.session_state.model = m
                            st.session_state.future = future
                            
                            st.success('✅ Model fitted successfully')
                            st.info(f"The model will produce forecasts up to {future['ds'].max().date()}")
                        except Exception as e:
                            st.error(f"⚠️ An error occurred while fitting the model: {str(e)}")
                else:
                    st.error('⚠️ Invalid configuration. Please check your settings.')

            if predict_button:
                if st.session_state.model is None:
                    st.error("⚠️ Please fit the model first by clicking 'Initialize and Fit Model'.")
                else:
                    try:
                        with st.spinner("🔮 Generating forecast..."):
                            m = st.session_state.model
                            future = st.session_state.future
                            forecast = m.predict(future)
                            st.success('✅ Prediction generated successfully')
                            st.session_state.forecast = forecast
                            
                            # Display forecast summary
                            st.subheader("Forecast Summary")
                            metric1, metric2, metric3 = st.columns(3)
                            metric1.metric("Forecast End Date", forecast['ds'].max().strftime('%Y-%m-%d'))
                            metric2.metric("Forecasted Average", f"{forecast['yhat'].mean():.2f}")
                            metric3.metric("Forecast Range", f"{forecast['yhat'].min():.2f} - {forecast['yhat'].max():.2f}")

                            # Interactive forecast plot
                            st.subheader("Interactive Forecast Plot")
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual', line=dict(color='blue')))
                            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='red')))
                            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(255,0,0,0.2)', name='Upper Bound'))
                            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)', name='Lower Bound'))
                            fig.update_layout(title='Time Series Forecast', xaxis_title='Date', yaxis_title='Value', hovermode='x unified')
                            st.plotly_chart(fig, use_container_width=True)

                            # Forecast components
                            st.subheader("Forecast Components")
                            fig3 = m.plot_components(forecast)
                            st.pyplot(fig3)

                            # Display changepoints if growth is linear
                            if growth == 'linear':
                                st.subheader("Changepoints in Trend")
                                fig2 = m.plot(forecast)
                                a = add_changepoints_to_plot(fig2.gca(), m, forecast)
                                st.pyplot(fig2)

                            # Forecast data table with download option
                            st.subheader("Forecast Data")
                            st.dataframe(forecast.style.highlight_max(axis=0, subset=['yhat', 'yhat_lower', 'yhat_upper']))
                            csv = forecast.to_csv(index=False)
                            st.download_button(
                                label="Download forecast data as CSV",
                                data=csv,
                                file_name="forecast_data.csv",
                                mime="text/csv",
                            )

                    except Exception as e:
                        st.error(f"⚠️ An error occurred while generating the forecast: {str(e)}")
        else:
            st.warning("Please upload a CSV file and configure the parameters before forecasting.")
    st.header('4. 🎯 Model Validation')
    st.write("In this section, you can perform cross-validation of the model and view performance metrics.")

    with st.expander("📘 Explanation"):
        st.markdown("""
        The Prophet library allows us to divide our historical data into training data and testing data for cross-validation. The main concepts are:
        - **Training data (initial)**: The amount of data set aside for training.
        - **Horizon**: The data set aside for validation.
        - **Cutoff (period)**: A forecast is made for every observed point between cutoff and cutoff + horizon.
        """)

    col1, col2, col3 = st.columns(3)
    with col1:
        initial = st.text_input("Initial training period", "365 days")
    with col2:
        period = st.text_input("Forecast period", "90 days")
    with col3:
        horizon = st.text_input("Horizon", "90 days")

    st.info(f"Cross-validation will assess prediction performance on a horizon of **{horizon}**, starting with **{initial}** of training data in the first cutoff and then making predictions every **{period}**.")

    if st.button('🔍 Perform Cross-Validation'):
        if 'model' in st.session_state and st.session_state.model is not None:
            with st.spinner("🕒 Performing cross-validation..."):
                try:
                    df_cv = cross_validation(st.session_state.model,
                                             initial=initial,
                                             period=period,
                                             horizon=horizon,
                                             parallel="threads")
                    df_p = performance_metrics(df_cv)
                    
                    # Convert timedelta to strings
                    for col in df_p.select_dtypes(include=['timedelta64']).columns:
                        df_p[col] = df_p[col].astype(str)
                    
                    st.session_state.df_cv = df_cv
                    st.session_state.df_p = df_p
                    st.success("✅ Cross-validation completed successfully!")
                except Exception as e:
                    st.error(f"❌ An error occurred during cross-validation: {str(e)}")
        else:
            st.warning("⚠️ Please load data and fit the model first.")

    st.subheader('📊 Performance Metrics')

    if 'df_p' in st.session_state and st.session_state.df_p is not None:
        st.dataframe(st.session_state.df_p.style.highlight_min(axis=0))
        
        st.markdown('**Metrics definition**')
        metrics_def = {
            "mse": "Mean squared error",
            "rmse": "Root mean squared error",
            "mae": "Mean absolute error",
            "mape": "Mean absolute percentage error",
            "mdape": "Median absolute percentage error",
            "coverage": "Prediction coverage"
        }
        for metric, definition in metrics_def.items():
            st.write(f"- **{metric}**: {definition}")
        
        metrics_options = ['Choose a metric'] + list(metrics_def.keys())
        selected_metric = st.selectbox("Select metric to plot", options=metrics_options)
        if selected_metric != metrics_options[0]:
            try:
                fig4 = plot_cross_validation_metric(st.session_state.df_cv, metric=selected_metric)
                st.pyplot(fig4)
            except Exception as e:
                st.error(f"Error plotting metric: {str(e)}")
                st.write("Try selecting a different metric or check your data.")
    else:
        st.info("Perform cross-validation to see metrics.")

    st.header('5. 🎛️ Hyperparameter Tuning')
    st.write("In this section, you can find the best combination of hyperparameters.")
    st.markdown("""For more information, visit the [documentation](https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning)""")

    param_grid = {
        'changepoint_prior_scale': [0.01, 0.1, 0.5, 1.0],
        'seasonality_prior_scale': [0.1, 1.0, 5.0, 10.0],
    }

    if st.button('🚀 Start Hyperparameter Tuning'):
        if 'df' in locals() and not df.empty and 'model' in st.session_state and st.session_state.model is not None:
            with st.spinner("🕒 Tuning hyperparameters... This may take a while."):
                try:
                    # Generate all combinations of parameters
                    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
                    rmses = []  # Store the RMSEs for each params here

                    # Use cross validation to evaluate all parameters
                    for params in all_params:
                        m = Prophet(**params).fit(df)  # Fit model with given params
                        df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon, parallel="threads")
                        df_p = performance_metrics(df_cv, rolling_window=1)
                        rmses.append(df_p['rmse'].values[0])

                    # Find the best parameters
                    tuning_results = pd.DataFrame(all_params)
                    tuning_results['rmse'] = rmses
                    best_params = all_params[np.argmin(rmses)]

                    st.success("✅ Hyperparameter tuning completed!")
                    st.subheader("Tuning Results")
                    st.dataframe(tuning_results.style.highlight_min(subset=['rmse']))
                    
                    st.subheader("Best Parameters")
                    for param, value in best_params.items():
                        st.info(f"**{param}**: {value}")

                except Exception as e:
                    st.error(f"❌ An error occurred during hyperparameter tuning: {str(e)}")
        else:
            st.warning("⚠️ Please load data and fit the model first.")          
                            

    st.header('6. 📤 Export Results')

    st.write("Export your forecast results, model configuration, and evaluation metrics.")

    if 'model' in st.session_state and st.session_state.model is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Forecast and Metrics")
            
            if st.button('📊 Export Forecast (.csv)', key='export_forecast'):
                try:
                    with st.spinner("Exporting forecast..."):
                        export_forecast = pd.DataFrame(st.session_state.forecast[['ds','yhat_lower','yhat','yhat_upper']])
                        csv = export_forecast.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="forecast.csv" class="btn">Download Forecast CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        st.success("Forecast exported successfully!")
                except Exception as e:
                    st.error(f"Error exporting forecast: {str(e)}")

            if st.button("📉 Export Model Metrics (.csv)", key='export_metrics'):
                try:
                    with st.spinner("Exporting metrics..."):
                        if 'df_p' in st.session_state and st.session_state.df_p is not None:
                            csv = st.session_state.df_p.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="metrics.csv" class="btn">Download Metrics CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
                            st.success("Metrics exported successfully!")
                        else:
                            st.warning("No metrics available. Please perform cross-validation first.")
                except Exception as e:
                    st.error(f"Error exporting metrics: {str(e)}")

        with col2:
            st.subheader("Model Configuration")
            
            if st.button('💾 Save Model Configuration (.json)', key='save_config'):
                try:
                    with st.spinner("Saving model configuration..."):
                        model_json = model_to_json(st.session_state.model)
                        b64 = base64.b64encode(json.dumps(model_json).encode()).decode()
                        href = f'<a href="data:file/json;base64,{b64}" download="model_config.json" class="btn">Download Model Config JSON</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        st.success("Model configuration saved successfully!")
                except Exception as e:
                    st.error(f"Error saving model configuration: {str(e)}")
                    
        # Add some CSS to style the download buttons
        st.markdown("""
        <style>
        .btn {
            display: inline-block;
            padding: 0.6em 1.2em;
            color: #FFFFFF;
            background-color: #4CAF50;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
            margin: 0.5em 0;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .btn:hover {
            background-color: #45a049;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            transform: translateY(-2px);
        }
        .btn:active {
            background-color: #3e8e41;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
            transform: translateY(1px);
        }
        </style>
    """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please generate a forecast before exporting results.")




    # Initialize session state variables
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'future' not in st.session_state:
        st.session_state.future = None


# Main app logic
if st.session_state.page == "Application":
    show_application_page()
elif st.session_state.page == "About":
    show_about_page()




