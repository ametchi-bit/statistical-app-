import streamlit as st  # Graphical User Interface
from streamlit_option_menu import option_menu
import pandas as pd  # Data exploration
import pickle
import io
import base64
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
# image library
from PIL import Image
# library for analysis
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statistics as stat

# styling plot
plt.style.use('_mpl-gallery')

choose = option_menu(
    menu_title="Dashboard",
    options=["Home", "Project"],
    icons=['house', 'folder', 'files'],
    menu_icon="list", default_index=0,
    orientation='horizontal',
    styles={
        "container-fluid": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "#5D4954", "font-size": "20px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
)
d_ex = Image.open(r'images/data_explo2.png')
d_vis = Image.open(r'images/data-visualization.png')
mch_l = Image.open(r'images/ml.png')

if choose == "Home":
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown(""" <style> @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Roboto:wght@300;400&display=swap');.font-primary {
        font-size:35px; font-family: 'Poppins'; font-weight: 600; color: #5D4954;}
        </style>""", unsafe_allow_html=True)
        st.markdown('<p class="font-primary">About NobleStat</p>', unsafe_allow_html=True)
    with col2:
        st.write('Logo')
    st.write("NobleStat is a Data Science application, which allow a user to perform statistical data analysis. ")

    col3_1, col3_2, col3_3 = st.columns(3)
    st.markdown(""" <style> @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Roboto:wght@300;400&display=swap');.font-secondary {
        font-size:20px; font-family: 'Poppins'; font-weight: 600; color: #5D4954;}
        </style>""", unsafe_allow_html=True)
    with col3_1:
        st.markdown('<p class="font-secondary">Data Exploration</p>', unsafe_allow_html=True)
        st.image(d_ex, width=200)
        st.write("In analysing data, Data exploration refers to the initial step of the analysis in which the data "
                 "analyst use statistical techniques to describe dataset characterization, such as size, quantity and "
                 "accuracy in order to better understand the nature of the data.")
    with col3_2:
        st.markdown('<p class="font-secondary">Data Visualisation</p>', unsafe_allow_html=True)
        st.image(d_vis, width=200)
        st.write(
            "Data visualisation is a form of data exploration which represent information in a form of chart, diagram "
            "picture or histogram.")
    with col3_3:
        st.markdown('<p class="font-secondary">Machine Learning</p>', unsafe_allow_html=True)
        st.image(mch_l, width=200)
        st.write("Machine learning is a type of artificial intelligence that allows software application to become "
                 "more accurate at predicting outcomes without being explicitly programmed to do so.")

elif choose == "Project":
    st.markdown(""" <style> @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Roboto:wght@300;400&display=swap');.font-primary {
        font-size:35px; font-family: 'Poppins'; font-weight: 600; color: #5D4954;}
        </style>""", unsafe_allow_html=True)
    st.markdown('<p class="font-primary">Data Exploration</p>', unsafe_allow_html=True)
    # Using streamlit file Uploader to allow user to upload dataset.
    file = st.file_uploader("Upload Dataset")
    if file is not None:
        extension = file.name.split('.')[1]
        if extension.upper() == 'CSV':
            df = pd.read_csv(file)
        elif extension.upper() == 'XLSX':
            df = pd.read_excel(file, engine='openpyxl')
        elif extension.upper() == 'PICKLE':
            df = pd.read_pickle(file)

        # Using tabs to separate tasks
        tab1, tab2, tab3 = st.tabs(["Dataset", "Visualisation", "Analysis"])

        with tab1:
            st.subheader("Dataset Uploaded")
            st.write(df)


            def pandas():
                # function to explore dataset
                def explore(df):
                    st.sidebar.subheader("Data Information")
                    col1, col2, col3 = st.sidebar.columns(3, gap="small")
                    with col1:
                        btn_info = st.button("Info")
                    with col2:
                        btn_sum = st.button("Summary")
                    with col3:
                        btn_profile = st.button("Report")

                    # Summary
                    if btn_sum:
                        st.write('SUMMARY')
                        st.write(df.describe())

                    # Info
                    if btn_info:
                        st.write('DATASET INFO')
                        buffer = io.StringIO()
                        df.info(buf=buffer)
                        s = buffer.getvalue()
                        st.text(s)

                    # Profile Report
                    if btn_profile:
                        pr = ProfileReport(df, explorative=True)
                        st_profile_report(pr)

                    # ****************** MEASURE OF CENTRAL TENDENCY ****************
                    st.sidebar.subheader("Measure of Central Tendency")
                    col1, col2, col3 = st.sidebar.columns(3)
                    with col1:
                        btn_mct1 = st.button("Mean")
                    with col2:
                        btn_mct2 = st.button("Median")
                    with col3:
                        btn_mct3 = st.button("Mode")
                    if btn_mct1:
                        st.write("Mean:")
                        st.write(df.mean())
                    if btn_mct2:
                        st.write("Median")
                        st.write(df.median())
                    if btn_mct3:
                        st.write("Mode")
                        st.write(df.mode())
                    # ***************** MEASURE OF DISPERSION ******************
                    st.sidebar.subheader("Measure of Dispersion")
                    col1, col2, col3 = st.sidebar.columns(3)
                    with col1:
                        btn_md1 = st.button("STD")
                    with col2:
                        btn_md2 = st.button("Skew")
                    with col3:
                        btn_md3 = st.button("Variance")
                    with col1:
                        btn_corr_coef = st.button("Corrcoef")
                    with col2:
                        btn_mad = st.button("MAD")

                    if btn_md1:
                        st.write("Standard deviation")
                        st.write(df.std())
                    if btn_md2:
                        st.write("Skew")
                        st.write(df.skew())
                    if btn_md3:
                        st.write("Variance")
                        st.write(df.var())
                    if btn_corr_coef:
                        st.write("Correlation Coefficient")
                        st.write(np.corrcoef(df))
                    if btn_mad:
                        st.write("Mean Absolute Deviation")
                        st.write(df.mad(axis=1, skipna=True))

                def download_file(df, types, new_types, extension):
                    for i, col in enumerate(df.columns):

                        new_type = types[new_types[i]]
                        if new_type:
                            try:
                                df[col] = df[col].astype(new_type)
                            except:
                                st.write('Could not convert', col, 'to', new_types[i])

                    # csv
                    if extension == 'csv':
                        csv = df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()

                    # pickle
                    else:
                        b = io.BytesIO()
                        pickle.dump(df, b)
                        b64 = base64.b64encode(b.getvalue()).decode()

                    # download link
                    href = f'<a href="data:file/csv;base64,{b64}" download="new_file.{extension}">Download {extension}</a>'
                    st.write(href, unsafe_allow_html=True)

                def transform(df):
                    # Select sample size
                    frac = st.slider('Random sample (%)', 1, 100, 100)
                    if frac < 100:
                        random_sp = df.sample(frac=frac / 100)

                        st.write(random_sp)
                    # Select columns
                    cols = st.multiselect('Columns',
                                          df.columns.tolist(),
                                          df.columns.tolist())
                    st.write(df[cols])

                    # ******************************* This section of the code is concern with the manipulation of the Table *************

                    cols2_1, cols2_2 = st.columns(2)
                    # Drop Columns
                    with cols2_1:
                        st.markdown("Drop Columns")
                        drop_cols = st.multiselect('Select Column(s)', df.columns)
                        to_drop = df.drop(drop_cols, axis=1)  # function to drop entire column
                        st.write(to_drop)

                    # Drop Row
                    with cols2_2:
                        st.write("Drop Rows")
                        drop_rows = st.multiselect('Select Row(s)', df.index)
                        to_drop = df.drop(drop_rows, axis=0)  # function to drop entire row
                        st.write(to_drop)  # display data new dataframe where the selected row(s) is/are dropped

                    # **************************** Start Missing Values ******************************

                    st.sidebar.write("Replace missing value using median")  # stating what is the section about

                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        btn_fill_1 = st.button("Fill by 0")
                    with col2:
                        btn_fill_2 = st.button("Fill by Median")

                    # Fill N/A by 0
                    if btn_fill_1:
                        st.write("Filled NA by 0:")
                        df.fillna(0, inplace=True)
                        st.write(df)
                    # Replace missing values by Median
                    if btn_fill_2:
                        st.write("Filled by NA by median")
                        median = df.median()
                        df.fillna(median, inplace=True)
                        st.write(df)

                    # ************************* CLose Missing Values Section ******************

                    # ************************* Rename Columns ********************************
                    st.write("Rename Column:")
                    with st.form(key="form"):
                        col_to_change = st.selectbox("Column to change", df.columns)
                        new_col_name = st.text_input("New name", value="")
                        submit_button = st.form_submit_button(label='Submit')

                    if submit_button:
                        df = df.rename(columns={col_to_change: new_col_name})
                        st.write("New Dataframe")
                        st.dataframe(df)

                    df.to_csv(file, sep=",", index=False)

                    # ************************** close Rename section ************************

                    # return df
                    types = {'-': None
                        , 'Boolean': '?'
                        , 'Byte': 'b'
                        , 'Integer': 'i'
                        , 'Floating point': 'f'
                        , 'Date Time': 'M'
                        , 'Time': 'm'
                        , 'Unicode String': 'U'
                        , 'Object': 'O'}
                    new_types = {}

                    expander_types = st.expander('Convert Data Types')
                    for i, col in enumerate(df.columns):
                        txt = 'Convert {} from {} to:'.format(col, df[col].dtypes)
                        expander_types.markdown(txt, unsafe_allow_html=True)
                        new_types[i] = expander_types.selectbox('Field to be converted:'
                                                                , [*types]
                                                                , index=0
                                                                , key=i)

                    st.text(" \n")  # break line
                    # first col 15% the size of the second
                    col1, col2 = st.columns([.15, 1])
                    with col1:
                        btn1 = st.button('Get CSV')
                    with col2:
                        btn2 = st.button('Get Pickle')
                    if btn1:
                        download_file(df, types, new_types, "csv")
                    if btn2:
                        download_file(df, types, new_types, "pickle")

                def my_radio():
                    task = st.sidebar.radio('Task', ['Explore', 'Transform'], 0)
                    if task == 'Explore':
                        explore(df)
                    else:
                        transform(df)

                my_radio()

        # ----------------------------------------- Data Visualisation start
        # -------------------------------------------------
        with tab2:
            def visualisation():
                st.sidebar.title("visualization Tools")
                cols = st.multiselect('Columns', df.columns.tolist(),
                                      df.columns.tolist())

                cols1, cols2, cols3 = st.sidebar.columns(3)
                with cols1:
                    title = st.text_input("Title:")
                with cols2:
                    xLabel = st.text_input("X label:")
                with cols3:
                    yLabel = st.text_input("y label:")

                selected = st.sidebar.selectbox("Basic Plot", ['None',
                                                               'Bar charts',
                                                               'Line chart',
                                                               'Area Charts'
                                                               ]
                                                )

                if selected == 'None':
                    st.write(None)
                if selected == 'Bar charts':
                    st.bar_chart(df[cols])
                if selected == 'Line chart':
                    st.line_chart(df[cols])
                if selected == 'Area Charts':
                    st.area_chart(df[cols])

                # ************************* Matplotlib ****************************************
                selected2 = st.sidebar.selectbox(
                    'Statistics plot',
                    [
                        'None',
                        'Line Chart',
                        'Histogram',
                        'Scatter plot',
                        'Boxplot',
                        'Violin Plot'
                    ]
                )

                if selected2 == 'None':
                    st.write(None)
                # Line Chart

                if selected2 == 'Line Chart':
                    lineFig, ax = plt.subplots()
                    ax.plot(df[cols], linewidth=2.0)
                    plt.title(title)
                    plt.xlabel(xLabel)
                    plt.ylabel(yLabel)
                    st.pyplot(lineFig)

                # Histogram
                if selected2 == 'Histogram':
                    histFig, ax = plt.subplots()
                    ax.hist(df[cols], bins=10)
                    plt.title(title)
                    plt.xlabel(xLabel)
                    plt.ylabel(yLabel)
                    st.pyplot(histFig)

                # Scatter plot
                if selected2 == 'Scatter plot':
                    # size and color:
                    scatter_Fig, ax = plt.subplots()
                    ax.scatter(x=df[cols], y=df[cols])
                    plt.title(title)
                    plt.xlabel(xLabel)
                    plt.ylabel(yLabel)
                    st.pyplot(scatter_Fig)

                # Boxplot
                if selected2 == 'Boxplot':
                    boxFig, ax = plt.subplots()
                    ax.boxplot(x=df[cols], widths=1.5, patch_artist=True)
                    plt.title(title)
                    plt.xlabel(xLabel)
                    plt.ylabel(yLabel)
                    st.pyplot(boxFig)

                # violin plot
                if selected2 == 'Violin Plot':
                    violFig, ax = plt.subplots()
                    ax.violinplot(df[cols])
                    plt.title(title)
                    plt.xlabel(xLabel)
                    plt.ylabel(yLabel)
                    st.pyplot(violFig)

                # ********************** seaborn plots *****************
                selected3 = st.sidebar.selectbox(
                    "Seaborn plot",
                    [
                        "None",
                        "HeatMap"
                    ]
                )

                mySeaFig = plt.figure(figsize=(10, 5))
                if selected3 == 'None':
                    st.write(None)
                elif selected3 == "HeatMap":
                    sns.heatmap(df[cols].corr(), annot=True)
                    plt.title(title)
                    plt.xlabel(xLabel)
                    plt.ylabel(yLabel)
                    st.pyplot(mySeaFig)
        # ----------------------------------------- End Data Visualisation ------------------------------------------
        with tab3:
            def my_model():
                st.sidebar.title("Model Toolbar")  # Title of model toolbar
                st.sidebar.subheader("Select model to perform analysis")

                def linear_reg_model():
                    col_1, col_2 = st.sidebar.columns(2)  # set columns

                    with col_1:
                        col_x = st.selectbox('set x', df.columns)
                    with col_2:
                        col_y = st.selectbox('set y', df.columns)

                    # assigning column to x and y
                    x = df[col_x].values.reshape(-1, 1)
                    y = df[col_y].values.reshape(-1, 1)

                    model = LinearRegression()  # create model
                    model.fit(x, y)  # Fit the model

                    with st.container():
                        with st.expander("Simple linear Regression"):
                            # st.write(np.corrcoef(x, y))
                            r_sq = model.score(x, y)  # coefficient of determination
                            st.markdown("Coefficient of determination $R^{2}$", unsafe_allow_html=True)
                            st.write(r_sq)
                            st.write(f"Intercept:{model.intercept_}")
                            st.write(f"Slope: {model.coef_}")
                            y_pred = model.predict(x)
                            st.write(f"Predicted response:\n{y_pred}")

                            # Plot linear regression

                            st.subheader("plot linear regression")
                            linear_plot = plt.figure(figsize=(10, 5))
                            sns.regplot(x, y, data=df)
                            st.pyplot(linear_plot)

                        # using train and test
                        SEED = 42
                        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
                        # st.write("xtrain", x_train)
                        # st.write("ytrain", y_train)

                        # Training linear model
                        model.fit(x_train, y_train)

                        with st.expander("Simple Linear Regression with X and Y Train"):
                            # st.markdown('$R^{2}$', unsafe_allow_html=True)
                            st.write(model.score(x_train, y_train))
                            st.write(model.intercept_)
                            st.write(model.coef_)

                            # plot linear regression with x_train and y_train
                            linear_plot = plt.figure(figsize=(10, 5))
                            sns.regplot(x_train, y_train, data=df)
                            st.pyplot(linear_plot)

                            # Making prediction
                            y_pred = model.predict(x_test)

                            # compare actual value with predicted value
                            st.write("Compare actual value  with predicted value")
                            if st.button("Compare"):
                                df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
                                st.write(df_preds)

                        with st.expander("Evaluating the Model"):
                            # Evaluate model
                            mae = mean_absolute_error(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)

                            # display evaluated model
                            st.write(f"Mean absolute error:{mae:2f}")
                            st.write(f"Mean squared error: {mse:2f}")
                            st.write(f"Root mean squared error:{rmse:2f}")

                    # Multiple regression
                    col_1, col_2 = st.sidebar.columns(2)  # set columns

                    with col_1:
                        x_col = st.multiselect('Select x', df.columns.tolist(), df.columns.tolist())
                    with col_2:
                        y_col = st.selectbox('Select y', df.columns)

                    # assigning column to x and y
                    y = df[y_col]
                    x = df[x_col]

                    x_train_m, x_test_m, y_train_m, y_test_m = train_test_split(x, y, test_size=0.2, random_state=SEED)

                    model.fit(x_train_m, y_train_m)

                    with st.expander("Multiple Linear Regression"):
                        # getting multiple linear Regression plot using seaborn
                        # variable = [x_col]
                        # for var in variable:
                        # multi_linear_plot = plt.figure()
                        # sns.regplot(x=var, y=y_col, data=df)
                        # st.pyplot(multi_linear_plot)

                        # getting correlation using heatmap
                        correlation = plt.figure(figsize=(10, 5))
                        sns.heatmap(df.corr(), annot=True)
                        st.pyplot(correlation)

                        st.write(model.intercept_)
                        # s t.write(model.coef_)
                        feature_names = x.columns
                        model_coefficient = model.coef_
                        coefficient_df = pd.DataFrame(data=model_coefficient,
                                                      index=feature_names,
                                                      columns=['Coefficient value'])
                        st.write(coefficient_df)
                        m_y_pred = model.predict(x_test_m)
                        results = pd.DataFrame({'Actual': y_test_m, 'Predicted': m_y_pred})
                        st.write(results)

                        # Evaluate model
                        mae = mean_absolute_error(y_test_m, y_pred)
                        mse = mean_squared_error(y_test_m, y_pred)
                        rmse = np.sqrt(mse)

                        # display evaluated model
                        st.write(f"Mean absolute error:{mae:2f}")
                        st.write(f"Mean squared error: {mse:2f}")
                        st.write(f"Root mean squared error:{rmse:2f}")

                #def logistic_reg_model():
                    # st.title("my logistics")

                def polynomial_reg_model():
                    # providing data

                    col_1, col_2 = st.columns(2)  # set columns

                    with col_1:
                        col_x = st.selectbox('set x', df.columns)
                    with col_2:
                        col_y = st.selectbox('set y', df.columns)

                    # assigning column to x and y
                    x = df[col_x].values.reshape(-1, 1)
                    y = df[col_y].values.reshape(-1, 1)

                    # transform input data

                    transformer = PolynomialFeatures(degree=2, include_bias=False)
                    transformer.fit(x)
                    new_x = transformer.transform(x)
                    new_x = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
                    st.write(new_x)

                    # create model and fit
                    polyno_model = LinearRegression()
                    polyno_model.fit(new_x, y)

                   # Printing Result
                    r_sq = polyno_model.score(new_x, y)
                    st.write(f"Coefficient of determination:{r_sq}")
                    st.write(f'intercept: {polyno_model.intercept_}')
                    st.write(f'Coefficients: {polyno_model.coef_}')

                    # Predict response
                    y_pred = polyno_model.predict(new_x)
                    st.write(f"Predicted response:\n{y_pred}")

                def my_radio_analysis():
                    task = st.sidebar.radio('Task', ['Linear Regression', 'Logistics Regression',
                                                     'Polynomial Regression'], 0)
                    if task == 'Linear Regression':
                        linear_reg_model()
                    #if task == 'Logistics Regression':
                        #logistic_reg_model()
                    if task == 'Polynomial Regression':
                        polynomial_reg_model()

                my_radio_analysis()


        def main():
            st.sidebar.title("ToolsBar")
            task = st.sidebar.radio('Task', ['Pandas', 'Visualization', 'Model'], 0)
            if task == 'Pandas':
                pandas()
            if task == 'Visualization':
                visualisation()
            if task == 'Model':
                my_model()


        main()

