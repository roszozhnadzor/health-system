##передается и файл, и специальность, показывается меню для терапевта

from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import dash
# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.packages import importr

import pandas as pd
import dash_bootstrap_components as dbc
import base64
import plotly.graph_objs as go
import numpy as np
import io
import plotly.io as pio 
from plotly.subplots import make_subplots
import plotly.express as px
# from upsetplot import UpSet
import numpy as np
import random
from sqlalchemy import create_engine
import pandas as pd
from dash.exceptions import PreventUpdate
import os


if not os.path.exists("images"):
    os.mkdir("images")

# Создание подключения к базе данных SQLite
engine = create_engine('sqlite:///mydatabase.db')


app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

# Layout for the first page
page_1_layout = dbc.Container([
    dbc.Row([
        html.Div('Система сопровождения диагностики заболеваний на основе медицинских анализов', 
                 className="text-primary text-center fs-3",
),
                 
    ], justify="center"),
    dbc.Row([
        dcc.Upload( id='output-data-status', className="text-primary text-center fs-3")
    ], justify="center"),
    
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Перетащите файл сюда или ',
                    html.A('выберите файл')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    #'margin': '10px',
                    'backgroundColor': '#78c2ad45',
                    'color': '#008001',
                    'font-size': '16px',
                    'font-family': 'Arial, sans-serif',
                    'cursor': 'pointer'
                },
                multiple=False
            ),
            dcc.Dropdown(
                id='specialty-dropdown',
                options=[
                    {'label': 'Педиатр', 'value': 'Педиатр'},
                    {'label': 'Офтальмолог', 'value': 'Офтальмолог'},
                    {'label': 'Аллерголог', 'value': 'Аллерголог'}
                ],
                placeholder="Выберите вашу специальность",
                style={'margin-top': '10px'}
            ),
            html.Button("Проанализировать результаты", id='analyze-results-button', className="btn btn-success mt-3")
        ], width=12),
    ]),
], fluid=True)


# Layout for the second page
page_2_layout = dbc.Container([
    dbc.Row([
        html.Div('Страница с результатами для специальности', className="text-primary text-center fs-3"),
        html.Div(id='specialty-text', className="text-primary text-center fs-3"),
    ]),
    html.Div(id='output-data-upload'),
    html.Div(id='dropdown-container'),  # Добавлен контейнер для выпадающих списков
    html.Div(id='output-graph'),
    dbc.Row([  
        dbc.Col([
            html.Div(id="download-buttons")], width=12, className="text-center mb-3"),
    
    ]),
    dcc.Download(id="download-image"),
    dbc.Row([
        dbc.Col([
            html.Button("К выбору специальности", id='return-to-specialty-selection-button', className="btn btn-primary mt-3")
        ], width=12, className="text-center bottom mb-3")
    ]),

], fluid=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    dcc.Store(id='file-store'),
    dcc.Store(id='file-store1'),
])


# Callback to switch between pages
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return html.Div([
            dcc.Link('Перейти на страницу с загрузкой файла', href='/page-1'),
        ])

# Callback to process uploaded data on the first page
@app.callback(
    Output('file-store1', 'data'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def process_uploaded_file(contents, filename):
    if contents is not None and filename is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'xlsx' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Возвращаем данные в хранилище
        return {'file_content': contents, 'file_name': filename, 'data': df.to_dict('records')}
    else:
        # Если файл не загружен, возвращаем None
        return None

# Callback to display uploaded file on the first page
@app.callback(
    Output('output-data-status', 'children'),
    [Input('file-store1', 'data')]
)
def display_uploaded_file_status(data):
    if data:
        return "Данные загружены"
    else:
        return "Данные отсутствуют"
    
# Callback to process uploaded data and redirect to the second page
@app.callback(
    [Output('url', 'pathname'),
     Output('file-store', 'data')
      #Output('dropdown-container', 'data')
      ],
    [Input('analyze-results-button', 'n_clicks')],
    [State('specialty-dropdown', 'value'),
     State('upload-data', 'contents'),
     State('upload-data', 'filename')
     ]
)
def update_output(n_clicks, specialty, contents, filename):
    dropdowns = []  # Список для элементов выпадающих списков
    
    print("n_clicks:", n_clicks)
    print("specialty:", specialty)
    print("contents:", contents)
    print("filename:", filename)
    
    if n_clicks and specialty and contents is not None and filename is not None:
        selected_specialty = {'Педиатр': 'Педиатр', 'Офтальмолог': 'Офтальмолог', 'Аллерголог': 'Аллерголог', 'Терапевт': 'Терапевт'}.get(specialty, '')
        
        print("selected_specialty:", selected_specialty)
        
        if selected_specialty == 'Педиатр':  # Проверка, является ли специальность "Педиатр"
            dropdowns = [  # Создание выпадающих списков только для "Педиатр"
                dcc.Dropdown(
                    id='dropdown-statistics',
                    options=[
                        {'label': 'Выберите визуализацию', 'value': ''},
                        {'label': 'Общая статистика', 'value': 'stat'},
                        {'label': 'Структура коморбидной патологии в исследовании', 'value': 'statkomorbid'},
                        {'label': 'Коморбидная патология', 'value': 'komorbid'},
                        {'label': 'Характеристика детей в исследовании', 'value': 'harakter'},
                        {'label': 'Средний рост по возрасту', 'value': 'height'},
                        {'label': 'Отклонение от нормального роста', 'value': 'height.sds'},
                        {'label': 'Средний вес по возрасту', 'value': 'weight'},
                        {'label': 'Средний индекс массы тела по возрасту', 'value': 'IMT'},
                        {'label': 'Отклонение от нормального ИМТ', 'value': 'IMT.sds'}
                    ],
                    value='',
                    clearable=False,
                  # style={'margin-left': '10px'}
                ),
                dcc.Dropdown(
                    id='radio-city',
                    options=[
                        {'label': 'Выбраны все города исследования', 'value': 'all'},
                        {'label': 'Челябинск', 'value': 'Челябинск'},
                        {'label': 'Вологда', 'value': 'Вологда'},
                        {'label': 'Казань', 'value': 'Казань'},
                        {'label': 'Якутск', 'value': 'Якутск'},
                        {'label': 'Смоленск', 'value': 'Смоленск'},
                        {'label': 'Ростов', 'value': 'Ростов'},
                        {'label': 'Ульяновск', 'value': 'Ульяновск'},
                        {'label': 'Томск', 'value': 'Томск'},
                        {'label': 'Тюмень', 'value': 'Тюмень'}
                    ],
                    value='all',
                    clearable=False,
                    style={'margin-top': '10px'}
                )
            ]
        
            print("dropdowns:", dropdowns)
        
            return '/page-2', {'file_content': contents, 'file_name': filename, 'specialty': selected_specialty, 'dropdowns': dropdowns}
        
        if selected_specialty == 'Офтальмолог':  # Проверка, является ли специальность "Офтальмолог"
            dropdowns = [  # Создание выпадающих списков только для "Офтальмолог"
                dcc.Dropdown(
                    id='dropdown-statistics',
                    options=[
                        {'label': 'Выберите визуализацию', 'value': ''},
                        {'label': 'Жалобы на зрение в исследуемой группе', 'value': 'sochetzrenie'},
                        {'label': 'Отклонение от идеального зрения по возрасту', 'value': 'otklzrenie'},
                        {'label': 'Заболевания зрения', 'value': 'zrenie'},
                       
                    ],
                    value='',
                    clearable=False,
                   # style={'margin-left': '10px'}
                ),
                dcc.Dropdown(
                    id='radio-city',
                    options=[
                        {'label': 'Выбраны все города исследования', 'value': 'all'},
                        {'label': 'Челябинск', 'value': 'Челябинск'},
                        {'label': 'Вологда', 'value': 'Вологда'},
                        {'label': 'Казань', 'value': 'Казань'},
                        {'label': 'Якутск', 'value': 'Якутск'},
                        {'label': 'Смоленск', 'value': 'Смоленск'},
                        {'label': 'Ростов', 'value': 'Ростов'},
                        {'label': 'Ульяновск', 'value': 'Ульяновск'},
                        {'label': 'Томск', 'value': 'Томск'},
                        {'label': 'Тюмень', 'value': 'Тюмень'}
                    ],
                    value='all',
                    clearable=False,
                    style={'margin-top': '10px'}
                )
            ]
        
            print("dropdowns:", dropdowns)
        
            return '/page-2', {'file_content': contents, 'file_name': filename, 'specialty': selected_specialty, 'dropdowns': dropdowns}
       
        if selected_specialty == 'Аллерголог':  # Проверка, является ли специальность "Аллерголог"
            dropdowns = [  # Создание выпадающих списков только для "Аллерголог"
                dcc.Dropdown(
                    id='dropdown-statistics',
                    options=[
                        {'label': 'Выберите визуализацию', 'value': ''},
                        {'label': 'Структура аллергической патологии в исследуемой группе', 'value': 'allerg'},
                        {'label': 'Сочетания аллергических заболеваний в исследуемой группе', 'value': 'sochetallerg'},
                    ],
                    value='',
                    clearable=False,
                   # style={'margin-left': '10px'}
                ),
                dcc.Dropdown(
                    id='radio-city',
                    options=[
                        {'label': 'Выбраны все города исследования', 'value': 'all'},
                        {'label': 'Челябинск', 'value': 'Челябинск'},
                        {'label': 'Вологда', 'value': 'Вологда'},
                        {'label': 'Казань', 'value': 'Казань'},
                        {'label': 'Якутск', 'value': 'Якутск'},
                        {'label': 'Смоленск', 'value': 'Смоленск'},
                        {'label': 'Ростов', 'value': 'Ростов'},
                        {'label': 'Ульяновск', 'value': 'Ульяновск'},
                        {'label': 'Томск', 'value': 'Томск'},
                        {'label': 'Тюмень', 'value': 'Тюмень'}
                    ],
                    value='all',
                    clearable=False,
                    style={'margin-top': '10px'}
                )
            ]
        
            print("dropdowns:", dropdowns)
        
            return '/page-2', {'file_content': contents, 'file_name': filename, 'specialty': selected_specialty, 'dropdowns': dropdowns}
        
        else: 
            return '/page-2', {'file_content': contents, 'file_name': filename, 'specialty': selected_specialty, 'dropdowns': dropdowns}
        
    print("Returning to page 1")
    
    return '/page-1', None# None  # Добавлено возвращаемое значение, когда условие не выполняется

@app.callback(
    [Output('dropdown-container', 'children'),  # Отображаем выпадающие списки в контейнере
     Output('specialty-text', 'children')],    # Отображаем выбранную специальность
    [Input('url', 'pathname')],
    [State('file-store', 'data')]
)
def display_dropdowns_and_specialty_text(pathname, data):
    dropdowns = []  # Переменная для хранения выпадающих списков

    if pathname == '/page-2' and data:
        specialty = data.get('specialty', '')  # Получаем выбранную специальность

        # Отображаем список выпадающих списков для врачей
        if specialty == 'Педиатр':
            dropdowns = data.get('dropdowns', [])
        if specialty == 'Офтальмолог':
            dropdowns = data.get('dropdowns', [])
        if specialty == 'Аллерголог':
            dropdowns = data.get('dropdowns', [])
        return dropdowns, specialty

    return [], ''  # Возвращаем пустые списки, если URL не соответствует странице-2 или нет данных в хранилище



@app.callback(
    [Output('url', 'pathname',allow_duplicate=True),
    Output('file-store1', 'data',allow_duplicate=True)],
    Input('return-to-specialty-selection-button', 'n_clicks'),
    State('file-store', 'data'),
    prevent_initial_call=True
)
def return_to_specialty_selection(n_clicks, data):
    if n_clicks:
        if data:
            return '/page-1', data
        else:
            return '/page-1', {'file_content': None, 'file_name': None, 'specialty': None, 'dropdowns': None}
    return dash.no_update

# Callback to display uploaded file on the second page
@app.callback(
    Output('output-data-upload', 'children'),
    [Input('file-store', 'data')]
)
def display_uploaded_file(data):
    if data:
        try:
            content_type, content_string = data['file_content'].split(',')
            decoded = base64.b64decode(content_string)
            if 'xlsx' in data['file_name']:
                df = pd.read_excel(io.BytesIO(decoded))
            elif 'csv' in data['file_name']:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            # Сохранение данных в переменную data
            data['data'] = df.to_dict('records')
            
            # Вывод сообщения о успешной загрузке
            return html.Div("Ваши данные успешно загружены", style={'color': 'mint', 'text-align': 'center', 'font-size': '1.5rem'})
        except Exception as e:
            print(e)
            return html.Div(
                className="alert alert-dismissible alert-primary",
                style={'width': 'fit-content', 'margin': 'auto','margin-top': '10px'},
                children=[
                    html.Button(type="button", className="btn"),
                    html.Strong("Ошибка при обработке файла"),
                ]
            )
    return html.Div()



# Callback для загрузки данных и создания визуализации
@app.callback(
   Output('output-graph', 'children'),
    [Input('output-data-upload', 'children'),
     Input('dropdown-statistics', 'value'),  # Добавляем Input для dropdown-statistics
     Input('radio-city', 'value')],          # Добавляем Input для radio-city
    [State('file-store', 'data')]
)
def update_graph(contents, dropdown_value, selected_city, data):

    print("мееееееу")
    print("data",data)
    #print("contents",contents)
    # Извлекаем значения dropdown_value и selected_city из структуры data
    #dropdown_value = data[0]['props']['value']
    #selected_city = data[1]['props']['value']

    # Теперь dropdown_value и selected_city содержат выбранные пользователем значения из выпадающих списков
    print("dropdown_value:", dropdown_value)
    print("selected_city:", selected_city)
    # Проверяем, что список dropdowns не пустой
    if data and data['specialty'] == 'Педиатр' and contents is not None:
        try:
            content_type, content_string = data['file_content'].split(',')
            decoded = base64.b64decode(content_string)
            if 'xlsx' in data['file_name']:
                df = pd.read_excel(io.BytesIO(decoded))
            elif 'csv' in data['file_name']:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                    
            df.dropna(subset=['weight'], inplace=True)
            df.dropna(subset=['height.sds кач'], inplace=True)
            df.dropna(subset=['bmi.sds кач'], inplace=True)
            df.dropna(subset=['height'], inplace=True) 
            df.dropna(subset=['IndexMassTela'], inplace=True) 
            df = df.loc[~df['AgeGradation'].isin(['9-летние', 'остальные','13-летние'])]
            df['AgeGradation'] = df['AgeGradation'].sort_values()
            if selected_city != 'all':
                df = df[df['city'] == selected_city]
            
          
            if dropdown_value == 'stat':
                grouped_df = df.groupby(['AgeGradation', 'sex']).size().reset_index(name='count')
                fig = px.bar(grouped_df, 
                             x='AgeGradation', 
                             y='count', 
                             color='sex', 
                             barmode='stack',
                             title='Количество детей каждого пола по возрасту',
                             labels={'count': 'Количество детей', 'AgeGradation': 'Возраст', 'sex': 'Пол'},
                             color_discrete_map={'мужской': '#66c5cc', 'женский': '#fe88b1'},
                             text = 'count'
                            )
            elif dropdown_value == 'harakter':
                grouped_median = df.groupby('sex')[['height', 'weight', 'IndexMassTela', 'age']].median().reset_index()
                grouped_median = grouped_median.rename(columns={'height': 'Медианный рост', 'weight': 'Медианный вес', 'IndexMassTela': 'Медианный индекс массы тела', 'age': 'Медианный возраст'})
               # Преобразуем данные в формат, подходящий для Plotly Express
                data_melted = grouped_median.melt(id_vars=['sex'], value_vars=['Медианный рост', 'Медианный вес', 'Медианный индекс массы тела', 'Медианный возраст'], var_name='Характеристика', value_name='Значение')
                
                # Создаем график
                fig = px.bar(data_melted,
                            x='Значение',
                            y='Характеристика',
                            color='sex',
                            orientation='h',
                            barmode='group',
                            title='Медианные значения по характеристикам и полу',
                            labels={'Значение': 'Значение', 'Характеристика': 'Характеристика','sex': 'Пол'},
                            color_discrete_map={'мужской': '#66c5cc', 'женский': '#fe88b1'},
                            text = 'Значение'
                            )

            elif dropdown_value == 'height':
                grouped_height_df = df.groupby(['AgeGradation', 'sex'])['height'].mean().reset_index()
                fig = px.bar(grouped_height_df, 
                             x='AgeGradation', 
                             y='height', 
                             color='sex', 
                             barmode='group',
                             title='Средний рост по возрасту с группировкой по полу',
                             labels={'height': 'Средний рост (см)', 'AgeGradation': 'Возраст','sex': 'Пол'},
                             color_discrete_map={'мужской': '#66c5cc', 'женский': '#fe88b1'},
                             text = 'height'

                            )
            
            elif dropdown_value == 'height.sds':
                grouped_height_df = df.groupby(['height.sds кач','sex']).size().reset_index(name='count')

                # Создаем график
                fig = px.bar(grouped_height_df,
                            x='height.sds кач',
                            y='count',
                            color='height.sds кач',
                            barmode='group',
                            facet_row='sex',
                            title='Отклонение от нормального роста',
                            labels={'count': 'Количество детей', 'height.sds кач': 'Рост','sex': 'Пол'},
                            #color_discrete_map={'норма': '#6baed6', 'низкорослость': '#fb9a99', 'высокорослость': '#08589e'}
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            text = 'count'
                            )
                fig.update_layout(yaxis = {"categoryorder":"total ascending"})
                                    
            elif dropdown_value == 'weight':
                grouped_weight_df = df.groupby(['AgeGradation', 'sex'])['weight'].mean().reset_index()
                fig = px.bar(grouped_weight_df, 
                             x='AgeGradation', 
                             y='weight', 
                             color='sex', 
                             barmode='group',
                             title='Средний вес по возрасту с группировкой по полу',
                             labels={'weight': 'Средний вес (кг)', 'AgeGradation': 'Возраст','sex': 'Пол'},
                             color_discrete_map={'мужской': '#66c5cc', 'женский': '#fe88b1'},
                             text = 'weight'
                            )
                    
            elif dropdown_value == 'IMT':
                grouped_weight_df = df.groupby(['AgeGradation', 'sex'])['IndexMassTela'].mean().reset_index()
                fig = px.bar(grouped_weight_df, 
                             x='AgeGradation', 
                             y='IndexMassTela', 
                             color='sex', 
                             barmode='group',
                             title='Средний индекс массы тела по возрасту с группировкой по полу',
                             labels={'IndexMassTela': 'Средний ИМТ', 'AgeGradation': 'Возраст','sex': 'Пол'},
                             color_discrete_map={'мужской': '#66c5cc', 'женский': '#fe88b1'},
                             text='IndexMassTela'  # Добавление текста к столбцам

                            )   
                

            elif dropdown_value == 'IMT.sds':
                grouped_height_df = df.groupby(['bmi.sds кач','sex']).size().reset_index(name='count')

                # Создаем график
                fig = px.bar(grouped_height_df,
                            x='bmi.sds кач',
                            y='count',
                            color='bmi.sds кач',
                            barmode='group',
                            facet_row='sex',
                            title='Отклонение от нормального индекса массы тела',
                            labels={'count': 'Количество детей', 'bmi.sds кач': 'ИМТ','sex': 'Пол'},
                            #color_discrete_map={'норма': '#6baed6', 'недостаточное питание': '#fb9a99', 'избыточная масса тела': '#08589e'}
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            text='count'  # Добавление текста к столбцам
                            )
                fig.update_layout(yaxis = {"categoryorder":"total ascending"})
            elif dropdown_value == 'komorbid':

                # Подготовка данных
                df.dropna(subset=['Рефракция'], inplace=True) 
                df.dropna(subset=['Кожные заболевания (при осмотре)'], inplace=True) 
                df.dropna(subset=['Изменения в легких'], inplace=True)
                df.dropna(subset=['Кардиологическая патология'], inplace=True) 
                # Создание столбцов для определения заболеваний
                # df['refraction_abnormal'] = df['Рефракция'] != 'норма'
                # df['cardio_abnormal'] = df['Кардиологическая патология'] != 'Норма'
                # df['skin_abnormal'] = df['Кожные заболевания (при осмотре)'] != 'отсутствие'
                # df['lungs_abnormal'] = df['Изменения в легких'] != 'Норма'
                df['counted_diseases'] = df[(df['Рефракция'] != 'норма') | (df['Кардиологическая патология'] != 'Норма') | (df['Кожные заболевания (при осмотре)'] != 'отсутствие') | (df['Изменения в легких'] != 'Норма')].apply(lambda row: ', '.join(row[['Рефракция', 'Кардиологическая патология', 'Кожные заболевания (при осмотре)', 'Изменения в легких']].dropna()), axis=1)
                df.dropna(subset=['counted_diseases'], inplace=True) 
                #print('counted',df['counted_diseases'])    
                # Подсчет уникальных комбинаций строк и их количества
                combination_counts = df['counted_diseases'].value_counts()

                df_combinations = pd.DataFrame(combination_counts.index, columns=['counted_diseases'])

                # Разделение столбца 'Сочетание заболеваний' на 4 столбца
                df_combinations[['Рефракция', 'Кардиологическая патология', 'Кожные заболевания (при осмотре)', 'Изменения в легких']] = df_combinations['counted_diseases'].str.split(',', expand=True)
               
                def get_cell_color(value):
                    value_list = [x.strip().lower() for x in value.split(',')]  # Приведение каждого значения к нижнему регистру и удаление пробелов
                    abnormal_values = {'норма', 'отсутствие', 'Норма'}
                    
                    if not all(val in abnormal_values for val in value_list):
                        return '#f6cf71'  # Цвет для аномальных значений
                    else:
                        return '#e5ecf6'  # Цвет для нормальных значений
                    
                # Создание объекта таблицы для отображения данных
                table = go.Table(
                    header=dict(
                    values=['<b>Рефракция<b>', '<b>Кардиологическая патология<b>', '<b>Кожные заболевания (при осмотре)<b>', '<b>Изменения в легких<b>', '<b>Количество детей<b>'],
                    line_color='#ffffff',  # Светлый цвет линий для всей таблицы
                    fill_color='#e5ecf6',    # Цвет заливки ячеек
                    align=['center', 'center'],
                    font=dict(color='rgb(42,63,94)', size=14),
                    height=30
                ),
                cells=dict(
                    values=[df_combinations['Рефракция'], df_combinations['Кардиологическая патология'], df_combinations['Кожные заболевания (при осмотре)'], df_combinations['Изменения в легких'], combination_counts.values],
                    fill_color=[
                    [get_cell_color(value) for value in df_combinations['Рефракция']],
                    [get_cell_color(value) for value in df_combinations['Кардиологическая патология']],
                    [get_cell_color(value) for value in df_combinations['Кожные заболевания (при осмотре)']],
                    [get_cell_color(value) for value in df_combinations['Изменения в легких']],
                    ['#e5ecf6' for value in combination_counts.values]]
                ))
                # Создание объекта Figure с таблицей
                fig = go.Figure(data=[table])

            elif dropdown_value == 'statkomorbid':
#миопия - заболевания пищеварительного тракта - паталогия сердечно-сосудистой системы - аллергические заболевания 
#- Паталогия ЛОР-органов - Заболевания нервной системы - Заболевания эндокринной системы
# - Заболевания опорно-двигательного аппарата - Заболевания мочевыделительной системы 
#- забоолевания дыхательной системы
                
                kolvodetey = df['age'].size
                print('counted',kolvodetey)    
                counts = {
                'Болезни зрения': ((df['Рефракция'] != 'норма') & (df['Рефракция'].notnull())).sum(), 
                'Кожные заболевания': ((df['Кожные заболевания (при осмотре)'] != 'отсутствие') & (df['Кожные заболевания (при осмотре)'].notnull())).sum(), 
                'Заболевания дыхания': ((df['Изменения в легких'] != 'Норма') & (df['Изменения в легких'].notnull())).sum(), 
                'Кардиологическая патология': ((df['Кардиологическая патология'] != 'Норма') & (df['Кардиологическая патология'].notnull())).sum(),
                'ЛОР-заболевания': ((df['Другое_x'].notnull()) & (df['Другое_x'] != 'отсутствуют') & (df['Другое_x'] != 'неизвестно') & (df['Другое_x'] != 'здоров')).sum(),
                'Неврологичсекие заболевание': df['NEVRdop'].notnull().sum(),
                'Аллергические заболевания': (df['Аллергические заболевания'] == 1).sum(),
                'Заболевания опорно-двигательного аппарата': df['Другое_x.2'].notnull().sum(),
                'Заболевания пищеварительной системы': df['GASTROdrugoe'].notnull().sum()
                }
                for count in counts.values():
                    print('counted',count)    

                # Создаем DataFrame из полученных данных
                data = {
                    'Отклонение от нормы': list(counts.keys()),
                    'Процент детей с отклонением': [(count * 100)/kolvodetey for count in counts.values()]
                }

                data['Текст процентов'] = [f'{percent:.2f}%' for percent in data['Процент детей с отклонением']]

          
                # Строим горизонтальную столбчатую диаграмму
                fig = px.bar(data, x='Процент детей с отклонением', y='Отклонение от нормы',
                            title='Процент детей с отклонениями в здоровье',
                            labels={'Процент детей с отклонением': 'Процент детей с отклонением', 'Отклонение от нормы': 'Отклонение от нормы'},
                            orientation='h',  # Горизонтальная ориентация
                            color = 'Отклонение от нормы',
                            text ='Текст процентов',  # Текстовые метки процентов
                            color_discrete_sequence=px.colors.qualitative.Pastel

                            )  
                fig.update_layout(yaxis = {"categoryorder":"total ascending"})

    

            return dcc.Graph(figure=fig)
        except Exception as e:
            print(e)
            return html.Div(
                className="alert alert-dismissible alert-primary",
                style={'width': 'fit-content', 'margin': 'auto','margin-top': '10px'},
                children=[
                    html.Button(type="button", className="btn"),
                    html.Strong("Выберите исследование, которое хотите визуализировать"),
                ]
            )
    if data and data['specialty'] == 'Офтальмолог' and contents is not None:
        try:
            content_type, content_string = data['file_content'].split(',')
            decoded = base64.b64decode(content_string)
            if 'xlsx' in data['file_name']:
                df = pd.read_excel(io.BytesIO(decoded))
            elif 'csv' in data['file_name']:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                    
            df.dropna(subset=['weight'], inplace=True)
            df.dropna(subset=['height.sds кач'], inplace=True)
            df.dropna(subset=['bmi.sds кач'], inplace=True)
            df.dropna(subset=['height'], inplace=True) 
            df.dropna(subset=['IndexMassTela'], inplace=True) 
            df = df.loc[~df['AgeGradation'].isin(['9-летние', 'остальные','13-летние'])]
            df['AgeGradation'] = df['AgeGradation'].sort_values()
            if selected_city != 'all':
                df = df[df['city'] == selected_city]
            
            if dropdown_value == 'zrenie':
                df.dropna(subset=['Рефракция'], inplace=True) 
             # Отфильтруем данные, оставив только случаи, где есть отклонение от нормы
                refraction_abnormal = df[df['Рефракция'] != 'норма']

                # Подсчитываем количество детей с каждым отклонением от нормы зрения
                refraction_counts = refraction_abnormal['Рефракция'].value_counts()

                # Создаем DataFrame с данными о количестве детей с каждым отклонением от нормы зрения
                data1 = {'Отклонение от нормы': refraction_counts.index.tolist(),
                        'Количество детей': refraction_counts.values}

                # Строим столбчатую диаграмму
                fig = px.bar(data1, x='Отклонение от нормы', y='Количество детей', 
                            title='Дети, страдающие отклонениями в зрении',
                            labels={'Количество детей': 'Количество детей', 'Отклонение от нормы': 'Отклонение от нормы'},
                            color ='Отклонение от нормы',
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            text=data1['Количество детей'],  # Подписи к столбцам

                            #template="simple_white"
                            )  # Белый фон
                # Обновляем макет для установки белого фона и сетки на заднем фоне
               # fig.update_layout(plot_bgcolor='white', yaxis=dict(showgrid=True, gridcolor='lightgray'))
                fig.update_layout(yaxis = {"categoryorder":"total ascending"})

            elif dropdown_value == 'sochetzrenie':

                df.dropna(subset=['Жалобы офтальмолог'], inplace=True) 
                oft_counts = df['Жалобы офтальмолог'].value_counts()

                # Создание датафрейма для построения диаграммы
                df_pie = pd.DataFrame({'Жалобы офтальмолог': oft_counts.index, 'Количество': oft_counts.values})
                # Построение круговой диаграммы
                fig = px.pie(df_pie, values='Количество', names='Жалобы офтальмолог', title='Жалобы на зрение', 
                             color_discrete_sequence=px.colors.qualitative.Pastel)

            elif dropdown_value == 'otklzrenie':

                df.dropna(subset=['Остр.зр.Л'], inplace=True) 
                df.dropna(subset=['Остр.зр.П'], inplace=True) 
                oft_counts = df['Остр.зр.П'].value_counts()
                
                # Определение отклонений от идеального зрения в одном из глаз
                df['Нарушение зрения'] = (df['Остр.зр.Л'] != 1) | (df['Остр.зр.П'] != 1)

                # Подсчет количества детей с отклонениями в каждой возрастной группе
                df_grouped = df.groupby('AgeGradation')['Нарушение зрения'].sum().reset_index()

                # Подсчет общего количества детей с отклонениями
                total_oft_children = df_grouped['Нарушение зрения'].sum()

                # Расчет процентного соотношения
                df_grouped['Процент детей с нарушением зрения'] = (df_grouped['Нарушение зрения'] / total_oft_children) * 100

                # Визуализация данных
                fig = px.bar(df_grouped, x='AgeGradation', y='Нарушение зрения', 
                            title='Количество детей с отклонениями зрения по возрастным группам',
                            labels={'value': 'Количество детей', 'variable': 'Нарушение зрения', 'AgeGradation': 'Возраст'},
                            barmode='group',
                            text=df_grouped['Нарушение зрения'],  # Добавление текста на столбцах с количеством
                            hover_data={'Процент детей с нарушением зрения': ':.2f%'},  # Отображение процентов при наведении курсора
                            color='AgeGradation',
                            color_discrete_sequence=px.colors.qualitative.Pastel)
          


            return dcc.Graph(figure=fig)
        except Exception as e:
            print(e)
            return html.Div(
                className="alert alert-dismissible alert-primary",
                style={'width': 'fit-content', 'margin': 'auto','margin-top': '10px'},
                children=[
                    html.Button(type="button", className="btn"),
                    html.Strong("Выберите исследование, которое хотите визуализировать"),
                ]
            )
    if data and data['specialty'] == 'Аллерголог' and contents is not None:
        try:
            content_type, content_string = data['file_content'].split(',')
            decoded = base64.b64decode(content_string)
            if 'xlsx' in data['file_name']:
                df = pd.read_excel(io.BytesIO(decoded))
            elif 'csv' in data['file_name']:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                    
            df.dropna(subset=['weight'], inplace=True)
            df.dropna(subset=['height.sds кач'], inplace=True)
            df.dropna(subset=['bmi.sds кач'], inplace=True)
            df.dropna(subset=['height'], inplace=True) 
            df.dropna(subset=['IndexMassTela'], inplace=True) 
            df = df.loc[~df['AgeGradation'].isin(['9-летние', 'остальные','13-летние'])]
            df['AgeGradation'] = df['AgeGradation'].sort_values()
            if selected_city != 'all':
                df = df[df['city'] == selected_city]
            
            if dropdown_value == 'allerg':

                kolvodetey = df['age'].size
                print('counted',kolvodetey)    
                counts = {
                'Пищевая аллергия':  ((df['Пищевая аллергия'] != 'Отсутствие') & (df['Пищевая аллергия'].notnull())).sum(), 
                'Аллергический ринит':((df['Аллергический ринит'] != 'Отсутствие') & (df['Аллергический ринит'].notnull())).sum(), 
                'Бронхиальная астма':  ((df['Бронхиальная астма'] != 'Отсутствие') & (df['Бронхиальная астма'].notnull())).sum(), 
                'Поллиноз': ((df['Поллиноз'] != 'Отсутствие') & (df['Поллиноз'].notnull())).sum(), 
                'Лекарственная аллергия': ((df['Лек. аллергия'] != 'Отсутствие') & (df['Лек. аллергия'].notnull())).sum(), 
                'Атопический дерматит': ((df['Атопический дерматит'] != 'Отсутствие') & (df['Атопический дерматит'].notnull())).sum(),
                }
                for count in counts.values():
                    print('counted',count)    

                # Создаем DataFrame из полученных данных
                data = {
                    'Отклонение от нормы': list(counts.keys()),
                    'Процент детей с отклонением': [(count * 100)/kolvodetey for count in counts.values()]
                }

                data['Текст процентов'] = [f'{percent:.2f}%' for percent in data['Процент детей с отклонением']]

          
                # Строим горизонтальную столбчатую диаграмму
                fig = px.bar(data, x='Процент детей с отклонением', y='Отклонение от нормы',
                            title='Процент детей с отклонениями в здоровье',
                            labels={'Процент детей с отклонением': 'Процент детей с отклонением', 'Отклонение от нормы': 'Отклонение от нормы'},
                            orientation='h',  # Горизонтальная ориентация
                            color = 'Отклонение от нормы',
                            text ='Текст процентов',  # Текстовые метки процентов
                            color_discrete_sequence=px.colors.qualitative.Pastel

                            )  
                fig.update_layout(yaxis = {"categoryorder":"total ascending"})
            if dropdown_value == 'sochetallerg':
                
                # Подготовка данных
                df.dropna(subset=['Пищевая аллергия'], inplace=True) 
                df.dropna(subset=['Аллергический ринит'], inplace=True) 
                df.dropna(subset=['Бронхиальная астма'], inplace=True)
                df.dropna(subset=['Поллиноз'], inplace=True) 
                df.dropna(subset=['Лек. аллергия'], inplace=True)
                df.dropna(subset=['Атопический дерматит'], inplace=True) 
                df['counted_diseases'] = df[(df['Пищевая аллергия'] == 'Наличие') | (df['Аллергический ринит'] == 'Наличие') | (df['Бронхиальная астма'] == 'Наличие') | (df['Поллиноз'] == 'Наличие') | (df['Лек. аллергия'] == 'Наличие') | (df['Атопический дерматит'] == 'Наличие')].apply(lambda row: ', '.join(row[['Пищевая аллергия', 'Аллергический ринит', 'Бронхиальная астма', 'Поллиноз', 'Лек. аллергия', 'Атопический дерматит']].astype(str).dropna()), axis=1)
                df.dropna(subset=['counted_diseases'], inplace=True) 
                print('counted',df['counted_diseases'])    
                # Подсчет уникальных комбинаций строк и их количества
                combination_counts = df['counted_diseases'].value_counts()

                df_combinations = pd.DataFrame(combination_counts.index, columns=['counted_diseases'])

                
                df_combinations[['Пищевая аллергия', 'Аллергический ринит', 'Бронхиальная астма', 'Поллиноз', 'Лек. аллергия', 'Атопический дерматит']] = df_combinations['counted_diseases'].str.split(',', expand=True)
               
                def get_cell_color(value):
                    value_list = [x.strip().lower() for x in value.split(',')]  # Приведение каждого значения к нижнему регистру и удаление пробелов
                    abnormal_values = {'наличие'}
                    
                    if not all(val in abnormal_values for val in value_list):
                        return '#e5ecf6'  # Цвет для аномальных значений
                    else:
                        return '#f6cf71'  # Цвет для нормальных значений
                    
                # Создание объекта таблицы для отображения данных
                table = go.Table(
                    header=dict(
                    values=['<b>Пищевая аллергия<b>', '<b>Аллергический ринит<b>', '<b>Бронхиальная астма<b>', '<b>Поллиноз<b>', '<b>Лек. аллергия<b>', '<b>Атопический дерматит<b>','<b>Количество детей<b>'],
                    line_color='#ffffff',  # Светлый цвет линий для всей таблицы
                    fill_color='#e5ecf6',    # Цвет заливки ячеек
                    align=['center', 'center'],
                    font=dict(color='rgb(42,63,94)', size=14),
                    height=30
                ),
                cells=dict(
                    values=[df_combinations['Пищевая аллергия'], df_combinations['Аллергический ринит'], df_combinations['Бронхиальная астма'], df_combinations['Поллиноз'], df_combinations['Лек. аллергия'], df_combinations['Атопический дерматит'], combination_counts.values],
                    fill_color=[
                    [get_cell_color(value) for value in df_combinations['Пищевая аллергия']],
                    [get_cell_color(value) for value in df_combinations['Аллергический ринит']],
                    [get_cell_color(value) for value in df_combinations['Бронхиальная астма']],
                    [get_cell_color(value) for value in df_combinations['Поллиноз']],
                    [get_cell_color(value) for value in df_combinations['Лек. аллергия']],
                    [get_cell_color(value) for value in df_combinations['Атопический дерматит']],
                    ['#e5ecf6' for value in combination_counts.values]]
                ))
                # Создание объекта Figure с таблицей
                fig = go.Figure(data=[table])

            return dcc.Graph(figure=fig)
        except Exception as e:
            print(e)
            return html.Div(
                className="alert alert-dismissible alert-primary",
                style={'width': 'fit-content', 'margin': 'auto','margin-top': '10px'},
                children=[
                    html.Button(type="button", className="btn"),
                    html.Strong("Выберите исследование, которое хотите визуализировать"),
                ]
            )
        
@app.callback(
    Output('download-image', 'data'),
    [Input('download-png-button', 'n_clicks'),
     Input('download-jpeg-button', 'n_clicks'),
     Input('download-svg-button', 'n_clicks')],
    [State('output-graph', 'children'),
     State('dropdown-statistics', 'value')]
)
def download_image(n_clicks_png, n_clicks_jpeg, n_clicks_svg, figure, selected_dropdown_value):

    if not figure:  # Проверяем, что график существует
        raise PreventUpdate

    
    print('new', selected_dropdown_value)

    if not n_clicks_png and not n_clicks_jpeg and not n_clicks_svg:
        raise PreventUpdate

    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

   
    print('new', selected_dropdown_value)
    fig = figure['props']['figure']

    print('data', fig)
    
    if button_id == 'download-png-button':
        filename = f"plot_{selected_dropdown_value}.png"
        pio.write_image(fig, filename) 
        return dcc.send_bytes(fig.to_image(format="png"), filename=filename)
    elif button_id == 'download-jpeg-button':
        filename = f"plot_{selected_dropdown_value}.jpeg"
        pio.write_image(fig, filename) 
        return dcc.send_bytes(fig.to_image(format="jpeg"), filename=filename)
    elif button_id == 'download-svg-button':
        filename = f"plot_{selected_dropdown_value}.svg"
        pio.write_image(fig, filename) 
        return dcc.send_bytes(fig.to_image(format="svg"), filename=filename)
    else:
        raise PreventUpdate


# Callback для отображения кнопок только после загрузки графика
@app.callback(
    Output('download-buttons', 'children'),
    [Input('output-graph', 'children')]
)
def show_download_buttons(graph_children):
    if graph_children:
        
        return html.Div([
            html.Button("Скачать PNG", id='download-png-button', className="btn btn-primary mt-3", style={ "margin-right": "10px"}),
            html.Button("Скачать JPEG", id='download-jpeg-button', className="btn btn-primary mt-3", style={"margin-right": "10px"}),
            html.Button("Скачать SVG", id='download-svg-button', className="btn btn-primary mt-3", style={ "margin-right": "10px"})
        ])
    else:
        return None

if __name__ == '__main__':
    app.run_server('host = 0.0.0.0', debug=True)
    # 185.130.114.94 
    
