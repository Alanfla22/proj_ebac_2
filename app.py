import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import pickle

custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
sns.set_theme(style='ticks', rc=custom_params)

base_dados = pd.read_csv('./data/previsao_de_renda.csv')
base_dados.drop_duplicates(inplace=True)
base_dados['empresario'] = base_dados['tipo_renda'] == 'Empresário'

# loading in the model to predict on the data
pickle_in = open('./modelo/modelo_reg.pkl', 'rb')
regressor = pickle.load(pickle_in)

# defining the function which will make the prediction using 
# the data which the user inputs
@st.cache_resource
def prediction(entrada):

  resultado = regressor.predict(entrada)

  return np.exp(resultado)[0]

# função para cálculo do intervalo de confiança
@st.cache_resource
def interval_conf(entrada, conf):

  desvio_padrao = np.std(regressor.resid)
  n = len(regressor.resid)
  media = np.mean(regressor.resid)
  g = n - 1

  # criando o intervalo

  intervalo = stats.t.interval(confidence=conf, df=g, loc=media, scale=desvio_padrao)

  resultado = regressor.predict(entrada)

  # valor mínimo do intervalo de confiança
  minimo = np.exp(resultado + intervalo[0])[0]
  # valor máximo do intervalo de confiança
  maximo = np.exp(resultado + intervalo[1])[0]

  return (minimo, maximo)

def welcome():
    return 'welcome all'
  
  
  
# this is the main function in which we define our webpage 
def main():    

    st.set_page_config(page_title = 'Renda Analisys', \
        page_icon = './images/telmarketing_icon.png',
        layout ='wide',
        initial_sidebar_state='expanded')
     
       
    st.title('Análise e Previsão de Renda')
    st.subheader('', divider='rainbow')
    
    st.subheader('Somatório das rendas')

    tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data",])
    with tab1:

      fig = px.histogram(base_dados, x='data_ref', y='renda', color_discrete_sequence=['#A9D0F5'], text_auto=True)
      fig.update_traces(textposition='outside')
      st.plotly_chart(fig, theme="streamlit", use_container_width=True)      
           
    with tab2:

      st.write(base_dados[['data_ref', 'renda']].groupby(by='data_ref').sum()) 

    st.subheader('Renda por característica do cliente')

    col1, col2 = st.columns(2)
    with col1:
      tab1, tab2, tab3, tab4 = st.tabs(["Sexo", "Posse de Imóvel", "Empresário", "Posse de Veículo"])
      with tab1:

        fig = px.pie(base_dados, values='renda', names='sexo', color_discrete_sequence=['#A9D0F5', '#6E6E6E'])
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)    
            
      with tab2:

        fig = px.pie(base_dados, values='renda', names='posse_de_imovel', color_discrete_sequence=['#A9D0F5', '#6E6E6E'])
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

      with tab3:
        fig = px.pie(base_dados, values='renda', names='empresario', color_discrete_sequence=['#A9D0F5', '#6E6E6E'])
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

      with tab4:
        fig = px.pie(base_dados, values='renda', names='posse_de_veiculo', color_discrete_sequence=['#A9D0F5', '#6E6E6E'])
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with col2:
      tab1, tab2 = st.tabs(["Idade", "Tempo de emprego"])
      with tab1:

        fig = px.scatter(base_dados, y='renda', x='idade')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
      with tab2:

        fig = px.scatter(base_dados, y='renda', x='tempo_emprego')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)     


    st.sidebar.write("## Faça sua previsão da renda aqui")

    with st.sidebar.form(key='my_form'):

      sexo = st.radio("Sexo", ['F', 'M'])
      col1, col2, = st.columns(2)
      with col1:
        posse_de_imovel = st.radio("Tem imóvel", [True, False])
      with col2:
        empresario = st.radio("Empresário", [True, False])
      col1, col2, = st.columns(2)
      with col1:
        tempo_emprego = st.number_input("Tempo de emprego", min_value=0, step=1)
      with col2:
        idade = st.number_input('Idade (anos)', min_value=18, step=1)
      confianca = st.slider("Nível de confiança", min_value=0.00, max_value=1.00, step=0.05)
    
      resultado =""
      intervalo =""

      entrada = {
      'sexo': sexo,
      'posse_de_imovel': posse_de_imovel,
      'tempo_emprego': tempo_emprego,
      'idade': idade,
      'empresario': empresario
      }
        
      if st.form_submit_button("Predict"):
          resultado = 'R$ %.2f' % (prediction(entrada))
          intervalo = 'A %i por cento de confiança o valor está situado entre %.2f e %.2f' % (confianca*100, interval_conf(entrada, confianca)[0], interval_conf(entrada, confianca)[1])
    
  
      st.write('#### Renda prevista:')
      st.metric(label="", value=resultado)
      st.write(intervalo)

if __name__=='__main__':
    main()