import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
import shap 
import streamlit.components.v1 as components

#cargar el modelo
modelo, preprocesador = load('Modelo_final/arbol_final.joblib')

def epaquetar(altura, estado_salud, inclinacion, especie, tipo_cazoleta,
              fase_vital, fuste, ahuecamiento, levantamiento_vereda, tipo_tierra,
              tipo_calle, luz_led, tipo_vereda, tipo_tendido, distancia_al_muro, 
              distancia_entre_ar):
    """Funcion que toma los parametros enviados, y los confirte en un dataSet,
    los pasa por un el procesador del modelo y hace la estimación.
    devuelve los datos porcesados y la estimación
    """
    X_pred = pd.DataFrame(columns=['altura', 'estado_salud', 'inclinacion', 'especie', 'tipo_cazoleta',
              'fase_vital', 'fuste', 'ahuecamiento', 'levantamiento_vereda', 'tipo_tierra',
              'tipo_calle', 'luz_led', 'tipo_vereda', 'tipo_tendido', 'distancia_al_muro', 
              'distancia_entre_ar'])
    X_pred.loc[0,:]=[altura, estado_salud, inclinacion, especie, tipo_cazoleta,
              fase_vital, fuste, ahuecamiento, levantamiento_vereda, tipo_tierra,
              tipo_calle, luz_led, tipo_vereda, tipo_tendido, distancia_al_muro, 
              distancia_entre_ar]
    X_pred_procesado=preprocesador.transform(X_pred)
    print(X_pred_procesado.shape)

    return X_pred_procesado, modelo.predict(X_pred_procesado), modelo.predict_proba(X_pred_procesado)

def st_shap(plot, heigth=None):
    js=shap.getjs()
    shap_html=f"<head>{js}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=heigth)


#crear la lista de opciones para
lista_features_orig=list(preprocesador.feature_names_in_)
lista_features=list(preprocesador.get_feature_names_out())

lista_altura= [x.split('cat__altura_')[1] for x in lista_features if 'altura' in x ]
lista_estado_salud= [x.split('cat__estado_salud_')[1] for x in lista_features if 'estado_salud' in x ]
lista_inclinacion = [x.split('cat__inclinacion_')[1] for x in lista_features if 'inclinacion' in x ]
lista_especie = [x.split('cat__especie_')[1] for x in lista_features if 'especie' in x ]
lista_tipo_cazoleta = [x.split('cat__tipo_cazoleta_')[1] for x in lista_features if 'tipo_cazoleta' in x ]
lista_fase_vital = [x.split('cat__fase_vital_')[1] for x in lista_features if 'fase_vital' in x ]
lista_fuste = [x.split('cat__fuste_')[1] for x in lista_features if 'fuste' in x ]
lista_ahuecamiento = [x.split('cat__ahuecamiento_')[1] for x in lista_features if 'ahuecamiento' in x ]
lista_levantamiento_vereda = [x.split('cat__levantamiento_vereda_')[1] for x in lista_features if 'levantamiento_vereda' in x ]
lista_tipo_tierra = [x.split('cat__tipo_tierra_')[1] for x in lista_features if 'tipo_tierra' in x ]
lista_tipo_calle = [x.split('cat__tipo_calle_')[1] for x in lista_features if 'tipo_calle' in x ]
lista_luz_led = [x.split('cat__luz_led_')[1] for x in lista_features if 'luz_led' in x ]
lista_tipo_vereda = [x.split('cat__tipo_vereda_')[1] for x in lista_features if 'tipo_vereda' in x ]
lista_tipo_tendido = [x.split('cat__tipo_tendido_')[1] for x in lista_features if 'tipo_tendido' in x ]

#creo la web
st.header('Esta es la web para predicción de caida de Arboles de Ctes')
#st.write(lista_features_orig)
with st.form(key='formulario'):
    altura = st.selectbox(
        'Altura', lista_altura
    )
    
    estado_salud = st.selectbox(
        'Estado del Salud', lista_estado_salud
    )
    
    inclinacion = st.selectbox(
        'Inclinacion', lista_inclinacion
    )
    
    especie = st.selectbox(
        'Especie', lista_especie
    )
    
    tipo_cazoleta = st.selectbox(
        'Tipo Cazoleta', lista_tipo_cazoleta
    )
    
    fase_vital = st.selectbox(
        'Fase Vital', lista_fase_vital
    )
    
    fuste = st.selectbox(
        'Fuste', lista_fuste
    )
    

    ahuecamiento = st.selectbox(
        'Ahuecamiento', lista_ahuecamiento
    )
    

    levantamiento_vereda = st.selectbox(
        'Levantamiento Vereda', lista_levantamiento_vereda
    )
    
    tipo_tierra = st.selectbox(
        'Tipo Tierra', lista_tipo_tierra
    )
    
    tipo_calle = st.selectbox(
        'Tipo Calle', lista_tipo_calle
    )
    
    luz_led = st.selectbox(
        'Luz Led', lista_luz_led
    )
    
    tipo_vereda = st.selectbox(
        'Tipo Vereda', lista_tipo_vereda
    )
    
    tipo_tendido = st.selectbox(
        'Tipo Tendido Electrico', lista_tipo_tendido
    )
    
    distancia_al_muro = st.slider('Distacia al Muro',0, 10, 1)
    st.write(distancia_al_muro)

    distancia_entre_ar = st.slider('Distacia entre Arboles',0, 10, 1)
    st.write(distancia_entre_ar)

    submitted =st.form_submit_button("Submit")
    if submitted:

        # Llamar a una función con los valores seleccionados
        X_pred, y_pred, prob = epaquetar(altura, estado_salud, inclinacion, especie, tipo_cazoleta,
              fase_vital, fuste, ahuecamiento, levantamiento_vereda, tipo_tierra,
              tipo_calle, luz_led, tipo_vereda, tipo_tendido, distancia_al_muro, 
              distancia_entre_ar)
        st.write(f"Resultado de la función: {y_pred}")

        st.write(f"la Porbabilidad calculada es: {prob}")

        explainer = shap.TreeExplainer(modelo[0])

        shap_value = explainer.shap_values(X_pred)

        class_labels = modelo.classes_
        class_index= int(np.where(class_labels == y_pred )[0]) 
        #sample_index = y_pred
        class_label=class_labels[class_index]
        #shap_values_for_class = shap_value[sample_index, : , class_index]
        st_shap(shap.force_plot(explainer.expected_value[0], shap_value[0, :, class_index], feature_names=lista_features))






    


