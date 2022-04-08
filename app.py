import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load
from Utils.model import predict
from Utils.data_formatter import train_formatter
# Keras Library
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
# 

cat_cols  = ['ethnicity', 'gender', 'hospital_admit_source', 'icu_admit_source',
       'icu_stay_type', 'icu_type', 'apache_3j_bodysystem',
       'apache_2_bodysystem']
num_cols = ["hospital_id", "age","bmi","elective_surgery","ethnicity","gender","height","hospital_admit_source",
    "icu_admit_source","icu_id","icu_stay_type","icu_type","pre_icu_los_days","readmission_status","weight",
    "apache_2_diagnosis","apache_3j_diagnosis","apache_post_operative","arf_apache","bun_apache","creatinine_apache",
    "gcs_eyes_apache","gcs_motor_apache","gcs_unable_apache","gcs_verbal_apache","glucose_apache","heart_rate_apache",
    "hematocrit_apache","intubated_apache","map_apache","resprate_apache","sodium_apache","temp_apache","ventilated_apache",
    "wbc_apache","d1_diasbp_max","d1_diasbp_min","d1_diasbp_noninvasive_max","d1_diasbp_noninvasive_min",
    "d1_heartrate_max","d1_heartrate_min","d1_mbp_max","d1_mbp_min","d1_mbp_noninvasive_max","d1_mbp_noninvasive_min",
    "d1_resprate_max","d1_resprate_min","d1_spo2_max","d1_spo2_min","d1_sysbp_max","d1_sysbp_min","d1_sysbp_noninvasive_max",
    "d1_sysbp_noninvasive_min","d1_temp_max","d1_temp_min","h1_diasbp_max","h1_diasbp_min","h1_diasbp_noninvasive_max",
    "h1_diasbp_noninvasive_min","h1_heartrate_max","h1_heartrate_min","h1_mbp_max","h1_mbp_min","h1_mbp_noninvasive_max",
    "h1_mbp_noninvasive_min","h1_resprate_max","h1_resprate_min","h1_spo2_max","h1_spo2_min","h1_sysbp_max",
    "h1_sysbp_min","h1_sysbp_noninvasive_max","h1_sysbp_noninvasive_min","h1_temp_max","h1_temp_min","d1_bun_max",
    "d1_bun_min","d1_calcium_max","d1_calcium_min","d1_creatinine_max","d1_creatinine_min","d1_glucose_max",
    "d1_glucose_min","d1_hco3_max","d1_hco3_min","d1_hemaglobin_max","d1_hemaglobin_min","d1_hematocrit_max",
    "d1_hematocrit_min","d1_platelets_max","d1_platelets_min","d1_potassium_max","d1_potassium_min",
    "d1_sodium_max","d1_sodium_min","d1_wbc_max","d1_wbc_min","apache_4a_hospital_death_prob",
    "apache_4a_icu_death_prob","aids","cirrhosis","diabetes_mellitus","hepatic_failure","immunosuppression","leukemia",
    "lymphoma","solid_tumor_with_metastasis","apache_3j_bodysystem","apache_2_bodysystem"]

image_path = "medical1.jpg"
image = Image.open(image_path)
features = num_cols + cat_cols
sc_path = r"Model/standardbscaler.bin"
sc = load(sc_path)

st.set_page_config(page_title="Patient Survival Detection App",
                   page_icon="⚕️", layout="centered")

st.image(image, caption='Detecting Survival Chances of the patient')

# page header
st.title(f"Patient Survival Detection")
 
with st.form("Prediction_form"):
    # form header
    st.header("Enter the below mentioned details in order for the App to detect the survival of the patient:")
    # input elements
    age =  st.slider('Select age : ',min_value=16.0, max_value=89.0)
    bmi = st.slider('Select body mass index of the person on unit admission : ',min_value=14.84, max_value=67.81)
    height = st.slider('Select height of the person on unit admission : ',min_value=137.2, max_value=195.59)
    pre_icu_los_days = st.slider('Select length of stay of the patient between hospital admission and unit admission : ',min_value=-24.94, max_value=160.0)
    weight = st.slider('Select weight of the person on unit admission : ',min_value=38.0, max_value=187.0)
    apache_2_diagnosis = st.slider('Select the APACHE II diagnosis for the ICU admission : ',min_value=101.0, max_value=308.0)
    apache_3j_diagnosis = st.slider('Select the APACHE III-J sub-diagnosis code which best describes the reason for the ICU admission : ',min_value=101.0, max_value=308.0)
    arf_apache = st.slider('Select whether the pateient had acute renal failure : ',min_value=0.0, max_value=1.0)
    bun_apache = st.slider('Select blood urea nitrogen concentration of the person on unit admission : ',min_value=4.0, max_value=127.0)
    creatinine_apache = st.slider('Select the creatine concentration of  the person on unit admission : ',min_value=0.3, max_value=11.18)
    gcs_eyes_apache = st.slider('Select the eye opening component of Glasgow Coma Scale of the person on unit admission : ',min_value=1.0, max_value=4.0)
    gcs_motor_apache = st.slider('Select the motor opening component of Glasgow Coma Scale of the person on unit admission : ',min_value=1.0, max_value=6.0)
    gcs_unable_apache = st.slider('Select whether Glasgow Coma Scale of the person was unable to be asseesed due to sedation : ',min_value=0.0, max_value=1.0)
    gcs_verbal_apache = st.slider('Select the verbal component of Glasgow Coma Scale of the person on unit admission : ',min_value=1.0, max_value=5.0)
    glucose_apache = st.slider('Select the glucose concentration of the person measured on unit admission : ',min_value=39.0, max_value=598.7)
    heart_rate_apache = st.slider('Select heart rate of the person measured : ',min_value=30.0, max_value=178.0)
    hematocrit_apache = st.slider('Select heamatocrit of the person measured : ',min_value=16.0, max_value=52.0)
    intubated_apache = st.slider('SelectWhether the patient was intubated at the time of the highest scoring arterial blood gas used in the oxygenation score : ',min_value=0.0, max_value=1.0)
    map_apache = st.slider('Select the mean arterial pressure measured during the first 24 hours which results in the highest APACHE III score : ',min_value=40.0, max_value=200.0)
    resprate_apache = st.slider('Select the respiratory rate measured during the first 24 hours which results in the highest APACHE III score : ',min_value=4.0, max_value=60.0)
    sodium_apache = st.slider('Select the sodium concentration measured during the first 24 hours which results in the highest APACHE III score : ',min_value=117.0, max_value=158.0)
    temp_apache = st.slider('Select the temperature measured during the first 24 hours which results in the highest APACHE III score : ',min_value=32.0, max_value=40.0)
    ventilated_apache = st.slider('Select Whether the patient was invasively ventilated at the time of the highest scoring arterial blood gas using the oxygenation scoring algorithm : ',min_value=0.0, max_value=1.0)
    wbc_apache = st.slider('Select the  white blood cell count measured during the first 24 hours which results in the highest APACHE III score : ',min_value=0.5, max_value=46.0)
    d1_diasbp_max = st.slider('Select the highest diastolic blood pressure during the first 24 hours of their unit stay, invasively measured : ',min_value=46.0, max_value=165.0)
    d1_diasbp_min = st.slider('Select the lowest diastolic blood pressure during the first 24 hours of their unit stay, invasively measured : ',min_value=13.0, max_value=90.0)
    d1_diasbp_noninvasive_max = st.slider('Select highest diastolic blood pressure during the first 24 hours of their unit stay, non-invasively measured : ',min_value=46.0, max_value=165.0)
    d1_diasbp_noninvasive_min = st.slider('Select lowest diastolic blood pressure during the first 24 hours of their unit stay, non-invasively measured : ',min_value=13.0, max_value=90.0)
    d1_heartrate_max = st.slider('Select  highest heart rate during the first 24 hours of their unit stay : ',min_value=58.0, max_value=177.0)
    d1_heartrate_min = st.slider('Select  lowest heart rate during the first 24 hours of their unit stay : ',min_value=0.0, max_value=175.0)
    d1_mbp_max = st.slider('Select  highest mean blood pressure during the first 24 hours of their unit stay, invasively measured : ',min_value=60.0, max_value=184.0)
    d1_mbp_min = st.slider('Select  lowest mean blood pressure during the first 24 hours of their unit stay, invasively measured : ',min_value=13.0, max_value=90.0)
    d1_mbp_noninvasive_max = st.slider('Select highest mean blood pressure during the first 24 hours of their unit stay, non-invasively measured : ',min_value=60.0, max_value=181.0)
    d1_mbp_noninvasive_min = st.slider('Select lowest mean blood pressure during the first 24 hours of their unit stay, invasively measured : ',min_value=22.0, max_value=112.0)
    d1_resprate_max = st.slider('Select the highest respiratory rate during the first 24 hours of their unit stay : ',min_value=14.0, max_value=92.0)
    d1_resprate_min = st.slider('Select the lowest respiratory rate during the first 24 hours of their unit stay : ',min_value=0.0, max_value=100.0)
    d1_spo2_max = st.slider('Select highest peripheral oxygen saturation during the first 24 hours of their unit stay : ',min_value=0.0, max_value=100.0)
    d1_spo2_min = st.slider('Select lowest peripheral oxygen saturation during the first 24 hours of their unit stay : ',min_value=0.0, max_value=100.0)
    d1_sysbp_max = st.slider('Select  highest systolic blood pressure during the first 24 hours of their unit stay, either non-invasively or invasively measured : ',min_value=90.0, max_value=232.0)
    d1_sysbp_min = st.slider('Select  lowest systolic blood pressure during the first 24 hours of their unit stay, either non-invasively or invasively measured : ',min_value=41.0, max_value=160.0)
    d1_sysbp_noninvasive_max = st.slider('Select highest systolic blood pressure during the first 24 hours of their unit stay, non-invasively measured : ',min_value=90.0, max_value=232.0)
    d1_sysbp_noninvasive_min = st.slider('Select lowest systolic blood pressure during the first 24 hours of their unit stay, non-invasively measured : ',min_value=41.0, max_value=160.0)
    d1_temp_max = st.slider('Select highest core temperature during the first 24 hours of their unit stay, invasively measured : ',min_value=35.0, max_value=40.0)
    d1_temp_min = st.slider('Select lowest core temperature during the first 24 hours of their unit stay, invasively measured : ',min_value=31.0, max_value=38.0)
    h1_diasbp_max = st.slider('Select highest diastolic blood pressure during the first hour of their unit stay, either non-invasively or invasively measured : ',min_value=37.0, max_value=143.0)
    h1_diasbp_min = st.slider('Select lowest diastolic blood pressure during the first hour of their unit stay, either non-invasively or invasively measured : ',min_value=22.0, max_value=113.0)
    h1_diasbp_noninvasive_max = st.slider('Select highest diastolic blood pressure during the first hour of their unit stay, non-invasively measured : ',min_value=37.0, max_value=144.0)
    h1_diasbp_noninvasive_min = st.slider('Select lowest diastolic blood pressure during the first hour of their unit stay,non-invasively  measured : ',min_value=22.0, max_value=114.0)
    h1_heartrate_max = st.slider('Select highest heart rate during the first hour of their unit stay : ',min_value=46.0, max_value=164.0)
    h1_heartrate_min = st.slider('Select lowest heart rate during the first hour of their unit stay : ',min_value=36.0, max_value=144.0)
    h1_mbp_max = st.slider('Select highest mean blood pressure during the first hour of their unit stay, either non-invasively or invasively measured : ',min_value=49.0, max_value=165.0)
    h1_mbp_min = st.slider('Select lowest mean blood pressure during the first hour of their unit stay, either non-invasively or invasively measured : ',min_value=32.0, max_value=138.0)
    h1_mbp_noninvasive_max = st.slider('Select highest mean blood pressure during the first hour of their unit stay, non-invasively measured : ',min_value=49.0, max_value=163.0)
    h1_mbp_noninvasive_min = st.slider('Select lowest mean blood pressure during the first hour of their unit stay, non-invasively measured : ',min_value=32.0, max_value=138.0)
    h1_resprate_max = st.slider('Select highest respiratory rate during the first hour of their unit stay : ',min_value=10.0, max_value=59.0)
    h1_resprate_min = st.slider('Select lowest respiratory rate during the first hour of their unit stay : ',min_value=0.0, max_value=189.0)
    h1_spo2_max = st.slider('Select highest peripheral oxygen saturation during the first hour of their unit stay : ',min_value=0.0, max_value=100.0)
    h1_spo2_min = st.slider('Select lowest peripheral oxygen saturation during the first hour of their unit stay : ',min_value=0.0, max_value=100.0)
    h1_sysbp_max = st.slider('Select highest systolic blood pressure during the first hour of their unit stay, either non-invasively or invasively measured : ',min_value=75.0, max_value=223.0)
    h1_sysbp_min = st.slider('Select lowest systolic blood pressure during the first hour of their unit stay, either non-invasively or invasively measured : ',min_value=53.0, max_value=194.0)
    h1_sysbp_noninvasive_max = st.slider('Select highest systolic blood pressure during the first hour of their unit stay,non-invasively measured : ',min_value=75.0, max_value=223.0)
    h1_sysbp_noninvasive_min = st.slider('Select lowest systolic blood pressure during the first hour of their unit stay, non-invasively measured : ',min_value=53.0, max_value=195.0)
    h1_temp_max = st.slider('Select highest core temperature during the first hour of their unit stay, invasively measured : ',min_value=33.4, max_value=39.5)
    h1_temp_min = st.slider('Select lowest core temperature during the first hour of their unit stay, invasively measured : ',min_value=4.0, max_value=126.0)
    d1_bun_max = st.slider('Select highest blood urea nitrogen concentration of the patient in their serum or plasma during the first 24 hours of their unit stay : ',min_value=4.0, max_value=126.0)
    d1_bun_min = st.slider('Select lowest blood urea nitrogen concentration of the patient in their serum or plasma during the first 24 hours of their unit stay : ',min_value=3.0, max_value=114.0)
    d1_calcium_max = st.slider('Select highest calcium concentration of the patient in their serum during the first 24 hours of their unit stay : ',min_value=6.2, max_value=10.8)
    d1_calcium_min = st.slider('Select lowest calcium concentration of the patient in their serum during the first 24 hours of their unit stay : ',min_value=5.5, max_value=10.3)
    d1_creatinine_max = st.slider('Select  highest creatinine concentration of the patient in their serum or plasma during the first 24 hours of their unit stay : ',min_value=0.34, max_value=11.11)
    d1_creatinine_min = st.slider('Select  lowest creatinine concentration of the patient in their serum or plasma during the first 24 hours of their unit stay : ',min_value=0.3, max_value=9.94)
    d1_glucose_max = st.slider('Select highest glucose concentration of the patient in their serum or plasma during the first 24 hours of their unit stay : ',min_value=73.0, max_value=611.0)
    d1_glucose_min = st.slider('Select lowest glucose concentration of the patient in their serum or plasma during the first 24 hours of their unit stay: ',min_value=33.0, max_value=288.0)
    d1_hco3_max  = st.slider('Select highest bicarbonate concentration for the patient in their serum or plasma during the first 24 hours of their unit stay : ',min_value=12.0, max_value=40.0)
    d1_hco3_min = st.slider('Select lowest bicarbonate concentration for the patient in their serum or plasma during the first 24 hours of their unit stay : ',min_value=7.0, max_value=39.0)
    d1_hemaglobin_max  = st.slider('Select highest hemoglobin concentration for the patient during the first 24 hours of their unit stay : ',min_value=6.8, max_value=17.2)
    d1_hemaglobin_min = st.slider('Select lowest hemoglobin concentration for the patient during the first 24 hours of their unit stay : ',min_value=5.3, max_value=16.7)
    d1_hematocrit_max  = st.slider('Select highest volume proportion of red blood cells in blood during the first 24 hours of their unit stay, expressed as a fraction : ',min_value=20.4, max_value=51.5)
    d1_hematocrit_min = st.slider('Select lowest volume proportion of red blood cells in blood during the first 24 hours of their unit stay, expressed as a fraction : ',min_value=16.1, max_value=50.0)
    d1_platelets_max = st.slider('Select highest platelet count for the patient during the first 24 hours of their unit stay : ',min_value=27.0, max_value=585.0)
    d1_platelets_min = st.slider('Select lowest platelet count for the patient during the first 24 hours of their unit stay : ',min_value=18.55, max_value=558.0)
    d1_potassium_max = st.slider('Select highest potassium concentration for the patient in their serum or plasma during the first 24 hours of their unit stay : ',min_value=3.0, max_value=7.0)
    d1_potassium_min = st.slider('Select lowest potassium concentration for the patient in their serum or plasma during the first 24 hours of their unit stay : ',min_value=2.0, max_value=6.0)
    d1_sodium_max = st.slider('Select highest sodium concentration for the patient in their serum or plasma during the first 24 hours of their unit stay : ',min_value=123.0, max_value=158.0)
    d1_sodium_min = st.slider('Select lowest sodium concentration for the patient in their serum or plasma during the first 24 hours of their unit stay : ',min_value=117.0, max_value=153.0)
    d1_wbc_max = st.slider('Select highest white blood cell count for the patient during the first 24 hours of their unit stay : ',min_value=1.2, max_value=47.0)
    d1_wbc_min = st.slider('Select lowest white blood cell count for the patient during the first 24 hours of their unit stay : ',min_value=0.9, max_value=40.9)
    apache_4a_hospital_death_prob  = st.slider('Select  APACHE IVa probabilistic prediction of in-hospital mortality for the patient which utilizes the APACHE III score and other covariates, including diagnosis : ',min_value=-1.0, max_value=0.98)
    apache_4a_icu_death_prob = st.slider('Select APACHE IVa probabilistic prediction of in ICU mortality for the patient which utilizes the APACHE III score and other covariates, including diagnosis : ',min_value=-1.0, max_value=0.97)
    aids = st.slider('Select Whether the patient has a definitive diagnosis of acquired immune deficiency syndrome (AIDS) (not HIV positive alone) : ',min_value=0.0, max_value=1.0)
    cirrhosis = st.slider('Select Whether the patient has a history of heavy alcohol use with portal hypertension and varices : ',min_value=0.0, max_value=1.0)
    diabetes_mellitus = st.slider('Select Whether the patient has been diagnosed with diabetes, either juvenile or adult onset, which requires medication : ',min_value=0.0, max_value=1.0)
    hepatic_failure = st.slider('Select Whether the patient hd a hepatic failure while admission : ',min_value=0.0, max_value=1.0)
    immunosuppression = st.slider('Select the patient has their immune system suppressed within six months prior to ICU admission : ',min_value=0.0, max_value=1.0)
    leukemia = st.slider('Select Whether the patient has been diagnosed with acute or chronic myelogenous leukemia, acute or chronic lymphocytic leukemia, or multiple myeloma : ',min_value=0.0, max_value=1.0)
    lymphoma = st.slider('Select Whether the patient has been diagnosed with non-Hodgkin lymphoma : ',min_value=0.0, max_value=1.0)
    solid_tumor_with_metastasis = st.slider('Select Whether the patient has been diagnosed with any solid tumor carcinoma (including malignant melanoma) which has evidence of metastasis : ',min_value=0.0, max_value=1.0)
    hospital_id = st.slider('Select Unique identifier associated with a hospital : ',min_value = 2, max_value=204)
    elective_surgery  = st.slider('Select Whether the patient was admitted to the hospital for an elective surgical operation : ',min_value = 2, max_value=204)
    icu_id  = st.slider('Select unique identifier for the unit to which the patient was admitted : ',min_value = 82, max_value=927)
    readmission_status = st.slider('Select Whether the current unit stay is the second (or greater) stay at an ICU within the same hospitalization : ',min_value = 0, max_value=1)
    apache_post_operative = st.slider('Select  APACHE operative status; 1 for post-operative, 0 for non-operative : ',min_value = 0, max_value=1)
    ethnicity = st.selectbox('Select ethnicity of the patient : ',('Caucasian','Hispanic','African American','Asian','Native American','Other/Unknown'))
    gender = st.selectbox('Select Gender of the patient : ',('M','F'))
    hospital_admit_source = st.selectbox('Select Source of admission : ',('Floor','Emergency Department','Operating Room','missing','Direct Admit',
                                                                          'Other Hospital','ICU to SDU','Recovery Room','Chest Pain Center',
                                                                          'Other ICU','Step-Down Unit (SDU)','PACU','Acute Care/Floor','ICU',
                                                                          'Other','Observation'))
    icu_admit_source = st.selectbox('Select Source of ICU admission : ',('Floor','Accident & Emergency','Operating Room / Recovery',
                                                                         'Other Hospital','Other ICU'))
    icu_stay_type = st.selectbox('Select ICU stya type : ',('admit','transfer','readmit'))
    icu_type = st.selectbox('Select ICU type : ',('Med-Surg ICU','CTICU','MICU','CCU-CTICU','SICU','Neuro ICU','Cardiac ICU','CSICU'))
    apache_3j_bodysystem = st.selectbox('Select Admission diagnosis group for APACHE III : ',('Respiratory','Metabolic','Cardiovascular','Neurological',
                                                                                              'Sepsis','Genitourinary','Gastrointestinal','Trauma',
                                                                                              'Musculoskeletal/Skin','Hematological','Gynecological'))
    apache_2_bodysystem = st.selectbox('Select Admission diagnosis group for APACHE II : ',('Respiratory','Metabolic','Cardiovascular','Neurologic',
                                                                                            'Renal/Genitourinary','Gastrointestinal','Trauma',
                                                                                            'Undefined diagnoses','Haematologic','Undefined Diagnoses'))
    
    # 
    submit = st.form_submit_button("Predict")
    #
    if submit :
        input_dict = {"hospital_id":hospital_id, 
                      "age":age,
                      "bmi":bmi,
                      "elective_surgery":elective_surgery,
                      "ethnicity":ethnicity,
                      "gender":gender,
                      "height":height,
                      "hospital_admit_source":hospital_admit_source,
                      "icu_admit_source":icu_admit_source,
                      "icu_id":icu_id,
                      "icu_stay_type":icu_stay_type,
                      "icu_type":icu_type,
                      "pre_icu_los_days":pre_icu_los_days,
                      "readmission_status":readmission_status,
                      "weight":weight,
                      "apache_2_diagnosis":apache_2_diagnosis,
                      "apache_3j_diagnosis":apache_3j_diagnosis,
                      "apache_post_operative":apache_post_operative,
                      "arf_apache":arf_apache,
                      "bun_apache":bun_apache,
                      "creatinine_apache":creatinine_apache,
                      "gcs_eyes_apache":gcs_eyes_apache,
                      "gcs_motor_apache":gcs_motor_apache,
                      "gcs_unable_apache":gcs_unable_apache,
                      "gcs_verbal_apache":gcs_verbal_apache,
                      "glucose_apache":glucose_apache,
                      "heart_rate_apache":heart_rate_apache,
                      "hematocrit_apache":hematocrit_apache,
                      "intubated_apache":intubated_apache,
                      "map_apache":map_apache,
                      "resprate_apache":resprate_apache,
                      "sodium_apache":sodium_apache,
                      "temp_apache":temp_apache,
                      "ventilated_apache":ventilated_apache,
                      "wbc_apache":wbc_apache,
                      "d1_diasbp_max":d1_diasbp_max,
                      "d1_diasbp_min":d1_diasbp_min,
                      "d1_diasbp_noninvasive_max":d1_diasbp_noninvasive_max,
                      "d1_diasbp_noninvasive_min":d1_diasbp_noninvasive_min,
                      "d1_heartrate_max":d1_heartrate_max,
                      "d1_heartrate_min":d1_heartrate_min,
                      "d1_mbp_max":d1_mbp_max,
                      "d1_mbp_min":d1_mbp_min,
                      "d1_mbp_noninvasive_max":d1_mbp_noninvasive_max,
                      "d1_mbp_noninvasive_min":d1_mbp_noninvasive_min,
                      "d1_resprate_max":d1_resprate_max,
                      "d1_resprate_min":d1_resprate_min,
                      "d1_spo2_max":d1_spo2_max,
                      "d1_spo2_min":d1_spo2_min,
                      "d1_sysbp_max":d1_sysbp_max,
                      "d1_sysbp_min":d1_sysbp_min,
                      "d1_sysbp_noninvasive_max":d1_sysbp_noninvasive_max,
                      "d1_sysbp_noninvasive_min":d1_sysbp_noninvasive_min,
                      "d1_temp_max":d1_temp_max,
                      "d1_temp_min":d1_temp_min,
                      "h1_diasbp_max":h1_diasbp_max,
                      "h1_diasbp_min":h1_diasbp_min,
                      "h1_diasbp_noninvasive_max": h1_diasbp_noninvasive_max,
                      "h1_diasbp_noninvasive_min":h1_diasbp_noninvasive_min,
                      "h1_heartrate_max":h1_heartrate_max,
                      "h1_heartrate_min":h1_heartrate_min,
                      "h1_mbp_max":h1_mbp_max,
                      "h1_mbp_min":h1_mbp_min,
                      "h1_mbp_noninvasive_max":h1_mbp_noninvasive_max,
                      "h1_mbp_noninvasive_min":h1_mbp_noninvasive_min,
                      "h1_resprate_max":h1_resprate_max,
                      "h1_resprate_min":h1_resprate_min,
                      "h1_spo2_max":h1_spo2_max,
                      "h1_spo2_min":h1_spo2_min,
                      "h1_sysbp_max":h1_sysbp_max,
                      "h1_sysbp_min":h1_sysbp_min,
                      "h1_sysbp_noninvasive_max":h1_sysbp_noninvasive_max,
                      "h1_sysbp_noninvasive_min":h1_sysbp_noninvasive_min,
                      "h1_temp_max":h1_temp_max,
                      "h1_temp_min":h1_temp_min,
                      "d1_bun_max":d1_bun_max,
                      "d1_bun_min":d1_bun_min,
                      "d1_calcium_max":d1_calcium_max,
                      "d1_calcium_min":d1_calcium_min,
                      "d1_creatinine_max":d1_creatinine_max,
                      "d1_creatinine_min":d1_creatinine_min,
                      "d1_glucose_max":d1_glucose_max,
                      "d1_glucose_min":d1_glucose_min,
                      "d1_hco3_max":d1_hco3_max,
                      "d1_hco3_min":d1_hco3_min,
                      "d1_hemaglobin_max":d1_hemaglobin_max,
                      "d1_hemaglobin_min":d1_hemaglobin_min,
                      "d1_hematocrit_max":d1_hematocrit_max,
                      "d1_hematocrit_min":d1_hematocrit_min,
                      "d1_platelets_max":d1_platelets_max,
                      "d1_platelets_min":d1_platelets_min,
                      "d1_potassium_max":d1_potassium_max,
                      "d1_potassium_min":d1_potassium_min,
                      "d1_sodium_max":d1_sodium_max,
                      "d1_sodium_min":d1_sodium_min,
                      "d1_wbc_max":d1_wbc_max,
                      "d1_wbc_min":d1_wbc_min,
                      "apache_4a_hospital_death_prob":apache_4a_hospital_death_prob,
                      "apache_4a_icu_death_prob":apache_4a_icu_death_prob,
                      "aids":aids,
                      "cirrhosis":cirrhosis,
                      "diabetes_mellitus":diabetes_mellitus,
                      "hepatic_failure":hepatic_failure,
                      "immunosuppression":immunosuppression,
                      "leukemia":leukemia,
                      "lymphoma":lymphoma,
                      "solid_tumor_with_metastasis":solid_tumor_with_metastasis,
                      "ethnicity":ethnicity,
                      "gender":gender, 
                      "hospital_admit_source":hospital_admit_source, 
                      "icu_admit_source":icu_admit_source,
                      "icu_stay_type":icu_stay_type, 
                      "icu_type":icu_type,
                      "apache_3j_bodysystem":apache_3j_bodysystem,
                      "apache_2_bodysystem":apache_2_bodysystem}
        
        df = pd.DataFrame(input_dict, index=[1])
        print(df)
        final_df = train_formatter(df)
        print(len(final_df))
        print(final_df.head())
        X_train_scaled = sc.transform(final_df)
        #
        prediction = predict(X_train_scaled)
        print(prediction)
        if prediction >=0.5:
            result =1
        else:
            result = 0
        map_acc_sev = { 0 :'Survive', 1: 'Demise'}
        value = map_acc_sev[result]
        # output header
        st.header("Predictions")
        # output results
        st.success(f"Survival Prediction : {value}")