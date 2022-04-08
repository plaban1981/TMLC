import pandas as pd


def create_train():
    with open(r"Data/features.txt","r") as f:
        data = f.read()
    data_list = [ d for d in data.split(",") if d > " "]
    data_dict = {k:0 for k in data_list}
    print(data_dict)
    train = pd.DataFrame(data_dict,index=[1])
    print(train.head())
    return train
#

#
def train_formatter(df):
    cat_cols  = ['ethnicity', 'gender', 'hospital_admit_source', 'icu_admit_source',
       'icu_stay_type', 'icu_type', 'apache_3j_bodysystem',
       'apache_2_bodysystem']
    num_cols = [col for col in df.columns if col not in cat_cols]
    train = create_train()
    print(train.columns)
    #train[num_cols] = df[num_cols]
    #ethinicity
    e = df['ethnicity'].values.tolist()[0]
    g = df["gender"].values.tolist()[0]
    h = df['hospital_admit_source'].values.tolist()[0]
    i = df['icu_admit_source'].values.tolist()[0]
    ic = df['icu_stay_type'].values.tolist()[0]
    ict = df['icu_type'].values.tolist()[0]
    a3 = df['apache_3j_bodysystem'].values.tolist()[0]
    a2 = df['apache_2_bodysystem'].values.tolist()[0]
    
    if e == 'Caucasian':
        train["ethnicity_Caucasian"] == 1
    elif e == 'Hispanic':
        train["ethnicity_Hispanic"] == 1
    elif e == 'African American':
        train["ethnicity_African American"] == 1
    elif e == 'Asian':
        train["ethnicity_Asian"] == 1
    elif e == 'Native American':
        train["ethnicity_Native American"] == 1
    else:
        train["ethnicity_Other/Unknown"] == 1
    #gender
    if g == 'F':
        train["gender_F"] = 1
    else:
        train["gender_M"] = 1
    #hospital_admit_source
    if h == 'Floor':
       train['hospital_admit_source_Floor'] = 1
    elif h == 'Emergency Department':
        train['hospital_admit_source_Emergency Department'] = 1
    elif h == 'Operating Room':
        train['hospital_admit_source_Operating Room'] = 1
    elif h == 'missing':
        train['hospital_admit_source_missing'] = 1
    elif  h == 'Direct Admit':
        train['hospital_admit_source_Direct Admit'] = 1
    elif h == 'Other Hospital':
        train['hospital_admit_source_Other Hospital'] = 1
    elif  h == 'ICU to SDU':
        train['hospital_admit_source_ICU to SDU'] = 1
    elif h == 'Recovery Room':
        train['hospital_admit_source_Recovery Room'] = 1
    elif h == 'Chest Pain Center':
        train['hospital_admit_source_Chest Pain Center'] = 1
    elif h == 'Other ICU':
        train['hospital_admit_source_Other ICU'] = 1
    elif  h == 'Step-Down Unit (SDU)':
        train['hospital_admit_source_Step-Down Unit (SDU)'] = 1
    elif  h == 'PACU':
        train['hospital_admit_source_PACU'] = 1
    elif  h == 'Acute Care/Floor':
        train['hospital_admit_source_Acute Care/Floor'] = 1
    elif  h == 'ICU':
        train['hospital_admit_source_ICU'] = 1
    elif h == 'Observation':
        train['hospital_admit_source_Observation'] = 1
    else:
        train['hospital_admit_source_Other'] = 1
 
    #icu_admit_source
    if i == 'Accident & Emergency':
        train["icu_admit_source_Accident & Emergency"] = 1
    elif i == 'Floor':
        train["icu_admit_source_Floor"] = 1
    elif i == 'Operating Room / Recovery':
        train["icu_admit_source_Operating Room / Recovery"] = 1
    elif i == 'Other Hospital':
        train["icu_admit_source_Other Hospital"] = 1
    elif i == 'Other ICU':
        train["icu_admit_source_Other ICU"] = 1
    else:
        train["icu_admit_source_missing"] = 1
        
    # icu_stay_type
    if ic == 'admit':
        train["icu_stay_type_admit"] = 1
    elif ic == 'readmit':
        train["icu_stay_type_readmit"] = 1
    else:
        train["icu_stay_type_transfer"] = 1

    # icu_type
    if ict == 'CCU-CTICU':
        train["icu_type_CCU-CTICU"] = 1
    elif ict == 'CSICU':
        train["icu_type_CSICU"] = 1
    elif ict == 'CTICU':
        train["icu_type_CTICU"] = 1
    elif ict == 'Cardiac ICU':
        train["icu_type_Cardiac ICU"] = 1
    elif ict == 'MICU':
        train["icu_type_MICU"] = 1
    elif ict == 'Med-Surg ICU':
        train["icu_type_Med-Surg ICU"] = 1
    elif ict == 'Neuro ICU':
        train["icu_type_Neuro ICU"] = 1
    else:
        train["icu_type_SICU"] = 1
        
    #apache_3j_bodysystem
    if a3 == 'Cardiovascular':
        train["apache_3j_bodysystem_Cardiovascular"] = 1
    elif a3 == 'Gastrointestinal':
        train["apache_3j_bodysystem_Gastrointestinal"] = 1
    elif a3 == 'Genitourinary':
        train["apache_3j_bodysystem_Genitourinary"] = 1
    elif a3 == 'Gynecological':
        train["apache_3j_bodysystem_Gynecological"] = 1
    elif a3 == 'Hematological':
        train["apache_3j_bodysystem_Hematological"] = 1
    elif a3 == 'Metabolic':
        train["apache_3j_bodysystem_Metabolic"] = 1
    elif a3 == 'Musculoskeletal/Skin':
        train["apache_3j_bodysystem_Musculoskeletal/Skin"] = 1
    elif a3 == 'Neurological':
        train["apache_3j_bodysystem_Neurological"] = 1
    elif a3 == 'Respiratory':
        train["apache_3j_bodysystem_Respiratory"] = 1
    elif a3 == 'Sepsis':
        train["apache_3j_bodysystem_Sepsis"] = 1
    elif a3 == 'Trauma':
        train["apache_3j_bodysystem_Trauma"] = 1
    else:
        train["apache_3j_bodysystem_missing"] = 1
    
    #apache_2_bodysystem
    if a2 == 'Cardiovascular':
        train["apache_2_bodysystem_Cardiovascular"] = 1
    elif a2 == 'Gastrointestinal':
        train["apache_2_bodysystem_Gastrointestinal"] = 1
    elif a2 == 'Haematologic':
        train["apache_2_bodysystem_Haematologic"] = 1
    elif a2 == 'Metabolic':
        train["apache_2_bodysystem_Metabolic"] = 1
    elif a2 == 'Neurologic':
        train["apache_2_bodysystem_Neurologic"] = 1
    elif a2 == 'Neurologic':
        train["apache_2_bodysystem_Neurologic"] = 1
    elif a2 == 'Renal/Genitourinary':
        train["apache_2_bodysystem_Renal/Genitourinary"] = 1
    elif a2 == 'Respiratory':
        train["apache_2_bodysystem_Respiratory"] = 1
    elif a2 == 'Trauma':
        train["apache_2_bodysystem_Trauma"] = 1
    elif a2 == 'Undefined Diagnoses':
        train["apache_2_bodysystem_Undefined Diagnoses"] = 1
    elif a2 == 'Undefined diagnoses':
        train["apache_2_bodysystem_Undefined diagnoses"] = 1
    else:
        train["apache_2_bodysystem_missing"] = 1
   
    return train