# Copyright (c) 2024 Houssem Ben Braiek, Emilio Rivera-Landos, IVADO, SEMLA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

from dotenv import load_dotenv
load_dotenv()
import fsspec
import fsspec.implementations.local
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

from bank_marketing.data.make_datasets import extract_credit_features, merge_defaults
from bank_marketing.sqlite_db.bank_marketing_DAL import BankMarketingDAL
from bank_marketing.helpers.file_loaders import load_fsspec_locally_temp

# Data sources
SERVER_API_URL = os.environ.get('SERVER_API_URL')
BANK_DB = os.environ.get('BANK_DB')
ECO_SOCIO_DF = os.environ.get('ECO_SOCIO_DF')
FUTURE_RES_DF = os.environ.get('FUTURE_RES_DF')
# --- APP SETTINGS ---
DISPLAYED_ATTRIBUTES = [
    'first_name',
    'last_name',
    'job',
    'age',
    'education',
    'marital',
    'housing',
    'loan',
    'phone',
]

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if '__file__' in locals() else Path.cwd()
css_file = current_dir / 'styles' / 'main.css'
# -------------- SETTINGS --------------
page_title = 'Advertising Phone Campaign'
page_icon = ':telephone:'  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = 'wide'

# --------- Page SETUP -----------
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
# --- LOAD CSS ---
with css_file.open() as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_resource
def connect_to_data(bank_db_file, eco_socio_data_file, future_results_data_file):
    # Detect if bank_db_file is local or not, so we can acquire (download)
    # the file if remote.
    db_url = bank_db_file
    _db = fsspec.open(bank_db_file)
    db_downloaded_file = None
    if not isinstance(_db.fs, fsspec.implementations.local.LocalFileSystem):
        db_downloaded_file = load_fsspec_locally_temp(bank_db_file, binary=True)
        db_url = db_downloaded_file.name
    bank_marketing_dl = BankMarketingDAL(db_url)
    # Load dataframes
    with fsspec.open(eco_socio_data_file) as f:
        socio_eco_indices_data = pd.read_csv(f, sep=";")
    with fsspec.open(future_results_data_file) as f:
        future_results_data = pd.read_csv(f, sep=";")
    return bank_marketing_dl, socio_eco_indices_data, future_results_data


def build_features(bank_marketing_dl, socio_eco_indices_data):
    current_date = st.session_state.current_date
    loans_df = bank_marketing_dl.loans.fetch_all(to_dataframe=True)
    mortgages_df = bank_marketing_dl.mortgages.fetch_all(to_dataframe=True)
    mortgages_df[['housing', 'default']] = mortgages_df.apply(
        extract_credit_features, axis=1, result_type='expand'
    )
    loans_df[['loan', 'default']] = loans_df.apply(
        extract_credit_features, axis=1, result_type='expand'
    )
    customers_df = bank_marketing_dl.customers.fetch_all(to_dataframe=True)
    campaign_missions_df = bank_marketing_dl.campaign_missions.fetch_by_date(
        current_date, to_dataframe=True
    )
    dataframe = pd.merge(customers_df, campaign_missions_df, left_on='id', right_on='customer_id')
    dataframe = pd.merge(dataframe, socio_eco_indices_data, left_on='comm_date', right_on='date')
    dataframe = pd.merge(
        dataframe, mortgages_df[['customer_id', 'housing', 'default']], on='customer_id'
    )
    dataframe = pd.merge(dataframe, loans_df[['customer_id', 'loan', 'default']], on='customer_id')
    dataframe['default'] = dataframe.apply(merge_defaults, axis=1)
    dataframe = dataframe.drop(columns=['default_x', 'default_y'])
    return dataframe


bank_marketing_dl, socio_eco_indices_data, future_results_data = connect_to_data(
    BANK_DB, ECO_SOCIO_DF, FUTURE_RES_DF
)

if 'curr_mission_ord' not in st.session_state:
    st.session_state.curr_mission_ord = 1

if 'campaign_missions' not in st.session_state:
    st.session_state.campaign_missions = []


def precedent_mission(missions_total):
    if st.session_state.curr_mission_ord > 1:
        st.session_state.curr_mission_ord -= 1
    elif st.session_state.curr_mission_ord == 1:
        st.session_state.curr_mission_ord = missions_total


def next_mission(missions_total):
    if st.session_state.curr_mission_ord < missions_total:
        st.session_state.curr_mission_ord += 1
    elif st.session_state.curr_mission_ord == missions_total:
        st.session_state.curr_mission_ord = 1


def reload_campaign_missions():
    ai_assistance = st.session_state.AI_assistance
    current_date = st.session_state.current_date
    if ai_assistance == 'Without' or ai_assistance == 'Partial':
        st.session_state.campaign_missions = bank_marketing_dl.campaign_missions.fetch_by_date(
            current_date, to_dict=True
        )
    elif ai_assistance == 'Full':
        data_features = build_features(bank_marketing_dl, socio_eco_indices_data)
        request_dict = {k: list(v.values()) for k, v in data_features.to_dict().items()}
        response = requests.post(SERVER_API_URL, json=request_dict)
        response_dict = {**request_dict, **response.json()}
        response_data = pd.DataFrame.from_dict(response_dict)
        success_data = response_data[response_data['prediction'] == 1]
        st.session_state.campaign_missions = success_data[
            bank_marketing_dl.campaign_missions.attrs
        ].to_dict('records')


profile_pic = current_dir / 'assets' / 'profile-pic.jpg'
# ---- SIDEBAR ----
profile_pic = Image.open(profile_pic)
st.sidebar.image(profile_pic, width=120)
st.sidebar.title('Employee Name')
future_dates = np.unique(
    [elt['comm_date'] for elt in bank_marketing_dl.campaign_missions.fetch_all_not_done(True)]
).tolist()

st.sidebar.selectbox('Date', future_dates, key='current_date')

st.sidebar.radio(
    'AI assistance',
    ('Without', 'Partial', 'Full'),
    key='AI_assistance',
    on_change=reload_campaign_missions,
)

st.sidebar.button('Start', key='start', on_click=reload_campaign_missions)

if st.session_state.campaign_missions:
    campaign_missions_count = len(st.session_state.campaign_missions)
    col1, _, col2, _, col3 = st.columns([2, 4, 2, 4, 2])
    col2.write(
        f'**< {st.session_state.curr_mission_ord}' + ' / ' + f'{campaign_missions_count} >**'
    )
    col1.button(
        'Prev.', on_click=precedent_mission, key='previous', args=(campaign_missions_count,)
    )
    col3.button('Next', on_click=next_mission, key='next', args=(campaign_missions_count,))
else:
    st.header('Click on **Start** to fetch prospects...')

if st.session_state.campaign_missions:
    curr_mission = st.session_state.campaign_missions[st.session_state.curr_mission_ord - 1]
    curr_customer = bank_marketing_dl.customers.fetch_one(curr_mission['customer_id'], True)

    col1, col2, col3 = st.columns([6, 2, 1])
    col1.title(f'Prospect: {curr_customer["first_name"]} {curr_customer["last_name"]}')
    col2.text('')
    col2.text('')
    col2.text('')
    AI_response = col2.empty()
    if st.session_state.AI_assistance == 'Partial':
        AI_response.text('AI Loading...')
    elif st.session_state.AI_assistance == 'Full':
        AI_response.markdown('**(AI Suggests: <g>CALL</g>)**', unsafe_allow_html=True)
    col3.text('')
    col3.text('')
    call = col3.button('Call', key='call', args=())

    if call:
        curr_customer_outcome = future_results_data[
            future_results_data['id'] == curr_mission['customer_id']
        ].iloc[0]
        answer = str(curr_customer_outcome['y'])
        duration = curr_customer_outcome['duration']
        if answer == 'yes':
            st.success(f'SUCCESS: The call lasts {duration//60} mins; {duration%60} seconds')
        elif answer == 'no':
            st.error(f'FAILURE: The call that lasts {duration//60} mins; {duration%60} seconds')

    st.text('')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'Age: {curr_customer["age"]}')
        st.markdown(f'Marital: {curr_customer["marital"]}')

    with col2:
        st.markdown(f'Mail: {curr_customer["email"]}')
        st.markdown(f'Phone: {curr_customer["phone"]}')

    with col3:
        st.markdown(f'Job: {curr_customer["job"]}')
        st.markdown(f'Education: {curr_customer["education"]}')

    st.divider()  # 👈 Another horizontal rule
    st.subheader('Campaign Infos:')
    st.text('')
    col1, col2, col3 = st.columns([3, 1, 3])
    with col1:
        st.markdown(f'Ongoing number of contacts: {curr_mission["curr_n_contact"]}')
        st.markdown(f'Preferred communication: {curr_mission["comm_type"]}')

    with col3:
        pdays = curr_mission['days_since_last_campaign']
        st.markdown(f'Days since last campaign: {"Non-Available" if pdays==999 else pdays}')
        st.markdown(f'Last campaign number of contacts: {curr_mission["last_n_contact"]}')
        st.markdown(f'Last campaign outcome: {curr_mission["last_outcome"]}')

    st.divider()  # 👈 Another horizontal rule
    st.subheader('Loans Data:')
    st.text('')
    loans_df = bank_marketing_dl.loans.fetch_by_customer(
        curr_mission['customer_id'], to_dataframe=True
    )
    st.dataframe(loans_df.drop(columns=['customer_id']))

    st.divider()  # 👈 Another horizontal rule
    st.subheader('Mortgages Data:')
    st.text('')
    mortgages_df = bank_marketing_dl.mortgages.fetch_by_customer(
        curr_mission['customer_id'], to_dataframe=True
    )
    st.dataframe(mortgages_df.drop(columns=['customer_id']))

    if st.session_state.AI_assistance == 'Partial':
        mortgages_df[['housing', 'default']] = mortgages_df.apply(
            extract_credit_features, axis=1, result_type='expand'
        )
        loans_df[['loan', 'default']] = loans_df.apply(
            extract_credit_features, axis=1, result_type='expand'
        )
        merged_dict = {**curr_mission, **curr_customer}
        data_features = pd.DataFrame().from_records([merged_dict])
        data_features = pd.merge(
            data_features, socio_eco_indices_data, left_on='comm_date', right_on='date'
        )
        data_features = pd.merge(
            data_features, mortgages_df[['customer_id', 'housing', 'default']], on='customer_id'
        )
        data_features = pd.merge(
            data_features, loans_df[['customer_id', 'loan', 'default']], on='customer_id'
        )
        data_features['default'] = data_features.apply(merge_defaults, axis=1)
        data_features = data_features.drop(columns=['default_x', 'default_y'])
        request_dict = {k: list(v.values()) for k, v in data_features.to_dict().items()}
        response = requests.post(SERVER_API_URL, json=request_dict)
        response_dict = response.json()
        suggestion = '<g>CALL</g>' if response_dict['prediction'][0] == 1 else '<r>SKIP</r>'
        AI_response.markdown(
            f'**(AI Suggests: {suggestion})**',
            unsafe_allow_html=True,
        )
