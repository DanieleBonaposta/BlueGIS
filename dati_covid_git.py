#!/usr/bin/python
# -*- coding: utf-8 -*-

# https://howto.webarea.it/linux/utilizzo-di-crontab-per-schedulare-processi-con-esempi-sotto-linux_1
# http://guide.debianizzati.org/index.php/Utilizzo_del_servizio_di_scheduling_Cron
# https://stackoverflow.com/questions/8727935/execute-python-script-via-crontab

import os, glob
import datetime
from datetime import date, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
import mapclassify
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_axes_aligner import align
from urllib.error import HTTPError

# variabili
folder = '/home/daniele/LAVORO/IN_CORSO/blue_GIS/Italia/Covid_19/' # per i salvataggi
oggi = date.today().strftime("%d")
mese = date.today().strftime("%B")
mese_n = int(date.today().strftime("%m"))
yesterday = date.today() - timedelta(days=1)
plotlist = []
# i dati del giorno corrente vengono elaborati solo se sono passate le 17, altrimenti ci si riferisce a ieri
if int(datetime.datetime.now().strftime("%H")) > 17:
    data_oggi = date.today().strftime("%Y-%m-%d")
else:
    data_oggi = (date.today() - timedelta(days = 1)).strftime("%Y-%m-%d")
# zangrillo
zang = 'Zangrillo, 31 maggio:\n"Il virus è clinicamente morto"'
bass = 'Bassetti, 3 maggio:\n"A giugno il virus sarà morto"'
zang2 = 'Zangrillo, 1 giugno:\n"Il coronavirus è sarà meno capace di replicarsi."'
zang3  = 'Zangrillo, 17 giugno:\n"Non voglio minimizzare, il virus esiste,\nma a livello subclinico"'
zang4  = 'Zangrillo, 19 giugno:\n"Tra poco potremo buttar via la mascherina,\nun positivo non è un malato."'
zang5  = 'Zangrillo, 14 luglio:\n"L\'emergenza covid? è finita da due mesi"'
zang6  = 'Zangrillo, 28 settembre:\n"Molto ottimisti, le cose vanno bene"'

# colori matplotlib
c1 = 'silver'
c2 = 'darkorange'
c3 = '#ffff66'
clist = ['#ff3300','#ffff00','#ff9933', '#ffffcc','#ffcc00']
clist2 = clist[:3]
clist3 = clist.copy()
clist3.append('#99ff66')
b1 = '#00004D' # blu scuro
b2 = '#6666FF' # azzurro
plt.rcParams['figure.facecolor'] = b1
plt.rcParams['figure.edgecolor'] = b2
plt.rcParams['axes.facecolor'] = b1
plt.rcParams['axes.edgecolor'] = b2
plt.rcParams['axes.labelcolor'] = b2
plt.rcParams['grid.color'] = b2
plt.rcParams['legend.facecolor'] = b1
plt.rcParams['legend.edgecolor'] = b2
plt.rcParams['text.color'] = b2
plt.rcParams['xtick.color'] = b2
plt.rcParams['ytick.color'] = b2
plt.rcParams['savefig.facecolor'] = b1
plt.rcParams['savefig.edgecolor'] = b2
#plt.rcParams



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# AGGIORNAMENTO DATI NAZIONALI
# scaricamento dati da GitHub
idt = pd.date_range('2020-02-24', data_oggi)
cols_andamento2 = 'data,ricoverati_con_sintomi,terapia_intensiva,totale_ospedalizzati,isolamento_domiciliare,totale_positivi,variazione_totale_positivi,nuovi_positivi,dimessi_guariti,deceduti,casi_da_sospetto_diagnostico,casi_da_screening,totale_casi,tamponi,casi_testati'.split(',')
url_andamento_ITA = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'
df_a = pd.read_csv(url_andamento_ITA, sep=',', usecols=cols_andamento2, encoding='cp1252', error_bad_lines=False)
df_a.set_index(idt, inplace=True)

# dati temporali
df_a['data'] = pd.to_datetime(df_a['data'], format='%Y-%m-%dT%H:%M:%S')
df_a.insert(loc=1, column='anno', value=df_a['data'].map(lambda x: int(x.year)))
df_a.insert(loc=2, column='mese', value=df_a['data'].map(lambda x: int(x.month)))
df_a.insert(loc=3, column='giorno', value=df_a['data'].map(lambda x: int(x.day)))
df_a.insert(loc=4, column='gg_sett', value=df_a['data'].map(lambda x: int(x.dayofweek)))
df_a['gg_sett'] = df_a['gg_sett'].map({0:'lun',1:'mar',2:'mer',3:'gio',4:'ven',5:'sab',6:'dom'})
df_a.insert(loc=5, column='n_sett', value=df_a['data'].map(lambda x: int(x.isocalendar()[1])))
df_a.drop('data', axis=1, inplace=True)

# estrazione dati giornalieri per i campi selezionati
cols = ['ricoverati_con_sintomi', 'terapia_intensiva', 'isolamento_domiciliare', 'nuovi_positivi', 'deceduti', 'tamponi']
for col in cols:
    pos = df_a.columns.get_loc(col) + 1
    nome = col + '_incr'
    df_a.insert(loc=pos, column=nome, value=0)
    c = 0
    for index, row in df_a.iterrows():
        if c == 0:
            old = df_a[col].loc[index]
        else:
            now = df_a[col].loc[index]
            df_a[nome].loc[index] = now - old
            old = df_a[col].loc[index]
        c += 1

# calcolo rapporto percentuali
df_a['pos_su_tamp'] = round((df_a['nuovi_positivi'] / df_a['tamponi_incr'])*100, 2)
df_a['osp_su_pos'] = round((df_a['totale_ospedalizzati'] / df_a['totale_positivi'])*100, 2)

# df degli incrementi giornalieri
df_a_incr = df_a.filter(like='_incr', axis=1)
df_a_incr = df_a_incr.cumsum()
df_a_incr.drop('tamponi_incr', axis=1, inplace=True)
df_a_incr = df_a_incr.join(df_a['gg_sett'])
df_a_incr['gg_sett'].loc[(df_a_incr['gg_sett'] == 'dom') | (df_a_incr['gg_sett'] == 'sab')] = 'we'

# posti letto T.I. per struttura ospedaliera, al 2018
cols = ['anno','cod_reg','nome_reg','cod_azienda','tipo_azienda','cod_struttura','subcodice','nome_struttura','indirizzo','cod_comune','comune','prov','cod_tipo_struttura','descr_tipo_struttura','cod_disciplina','descr_disciplina','tipo_disciplina','n_reparti','p_l_degenza_ord','p_l_degenza_pagamento','p_l_day_hospital','p_l_day_surgery','tot_posti_letto']
df_h = pd.read_csv('/home/daniele/LAVORO/IN_CORSO/blue_GIS/Italia/Covid_19/SSN_posti_letto.csv', sep=';', names=cols, skiprows=[0])
df_h_2018 = df_h.loc[(df_h['anno'] == 2018) & (df_h['descr_disciplina'] == "TERAPIA INTENSIVA".ljust(40))]
df_h_2018 = df_h_2018[['nome_reg','n_reparti', 'p_l_degenza_ord', 'p_l_degenza_pagamento', 'p_l_day_hospital', 'p_l_day_surgery', 'tot_posti_letto']]

# https://tg24.sky.it/cronaca/2020/10/29/covid-terapie-intensive-italia#08
df_h_2018_reg = df_h_2018.groupby('nome_reg').sum()
ti_attuali = {'ABRUZZO':133, 'BASILICATA':73, 'CALABRIA':152, 'CAMPANIA':427, 'EMILIA ROMAGNA':516,
       'FRIULI VENEZIA GIULIA':175, 'LAZIO':747, 'LIGURIA':209, 'LOMBARDIA':983, 'MARCHE':129,
       'MOLISE':34, 'PIEMONTE':367, 'PROV. AUTON. BOLZANO':55, 'PROV. AUTON. TRENTO':51,
       'PUGLIA':366, 'SARDEGNA':175, 'SICILIA':538, 'TOSCANA':415, 'UMBRIA':97, 'VALLE D`AOSTA':20,
       'VENETO':825}
df_h_2020_reg = df_h_2018_reg.join(pd.Series(ti_attuali, name='p_l_2020'))

# codici regionali
cod_regioni = {'ABRUZZO':13, 'BASILICATA':17, 'CALABRIA':18, 'CAMPANIA':15, 'EMILIA ROMAGNA':8,
       'FRIULI VENEZIA GIULIA':6, 'LAZIO':12, 'LIGURIA':7, 'LOMBARDIA':3, 'MARCHE':11,
       'MOLISE':14, 'PIEMONTE':1, 'PROV. AUTON. BOLZANO':21, 'PROV. AUTON. TRENTO':22,
       'PUGLIA':16, 'SARDEGNA':20, 'SICILIA':19, 'TOSCANA':9, 'UMBRIA':10, 'VALLE D`AOSTA':2,
       'VENETO':5}
df_h_2020_reg = df_h_2020_reg.join(pd.Series(cod_regioni, name='cod_reg'))



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GRAFICI DATI NAZIONALI
# grafico degli incrementi giornalieri
plt.figure(figsize=(16, 9))
for e, i in enumerate(df_a_incr.columns.tolist()):
    if i == 'gg_sett' or i == 'isolamento_domiciliare_incr':
        pass
    else:
        plt.plot(df_a_incr.index, df_a_incr[i].values, label=i[:len(i)-5].replace('_', ' '), c=clist[e])
plt.xticks(rotation=90, size=9)
plt.title('Dichiarazioni Zangrillo e incrementi giornalieri')
plt.ylabel('N. casi giornalieri (log)')
#plt.yscale('log')
plt.xlabel('Data')
#plt.grid(alpha=0.5, ls='--')
w=0.05
dt = ['2020-05-31','2020-06-01','2020-06-17','2020-06-19','2020-07-14','2020-07-28']
shift = [8000, 5500, 4500, -1500, 2500, -3000]
for e, (d, z) in enumerate(zip(dt, [zang, zang2, zang3, zang4, zang5, zang6])):
    d_dt = pd.to_datetime(d, format='%Y-%m-%d', errors='ignore')
    plt.scatter(d_dt, df_a_incr.loc[d, 'deceduti_incr'], marker='o', color=clist[4])
    plt.annotate(z[11:], xy=(d_dt, df_a_incr.loc[d, 'deceduti_incr']-shift[e]))
plt.legend(loc=2)
plt.savefig(folder+'plot_zangrillo.jpg')
plotlist.append('plot_zangrillo.jpg')

# grafico degli incrementi giornalieri dal 1o settembre
ti_oggi = df_a.loc[data_oggi,'terapia_intensiva']
np_oggi = df_a_incr.loc[data_oggi,'nuovi_positivi_incr']
plt.figure(figsize=(16, 9))
plt.plot(df_a_incr.index, df_a_incr['nuovi_positivi_incr'].values, label='nuovi positivi ({})'.format(np_oggi), c='#ff6600')
plt.plot(df_a_incr.index, df_a_incr['terapia_intensiva_incr'].values, label='terapia intensiva ({})'.format(ti_oggi), c='#ffff00')
plt.fill_between(df_a_incr.index, df_a_incr['nuovi_positivi_incr'].values, where=df_a_incr['gg_sett']=='we', color='red', alpha=0.2, label='we')
plt.xticks(rotation=90, size=9)
plt.title('Incrementi giornalieri dal 1o settembre')
plt.ylabel('N. casi giornalieri')
plt.xlabel('Data')
plt.xlim(pd.to_datetime('2020-09-01', format='%Y-%m-%d'), pd.to_datetime(data_oggi, format='%Y-%m-%d'))
plt.legend()
plt.grid(alpha=0.5, ls='--')
plt.savefig(folder+'plot_incrementi_2a_ondata.jpg')
plotlist.append('plot_incrementi_2a_ondata.jpg')

# grafico dei positivi sui tamponi
lw = 3
plt.figure(figsize=(16, 9))
plt.plot(df_a.index, df_a['pos_su_tamp'].values, label='% positivi su tamponi', color='w')
plt.fill_between(df_a.index, df_a['pos_su_tamp'].values, where=df_a['tamponi_incr']<25000, facecolor='gray', lw=lw, edgecolor="gray", label='meno di 25.000 tamponi')
plt.fill_between(df_a.index, df_a['pos_su_tamp'].values, where=(df_a['tamponi_incr']>25000) & (df_a['tamponi_incr']<50000), facecolor='yellow', lw=lw, edgecolor="yellow", label='tra 25.000 e 50.000 tamponi')
plt.fill_between(df_a.index, df_a['pos_su_tamp'].values, where=(df_a['tamponi_incr']>50000) & (df_a['tamponi_incr']<100000), facecolor='orange', lw=lw, edgecolor="orange", label='tra 50.000 e 100.000 tamponi')
plt.fill_between(df_a.index, df_a['pos_su_tamp'].values, where=(df_a['tamponi_incr']>100000) & (df_a['tamponi_incr']<200000), facecolor='purple', lw=lw, edgecolor="purple", label='tra 100.000 e 200.000 tamponi')
plt.fill_between(df_a.index, df_a['pos_su_tamp'].values, where=df_a['tamponi_incr']>200000, facecolor='red', lw=lw, edgecolor="red", label='più di 200.000 tamponi')
plt.vlines(pd.to_datetime('2020-03-08', format='%Y-%m-%d'), 0, 50, color=c1, ls='--')
plt.annotate("lock down", xy=(pd.to_datetime('2020-02-28', format='%Y-%m-%d'), 50.5), color=c1, size=12)
plt.vlines(pd.to_datetime('2020-05-04', format='%Y-%m-%d'), 0, 20, color=c1, ls='--')
plt.annotate("fine lock down", xy=(pd.to_datetime('2020-04-24', format='%Y-%m-%d'), 20.5), color=c1, size=12)
plt.vlines(pd.to_datetime('2020-09-07', format='%Y-%m-%d'), 0, 10, color=c1, ls='--')
plt.vlines(pd.to_datetime('2020-09-24', format='%Y-%m-%d'), 0, 10, color=c1, ls='--')
plt.annotate("aperture scuole", xy=(pd.to_datetime('2020-09-02', format='%Y-%m-%d'), 10.5), color=c1, size=12)
plt.annotate(str(df_a.loc[data_oggi,'pos_su_tamp']), xy=(pd.to_datetime(data_oggi, format='%Y-%m-%d'), int(df_a.loc[data_oggi,'pos_su_tamp'])+2), color=c1, size=12)
plt.xticks(rotation=90, size=9)
plt.title('Positivi su tamponi effettuati')
plt.ylabel('% positivi')
plt.xlabel('Data')
plt.legend(loc=1)
plt.grid(alpha=0.5, ls='--')
plt.savefig(folder+'plot_incrementi_pct.jpg')
plotlist.append('plot_incrementi_pct.jpg')

# grafico degli ospedalizzati sui positivi
lw = 3
fig, ax1 = plt.subplots(figsize=(12, 8))
ax1.plot(df_a.index, df_a['osp_su_pos'].values, label='% ospedalizzati sui positivi', color=c3)
ax1.set_ylim(0, 80)
ax1.vlines(pd.to_datetime('2020-03-08', format='%Y-%m-%d'), 0, 75, color=c1, ls='--')
ax1.annotate("lock down", xy=(pd.to_datetime('2020-02-28', format='%Y-%m-%d'), 76), color=c1, size=12)
ax1.vlines(pd.to_datetime('2020-05-04', format='%Y-%m-%d'), 0, 50, color=c1, ls='--')
ax1.annotate("fine lock down", xy=(pd.to_datetime('2020-04-24', format='%Y-%m-%d'), 51), color=c1, size=12)
ax1.vlines(pd.to_datetime('2020-09-07', format='%Y-%m-%d'), 0, 10, color=c2, ls='--')
ax1.vlines(pd.to_datetime('2020-09-24', format='%Y-%m-%d'), 0, 10, color=c2, ls='--')
ax1.annotate("aperture scuole", xy=(pd.to_datetime('2020-09-01', format='%Y-%m-%d'), 10.5), color=c2, size=12)
ax1.set(title='Ospedalizzati sui positivi', ylabel='% ospedalizzati', xlabel='Data')
ax1.legend(loc=3)
ax2 = ax1.twinx()  # secondo axes che condivide lo stesso asse
ax2.bar(df_a.index, df_a['terapia_intensiva'].values, label='terapia intensiva', color=c2, alpha=0.25)
ax2.set_ylim(0, 5000)
ax2.legend(loc=4)
# Adjust the plotting range of two y axes
org1 = 0.0  # Origin of first axis
org2 = 0.0  # Origin of second axis
pos = 0.1  # Position the two origins are aligned
align.yaxes(ax1, org1, ax2, org2, pos)
plt.savefig(folder+'plot_incrementi_pct2.jpg')
plotlist.append('plot_incrementi_pct2.jpg')

# grafico crescita parametri
x = df_a.index.values
y1 = df_a[['ricoverati_con_sintomi','terapia_intensiva', 'isolamento_domiciliare']].values
y2 = df_a[['ricoverati_con_sintomi_incr','terapia_intensiva_incr', 'isolamento_domiciliare_incr']].values
lab = ['ricoverati con sintomi','terapia intensiva', 'isolamento domiciliare']
fig, ax = plt.subplots(2, 1, figsize=(14, 12))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
plt.suptitle('Andamento ricoveri', fontsize=16)
# grafico in alto
ax1.stackplot(x, y1.T, labels=lab, colors=clist2) #baseline='wiggle')
ax1.vlines(pd.to_datetime('2020-03-08', format='%Y-%m-%d'), 0, 100000, color=c1, ls='--')
ax1.annotate("lock down", xy=(pd.to_datetime('2020-02-28', format='%Y-%m-%d'), 102000), color=c1, size=12)
ax1.vlines(pd.to_datetime('2020-05-04', format='%Y-%m-%d'), 0, 145000, color=c1, ls='--')
ax1.annotate("fine lock down", xy=(pd.to_datetime('2020-04-24', format='%Y-%m-%d'), 150000), color=c1, size=12)
ax1.vlines(pd.to_datetime('2020-09-07', format='%Y-%m-%d'), 0, 80000, color=c1, ls='--')
ax1.vlines(pd.to_datetime('2020-09-24', format='%Y-%m-%d'), 0, 80000, color=c1, ls='--')
ax1.annotate("aperture scuole", xy=(pd.to_datetime('2020-09-01', format='%Y-%m-%d'), 90000), color=c1, size=12)
ax1.set(title='Incrementi cumulati', ylabel='N. ricoverati', xlabel='Data')
ax1.legend(loc='upper center')
# grafico in basso
ax2.stackplot(x, y2.T, labels=lab, colors=clist2) #baseline='wiggle')
ax2.vlines(pd.to_datetime('2020-03-08', format='%Y-%m-%d'), -6000, 6000, color=c1, ls='--')
ax2.annotate("lock down", xy=(pd.to_datetime('2020-02-28', format='%Y-%m-%d'), 6200), color=c1, size=12)
ax2.vlines(pd.to_datetime('2020-05-04', format='%Y-%m-%d'), -6000, 6000, color=c1, ls='--')
ax2.annotate("fine lock down", xy=(pd.to_datetime('2020-04-24', format='%Y-%m-%d'), 6200), color=c1, size=12)
ax2.vlines(pd.to_datetime('2020-09-07', format='%Y-%m-%d'), 0, 5000, color=c1, ls='--')
ax2.vlines(pd.to_datetime('2020-09-24', format='%Y-%m-%d'), 0, 5000, color=c1, ls='--')
ax2.annotate("aperture scuole", xy=(pd.to_datetime('2020-09-01', format='%Y-%m-%d'), 5500), color=c1, size=12)
ax2.set(title='Incrementi giornalieri', ylabel='N. ricoverati', xlabel='Data')
ax2.legend(loc='lower center')
plt.tight_layout(pad=6, h_pad=2)
# save
plt.savefig(folder+'plot_incrementi_ricoveri.jpg')
plotlist.append('plot_incrementi_ricoveri.jpg')



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# AGGIORNAMENTO DATI REGIONALI
# scaricamento dati da GitHub
cols_andamento3 = 'data,codice_regione,denominazione_regione,lat,long,ricoverati_con_sintomi,terapia_intensiva,totale_ospedalizzati,isolamento_domiciliare,totale_positivi,variazione_totale_positivi,nuovi_positivi,dimessi_guariti,deceduti,casi_da_sospetto_diagnostico,casi_da_screening,totale_casi,tamponi,casi_testati'.split(',')
url_andamento_reg = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv'
df_r = pd.read_csv(url_andamento_reg, sep=',', usecols=cols_andamento3, encoding='cp1252', error_bad_lines=False)
identificativi_reg = df_r[['codice_regione','denominazione_regione']].drop_duplicates()

# dati temporali
df_r.insert(loc=1, column='aaaammgg', value=df_r['data'].map(lambda x: x[:10]))
df_r['data'] = pd.to_datetime(df_r['data'], format='%Y-%m-%dT%H:%M:%S')
df_r.insert(loc=2, column='anno', value=df_r['data'].map(lambda x: int(x.year)))
df_r.insert(loc=3, column='mese', value=df_r['data'].map(lambda x: int(x.month)))
df_r.insert(loc=4, column='giorno', value=df_r['data'].map(lambda x: int(x.day)))
df_r.insert(loc=5, column='gg_sett', value=df_r['data'].map(lambda x: int(x.dayofweek)))
df_r['gg_sett'] = df_r['gg_sett'].map({0:'lun',1:'mar',2:'mer',3:'gio',4:'ven',5:'sab',6:'dom'})
df_r.insert(loc=6, column='n_sett', value=df_r['data'].map(lambda x: int(x.isocalendar()[1])))
df_r.drop('data', axis=1, inplace=True)
df_r.rename(columns={"aaaammgg": "data"}, inplace=True)

# pct positivi su tampone
df_r_incr = df_r[['data','denominazione_regione','totale_positivi','tamponi']].copy().set_index('data')
df_r_incr = df_r_incr.loc[data_oggi,:].set_index('denominazione_regione')
df_r_incr['pos_su_tamp'] = round((df_r_incr['totale_positivi'] / df_r_incr['tamponi'])*100, 2)
# km2 per regione
df_r_tot_casi = df_r[['data', 'denominazione_regione', 'totale_casi']].groupby(['data', 'denominazione_regione']).sum().unstack(0)
sup_regioni = {'Abruzzo':10800, 'Basilicata':10000, 'Calabria':15000, 'Campania':13600, 'Emilia-Romagna':22500,
       'Friuli Venezia Giulia':7900, 'Lazio':17200, 'Liguria':5400, 'Lombardia':23900, 'Marche':9400,
       'Molise':4400, 'Piemonte':25400, 'P.A. Bolzano':7400, 'P.A. Trento':6200, 'Puglia':19300, 'Sardegna':24100,
       'Sicilia':25700, 'Toscana':23000, 'Umbria':8500, "Valle d'Aosta":3200, 'Veneto':18400}
df_r_tot_casi.columns = df_r_tot_casi.columns.droplevel()
df_r_tot_casi = df_r_tot_casi.join(pd.Series(sup_regioni, name='km2'))
df_r_tot_casi = df_r_tot_casi.loc[:, [data_oggi, 'km2']]
df_r_tot_casi['densita_casi'] = df_r_tot_casi[data_oggi] / df_r_tot_casi['km2']

# confronti con la 1a ondata
df_r_dataind = df_r.set_index('data')
dati_diz = {}
for r in df_r_dataind.denominazione_regione.unique().tolist():
    df_r_dataind_sub = df_r_dataind.loc[df_r_dataind['denominazione_regione'] == r]
    dati_diz[r] = {}
    for c in ['ricoverati_con_sintomi','terapia_intensiva','totale_ospedalizzati','isolamento_domiciliare','totale_positivi','deceduti','totale_casi', 'tamponi']:
        max1 = df_r_dataind_sub.loc['2020-02-24':'2020-08-01', c].nlargest(1).values[0]
        max2 = df_r_dataind_sub.loc[data_oggi, c]
        dati_diz[r][c] = {'prima_ondata':max1, 'oggi':max2}
        #print('{}, {}, {} --> {}'.format(r, c, max1, max2))

df_ondate = pd.DataFrame.from_dict({(i,j): dati_diz[i][j] for i in dati_diz.keys() for j in dati_diz[i].keys()}, orient='index')
df_ondate['var_pct'] = round(((df_ondate['oggi']-df_ondate['prima_ondata'])/df_ondate['prima_ondata'])*100, 2)
# preparazione df per l'export
# eliminazione multiindex delle colonne
df_ondate_csv = df_ondate.unstack().droplevel(0, axis=1)
ond_cols = ['deceduti1','is_domic1','ric_sint1','tamponi1','ter_int1','tot_casi1','tot_osp1','tot_pos1','deceduti2','is_domic2','ric_sint2','tamponi2','ter_int2','tot_casi2','tot_osp2','tot_pos2','decedutiPCT','is_domicPCT','ric_sintPCT','tamponiPCT','ter_intPCT','tot_casiPCT','tot_ospPCT','tot_posPCT']
df_ondate_csv.columns = ond_cols
df_ondate_csv = df_ondate_csv.join(identificativi_reg.set_index('denominazione_regione'))
df_ondate_csv.set_index('codice_regione', inplace=True)
df_ondate_csv.to_csv(folder + '/confronto_ondate.csv', sep=';')

# terapie intensive regionali
df_reg_ti = df_r[['data', 'codice_regione', 'denominazione_regione', 'ricoverati_con_sintomi','terapia_intensiva', 'totale_ospedalizzati']]
df_reg_ti = df_reg_ti.merge(df_h_2020_reg[['p_l_2020', 'cod_reg']], left_on='codice_regione', right_on='cod_reg')
df_reg_ti.drop(['cod_reg'], axis=1, inplace=True)
df_reg_ti['ti_occupate_pct'] = round(df_reg_ti['terapia_intensiva'] / df_reg_ti['p_l_2020'] *100, 2)

df_reg_ti_oggi = df_r[['data', 'codice_regione', 'denominazione_regione', 'ricoverati_con_sintomi','terapia_intensiva', 'totale_ospedalizzati']]
df_reg_ti_oggi = df_reg_ti_oggi.loc[df_reg_ti_oggi['data'] == data_oggi]
df_reg_ti_oggi = df_reg_ti_oggi.merge(df_h_2020_reg[['p_l_2020', 'cod_reg']], left_on='codice_regione', right_on='cod_reg')
df_reg_ti_oggi.drop(['data', 'cod_reg'], axis=1, inplace=True)
df_reg_ti_oggi['ti_occupate_pct'] = round(df_reg_ti_oggi['terapia_intensiva'] / df_reg_ti_oggi['p_l_2020'] *100, 2)

# casi per data e regione
reg_time_group = df_r.groupby(['data', 'denominazione_regione']).sum()
reg_time_group.reset_index(inplace=True)
reg_time_group.drop(['anno', 'mese', 'giorno', 'n_sett', 'codice_regione', 'lat', 'long'], axis=1, inplace=True)
reg_time_group.groupby(['denominazione_regione'])['totale_casi'].max()
reg_time_group_small = reg_time_group.drop(['ricoverati_con_sintomi','totale_ospedalizzati', 'variazione_totale_positivi',
                'nuovi_positivi', 'casi_da_sospetto_diagnostico', 'casi_da_screening', 'casi_testati'], axis=1)

# top five in base ai casi di ieri
top5 = reg_time_group_small.loc[reg_time_group_small.data == data_oggi].nlargest(5,['totale_casi'])
top5_list = top5.denominazione_regione.unique().tolist()
reg_time_group_top5 = reg_time_group_small.copy()
reg_time_group_top5.loc[~reg_time_group_small['denominazione_regione'].isin(top5_list), 'denominazione_regione'] = 'altre regioni'
reg_time_group_top5 = reg_time_group_top5.groupby(['data', 'denominazione_regione']).sum()
reg_time_group_top5.reset_index(level=1, inplace=True)



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GRAFICI DATI REGIONALI
# grafico tutte le regioni VS top5
r_list = reg_time_group_top5.denominazione_regione.unique().tolist()
sty = '-'
plt.figure(figsize=(16, 9))
for r in r_list:
    dfi = reg_time_group_top5.loc[reg_time_group_top5['denominazione_regione'] == r]
    if r == 'altre regioni':
        sty = '--'
    plt.plot(dfi.index, dfi.totale_casi.values, color=clist3[r_list.index(r)], linestyle=sty, label=r)
plt.axvspan('2020-09-07', '2020-09-24', color='red', alpha=0.25)
plt.annotate("apertura\nscuole", xy=('2020-09-08', 120000), color='r', size=12)
plt.axvspan('2020-03-08', '2020-05-04', color=c1, alpha=0.25)
plt.annotate("lock down", xy=('2020-03-28', 120000), color='k', size=12)
plt.title('Casi totali: top 5 Regioni')
plt.ylabel('N. casi totali')
plt.xlabel('Data')
plt.xticks(rotation=90, size=5)
plt.legend()
plt.savefig(folder+'plot_reg_top5.jpg')
plotlist.append('plot_reg_top5.jpg')

# occupazione % t.i.
df_reg_ti_oggi.sort_values(by='ti_occupate_pct', ascending=False, inplace=True)
fig, ax = plt.subplots(figsize=(16, 9))
y_pos = np.arange(df_reg_ti_oggi.index.shape[0])
vals = df_reg_ti_oggi['ti_occupate_pct'].values
nomi = df_reg_ti_oggi['denominazione_regione'].values.tolist()
clist = ['colors']*len(vals)
for i, v in enumerate(vals):
    if v > 50:
        clist[i] = '#cc00cc'
    elif v > 25 and v <= 50:
        clist[i] = '#9900ff'
    elif v > 10 and v <= 25:
        clist[i] = '#6600ff'
    else:
        clist[i] = '#9933ff'
    ax.text(v-2.5, i+0.25, str(v), color=c1, fontweight='bold', va='bottom')
ax.barh(y_pos, vals, color=clist, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(nomi)
ax.invert_yaxis()  # labels read top-to-bottom
ax.grid(axis='x', alpha=0.5, ls='--', c='gray')
ax.set_xlabel('% occupazione')
ax.set_title('Occupazione terapie intensive regionali')
plt.savefig(folder+'plot_reg_TI.jpg')
plotlist.append('plot_reg_TI.jpg')

# variazione dei positivi
df_r_var_ti = df_r[['data', 'terapia_intensiva']].groupby(['data']).sum()
M1ti = df_r_var_ti.loc['2020-02-28':'2020-07-01', 'terapia_intensiva'].nlargest(1).values[0]
M2ti = df_r_var_ti.loc['2020-07-01':data_oggi, 'terapia_intensiva'].nlargest(1).values[0]
df_r_var = df_r[['data', 'variazione_totale_positivi']].groupby(['data']).sum()
M1 = df_r_var.loc['2020-02-28':'2020-07-01', 'variazione_totale_positivi'].nlargest(1).values[0]
M2 = round(df_r_var['variazione_totale_positivi'].rolling(7).mean().nlargest(1).values[0], 0)
su_M1 = round((((M2-M1)/M1)*100),2)
plt.figure(figsize=(16, 9))
plt.plot(df_r_var.index, df_r_var['variazione_totale_positivi'], color=c1, lw=0.75, label='Nuovi positivi (var. giornaliera)')
plt.plot(df_r_var.index, df_r_var['variazione_totale_positivi'].rolling(7).mean(), '#66ccff', label='Nuovi positivi (media mobile a 7 giorni)')
plt.plot(df_r_var_ti.index, df_r_var_ti['terapia_intensiva'], color='#ffff66', lw=1, label='Terapia intensiva (val. assoluto)')
plt.hlines(M1, '2020-02-28', data_oggi, color='darkorange', ls='--', lw=1)
plt.annotate('massimo prima ondata: {}'.format(M1), xy=('2020-03-10', M1+500), c='darkorange')
plt.annotate('rispetto al massimo\nprima ondata:\n+{}%'.format(su_M1), xy=(data_oggi, M2+200), c='#66ccff')
plt.annotate('{}'.format(M1ti), xy=('2020-04-09', M1ti-200), c='#ffff66')
plt.annotate('{}'.format(M2ti), xy=(data_oggi, M2ti+200), c='#ffff66')
plt.title('Variazione giornaliera positivi')
plt.ylabel('variazione nuovi positivi\nposti letto t.i. occupati')
plt.xlabel('Data')
plt.xticks(rotation=90, size=5)
plt.legend(loc='upper left')
plt.savefig(folder+'plot_reg_var_pos_TI.jpg')
plotlist.append('plot_reg_var_pos_TI.jpg')

# pct su densità
y = df_r_incr.pos_su_tamp.values
x = df_r_tot_casi.densita_casi.values
plt.scatter(x, y)



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# AGGIORNAMENTO DATI PROVINCIALI
# scaricamento dati da GitHub
cols_andamento4 = 'data,codice_regione,denominazione_regione,codice_provincia,denominazione_provincia,sigla_provincia,lat,long,totale_casi'.split(',')
url_andamento_pro = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv'
df_p = pd.read_csv(url_andamento_pro, sep=',', usecols=cols_andamento4, encoding='cp1252', error_bad_lines=False)
df_p = df_p.drop(df_p[(df_p.denominazione_provincia == 'In fase di definizione/aggiornamento') | (df_p.denominazione_provincia == 'Fuori Regione / Provincia Autonoma')].index)
identificativi_prov = df_p[['codice_provincia','denominazione_provincia']].drop_duplicates()

# dati temporali
df_p.insert(loc=1, column='aaaammgg', value=df_p['data'].map(lambda x: x[:10]))
df_p['data'] = pd.to_datetime(df_p['data'], format='%Y-%m-%dT%H:%M:%S')
df_p.insert(loc=2, column='anno', value=df_p['data'].map(lambda x: int(x.year)))
df_p.insert(loc=3, column='mese', value=df_p['data'].map(lambda x: int(x.month)))
df_p.insert(loc=4, column='giorno', value=df_p['data'].map(lambda x: int(x.day)))
df_p.insert(loc=5, column='gg_sett', value=df_p['data'].map(lambda x: int(x.dayofweek)))
df_p['gg_sett'] = df_p['gg_sett'].map({0:'lun',1:'mar',2:'mer',3:'gio',4:'ven',5:'sab',6:'dom'})
df_p.insert(loc=6, column='n_sett', value=df_p['data'].map(lambda x: int(x.isocalendar()[1])))
df_p.drop('data', axis=1, inplace=True)
df_p.rename(columns={"aaaammgg": "data"}, inplace=True)

# confronti con la 1a ondata
df_p_dataind = df_p.set_index('data')
dati_diz2 = {}
for p in df_p_dataind.denominazione_provincia.unique().tolist():
    df_p_dataind_sub = df_p_dataind.loc[df_p_dataind['denominazione_provincia'] == p]
    dati_diz2[p] = {}
    max1 = df_p_dataind_sub.loc['2020-02-24':'2020-08-01', 'totale_casi'].nlargest(1).values[0]
    max2 = df_p_dataind_sub.loc[data_oggi, 'totale_casi']
    dati_diz2[p]['totale_casi'] = {'prima_ondata':max1, 'oggi':max2}

df2_ondate = pd.DataFrame.from_dict({(i): dati_diz2[i]['totale_casi'] for i in dati_diz2.keys()}, orient='index')
df2_ondate['var_pct'] = round(((df2_ondate['oggi']-df2_ondate['prima_ondata'])/df2_ondate['prima_ondata'])*100, 2)
# preparazione df per l'export
ond2_cols = ['tot_casi1','tot_casi2','tot_casiPCT']
df2_ondate.columns = ond2_cols
df2_ondate_csv = df2_ondate.join(identificativi_prov.set_index('denominazione_provincia'))
df2_ondate_csv.set_index('codice_provincia', inplace=True)
df2_ondate_csv.to_csv(folder + '/confronto_ondate_prov.csv', sep=';')

# casi per data e provincia
prov_time_group = df_p.groupby(['data', 'denominazione_provincia']).sum()
prov_time_group.reset_index(inplace=True)
prov_time_group.drop(['anno', 'mese', 'giorno', 'n_sett', 'codice_regione', 'codice_provincia', 'lat', 'long'], axis=1, inplace=True)
prov_time_group.groupby(['denominazione_provincia'])['totale_casi'].max()

# top five in base ai casi di ieri
top5p = prov_time_group.loc[prov_time_group.data == data_oggi].nlargest(5,['totale_casi'])
top5p_list = top5p.denominazione_provincia.unique().tolist()
prov_time_group_top5 = prov_time_group.copy()
prov_time_group_top5.loc[~prov_time_group['denominazione_provincia'].isin(top5p_list), 'denominazione_provincia'] = 'altre province'
prov_time_group_top5 = prov_time_group_top5.groupby(['data', 'denominazione_provincia']).sum()
prov_time_group_top5.reset_index(level=1, inplace=True)



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GRAFICI DATI PROVINCIALI
# grafico tutte le province VS top5
fig, ax = plt.subplots(2, 1, figsize=(16, 19))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
plt.suptitle('Andamento casi totali per provincia', fontsize=16)
# grafico in alto
n = len(df_p.denominazione_regione.unique())
color=cm.rainbow(np.linspace(0,1,n))
for i, c in zip(range(n),color):
    ax1.plot(df_p.data.values, df_p.totale_casi.values, color=c)
plt.xticks(rotation=90, size=5)
# grafico in basso
p_list = prov_time_group_top5.denominazione_provincia.unique().tolist()
sty = '-'
for p in p_list:
    dfi = prov_time_group_top5.loc[prov_time_group_top5['denominazione_provincia'] == p]
    if p == 'altre province':
        sty = '--'
    ax2.plot(dfi.index, dfi.totale_casi.values, color=clist3[p_list.index(p)], linestyle=sty, label=p)
ax2.axvspan('2020-09-07', '2020-09-24', color='red', alpha=0.25)
ax2.annotate("apertura\nscuole", xy=('2020-09-08', 120000), color='r', size=12)
ax2.axvspan('2020-03-08', '2020-05-04', color=c1, alpha=0.25)
ax2.annotate("lock down", xy=('2020-03-28', 180000), color='k', size=12)
ax2.set(title='Casi totali: top 5 Province  -  {}'.format(data_oggi), ylabel='N. casi totali', xlabel='Data')
plt.xticks(rotation=90, size=5)
ax2.legend(loc='upper center')
plt.savefig(folder+'plot_prov.jpg')



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GEOPANDAS
# shape regioni
italy_r = gpd.read_file('/home/daniele/LAVORO/IN_CORSO/blue_GIS/Italia/Covid_19/limiti_amministrativi/regioni_ita.shp')
df_reg_dati = df_reg_ti_oggi.merge(df_ondate.loc[pd.IndexSlice[:, 'terapia_intensiva'], :].reset_index(level=1, drop=True), left_on='denominazione_regione', right_index=True)
df_reg_dati = df_reg_dati.merge(df_r_incr, on='denominazione_regione')
ita_reg_dati = italy_r.merge(df_reg_dati, left_on='COD_REG', right_on='codice_regione')
ita_reg_dati.drop(['SHAPE_Leng','SHAPE_Area','str_osped','TI_ogni'], axis=1, inplace=True)
ita_reg_dati.head()

# shape province
italy_p = gpd.read_file('/home/daniele/LAVORO/IN_CORSO/blue_GIS/Italia/Covid_19/limiti_amministrativi/province_ita.shp')
df_p_s = df_p.loc[df_p['data'] == data_oggi]
df_p_s = df_p_s[['codice_provincia', 'totale_casi']]
ita_prov_dati = italy_p.merge(df_p_s, left_on='COD_PRO', right_on='codice_provincia', how='left')

# mappe 1
miss = {'color':'lightgrey', 'hatch':'////', 'label':'nd'}
fig, (axs, axd) = plt.subplots(1, 2, figsize=(24, 12))
#plt.suptitle('Dati totali', fontsize=16)
# mappa sinistra
vard = 'totale_positivi'
vmin, vmax = 120, 220
ita_reg_dati.plot(column=vard, legend=True, scheme='natural_breaks', cmap='YlOrRd', lw=0.8, ax=axs)
axs.set(title=vard.upper().replace('_', ' '))
# mappa destra
vars = 'totale_casi'
vmin, vmax = 120, 220
ita_prov_dati.plot(column=vars, legend=True, scheme='natural_breaks', cmap='YlOrRd', missing_kwds=miss, lw=0.8, ax=axd)
axd.set(title=vars.upper().replace('_', ' '))
plt.savefig(folder+'plot_map1.jpg')
plotlist.append('plot_map1.jpg')

# mappe 2
fig, (axs, axd) = plt.subplots(1, 2, figsize=(24, 12))
#plt.suptitle('Dati totali', fontsize=16)
# mappa sinistra
vard = 'ti_occupate_pct'
vmin, vmax = 120, 220
ita_reg_dati.plot(column=vard, legend=True, scheme='natural_breaks', cmap='YlOrRd', lw=0.8, ax=axs)
axs.set(title='Occupazione t.i. (%)')
# mappa destra
vars = 'pos_su_tamp'
vmin, vmax = 120, 220
ita_reg_dati.plot(column=vars, legend=True, scheme='natural_breaks', cmap='YlOrRd', lw=0.8, ax=axd)
axd.set(title='Positivi su tamponi (%)')
plt.savefig(folder+'plot_map2.jpg')
plotlist.append('plot_map2.jpg')



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TRASFERIMENTO GRAFICI SUL SERVER
import paramiko

hostname, username, password = '80.211.85.175', 'root', 'p455w0rd_53Rv3R'
srv_folder = '/var/www/html/covid/'
# connessione al server
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname=hostname,username=username,password=password)
# upload file
ftp_client = ssh_client.open_sftp()
for item in plotlist:
    ftp_client.put(os.path.join(folder, item), os.path.join(srv_folder, item))

ftp_client.close()
ssh_client.close()


#
