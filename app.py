import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
from pandas.core.series import Series

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse de Tendance Exponentielle et Volatilité",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constantes globales pour les calculs ---
WINDOW_MA_SHORT = 50
WINDOW_MA_LONG = 200
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
WEEKS_PER_MONTH = 4.33
WEEKS_PER_YEAR = 52

# --- NOUVELLES FONCTIONS D'AIDE LTM / T0-T4 ---

def get_ltm_sum(q_data: pd.DataFrame, key: str) -> float:
    """Somme les 4 derniers trimestres (LTM) pour un élément de flux (Income/Cash Flow)."""
    if key not in q_data.index:
        # Gère 'Total Revenue' vs 'Total Revenues'
        if key == 'Total Revenue' and 'Total Revenues' in q_data.index:
            key = 'Total Revenues'
        else:
            # Pour les éléments optionnels ou manquants, renvoie 0.0 si moins de 4 trimestres ou clé absente.
            if key in ['Income Tax Expense', 'Pretax Income', 'EBIT', 'Net Income']:
                return 0.0 
            raise KeyError(f"Donnée financière LTM manquante: {key}")
            
    if len(q_data.columns) < 4:
        raise ValueError("Données trimestrielles insuffisantes (moins de 4) pour le calcul LTM.")

    # Somme des 4 dernières colonnes (iloc[0] à iloc[3])
    return q_data.loc[key].iloc[0:4].sum()

def get_balance_value(q_balance: pd.DataFrame, key: str, index: int = 0, default_val: float = 0.0) -> float:
    """Récupère une valeur de bilan spécifique à un index (0=T0, 4=T-4)."""
    if key not in q_balance.index:
        # Pour les intérêts minoritaires optionnels
        if key in ['Minority Interest', 'Non Controlling Interest']:
            return default_val
        raise KeyError(f"Donnée de bilan manquante: {key}")
        
    if len(q_balance.columns) <= index:
        raise ValueError(f"Données de bilan insuffisantes (moins de {index+1} trimestres) pour obtenir l'index {index}.")
    
    return q_balance.loc[key].iloc[index]


# --- FONCTION DE CALCUL POUR L'ANALYSE DUPONT (ROE LTM) ---
def calculate_dupont(q_financials: pd.DataFrame, q_balance: pd.DataFrame) -> dict | None:
    """
    Calcule l'analyse DuPont (ROE) en utilisant les données LTM.
    Les moyennes du bilan sont basées sur le dernier trimestre (T0) et T-4 (il y a 12 mois).
    """
    # Nécessite 4 Q de financials (LTM) et 5 Q de balance (T0 et T-4)
    if q_financials.empty or q_balance.empty or len(q_financials.columns) < 4 or len(q_balance.columns) < 5:
        return None

    try:
        # 1. Extraction des Données LTM (Flux)
        ltm_revenu = get_ltm_sum(q_financials, 'Total Revenue')
        ltm_resultat_net = get_ltm_sum(q_financials, 'Net Income')
        
        # 2. Extraction des Bilans T0 et T-4 (Stock)
        actifs_t0 = get_balance_value(q_balance, 'Total Assets', index=0)
        actifs_t4 = get_balance_value(q_balance, 'Total Assets', index=4)
        capitaux_propres_t0 = get_balance_value(q_balance, 'Common Stock Equity', index=0)
        capitaux_propres_t4 = get_balance_value(q_balance, 'Common Stock Equity', index=4)

    except (KeyError, ValueError) as ke:
        return {'error': f"Donnée financière/bilan manquante ou insuffisante (LTM / T0-T-4) : {ke}"}
    except Exception as e:
        return {'error': f"Erreur inattendue dans l'analyse DuPont (LTM) : {e}"}

    # --- Calcul des Moyennes (utilisant T0 et T-4) ---
    actifs_moyens = (actifs_t0 + actifs_t4) / 2
    capitaux_propres_moyens = (capitaux_propres_t0 + capitaux_propres_t4) / 2

    # Prévention de la division par zéro
    if ltm_revenu <= 0 or actifs_moyens <= 0 or capitaux_propres_moyens <= 0:
        return {
            'Marge_Nette': 0.0, 'Rotation_Actif': 0.0,
            'Multiplicateur_CE': 0.0, 'ROE': 0.0,
            'Dates_LTM': q_financials.columns[0:4].strftime('%Y-%m-%d').tolist(),
            'Dates_Bilan': [q_balance.columns[0].strftime('%Y-%m-%d'), q_balance.columns[4].strftime('%Y-%m-%d')]
        }

    # --- Composantes DuPont (LTM) ---
    marge_nette = ltm_resultat_net / ltm_revenu
    rotation_actif = ltm_revenu / actifs_moyens
    multiplicateur_ce = actifs_moyens / capitaux_propres_moyens
    roe = marge_nette * rotation_actif * multiplicateur_ce

    return {
        'Marge_Nette': marge_nette,
        'Rotation_Actif': rotation_actif,
        'Multiplicateur_CE': multiplicateur_ce,
        'ROE': roe,
        'Dates_LTM': q_financials.columns[0:4].strftime('%Y-%m-%d').tolist(),
        'Dates_Bilan': [q_balance.columns[0].strftime('%Y-%m-%d'), q_balance.columns[4].strftime('%Y-%m-%d')]
    }


# --- FONCTION DE CALCUL POUR L'ANALYSE ROIC (LTM) ---
def calculate_roic(q_financials: pd.DataFrame, q_balance: pd.DataFrame) -> dict | None:
    """
    Calcule la décomposition du ROIC (Return on Invested Capital) en utilisant les données LTM.
    Le capital investi moyen est basé sur T0 et T-4.
    """
    # Nécessite 4 Q de financials (LTM) et 5 Q de balance (T0 et T-4)
    if q_financials.empty or q_balance.empty or len(q_financials.columns) < 4 or len(q_balance.columns) < 5:
        return None

    try:
        # 1. Extraction des Données LTM (Flux)
        ltm_revenu = get_ltm_sum(q_financials, 'Total Revenue')
        ltm_ebit = get_ltm_sum(q_financials, 'EBIT')
        ltm_impots = get_ltm_sum(q_financials, 'Income Tax Expense')
        ltm_pretax_income = get_ltm_sum(q_financials, 'Pretax Income')

        # Calcul du NOPAT (LTM)
        if ltm_pretax_income > 0 and ltm_ebit > 0:
            taux_impot_ltm = ltm_impots / ltm_pretax_income
        else:
            taux_impot_ltm = 0.25 # Taux par défaut si le calcul est impossible.
        
        ltm_nopat = ltm_ebit * (1 - taux_impot_ltm)
        
        # 2. Extraction des Bilans T0 et T-4 pour Capital Investi (Stock)
        # CI = Total Dette + Capitaux Propres + Intérêts Minoritaires
        
        # T0
        total_dette_t0 = get_balance_value(q_balance, 'Total Debt', index=0)
        capitaux_propres_t0 = get_balance_value(q_balance, 'Common Stock Equity', index=0)
        interet_minoritaire_t0 = get_balance_value(q_balance, 'Minority Interest', index=0, default_val=0) 
        
        # T-4
        total_dette_t4 = get_balance_value(q_balance, 'Total Debt', index=4)
        capitaux_propres_t4 = get_balance_value(q_balance, 'Common Stock Equity', index=4)
        interet_minoritaire_t4 = get_balance_value(q_balance, 'Minority Interest', index=4, default_val=0)
        
    except (KeyError, ValueError) as ke:
        return {'error': f"Donnée financière/bilan manquante ou insuffisante (LTM / T0-T-4) : {ke}"}
    except Exception as e:
        return {'error': f"Erreur inattendue dans l'analyse ROIC (LTM) : {e}"}

    # --- Calcul du Capital Investi Moyen (utilisant T0 et T-4) ---
    capital_investi_t0 = total_dette_t0 + capitaux_propres_t0 + interet_minoritaire_t0
    capital_investi_t4 = total_dette_t4 + capitaux_propres_t4 + interet_minoritaire_t4

    capital_investi_moyen = (capital_investi_t0 + capital_investi_t4) / 2

    # Prévention de la division par zéro
    if ltm_revenu <= 0 or capital_investi_moyen <= 0:
        return {
            'Marge_NOPAT': 0.0, 'Rotation_CI': 0.0,
            'ROIC': 0.0,
            'Dates_LTM': q_financials.columns[0:4].strftime('%Y-%m-%d').tolist(),
            'Dates_Bilan': [q_balance.columns[0].strftime('%Y-%m-%d'), q_balance.columns[4].strftime('%Y-%m-%d')]
        }

    # --- Composantes ROIC (LTM) ---
    marge_nopat = ltm_nopat / ltm_revenu
    rotation_ci = ltm_revenu / capital_investi_moyen
    roic = marge_nopat * rotation_ci

    return {
        'Marge_NOPAT': marge_nopat,
        'Rotation_CI': rotation_ci,
        'ROIC': roic,
        'Dates_LTM': q_financials.columns[0:4].strftime('%Y-%m-%d').tolist(),
        'Dates_Bilan': [q_balance.columns[0].strftime('%Y-%m-%d'), q_balance.columns[4].strftime('%Y-%m-%d')]
    }


# --- Fonction Principale pour l'Application ---
def run_app():
    # --- Code inchangé pour la barre latérale et la sélection des données ---
    st.sidebar.header("Options d'Analyse")

    ticker_input = st.sidebar.text_input(
        "Code Action (Ticker) :",
        value='GOOGL'
    ).upper()

    period_choice = st.sidebar.selectbox(
        "Période d'Agrégation :",
        options=["Hebdomadaire", "Mensuelle"],
        index=0
    )

    end_date_dt = pd.to_datetime('today')
    end_date = end_date_dt.strftime('%Y-%m-%d')

    period_options = {
        "Dernières 3 Années": 3,
        "Dernières 5 Années": 5,
        "Dernières 10 Années": 10,
        "Dernières 15 Années": 15,
        "Dernières 20 Années": 20
    }

    selected_period_label = st.sidebar.selectbox(
        "Sélectionner la Période :",
        options=list(period_options.keys()),
        index=2
    )

    years_offset = period_options[selected_period_label]
    start_date_dt = end_date_dt - pd.DateOffset(years=years_offset)

    if not isinstance(start_date_dt, pd.Timestamp):
        start_date_dt = pd.to_datetime(start_date_dt)

    start_date = start_date_dt.strftime('%Y-%m-%d')

    if period_choice == "Hebdomadaire":
        interval = "1wk"
        period_label = "Semaine"
    else:
        interval = "1mo"
        period_label = "Mois"

    st.sidebar.markdown("---")
    st.sidebar.caption(f"EMA Courte: {WINDOW_MA_SHORT} Périodes ({period_label}s)")
    st.sidebar.caption(f"EMA Longue: {WINDOW_MA_LONG} Périodes ({period_label}s)")
    st.sidebar.caption(f"Intervalle YFinance: **{interval}**")
    st.sidebar.write(f"Période: **{start_date}** à **{end_date}**")

    st.title("Analyse de Tendance Exponentielle, Volatilité & MACD")
    st.markdown(f"**Action:** {ticker_input} | **Période d'Agrégation:** {period_choice}")
    st.markdown("---")

    # --- Téléchargement et Traitement des Données ---
    try:
        # Récupération des infos de l'entreprise et création de l'objet ticker
        ticker_obj = yf.Ticker(ticker_input)
        company_info = ticker_obj.info
        company_name = company_info.get('longName', ticker_input)
        currency = company_info.get('currency', '$')

        # Téléchargement des données de prix (Code inchangé)
        with st.spinner(f"Téléchargement des données pour **{company_name}** ({ticker_input}) en intervalle **{interval}**..."):
            data = yf.download(ticker_input, start=start_date, end=end_date, auto_adjust=True, interval=interval)

        if data.empty:
            st.error(f"Erreur: Aucune donnée trouvée pour le ticker **{ticker_input}** sur la période {start_date} à {end_date} avec l'intervalle {interval}.")
            return

        st.subheader(f"Graphique de l'Action : {company_name} ({ticker_input})")

        # --- CALCULS DES INDICATEURS (Code inchangé) ---
        data['Pct_Change'] = data['Close'].pct_change() * 100
        max_gain = data['Pct_Change'].max()
        min_loss = data['Pct_Change'].min()
        date_max_gain = data['Pct_Change'].idxmax().strftime('%Y-%m-%d')
        date_min_loss = data['Pct_Change'].idxmin().strftime('%Y-%m-%d')

        data[f'EMA_{WINDOW_MA_SHORT}'] = data['Close'].ewm(span=WINDOW_MA_SHORT, adjust=False).mean()
        data[f'EMA_{WINDOW_MA_LONG}'] = data['Close'].ewm(span=WINDOW_MA_LONG, adjust=False).mean()

        data['EMA_Fast'] = data['Close'].ewm(span=MACD_FAST_PERIOD, adjust=False).mean()
        data['EMA_Slow'] = data['Close'].ewm(span=MACD_SLOW_PERIOD, adjust=False).mean()
        data['MACD'] = data['EMA_Fast'] - data['EMA_Slow']
        data['Signal'] = data['MACD'].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
        data['Histogram'] = data['MACD'] - data['Signal']

        data['Periods'] = np.arange(len(data))
        data['Log_Close'] = np.log(data['Close'])

        X = data[['Periods']]
        y_log = data['Log_Close'].squeeze()
        y_price = data['Close'].squeeze()

        model_log = LinearRegression()
        if len(data) < 2:
            st.warning("Pas assez de données pour effectuer la régression log-linéaire.")
            return

        model_log.fit(X, y_log)
        r_squared = model_log.score(X, y_log)

        data['Predicted_Log_Price'] = model_log.predict(X)
        data['Predicted_Price'] = np.exp(data['Predicted_Log_Price'])

        data['Log_Residuals'] = y_log - data['Predicted_Log_Price']
        sigma_log = data['Log_Residuals'].std()

        data['Upper_1sigma'] = np.exp(data['Predicted_Log_Price'] + sigma_log)
        data['Lower_1sigma'] = np.exp(data['Predicted_Log_Price'] - sigma_log)
        data['Upper_2sigma'] = np.exp(data['Predicted_Log_Price'] + 2 * sigma_log)
        data['Lower_2sigma'] = np.exp(data['Predicted_Log_Price'] - 2 * sigma_log)

        pente_log_periode = model_log.coef_[0]
        taux_croissance_periode = (np.exp(pente_log_periode) - 1) * 100

        if period_choice == "Hebdomadaire":
            multiplier = WEEKS_PER_YEAR
        else:
            multiplier = 12

        pente_log_annuelle = pente_log_periode * multiplier
        taux_croissance_annuel = (np.exp(pente_log_annuelle) - 1) * 100

        prix_initial_estime = np.exp(model_log.intercept_)

        sigma_percent_1 = (np.exp(sigma_log) - 1) * 100
        sigma_percent_2 = (np.exp(2 * sigma_log) - 1) * 100

        # --- Affichage des Métriques Clés (Code inchangé) ---
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label=f"Taux de Croissance Annuel Estimé (composé)",
                value=f"{taux_croissance_annuel:.2f} %"
            )
        with col2:
            st.metric(
                label=f"Volatilité (Écart de Prix +/-1σ par {period_label})",
                value=f"{sigma_percent_1:.2f} %"
            )
        with col3:
            st.metric(
                label=f"Volatilité (Écart de Prix +/-2σ par {period_label})",
                value=f"{sigma_percent_2:.2f} %"
            )
        with col4:
            st.metric(
                label=f"R² du Modèle (sur Log-Prix)",
                value=f"{r_squared:.4f}"
            )

        st.markdown("---")

        # --- Graphique Interactif (Code inchangé) ---
        # 1. Créer la figure avec 2 subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3], # 70% pour le prix, 30% pour le MACD
            specs=[[{"type": "scatter", "secondary_y": False, "rowspan": 1}],
                   [{"type": "scatter", "secondary_y": False, "rowspan": 1}]]
        )

        # --- SUBPLOT 1: PRIX, TENDANCE, EMA, VOLATILITÉ (row=1, col=1) ---

        # Bandes de Volatilité (+/- 2 et 1 sigma)
        fig.add_trace(go.Scatter(x=data.index, y=data['Upper_2sigma'], mode='lines', name=f'+2σ ({data["Upper_2sigma"].iloc[-1]:.2f})', line=dict(color='orange', width=0.5, dash='dot'), legendgroup='prix', showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Upper_1sigma'], mode='lines', name=f'+1σ ({data["Upper_1sigma"].iloc[-1]:.2f})', line=dict(color='green', width=1, dash='dash'), legendgroup='prix', showlegend=True), row=1, col=1)

        # Prix de clôture et Tendance Exponentielle
        fig.add_trace(go.Scatter(x=data.index, y=y_price, mode='lines', name=f'Prix de clôture: {y_price.iloc[-1]:.2f}', line=dict(color='black', width=2), legendgroup='prix', showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Predicted_Price'], mode='lines', name=f'Tendance Exp.: {data["Predicted_Price"].iloc[-1]:.2f}', line=dict(color='red', width=2), legendgroup='prix', showlegend=True), row=1, col=1)

        # Lignes -1 et -2 sigma inférieures
        fig.add_trace(go.Scatter(x=data.index, y=data['Lower_1sigma'], mode='lines', name=f'-1σ ({data["Lower_1sigma"].iloc[-1]:.2f})', line=dict(color='green', width=1, dash='dash'), legendgroup='prix', showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Lower_2sigma'], mode='lines', name=f'-2σ ({data["Lower_2sigma"].iloc[-1]:.2f})', line=dict(color='orange', width=0.5, dash='dot'), legendgroup='prix', showlegend=True), row=1, col=1)

        # Moyennes Mobiles Exponentielles (EMA)
        ma_long_label = f'EMA {WINDOW_MA_LONG} {period_label}s: {data[f"EMA_{WINDOW_MA_LONG}"].iloc[-1]:.2f}'
        fig.add_trace(go.Scatter(x=data.index, y=data[f'EMA_{WINDOW_MA_LONG}'], mode='lines', name=ma_long_label, line=dict(color='purple', width=2, dash='solid'), legendgroup='prix', showlegend=True), row=1, col=1)

        ma_short_label = f'EMA {WINDOW_MA_SHORT} {period_label}s: {data[f"EMA_{WINDOW_MA_SHORT}"].iloc[-1]:.2f}'
        fig.add_trace(go.Scatter(x=data.index, y=data[f'EMA_{WINDOW_MA_SHORT}'], mode='lines', name=ma_short_label, line=dict(color='blue', width=1, dash='solid'), legendgroup='prix', showlegend=True), row=1, col=1)


        # --- SUBPLOT 2: MACD (row=2, col=1) ---

        # Histogramme MACD
        fig.add_trace(go.Bar(
            x=data.index, y=data['Histogram'], name='Histogramme MACD',
            marker_color=np.where(data['Histogram'] >= 0, 'rgba(0, 150, 0, 0.7)', 'rgba(200, 0, 0, 0.7)'), # Vert ou Rouge
            legendgroup='macd', showlegend=True
        ), row=2, col=1)

        # Ligne MACD
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MACD'], mode='lines', name=f'MACD: {data["MACD"].iloc[-1]:.2f}',
            line=dict(color='black', width=1.5),
            legendgroup='macd', showlegend=True
        ), row=2, col=1)

        # Ligne de Signal
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Signal'], mode='lines', name=f'Signal: {data["Signal"].iloc[-1]:.2f}',
            line=dict(color='red', width=1),
            legendgroup='macd', showlegend=True
        ), row=2, col=1)

        # Ligne Zéro
        fig.add_trace(go.Scatter(
            x=data.index, y=[0]*len(data.index), mode='lines', name='Ligne Zéro',
            line=dict(color='gray', width=0.5, dash='dot'),
            legendgroup='macd', showlegend=False
        ), row=2, col=1)


        # --- Mise en page finale (Code inchangé) ---
        fig.update_layout(
            title={
                'text': f'Analyse {company_name} ({ticker_input}) ({period_choice}): Tendance Exp. & MACD',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            hovermode="x unified",
            template="plotly_white",
            height=850
        )

        fig.update_yaxes(title_text=f"Prix ({currency}) (Log)", row=1, col=1, type="log")
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # --- Affichage des autres résultats du modèle (Code inchangé) ---
        st.markdown("### Détails de la Régression Log-Linéaire et Extrêmes")

        details = f"""
        - **Taux de Croissance par {period_label} (composé):** `{taux_croissance_periode:.3f}%`
        - **Volatilité (Écart-type des résidus log):** `{sigma_log:.6f}`
        - **Prix de départ estimé (Intercept):** `{prix_initial_estime:.2f} {currency}`
        - **Période de Max Gain ({date_max_gain}):** `{max_gain:.2f} %`
        - **Période de Max Perte ({date_min_loss}):** `{min_loss:.2f} %`
        """
        st.markdown(details)

        # --- NOUVELLE SECTION : ANALYSE DUPONT (ROE LTM) ---
        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>Analyse DuPont (Return on Equity) - LTM</h2>", unsafe_allow_html=True)
        
        try:
            # Récupération des données financières trimestrielles
            q_financials = ticker_obj.quarterly_financials
            q_balance = ticker_obj.quarterly_balance_sheet
            
            # Effectuer le calcul ROE (LTM)
            dupont_results = calculate_dupont(q_financials, q_balance)
            
            if dupont_results and 'error' not in dupont_results:
                # Affichage des dates utilisées pour le contexte
                dates_ltm_start = dupont_results['Dates_LTM'][-1]
                dates_ltm_end = dupont_results['Dates_LTM'][0]
                date_bilan_t0 = dupont_results['Dates_Bilan'][0]
                date_bilan_t4 = dupont_results['Dates_Bilan'][1]
                
                st.caption(f"Le ROE LTM est basé sur les flux financiers du {dates_ltm_start} au {dates_ltm_end}. Le bilan moyen est calculé entre le {date_bilan_t0} (T0) et le {date_bilan_t4} (T-4).")

                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric(
                        label="1. Marge Nette (LTM)",
                        value=f"{dupont_results['Marge_Nette'] * 100:.2f} %"
                    )
                with col_b:
                    st.metric(
                        label="2. Rotation de l'Actif (LTM)",
                        value=f"{dupont_results['Rotation_Actif']:.2f} x"
                    )
                with col_c:
                    st.metric(
                        label="3. Multiplicateur de CE (T0/T-4)",
                        value=f"{dupont_results['Multiplicateur_CE']:.2f} x"
                    )
                with col_d:
                    # Final ROE (produit des 3)
                    st.metric(
                        label="ROE (Return on Equity) LTM",
                        value=f"{dupont_results['ROE'] * 100:.2f} %",
                        delta=f"Produit de (1) x (2) x (3)"
                    )
            elif dupont_results and 'error' in dupont_results:
                st.info(f"Analyse DuPont (ROE LTM) non disponible: {dupont_results['error']}")
            else:
                st.info("Les données financières trimestrielles nécessaires à l'Analyse DuPont (ROE LTM) ne sont pas suffisantes (nécessite 4 Q de résultats et 5 Q de bilan).")
                
        except Exception as e:
            st.error(f"Erreur inattendue lors de l'accès aux données financières de YFinance pour ROE LTM: {e}")
            
        # --- NOUVELLE SECTION : ANALYSE ROIC (LTM) ---
        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>Décomposition du ROIC (Return on Invested Capital) - LTM</h2>", unsafe_allow_html=True)

        try:
            # Effectuer le calcul ROIC (LTM)
            roic_results = calculate_roic(q_financials, q_balance)
            
            if roic_results and 'error' not in roic_results:
                # Affichage des dates utilisées pour le contexte
                dates_ltm_start = roic_results['Dates_LTM'][-1]
                dates_ltm_end = roic_results['Dates_LTM'][0]
                date_bilan_t0 = roic_results['Dates_Bilan'][0]
                date_bilan_t4 = roic_results['Dates_Bilan'][1]

                st.caption(f"Le ROIC LTM est basé sur les flux financiers du {dates_ltm_start} au {dates_ltm_end}. Le capital investi moyen est calculé entre le {date_bilan_t0} (T0) et le {date_bilan_t4} (T-4).")

                col_e, col_f, col_g = st.columns([1, 1, 2]) # 3 colonnes pour les 3 métriques

                with col_e:
                    st.metric(
                        label="1. Marge NOPAT (LTM)",
                        value=f"{roic_results['Marge_NOPAT'] * 100:.2f} %"
                    )
                with col_f:
                    st.metric(
                        label="2. Rotation du CI (LTM)",
                        value=f"{roic_results['Rotation_CI']:.2f} x"
                    )
                with col_g:
                    # Final ROIC (produit des 2)
                    st.metric(
                        label="ROIC (Return on Invested Capital) LTM",
                        value=f"{roic_results['ROIC'] * 100:.2f} %",
                        delta=f"Produit de (1) x (2)"
                    )
            elif roic_results and 'error' in roic_results:
                 st.info(f"Analyse ROIC (LTM) non disponible: {roic_results['error']}")
            else:
                st.info("Les données financières trimestrielles nécessaires à la décomposition du ROIC LTM ne sont pas suffisantes (nécessite 4 Q de résultats et 5 Q de bilan).")
                
        except Exception as e:
            st.error(f"Erreur inattendue lors de l'accès aux données financières de YFinance pour ROIC LTM: {e}")

    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement des données ou du téléchargement: {e}")
        st.caption("Vérifiez que le code de l'action (ticker) est correct.")


# Exécuter l'application
if __name__ == "__main__":
    run_app()