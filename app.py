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

# --- Fonction de Calcul pour l'Analyse DuPont ---
def calculate_dupont(q_financials: pd.DataFrame, q_balance: pd.DataFrame) -> dict | None:
    """
    Calcule l'analyse DuPont (ROE) à partir des données financières trimestrielles de yfinance.
    Utilise les deux trimestres les plus récents de la Balance Sheet pour calculer les moyennes.
    """
    try:
        if q_financials.empty or q_balance.empty or len(q_balance.columns) < 2:
            return None

        # Helper function pour récupérer la valeur en gérant les clés alternatives (Revenues)
        def get_value(series: Series, key: str):
            if key in series.index:
                return series.loc[key]
            if key == 'Total Revenue' and 'Total Revenues' in series.index:
                return series.loc['Total Revenues']
            raise KeyError(key)

        # T0: Dernier Trimestre connu (colonne 0)
        q0_financials = q_financials.iloc[:, 0]
        q0_balance = q_balance.iloc[:, 0]
        
        # T1: Trimestre Précédent (colonne 1), utilisé pour les moyennes (simplification de T-4)
        q1_balance = q_balance.iloc[:, 1]
        
        # --- Extraction des Données ---
        revenu_q0 = get_value(q0_financials, 'Total Revenue')
        resultat_net_q0 = get_value(q0_financials, 'Net Income')
        actifs_q0 = get_value(q0_balance, 'Total Assets')
        capitaux_propres_q0 = get_value(q0_balance, 'Common Stock Equity')
        
        actifs_q1 = get_value(q1_balance, 'Total Assets')
        capitaux_propres_q1 = get_value(q1_balance, 'Common Stock Equity')
        
    except KeyError as ke:
        return {'error': f"Donnée financière manquante pour DuPont: {ke}"}
    except Exception as e:
        return {'error': f"Erreur inattendue dans l'analyse DuPont: {e}"}

    # --- Calcul des Moyennes (utilisant T0 et T1) ---
    actifs_moyens = (actifs_q0 + actifs_q1) / 2
    capitaux_propres_moyens = (capitaux_propres_q0 + capitaux_propres_q1) / 2

    # Prévention de la division par zéro
    if revenu_q0 == 0 or actifs_moyens == 0 or capitaux_propres_moyens == 0:
        return {
            'Marge_Nette': 0.0, 'Rotation_Actif': 0.0,
            'Multiplicateur_CE': 0.0, 'ROE': 0.0,
            'Dates': q_balance.columns[0:2].strftime('%Y-%m-%d').tolist()
        }

    # --- Composantes DuPont ---
    marge_nette = resultat_net_q0 / revenu_q0
    rotation_actif = revenu_q0 / actifs_moyens
    multiplicateur_ce = actifs_moyens / capitaux_propres_moyens
    roe = marge_nette * rotation_actif * multiplicateur_ce

    return {
        'Marge_Nette': marge_nette,
        'Rotation_Actif': rotation_actif,
        'Multiplicateur_CE': multiplicateur_ce,
        'ROE': roe,
        'Dates': q_balance.columns[0:2].strftime('%Y-%m-%d').tolist()
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

        # Téléchargement des données de prix
        with st.spinner(f"Téléchargement des données pour **{company_name}** ({ticker_input}) en intervalle **{interval}**..."):
            data = yf.download(ticker_input, start=start_date, end=end_date, auto_adjust=True, interval=interval)

        if data.empty:
            st.error(f"Erreur: Aucune donnée trouvée pour le ticker **{ticker_input}** sur la période {start_date} à {end_date} avec l'intervalle {interval}.")
            return

        st.subheader(f"Graphique de l'Action : {company_name} ({ticker_input})")

        # --- CALCULS DES INDICATEURS (Log-Linéaire, EMA, MACD, etc. - Code inchangé) ---
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
        # ... (Le code du graphique reste ici, non inclus pour la concision) ...
        # [GRAPHIQUE PLOTLY]
        
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


        # --- Mise en page finale ---

        # Titre général
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

        # Mise à jour des axes du Subplot 1 (Prix)
        fig.update_yaxes(title_text=f"Prix ({currency}) (Log)", row=1, col=1, type="log")

        # Mise à jour des axes du Subplot 2 (MACD)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

        # Affichage du graphique dans Streamlit
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

        # --- NOUVELLE SECTION : ANALYSE DUPONT (ROE) ---
        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>Analyse DuPont (Return on Equity)</h2>", unsafe_allow_html=True)
        
        try:
            # Récupération des données financières trimestrielles
            q_financials = ticker_obj.quarterly_financials
            q_balance = ticker_obj.quarterly_balance_sheet
            
            # Effectuer le calcul
            dupont_results = calculate_dupont(q_financials, q_balance)
            
            if dupont_results and 'error' not in dupont_results:
                # Affichage des dates utilisées pour le contexte
                date_actuel, date_prec = dupont_results['Dates']
                st.caption(f"Basé sur le dernier trimestre ({date_actuel}) et le trimestre précédent ({date_prec}) pour le calcul des moyennes d'actifs et de capitaux propres.")

                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric(
                        label="1. Marge Nette (Net Profit Margin)",
                        value=f"{dupont_results['Marge_Nette'] * 100:.2f} %"
                    )
                with col_b:
                    st.metric(
                        label="2. Rotation de l'Actif (Asset Turnover)",
                        value=f"{dupont_results['Rotation_Actif']:.2f} x"
                    )
                with col_c:
                    st.metric(
                        label="3. Multiplicateur de CE (Equity Multiplier)",
                        value=f"{dupont_results['Multiplicateur_CE']:.2f} x"
                    )
                with col_d:
                    # Final ROE (produit des 3)
                    st.metric(
                        label="ROE (Return on Equity)",
                        value=f"{dupont_results['ROE'] * 100:.2f} %",
                        delta=f"Produit de (1) x (2) x (3)"
                    )
            elif dupont_results and 'error' in dupont_results:
                st.info(f"Analyse DuPont non disponible: {dupont_results['error']}")
            else:
                st.info("Les données financières trimestrielles nécessaires à l'Analyse DuPont ne sont pas suffisantes ou complètes (nécessite au moins 2 trimestres de bilan).")
                
        except Exception as e:
            st.error(f"Erreur inattendue lors de l'accès aux données financières de YFinance: {e}")

    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement des données ou du téléchargement: {e}")
        st.caption("Vérifiez que le code de l'action (ticker) est correct.")


# Exécuter l'application
if __name__ == "__main__":
    run_app()