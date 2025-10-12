import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
from datetime import date 

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse de Tendance Exponentielle et Volatilité",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constantes globales pour les calculs ---
# Fenêtres pour les Moyennes Mobiles (en nombre de périodes)
WINDOW_MA_SHORT = 50
WINDOW_MA_LONG = 200

# Paramètres MACD standard
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# Constantes pour les conversions de période
WEEKS_PER_MONTH = 4.33 # Approximation pour les conversions de taux si intervalle="1wk"
WEEKS_PER_YEAR = 52     # Nombre de semaines dans une année

# --- Fonction Principale pour l'Application ---
def run_app():
    # --- Barre Latérale de Contrôle ---
    st.sidebar.header("Options d'Analyse")

    # 1. Sélection du Ticker
    ticker_input = st.sidebar.text_input(
        "Code Action (Ticker) :",
        value='GOOGL'
    ).upper()

    # 2. Sélection de la Période d'Agrégation
    period_choice = st.sidebar.selectbox(
        "Période d'Agrégation :",
        options=["Hebdomadaire", "Mensuelle"],
        index=0  # Défaut sur Hebdomadaire
    )
    
    # 3. LOGIQUE DE SÉLECTION DES DATES SIMPLIFIÉE
    
    # Définition de la date de fin (aujourd'hui)
    end_date_dt = pd.to_datetime('today')
    end_date = end_date_dt.strftime('%Y-%m-%d')
    
    # Options pour la selectbox de période (SIMPLIFIÉES)
    period_options = {
        "Dernières 3 Années": 3,
        "Dernières 5 Années": 5,
        "Dernières 10 Années": 10,
        "Dernières 15 Années": 15,
        "Dernières 20 Années": 20
    }

    # Selectbox pour le choix de la période
    selected_period_label = st.sidebar.selectbox(
        "Sélectionner la Période :",
        options=list(period_options.keys()),
        index=2 # "Dernières 10 Années" par défaut
    )
    
    # Calcul de la date de début pour les options prédéfinies
    years_offset = period_options[selected_period_label]
    start_date_dt = end_date_dt - pd.DateOffset(years=years_offset)
        
    # S'assurer que start_date_dt est un objet pd.Timestamp pour le formatage
    if not isinstance(start_date_dt, pd.Timestamp):
        start_date_dt = pd.to_datetime(start_date_dt)
        
    start_date = start_date_dt.strftime('%Y-%m-%d')
    
    # --- Fin de la logique de dates simplifiée ---
    
    # Convertir le choix de l'utilisateur au format yfinance
    if period_choice == "Hebdomadaire":
        interval = "1wk"
        period_label = "Semaine"
    else: # Mensuelle
        interval = "1mo"
        period_label = "Mois"

    st.sidebar.markdown("---")
    st.sidebar.caption(f"MM Courte: {WINDOW_MA_SHORT} Périodes ({period_label}s)")
    st.sidebar.caption(f"MM Longue: {WINDOW_MA_LONG} Périodes ({period_label}s)")
    st.sidebar.caption(f"Intervalle YFinance: **{interval}**")
    st.sidebar.write(f"Période: **{start_date}** à **{end_date}**") # Affichage de la période utilisée

    # --- Titre Principal ---
    st.title("Analyse de Tendance Exponentielle, Volatilité & MACD")
    st.markdown(f"**Action:** {ticker_input} | **Période d'Agrégation:** {period_choice}")
    st.markdown("---")

    # --- Téléchargement et Traitement des Données ---
    try:
        # Récupération des infos de l'entreprise
        ticker_obj = yf.Ticker(ticker_input)
        company_info = ticker_obj.info
        company_name = company_info.get('longName', ticker_input)
        currency = company_info.get('currency', '$')

        # Téléchargement des données
        with st.spinner(f"Téléchargement des données pour **{company_name}** ({ticker_input}) en intervalle **{interval}**..."):
            data = yf.download(ticker_input, start=start_date, end=end_date, auto_adjust=True, interval=interval)

        if data.empty:
            st.error(f"Erreur: Aucune donnée trouvée pour le ticker **{ticker_input}** sur la période {start_date} à {end_date} avec l'intervalle {interval}.")
            return

        # Affichage du nom complet
        st.subheader(f"Graphique de l'Action : {company_name} ({ticker_input})")

        # --- CALCULS DES INDICATEURS (AUCUN CHANGEMENT) ---

        # 1. Rendements (pour les extrêmes)
        data['Pct_Change'] = data['Close'].pct_change() * 100
        max_gain = data['Pct_Change'].max()
        min_loss = data['Pct_Change'].min()
        date_max_gain = data['Pct_Change'].idxmax().strftime('%Y-%m-%d')
        date_min_loss = data['Pct_Change'].idxmin().strftime('%Y-%m-%d')

        # 2. Moyennes Mobiles
        data[f'MA_{WINDOW_MA_SHORT}'] = data['Close'].rolling(window=WINDOW_MA_SHORT).mean()
        data[f'MA_{WINDOW_MA_LONG}'] = data['Close'].rolling(window=WINDOW_MA_LONG).mean()

        # 3. CALCUL DU MACD
        data['EMA_Fast'] = data['Close'].ewm(span=MACD_FAST_PERIOD, adjust=False).mean()
        data['EMA_Slow'] = data['Close'].ewm(span=MACD_SLOW_PERIOD, adjust=False).mean()
        data['MACD'] = data['EMA_Fast'] - data['EMA_Slow']
        data['Signal'] = data['MACD'].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
        data['Histogram'] = data['MACD'] - data['Signal']

        # 4. Préparation pour la Régression Log-Linéaire
        data['Periods'] = np.arange(len(data))
        data['Log_Close'] = np.log(data['Close'])

        X = data[['Periods']]
        y_log = data['Log_Close'].squeeze()
        y_price = data['Close'].squeeze()

        # 5. Régression Log-Linéaire
        model_log = LinearRegression()
        if len(data) < 2:
            st.warning("Pas assez de données pour effectuer la régression log-linéaire.")
            return

        model_log.fit(X, y_log)
        r_squared = model_log.score(X, y_log) # Calcul de R²

        # Prédictions
        data['Predicted_Log_Price'] = model_log.predict(X)
        data['Predicted_Price'] = np.exp(data['Predicted_Log_Price'])

        # 6. Volatilité (Écart-type des résidus)
        data['Log_Residuals'] = y_log - data['Predicted_Log_Price']
        sigma_log = data['Log_Residuals'].std()

        # Calcul des bandes de volatilité
        data['Upper_1sigma'] = np.exp(data['Predicted_Log_Price'] + sigma_log)
        data['Lower_1sigma'] = np.exp(data['Predicted_Log_Price'] - sigma_log)
        data['Upper_2sigma'] = np.exp(data['Predicted_Log_Price'] + 2 * sigma_log)
        data['Lower_2sigma'] = np.exp(data['Predicted_Log_Price'] - 2 * sigma_log)
        
        # 7. Calcul des Taux de Croissance
        pente_log_periode = model_log.coef_[0]
        taux_croissance_periode = (np.exp(pente_log_periode) - 1) * 100

        # Taux Annualisé (Pour la comparaison)
        if period_choice == "Hebdomadaire":
            multiplier = WEEKS_PER_YEAR # 52 semaines
        else: # Mensuelle
            multiplier = 12 # 12 mois
            
        pente_log_annuelle = pente_log_periode * multiplier
        taux_croissance_annuel = (np.exp(pente_log_annuelle) - 1) * 100
        
        # Prix Initial Estimé (au Jour 0, où Periods=0)
        prix_initial_estime = np.exp(model_log.intercept_)

        # Volatilité en pourcentage d'écart
        sigma_percent_1 = (np.exp(sigma_log) - 1) * 100
        sigma_percent_2 = (np.exp(2 * sigma_log) - 1) * 100

        # --- Affichage des Métriques Clés ---
        
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
        
        # Volatilité à +/- 2 sigma
        with col3:
            st.metric(
                label=f"Volatilité (Écart de Prix +/-2σ par {period_label})",
                value=f"{sigma_percent_2:.2f} %"
            )
        # R² du modèle
        with col4:
            st.metric(
                label=f"R² du Modèle (sur Log-Prix)",
                value=f"{r_squared:.4f}"
            )
            
        st.markdown("---")

        # --- Graphique Interactif avec Plotly ---
        
        # 1. Créer la figure avec 2 subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3], # 70% pour le prix, 30% pour le MACD
            specs=[[{"type": "scatter", "secondary_y": False, "rowspan": 1}],
                    [{"type": "scatter", "secondary_y": False, "rowspan": 1}]]
        )

        # --- SUBPLOT 1: PRIX, TENDANCE, MM, VOLATILITÉ (row=1, col=1) ---
        
        # Bandes de Volatilité (+/- 2 et 1 sigma)
        fig.add_trace(go.Scatter(x=data.index, y=data['Upper_2sigma'], mode='lines', name=f'+2σ ({data["Upper_2sigma"].iloc[-1]:.2f})', line=dict(color='orange', width=0.5, dash='dot'), legendgroup='prix', showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Upper_1sigma'], mode='lines', name=f'+1σ ({data["Upper_1sigma"].iloc[-1]:.2f})', line=dict(color='green', width=1, dash='dash'), legendgroup='prix', showlegend=True), row=1, col=1)

        # Prix de clôture et Tendance Exponentielle
        fig.add_trace(go.Scatter(x=data.index, y=y_price, mode='lines', name=f'Prix de clôture: {y_price.iloc[-1]:.2f}', line=dict(color='black', width=2), legendgroup='prix', showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Predicted_Price'], mode='lines', name=f'Tendance Exp.: {data["Predicted_Price"].iloc[-1]:.2f}', line=dict(color='red', width=2), legendgroup='prix', showlegend=True), row=1, col=1)

        # Lignes -1 et -2 sigma inférieures
        fig.add_trace(go.Scatter(x=data.index, y=data['Lower_1sigma'], mode='lines', name=f'-1σ ({data["Lower_1sigma"].iloc[-1]:.2f})', line=dict(color='green', width=1, dash='dash'), legendgroup='prix', showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Lower_2sigma'], mode='lines', name=f'-2σ ({data["Lower_2sigma"].iloc[-1]:.2f})', line=dict(color='orange', width=0.5, dash='dot'), legendgroup='prix', showlegend=True), row=1, col=1)
        
        # Moyennes Mobiles
        ma_long_label = f'MM {WINDOW_MA_LONG} {period_label}s: {data[f"MA_{WINDOW_MA_LONG}"].iloc[-1]:.2f}'
        fig.add_trace(go.Scatter(x=data.index, y=data[f'MA_{WINDOW_MA_LONG}'], mode='lines', name=ma_long_label, line=dict(color='purple', width=2, dash='solid'), legendgroup='prix', showlegend=True), row=1, col=1)

        ma_short_label = f'MM {WINDOW_MA_SHORT} {period_label}s: {data[f"MA_{WINDOW_MA_SHORT}"].iloc[-1]:.2f}'
        fig.add_trace(go.Scatter(x=data.index, y=data[f'MA_{WINDOW_MA_SHORT}'], mode='lines', name=ma_short_label, line=dict(color='blue', width=1, dash='solid'), legendgroup='prix', showlegend=True), row=1, col=1)

        
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
            height=850 # Augmenter la hauteur pour accommoder le subplot
        )
        
        # Mise à jour des axes du Subplot 1 (Prix)
        fig.update_yaxes(title_text=f"Prix ({currency}) (Log)", row=1, col=1, type="log")
        
        # Mise à jour des axes du Subplot 2 (MACD)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1) # Ajoute le titre X seulement au dernier subplot


        # Affichage du graphique dans Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # --- Affichage des autres résultats du modèle ---
        st.markdown("### Détails de la Régression Log-Linéaire et Extrêmes")
        
        details = f"""
        - **Taux de Croissance par {period_label} (composé):** `{taux_croissance_periode:.3f}%`
        - **Volatilité (Écart-type des résidus log):** `{sigma_log:.6f}`
        - **Prix de départ estimé (Intercept):** `{prix_initial_estime:.2f} {currency}`
        - **Période de Max Gain ({date_max_gain}):** `{max_gain:.2f} %`
        - **Période de Max Perte ({date_min_loss}):** `{min_loss:.2f} %`
        """
        st.markdown(details)

    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement des données ou du téléchargement: {e}")
        st.caption("Vérifiez que le code de l'action (ticker) est correct.")


# Exécuter l'application
if __name__ == "__main__":
    run_app()