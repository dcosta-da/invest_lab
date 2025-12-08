import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import date
from pandas.core.series import Series

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse de Tendance Exponentielle et Volatilité",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constantes globales pour les calculs ---
WINDOW_MA_SHORT = 30
WINDOW_MA_LONG = 200
WEEKS_PER_MONTH = 4.33
WEEKS_PER_YEAR = 52

# --- NOUVELLES FONCTIONS D'AIDE LTM / T0-T4 (INCHANGÉES) ---

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
        return default_val
    
    return q_balance.loc[key].iloc[index]


def get_balance_value_5Q_average(q_balance: pd.DataFrame, keys: list[str], min_length: int = 5) -> float:
    """Calcule la moyenne de 5 trimestres (T0 à T-4) pour un ou plusieurs éléments du bilan (sommés)."""
    if len(q_balance.columns) < min_length:
        raise ValueError(f"Données de bilan insuffisantes (moins de {min_length} trimestres) pour la moyenne 5Q.")
    
    # 1. Obtenir les 5 valeurs pour chaque clé
    total_5q_sum = 0.0
    
    for key in keys:
        try:
            if key not in q_balance.index:
                if key in ['Minority Interest', 'Non Controlling Interest', 'Total Debt']:
                    continue # On passe si la clé optionnelle n'existe pas
                raise KeyError(f"Clé de bilan essentielle manquante pour la moyenne 5Q: {key}")

            # Somme des 5 trimestres pour cette clé
            total_5q_sum += q_balance.loc[key].iloc[0:5].sum()
            
        except KeyError:
            raise
    
    # 2. Calculer la moyenne (Somme des 5 points / 5)
    return total_5q_sum / min_length


# --- FONCTION DE CALCUL POUR L'ANALYSE DUPONT (ROE LTM) (INCHANGÉE) ---
def calculate_dupont(q_financials: pd.DataFrame, q_balance: pd.DataFrame) -> dict | None:
    """
    Calcule l'analyse DuPont (ROE) en utilisant les données LTM.
    Les moyennes du bilan sont basées sur la moyenne des 5 trimestres (T0 à T-4).
    """
    if q_financials.empty or q_balance.empty or len(q_financials.columns) < 4 or len(q_balance.columns) < 5:
        return None

    try:
        ltm_revenu = get_ltm_sum(q_financials, 'Total Revenue')
        ltm_resultat_net = get_ltm_sum(q_financials, 'Net Income')
        
        actifs_moyens = get_balance_value_5Q_average(q_balance, ['Total Assets'])
        
        capitaux_propres_moyens = get_balance_value_5Q_average(q_balance, ['Common Stock Equity'])

        date_bilan_t0 = q_balance.columns[0].strftime('%Y-%m-%d')
        date_bilan_t4 = q_balance.columns[4].strftime('%Y-%m-%d')
        

    except (KeyError, ValueError) as ke:
        return {'error': f"Donnée financière/bilan manquante ou insuffisante (LTM / T0-T-4) : {ke}"}
    except Exception as e:
        return {'error': f"Erreur inattendue dans l'analyse DuPont (LTM) : {e}"}

    if ltm_revenu <= 0 or actifs_moyens <= 0 or capitaux_propres_moyens <= 0:
        return {
            'Marge_Nette': 0.0, 'Rotation_Actif': 0.0,
            'Multiplicateur_CE': 0.0, 'ROE': 0.0,
            'Dates_LTM': q_financials.columns[0:4].strftime('%Y-%m-%d').tolist(),
            'Dates_Bilan': [date_bilan_t0, date_bilan_t4]
        }

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
        'Dates_Bilan': [date_bilan_t0, date_bilan_t4]
    }


# --- FONCTION DE SIMULATION MONTE CARLO ---
def run_monte_carlo_simulation(
    initial_price: float,
    expected_return_period: float,  # rendement attendu par période (log)
    volatility_period: float,       # volatilité par période (écart-type log)
    num_simulations: int,
    num_periods: int,
    seed: int = None  # None = aléatoire à chaque exécution
) -> np.ndarray:
    """
    Exécute une simulation de Monte Carlo pour la projection des prix.
    Utilise le modèle de Mouvement Brownien Géométrique (GBM).
    
    Returns:
        np.ndarray: Matrice de prix (num_periods + 1, num_simulations)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Le drift est directement le rendement log attendu par période
    # pente_log_periode de la régression = E[log(S_t+1/S_t)] = μ - σ²/2
    # Donc on l'utilise directement sans ajustement supplémentaire
    drift = expected_return_period
    
    # Génération des chocs aléatoires
    random_shocks = np.random.normal(0, 1, (num_periods, num_simulations))
    
    # Calcul des rendements logarithmiques
    log_returns = drift + volatility_period * random_shocks
    
    # Construction des trajectoires de prix
    price_paths = np.zeros((num_periods + 1, num_simulations))
    price_paths[0] = initial_price
    
    for t in range(1, num_periods + 1):
        price_paths[t] = price_paths[t-1] * np.exp(log_returns[t-1])
    
    return price_paths


def calculate_monte_carlo_statistics(price_paths: np.ndarray) -> dict:
    """
    Calcule les statistiques clés des simulations Monte Carlo.
    Retourne les vrais percentiles sans plafonnement.
    """
    final_prices = price_paths[-1, :]
    initial_price = price_paths[0, 0]
    
    return {
        'mean_final': np.mean(final_prices),
        'median_final': np.median(final_prices),
        'std_final': np.std(final_prices),
        'min_final': np.min(final_prices),
        'max_final': np.max(final_prices),
        'percentile_10': np.percentile(final_prices, 10),
        'percentile_25': np.percentile(final_prices, 25),
        'percentile_75': np.percentile(final_prices, 75),
        'percentile_90': np.percentile(final_prices, 90),
        'prob_gain': np.mean(final_prices > initial_price) * 100,
        'prob_double': np.mean(final_prices > 2 * initial_price) * 100,
        'prob_loss_50': np.mean(final_prices < 0.5 * initial_price) * 100,
    }


# --- FONCTION DE CALCUL POUR L'ANALYSE ROIC (LTM) (INCHANGÉE) ---
def calculate_roic(q_financials: pd.DataFrame, q_balance: pd.DataFrame) -> dict | None:
    """
    Calcule la décomposition du ROIC (Return on Invested Capital) en utilisant les données LTM.
    Le capital investi moyen est basé sur la moyenne des 5 trimestres (T0 à T-4).
    """
    if q_financials.empty or q_balance.empty or len(q_financials.columns) < 4 or len(q_balance.columns) < 5:
        return None

    try:
        ltm_revenu = get_ltm_sum(q_financials, 'Total Revenue')
        ltm_ebit = get_ltm_sum(q_financials, 'EBIT')
        ltm_impots = get_ltm_sum(q_financials, 'Income Tax Expense')
        ltm_pretax_income = get_ltm_sum(q_financials, 'Pretax Income')

        if ltm_pretax_income > 0 and ltm_ebit > 0:
            taux_impot_ltm = ltm_impots / ltm_pretax_income
        else:
            taux_impot_ltm = 0.25 
        
        ltm_nopat = ltm_ebit * (1 - taux_impot_ltm)
        
        capital_investi_moyen = get_balance_value_5Q_average(
            q_balance, 
            ['Common Stock Equity', 'Total Debt', 'Minority Interest'] 
        )

        date_bilan_t0 = q_balance.columns[0].strftime('%Y-%m-%d')
        date_bilan_t4 = q_balance.columns[4].strftime('%Y-%m-%d')
        
    except (KeyError, ValueError) as ke:
        return {'error': f"Donnée financière/bilan manquante ou insuffisante (LTM / T0-T-4) : {ke}"}
    except Exception as e:
        return {'error': f"Erreur inattendue dans l'analyse ROIC (LTM) : {e}"}


    if ltm_revenu <= 0 or capital_investi_moyen <= 0:
        return {
            'Marge_NOPAT': 0.0, 'Rotation_CI': 0.0,
            'ROIC': 0.0,
            'Dates_LTM': q_financials.columns[0:4].strftime('%Y-%m-%d').tolist(),
            'Dates_Bilan': [date_bilan_t0, date_bilan_t4]
        }

    marge_nopat = ltm_nopat / ltm_revenu
    rotation_ci = ltm_revenu / capital_investi_moyen
    roic = marge_nopat * rotation_ci

    return {
        'Marge_NOPAT': marge_nopat,
        'Rotation_CI': rotation_ci,
        'ROIC': roic,
        'Dates_LTM': q_financials.columns[0:4].strftime('%Y-%m-%d').tolist(),
        'Dates_Bilan': [date_bilan_t0, date_bilan_t4]
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
    st.sidebar.caption(f"SMA Courte: {WINDOW_MA_SHORT} Périodes ({period_label}s)")
    st.sidebar.caption(f"SMA Longue: {WINDOW_MA_LONG} Périodes ({period_label}s)")
    st.sidebar.caption(f"Intervalle YFinance: **{interval}**")
    st.sidebar.write(f"Période: **{start_date}** à **{end_date}**")


    st.title("Analyse de Tendance Exponentielle et Volatilité")
    st.markdown(f"**Action:** {ticker_input} | **Période d'Agrégation:** {period_choice}")
    st.markdown("---")

    # --- Téléchargement et Traitement des Données ---
    try:
        ticker_obj = yf.Ticker(ticker_input)
        company_info = ticker_obj.info
        company_name = company_info.get('longName', ticker_input)
        currency = company_info.get('currency', '$')

        with st.spinner(f"Téléchargement des données pour **{company_name}** ({ticker_input}) en intervalle **{interval}**..."):
            data = yf.download(ticker_input, start=start_date, end=end_date, auto_adjust=True, interval=interval)

        if data.empty:
            st.error(f"Erreur: Aucune donnée trouvée pour le ticker **{ticker_input}** sur la période {start_date} à {end_date} avec l'intervalle {interval}.")
            return

        st.subheader(f"Graphique de l'Action : {company_name} ({ticker_input})")

        # --- CALCULS DES INDICATEURS ---
        data['Pct_Change'] = data['Close'].pct_change() * 100
        max_gain = data['Pct_Change'].max()
        min_loss = data['Pct_Change'].min()
        date_max_gain = data['Pct_Change'].idxmax().strftime('%Y-%m-%d')
        date_min_loss = data['Pct_Change'].idxmin().strftime('%Y-%m-%d')

        data[f'SMA_{WINDOW_MA_SHORT}'] = data['Close'].rolling(window=WINDOW_MA_SHORT).mean()
        data[f'SMA_{WINDOW_MA_LONG}'] = data['Close'].rolling(window=WINDOW_MA_LONG).mean()

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

        
        # 1. Créer la figure (Sans make_subplots)
        fig = go.Figure()

        # Bandes de Volatilité (+/- 2 et 1 sigma)
        fig.add_trace(go.Scatter(x=data.index, y=data['Upper_2sigma'], mode='lines', name=f'+2σ ({data["Upper_2sigma"].iloc[-1]:.2f})', line=dict(color='grey', width=0.5, dash='dot'), legendgroup='prix', showlegend=True))
        fig.add_trace(go.Scatter(x=data.index, y=data['Upper_1sigma'], mode='lines', name=f'+1σ ({data["Upper_1sigma"].iloc[-1]:.2f})', line=dict(color='grey', width=1, dash='dash'), legendgroup='prix', showlegend=True))

        # Prix de clôture et Tendance Exponentielle
        fig.add_trace(go.Scatter(x=data.index, y=y_price, mode='lines', name=f'Prix de clôture: {y_price.iloc[-1]:.2f}', line=dict(color='#186ddd', width=2), legendgroup='prix', showlegend=True))
        fig.add_trace(go.Scatter(x=data.index, y=data['Predicted_Price'], mode='lines', name=f'Tendance Exp.: {data["Predicted_Price"].iloc[-1]:.2f}', line=dict(color='#e4c00a', width=2), legendgroup='prix', showlegend=True))

        # Lignes -1 et -2 sigma inférieures
        fig.add_trace(go.Scatter(x=data.index, y=data['Lower_1sigma'], mode='lines', name=f'-1σ ({data["Lower_1sigma"].iloc[-1]:.2f})', line=dict(color='grey', width=1, dash='dash'), legendgroup='prix', showlegend=True))
        fig.add_trace(go.Scatter(x=data.index, y=data['Lower_2sigma'], mode='lines', name=f'-2σ ({data["Lower_2sigma"].iloc[-1]:.2f})', line=dict(color='grey', width=0.5, dash='dot'), legendgroup='prix', showlegend=True))

        # Moyennes Mobiles Exponentielles (SMA)
        ma_long_label = f'SMA {WINDOW_MA_LONG} {period_label}s: {data[f"SMA_{WINDOW_MA_LONG}"].iloc[-1]:.2f}'
        fig.add_trace(go.Scatter(x=data.index, y=data[f'SMA_{WINDOW_MA_LONG}'], mode='lines', name=ma_long_label, line=dict(color='#ff0195', width=2, dash='solid'), legendgroup='prix', showlegend=True))

        ma_short_label = f'SMA {WINDOW_MA_SHORT} {period_label}s: {data[f"SMA_{WINDOW_MA_SHORT}"].iloc[-1]:.2f}'
        fig.add_trace(go.Scatter(x=data.index, y=data[f'SMA_{WINDOW_MA_SHORT}'], mode='lines', name=ma_short_label, line=dict(color='#00c2ff', width=1, dash='solid'), legendgroup='prix', showlegend=True))

        # --- Mise en page finale ---
        fig.update_layout(
            title={
                'text': f'Analyse {company_name} ({ticker_input}) ({period_choice}): Tendance Exponentielle et Volatilité',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            hovermode="x unified",
            template="plotly_white",
            height=600 # Hauteur ajustée pour un seul graphique
        )

        fig.update_yaxes(title_text=f"Prix ({currency}) (Log)", type="log")
        fig.update_xaxes(title_text="Date") 
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
        st.markdown("<h2 style='text-align: center;'>🔎 Analyse DuPont (Rentabilité des Capitaux Propres) - LTM</h2>", unsafe_allow_html=True)
        
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
                
                st.caption(f"Le ROE LTM est basé sur les flux financiers du {dates_ltm_start} au {dates_ltm_end}. Le bilan moyen est calculé sur 5 trimestres, du {date_bilan_t0} (T0) au {date_bilan_t4} (T-4).")

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
                        label="3. Multiplicateur de CE (5Q Moyen)",
                        value=f"{dupont_results['Multiplicateur_CE']:.2f} x"
                    )
                with col_d:
                    st.metric(
                        label="ROE (Return on Equity) LTM",
                        value=f"{dupont_results['ROE'] * 100:.2f} %",
                        delta=f"Produit de (1) x (2) x (3)"
                    )
                    
                # EXPLICATION DUPONT DANS UN EXPANDER
                with st.expander("📚 Comprendre la Décomposition DuPont (cliquez pour voir)"):
                    st.markdown("""
                    ## Qu'est-ce que l'Analyse DuPont ?
                    
                    L'analyse **DuPont** décompose le **ROE (Return on Equity)** en trois leviers fondamentaux :
                    
                    $$\\text{ROE} = \\text{Marge Nette} \\times \\text{Rotation des Actifs} \\times \\text{Multiplicateur CE}$$
                    
                    ---
                    
                    ### 1️⃣ Marge Nette (Résultat Net / Chiffre d'Affaires)
                    
                    **Ce que ça mesure :** Combien de profit l'entreprise garde pour chaque euro de vente.
                    
                    | Secteur | Marge Nette Typique |
                    |---------|---------------------|
                    | Luxe (LVMH, Hermès) | 15-25% |
                    | Tech (Google, Microsoft) | 20-35% |
                    | Grande distribution (Carrefour) | 1-3% |
                    | Automobile | 3-8% |
                    
                    **Exemple concret :**
                    - Chiffre d'affaires : **100 M€**
                    - Résultat net : **15 M€**
                    - Marge nette = 15/100 = **15%**
                    
                    ✅ **Bonne marge** = Pouvoir de fixation des prix, efficacité opérationnelle  
                    ❌ **Faible marge** = Forte concurrence, coûts élevés
                    
                    ---
                    
                    ### 2️⃣ Rotation des Actifs (Chiffre d'Affaires / Total Actifs)
                    
                    **Ce que ça mesure :** Combien de revenus chaque euro d'actif génère.
                    
                    | Secteur | Rotation Typique |
                    |---------|------------------|
                    | Grande distribution | 2.0 - 3.0x |
                    | Restauration rapide | 1.5 - 2.5x |
                    | Industrie lourde | 0.5 - 1.0x |
                    | Utilities (électricité) | 0.3 - 0.5x |
                    
                    **Exemple concret :**
                    - Chiffre d'affaires : **100 M€**
                    - Total des actifs : **50 M€**
                    - Rotation = 100/50 = **2.0x**
                    
                    ✅ **Rotation élevée** = Utilisation intensive des actifs (ex: supermarché)  
                    ❌ **Rotation faible** = Actifs lourds peu utilisés (ex: usine)
                    
                    ---
                    
                    ### 3️⃣ Multiplicateur de Capitaux Propres (Actifs / Capitaux Propres)
                    
                    **Ce que ça mesure :** Le niveau d'endettement (effet de levier financier).
                    
                    | Multiplicateur | Signification |
                    |----------------|---------------|
                    | 1.0x | Pas de dette (100% fonds propres) |
                    | 2.0x | 50% dette, 50% fonds propres |
                    | 3.0x | 67% dette, 33% fonds propres |
                    | 5.0x | 80% dette, 20% fonds propres |
                    
                    **Exemple concret :**
                    - Total des actifs : **100 M€**
                    - Capitaux propres : **40 M€**
                    - Multiplicateur = 100/40 = **2.5x** (60% de dette)
                    
                    ✅ **Levier modéré (1.5-2.5x)** = Optimisation du rendement  
                    ⚠️ **Levier élevé (>3x)** = Risque financier accru
                    
                    ---
                    
                    ### 🎯 Exemple Complet : Comparaison de 2 Entreprises
                    
                    | Métrique | Entreprise A | Entreprise B |
                    |----------|--------------|--------------|
                    | Marge Nette | 10% | 5% |
                    | Rotation Actifs | 1.0x | 2.0x |
                    | Multiplicateur CE | 2.0x | 2.0x |
                    | **ROE** | **20%** | **20%** |
                    
                    **Même ROE, mais :**
                    - **Entreprise A** : Marges élevées, modèle premium
                    - **Entreprise B** : Volume élevé, marges faibles
                    
                    L'analyse DuPont révèle **comment** le ROE est généré, pas seulement sa valeur !
                    """)
                
            elif dupont_results and 'error' in dupont_results:
                st.info(f"Analyse DuPont (ROE LTM) non disponible: {dupont_results['error']}")
            else:
                st.info("Les données financières trimestrielles nécessaires à l'Analyse DuPont (ROE LTM) ne sont pas suffisantes (nécessite 4 Q de résultats et 5 Q de bilan).")
                
        except Exception as e:
            st.error(f"Erreur inattendue lors de l'accés aux données financiéres de YFinance pour ROE LTM: {e}")
            
        # --- NOUVELLE SECTION : ANALYSE ROIC (LTM) ---
        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>🛠️ Décomposition du ROIC (Return on Invested Capital) - LTM</h2>", unsafe_allow_html=True)

        try:
            # Effectuer le calcul ROIC (LTM)
            roic_results = calculate_roic(q_financials, q_balance)
            
            if roic_results and 'error' not in roic_results:
                dates_ltm_start = roic_results['Dates_LTM'][-1]
                dates_ltm_end = roic_results['Dates_LTM'][0]
                date_bilan_t0 = roic_results['Dates_Bilan'][0]
                date_bilan_t4 = roic_results['Dates_Bilan'][1]

                st.caption(f"Le ROIC LTM est basé sur les flux financiers du {dates_ltm_start} au {dates_ltm_end}. Le capital investi moyen est calculé sur 5 trimestres, du {date_bilan_t0} (T0) au {date_bilan_t4} (T-4).")

                col_e, col_f, col_g = st.columns([1, 1, 2]) 

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
                    st.metric(
                        label="ROIC (Return on Invested Capital) LTM",
                        value=f"{roic_results['ROIC'] * 100:.2f} %",
                        delta=f"Produit de (1) x (2)"
                    )
                    
                # EXPLICATION ROIC DANS UN EXPANDER
                with st.expander("📚 Comprendre la Décomposition du ROIC (cliquez pour voir)"):
                    st.markdown("""
                    ## Qu'est-ce que le ROIC ?
                    
                    Le **ROIC (Return on Invested Capital)** mesure la rentabilité du capital total investi 
                    dans l'entreprise (dette + capitaux propres), indépendamment de la structure de financement.
                    
                    $$\\text{ROIC} = \\text{Marge NOPAT} \\times \\text{Rotation du Capital Investi}$$
                    
                    ---
                    
                    ### 1️⃣ Marge NOPAT (NOPAT / Chiffre d'Affaires)
                    
                    **NOPAT** = Net Operating Profit After Taxes = EBIT × (1 - Taux d'imposition)
                    
                    **Ce que ça mesure :** La rentabilité opérationnelle pure, sans l'effet du financement (intérêts).
                    
                    | Secteur | Marge NOPAT Typique |
                    |---------|---------------------|
                    | Tech/Software | 20-30% |
                    | Pharma | 15-25% |
                    | Industrie | 8-15% |
                    | Distribution | 3-8% |
                    
                    **Exemple concret :**
                    - Chiffre d'affaires : **100 M€**
                    - EBIT : **20 M€**
                    - Taux d'imposition : **25%**
                    - NOPAT = 20 × (1 - 0.25) = **15 M€**
                    - Marge NOPAT = 15/100 = **15%**
                    
                    ✅ **Haute marge NOPAT** = Excellence opérationnelle  
                    ❌ **Faible marge NOPAT** = Problèmes structurels de rentabilité
                    
                    ---
                    
                    ### 2️⃣ Rotation du Capital Investi (CA / Capital Investi)
                    
                    **Capital Investi** = Capitaux Propres + Dette Financière Nette
                    
                    **Ce que ça mesure :** L'efficacité avec laquelle le capital est utilisé pour générer des ventes.
                    
                    | Secteur | Rotation CI Typique |
                    |---------|---------------------|
                    | Services/Conseil | 2.0 - 4.0x |
                    | Distribution | 1.5 - 2.5x |
                    | Industrie | 0.8 - 1.5x |
                    | Utilities | 0.3 - 0.6x |
                    
                    **Exemple concret :**
                    - Chiffre d'affaires : **100 M€**
                    - Capitaux propres : **30 M€**
                    - Dette nette : **20 M€**
                    - Capital investi = 30 + 20 = **50 M€**
                    - Rotation CI = 100/50 = **2.0x**
                    
                    ✅ **Rotation élevée** = Capital utilisé efficacement  
                    ❌ **Rotation faible** = Capital "dormant" ou mal alloué
                    
                    ---
                    
                    ### 🎯 Pourquoi le ROIC est-il Important ?
                    
                    | Critère | ROE | ROIC |
                    |---------|-----|------|
                    | Prend en compte la dette | ❌ Non (gonflé par le levier) | ✅ Oui |
                    | Comparable entre secteurs | ⚠️ Difficile | ✅ Plus facile |
                    | Mesure la création de valeur | ⚠️ Partiel | ✅ Oui |
                    
                    **Règle de création de valeur :**
                    - Si **ROIC > Coût du Capital (WACC)** → L'entreprise **crée** de la valeur
                    - Si **ROIC < WACC** → L'entreprise **détruit** de la valeur
                    
                    ---
                    
                    ### 📊 Exemple Complet
                    
                    | Métrique | Valeur |
                    |----------|--------|
                    | CA | 100 M€ |
                    | NOPAT | 12 M€ |
                    | Capital Investi | 60 M€ |
                    | Marge NOPAT | 12% |
                    | Rotation CI | 1.67x |
                    | **ROIC** | **20%** |
                    
                    Si le WACC est de 10%, cette entreprise crée **10% de valeur** au-delà du coût de son capital !
                    
                    ---
                    
                    ### 🏆 Benchmarks ROIC par Secteur
                    
                    | Secteur | ROIC Médian | Top Performers |
                    |---------|-------------|----------------|
                    | Tech/Software | 15-25% | >40% (Google, Microsoft) |
                    | Biens de consommation | 12-18% | >25% (L'Oréal, P&G) |
                    | Industrie | 8-12% | >15% |
                    | Utilities | 5-8% | >10% |
                    | Airlines | 2-6% | >10% (rare) |
                    """)
                
            elif roic_results and 'error' in roic_results:
                    st.info(f"Analyse ROIC (LTM) non disponible: {roic_results['error']}")
            else:
                st.info("Les données financières trimestrielles nécessaires à la décomposition du ROIC LTM ne sont pas suffisantes (nécessite 4 Q de résultats et 5 Q de bilan).")
                
        except Exception as e:
            st.error(f"Erreur inattendue lors de l'accés aux données financiéres de YFinance pour ROIC LTM: {e}")

        # --- NOUVELLE SECTION : SIMULATION MONTE CARLO ---
        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>🎲 Simulation Monte Carlo - Projection des Prix</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        La simulation **Monte Carlo** utilise le rendement et la volatilité historiques pour générer 
        des milliers de trajectoires de prix possibles, permettant d'estimer la distribution 
        probabiliste des prix futurs.
        
        ⚠️ **Bornes appliquées** : Les scénarios extrêmes sont plafonnés à des rendements 
        annuels composés (CAGR) de **+50%** (optimiste) et **-50%** (pessimiste).
        """)
        
        # Paramètres de la simulation dans la sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎲 Paramètres Monte Carlo")
        
        num_simulations = st.sidebar.select_slider(
            "Nombre de Simulations",
            options=[1000, 5000, 10000, 50000, 100000],
            value=10000,
            format_func=lambda x: f"{x:,}".replace(",", " "),
            help="Plus de simulations = résultats plus précis mais calcul plus long"
        )
        
        # Horizon de projection (1, 3, 5, 10 ans uniquement)
        if period_choice == "Hebdomadaire":
            horizon_options = {
                "1 an (~52 semaines)": 52,
                "3 ans (~156 semaines)": 156,
                "5 ans (~260 semaines)": 260,
                "10 ans (~520 semaines)": 520
            }
        else:
            horizon_options = {
                "1 an (12 mois)": 12,
                "3 ans (36 mois)": 36,
                "5 ans (60 mois)": 60,
                "10 ans (120 mois)": 120
            }
        
        selected_horizon = st.sidebar.selectbox(
            "Horizon de Projection",
            options=list(horizon_options.keys()),
            index=0  # Par défaut: 1 an
        )
        
        num_periods = horizon_options[selected_horizon]
        
        # Déterminer le nombre d'années pour les caps réalistes
        if period_choice == "Hebdomadaire":
            num_years = num_periods / 52
        else:
            num_years = num_periods / 12
        
        # Définir des bornes réalistes basées sur des CAGR max/min historiques
        # CAGR max ~25-30% (performance exceptionnelle type top hedge funds)
        # CAGR min ~-15% (scénario très négatif prolongé)
        MAX_CAGR = 0.50  # 50% par an (très optimiste)
        MIN_CAGR = -0.50  # -50% par an (très pessimiste)
        
        max_realistic_multiple = (1 + MAX_CAGR) ** num_years
        min_realistic_multiple = max((1 + MIN_CAGR) ** num_years, 0.05)  # Plancher à 5% du prix
        
        # Récupération des paramètres du modèle de tendance
        current_price = float(y_price.iloc[-1])
        
        # Prix min/max réalistes
        max_realistic_price = current_price * max_realistic_multiple
        min_realistic_price = current_price * min_realistic_multiple
        
        # Lancer la simulation
        with st.spinner(f"Simulation de {num_simulations} trajectoires sur {selected_horizon}..."):
            price_paths = run_monte_carlo_simulation(
                initial_price=current_price,
                expected_return_period=pente_log_periode,
                volatility_period=sigma_log,
                num_simulations=num_simulations,
                num_periods=num_periods
            )
            
            # Calcul des statistiques (vrais percentiles, sans plafonnement)
            mc_stats = calculate_monte_carlo_statistics(price_paths)
        
        # Génération des dates futures pour l'axe X
        last_date = data.index[-1]
        if period_choice == "Hebdomadaire":
            future_dates = pd.date_range(start=last_date, periods=num_periods + 1, freq='W')
        else:
            future_dates = pd.date_range(start=last_date, periods=num_periods + 1, freq='ME')
        
        # Calcul des percentiles pour les bandes (P10-P90 pour éviter les extrêmes)
        # Puis application des bornes réalistes
        percentile_10_raw = np.percentile(price_paths, 10, axis=1)
        percentile_25_raw = np.percentile(price_paths, 25, axis=1)
        percentile_50 = np.percentile(price_paths, 50, axis=1)
        percentile_75_raw = np.percentile(price_paths, 75, axis=1)
        percentile_90_raw = np.percentile(price_paths, 90, axis=1)
        mean_path = np.mean(price_paths, axis=1)
        
        # Appliquer les bornes réalistes aux percentiles
        percentile_10 = np.clip(percentile_10_raw, min_realistic_price, max_realistic_price)
        percentile_25 = np.clip(percentile_25_raw, min_realistic_price, max_realistic_price)
        percentile_75 = np.clip(percentile_75_raw, min_realistic_price, max_realistic_price)
        percentile_90 = np.clip(percentile_90_raw, min_realistic_price, max_realistic_price)
        
        # === CRÉATION DU GRAPHIQUE MONTE CARLO (Style Cône de Projection) ===
        fig_mc = go.Figure()
        
        # Définir les limites Y basées sur les bornes réalistes (avec marge)
        y_min = min_realistic_price * 0.85
        y_max = max_realistic_price * 1.15
        
        # --- TRAJECTOIRES INDIVIDUELLES (en gris léger) ---
        num_paths_to_show = min(50, num_simulations)  # Limiter à 50 pour la lisibilité
        sample_indices = np.random.choice(num_simulations, num_paths_to_show, replace=False)
        
        for idx in sample_indices:
            fig_mc.add_trace(go.Scatter(
                x=future_dates,
                y=price_paths[:, idx],
                mode='lines',
                line=dict(color='rgba(120, 120, 120, 0.2)', width=0.5),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # --- ZONE 1: Intervalle 80% (P10-P90) - Zone extérieure ---
        fig_mc.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(percentile_90) + list(percentile_10[::-1]),
            fill='toself',
            fillcolor='rgba(65, 105, 225, 0.15)',  # Bleu royal transparent
            line=dict(color='rgba(0,0,0,0)'),
            name='Intervalle 80% (P10-P90)',
            hoverinfo='skip'
        ))
        
        # --- ZONE 2: Intervalle 50% (P25-P75) - Zone centrale ---
        fig_mc.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(percentile_75) + list(percentile_25[::-1]),
            fill='toself',
            fillcolor='rgba(65, 105, 225, 0.35)',  # Bleu plus dense
            line=dict(color='rgba(0,0,0,0)'),
            name='Intervalle 50% (P25-P75)',
            hoverinfo='skip'
        ))
        
        # --- Ligne P90 (Optimiste) ---
        fig_mc.add_trace(go.Scatter(
            x=future_dates,
            y=percentile_90,
            mode='lines',
            name=f'P90 (Optimiste): {percentile_90[-1]:.2f} {currency}',
            line=dict(color='#27AE60', width=2, dash='dot'),
            hovertemplate='P90: %{y:.2f}<extra></extra>'
        ))
        
        # --- Ligne P10 (Pessimiste) ---
        fig_mc.add_trace(go.Scatter(
            x=future_dates,
            y=percentile_10,
            mode='lines',
            name=f'P10 (Pessimiste): {percentile_10[-1]:.2f} {currency}',
            line=dict(color='#E74C3C', width=2, dash='dot'),
            hovertemplate='P10: %{y:.2f}<extra></extra>'
        ))
        
        # --- Ligne Médiane (P50) ---
        fig_mc.add_trace(go.Scatter(
            x=future_dates,
            y=percentile_50,
            mode='lines',
            name=f'Médiane (P50): {percentile_50[-1]:.2f} {currency}',
            line=dict(color='#2980B9', width=3),
            hovertemplate='Médiane: %{y:.2f}<extra></extra>'
        ))
        
        # --- Ligne horizontale du prix actuel ---
        fig_mc.add_trace(go.Scatter(
            x=future_dates,
            y=[current_price] * len(future_dates),
            mode='lines',
            name=f'Prix actuel: {current_price:.2f} {currency}',
            line=dict(color='#F39C12', width=2, dash='dash'),
            hovertemplate='Prix actuel: %{y:.2f}<extra></extra>'
        ))
        
        # --- Points de départ et d'arrivée ---
        # Point de départ
        fig_mc.add_trace(go.Scatter(
            x=[future_dates[0]],
            y=[current_price],
            mode='markers',
            name='Départ',
            marker=dict(color='#F39C12', size=14, symbol='diamond', 
                       line=dict(color='white', width=2)),
            showlegend=False,
            hovertemplate=f'Départ: {current_price:.2f} {currency}<extra></extra>'
        ))
        
        # Points finaux sur les lignes clés
        final_points_x = [future_dates[-1]] * 3
        final_points_y = [percentile_10[-1], percentile_50[-1], percentile_90[-1]]
        final_colors = ['#E74C3C', '#2980B9', '#27AE60']
        
        fig_mc.add_trace(go.Scatter(
            x=final_points_x,
            y=final_points_y,
            mode='markers+text',
            marker=dict(color=final_colors, size=12, symbol='circle',
                       line=dict(color='white', width=2)),
            text=[f'{v:.0f}' for v in final_points_y],
            textposition='middle right',
            textfont=dict(size=11, color=final_colors),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # --- Mise en page ---
        fig_mc.update_layout(
            title={
                'text': f'📈 Projection Monte Carlo - Horizon: {selected_horizon}',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18)
            },
            xaxis_title="Date",
            yaxis_title=f"Prix ({currency})",
            hovermode="x unified",
            template="plotly_white",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            ),
            margin=dict(t=100, r=80)
        )
        
        # Échelle logarithmique (plus adaptée aux prix d'actions)
        fig_mc.update_yaxes(
            type="log",
            range=[np.log10(y_min), np.log10(y_max)],
            tickformat='.0f',
            gridcolor='rgba(0,0,0,0.1)'
        )
        
        fig_mc.update_xaxes(
            gridcolor='rgba(0,0,0,0.1)'
        )
        
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # --- CARTES SCÉNARIOS DE PRIX (juste après le graphique MC) ---
        st.markdown("#### 📋 Scénarios de Prix Projetés")
        
        # Valeurs brutes des percentiles
        p10_raw = mc_stats['percentile_10']
        p50_raw = mc_stats['median_final']
        p90_raw = mc_stats['percentile_90']
        
        # Appliquer des bornes réalistes pour l'AFFICHAGE uniquement
        # CAGR: +50% max, -50% min
        max_cagr_display = 0.50  # 50% par an
        min_cagr_display = -0.50  # -50% par an
        
        max_display_price = current_price * ((1 + max_cagr_display) ** num_years)
        min_display_price = current_price * ((1 + min_cagr_display) ** num_years)
        
        # Valeurs affichées (plafonnées si nécessaire)
        p10_display = max(p10_raw, min_display_price)
        p90_display = min(p90_raw, max_display_price)
        
        # Détecter si des valeurs sont plafonnées
        p10_capped = p10_raw < min_display_price
        p90_capped = p90_raw > max_display_price
        
        # Calcul des rendements pour les cartes
        p10_ret_card = ((p10_display / current_price) - 1) * 100
        p50_ret_card = ((p50_raw / current_price) - 1) * 100
        p90_ret_card = ((p90_display / current_price) - 1) * 100
        
        col_card1, col_card2, col_card3 = st.columns(3)
        
        with col_card1:
            cap_note_p10 = " ⚠️" if p10_capped else ""
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); padding: 15px; border-radius: 10px; text-align: center;">
                <h4 style="color: #c62828; margin: 0;">🔴 Pessimiste (P10){cap_note_p10}</h4>
                <h2 style="margin: 10px 0;">{p10_display:.2f} {currency}</h2>
                <p style="margin: 0; color: #c62828; font-weight: bold;">{p10_ret_card:+.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_card2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 15px; border-radius: 10px; text-align: center;">
                <h4 style="color: #1565c0; margin: 0;">🔵 Médiane (P50)</h4>
                <h2 style="margin: 10px 0;">{p50_raw:.2f} {currency}</h2>
                <p style="margin: 0; color: #1565c0; font-weight: bold;">{p50_ret_card:+.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_card3:
            cap_note_p90 = " ⚠️" if p90_capped else ""
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); padding: 15px; border-radius: 10px; text-align: center;">
                <h4 style="color: #2e7d32; margin: 0;">🟢 Optimiste (P90){cap_note_p90}</h4>
                <h2 style="margin: 10px 0;">{p90_display:.2f} {currency}</h2>
                <p style="margin: 0; color: #2e7d32; font-weight: bold;">{p90_ret_card:+.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Note explicative si des valeurs sont plafonnées
        if p10_capped or p90_capped:
            st.caption(f"⚠️ Valeurs plafonnées aux rendements réalistes (CAGR {min_cagr_display*100:+.0f}% à {max_cagr_display*100:+.0f}%/an). "
                      f"Valeurs brutes: P10={p10_raw:.2f}, P90={p90_raw:.2f}")
        
        st.caption(f"💡 Prix actuel: {current_price:.2f} {currency} | Probabilité de gain: {mc_stats['prob_gain']:.0f}%")
        
        # Affichage des statistiques Monte Carlo
        st.markdown("### 📊 Statistiques de la Simulation")
        
        col_mc1, col_mc2, col_mc3, col_mc4 = st.columns(4)
        
        with col_mc1:
            rendement_median = ((mc_stats['median_final'] / current_price) - 1) * 100
            st.metric(
                label=f"Prix Médian à {selected_horizon}",
                value=f"{mc_stats['median_final']:.2f} {currency}",
                delta=f"{rendement_median:+.1f}%"
            )
        
        with col_mc2:
            st.metric(
                label="Intervalle 80% (P10-P90)",
                value=f"{mc_stats['percentile_10']:.2f} - {mc_stats['percentile_90']:.2f}",
                delta=None
            )
        
        with col_mc3:
            st.metric(
                label="Probabilité de Gain",
                value=f"{mc_stats['prob_gain']:.1f}%",
                delta=None
            )
        
        with col_mc4:
            st.metric(
                label="Prob. de Doubler",
                value=f"{mc_stats['prob_double']:.1f}%",
                delta=f"Prob. -50%: {mc_stats['prob_loss_50']:.1f}%" if mc_stats['prob_loss_50'] > 0 else None,
                delta_color="inverse"
            )
        
        # Tableau détaillé des percentiles
        st.markdown("### 📈 Distribution des Prix Finaux")
        
        col_table1, col_table2 = st.columns(2)
        
        with col_table1:
            percentile_data = pd.DataFrame({
                'Percentile': ['10% (Pessimiste)', '25%', '50% (Médiane)', '75%', '90% (Optimiste)'],
                f'Prix ({currency})': [
                    f"{mc_stats['percentile_10']:.2f}",
                    f"{mc_stats['percentile_25']:.2f}",
                    f"{mc_stats['median_final']:.2f}",
                    f"{mc_stats['percentile_75']:.2f}",
                    f"{mc_stats['percentile_90']:.2f}"
                ],
                'Rendement': [
                    f"{((mc_stats['percentile_10'] / current_price) - 1) * 100:+.1f}%",
                    f"{((mc_stats['percentile_25'] / current_price) - 1) * 100:+.1f}%",
                    f"{((mc_stats['median_final'] / current_price) - 1) * 100:+.1f}%",
                    f"{((mc_stats['percentile_75'] / current_price) - 1) * 100:+.1f}%",
                    f"{((mc_stats['percentile_90'] / current_price) - 1) * 100:+.1f}%"
                ]
            })
            st.dataframe(percentile_data, use_container_width=True, hide_index=True)
        
        with col_table2:
            st.markdown("""
            **Interprétation des résultats:**
            - **P10 (Pessimiste)**: 90% des simulations dépassent ce prix
            - **P50 (Médiane)**: 50% des simulations au-dessus/en-dessous
            - **P90 (Optimiste)**: Seulement 10% des simulations dépassent ce prix
            
            *Les scénarios extrêmes (P5/P95) sont exclus car ils représentent 
            des événements rares peu pertinents pour la planification.*
            """)
        
        # === HISTOGRAMME SIMPLIFIÉ ===
        st.markdown("### 📊 Distribution des Résultats à l'Horizon")
        
        final_prices_all = price_paths[-1, :]
        
        # Calcul des rendements en %
        returns_all = ((final_prices_all / current_price) - 1) * 100
        
        # Percentiles clés
        p10_ret = ((mc_stats['percentile_10'] / current_price) - 1) * 100
        p50_ret = ((mc_stats['median_final'] / current_price) - 1) * 100
        p90_ret = ((mc_stats['percentile_90'] / current_price) - 1) * 100
        
        # Filtrer P10-P90 pour un affichage compact
        p10_val = np.percentile(returns_all, 10)
        p90_val = np.percentile(returns_all, 90)
        returns_filtered = returns_all[(returns_all >= p10_val) & (returns_all <= p90_val)]
        
        # Filtrer les prix P10-P90
        p10_price_val = np.percentile(final_prices_all, 10)
        p90_price_val = np.percentile(final_prices_all, 90)
        prices_filtered = final_prices_all[(final_prices_all >= p10_price_val) & (final_prices_all <= p90_price_val)]
        
        # --- HISTOGRAMME 1: DISTRIBUTION DES PRIX ---
        fig_hist_price = go.Figure()
        
        fig_hist_price.add_trace(go.Histogram(
            x=prices_filtered,
            nbinsx=35,
            name='Simulations (P10-P90)',
            marker_color='rgba(46, 204, 113, 0.7)',
            marker_line_color='rgba(46, 204, 113, 1)',
            marker_line_width=1
        ))
        
        # Lignes de référence avec légende
        fig_hist_price.add_trace(go.Scatter(
            x=[current_price, current_price], y=[0, 0], mode='lines',
            name=f'🟡 Prix actuel: {current_price:.2f} {currency}',
            line=dict(color='#F39C12', width=3, dash='solid')
        ))
        fig_hist_price.add_vline(x=current_price, line_dash="solid", line_color="#F39C12", line_width=3)
        
        fig_hist_price.add_trace(go.Scatter(
            x=[mc_stats['percentile_10'], mc_stats['percentile_10']], y=[0, 0], mode='lines',
            name=f'🔴 P10: {mc_stats["percentile_10"]:.2f} {currency}',
            line=dict(color='#E74C3C', width=2, dash='dot')
        ))
        fig_hist_price.add_vline(x=mc_stats['percentile_10'], line_dash="dot", line_color="#E74C3C", line_width=2)
        
        fig_hist_price.add_trace(go.Scatter(
            x=[mc_stats['median_final'], mc_stats['median_final']], y=[0, 0], mode='lines',
            name=f'🔵 P50: {mc_stats["median_final"]:.2f} {currency}',
            line=dict(color='#2980B9', width=2, dash='dash')
        ))
        fig_hist_price.add_vline(x=mc_stats['median_final'], line_dash="dash", line_color="#2980B9", line_width=2)
        
        fig_hist_price.add_trace(go.Scatter(
            x=[mc_stats['percentile_90'], mc_stats['percentile_90']], y=[0, 0], mode='lines',
            name=f'🟢 P90: {mc_stats["percentile_90"]:.2f} {currency}',
            line=dict(color='#27AE60', width=2, dash='dot')
        ))
        fig_hist_price.add_vline(x=mc_stats['percentile_90'], line_dash="dot", line_color="#27AE60", line_width=2)
        
        fig_hist_price.update_layout(
            title=f'Distribution des Prix à {selected_horizon}',
            xaxis_title=f"Prix ({currency})",
            yaxis_title="Nombre de simulations",
            template="plotly_white",
            height=380,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)"
            ),
            margin=dict(t=70, b=50)
        )
        
        margin_price = (p90_price_val - p10_price_val) * 0.15
        fig_hist_price.update_xaxes(range=[p10_price_val - margin_price, p90_price_val + margin_price])
        
        st.plotly_chart(fig_hist_price, use_container_width=True)
        
        # --- HISTOGRAMME 2: DISTRIBUTION DES RENDEMENTS ---
        fig_hist_ret = go.Figure()
        
        fig_hist_ret.add_trace(go.Histogram(
            x=returns_filtered,
            nbinsx=35,
            name='Simulations (P10-P90)',
            marker_color='rgba(52, 152, 219, 0.7)',
            marker_line_color='rgba(52, 152, 219, 1)',
            marker_line_width=1
        ))
        
        # Lignes de référence avec légende
        fig_hist_ret.add_trace(go.Scatter(
            x=[0, 0], y=[0, 0], mode='lines',
            name=f'🟡 Breakeven: 0%',
            line=dict(color='#F39C12', width=3, dash='solid')
        ))
        fig_hist_ret.add_vline(x=0, line_dash="solid", line_color="#F39C12", line_width=3)
        
        fig_hist_ret.add_trace(go.Scatter(
            x=[p10_ret, p10_ret], y=[0, 0], mode='lines',
            name=f'🔴 P10: {p10_ret:+.1f}%',
            line=dict(color='#E74C3C', width=2, dash='dot')
        ))
        fig_hist_ret.add_vline(x=p10_ret, line_dash="dot", line_color="#E74C3C", line_width=2)
        
        fig_hist_ret.add_trace(go.Scatter(
            x=[p50_ret, p50_ret], y=[0, 0], mode='lines',
            name=f'🔵 P50: {p50_ret:+.1f}%',
            line=dict(color='#2980B9', width=2, dash='dash')
        ))
        fig_hist_ret.add_vline(x=p50_ret, line_dash="dash", line_color="#2980B9", line_width=2)
        
        fig_hist_ret.add_trace(go.Scatter(
            x=[p90_ret, p90_ret], y=[0, 0], mode='lines',
            name=f'🟢 P90: {p90_ret:+.1f}%',
            line=dict(color='#27AE60', width=2, dash='dot')
        ))
        fig_hist_ret.add_vline(x=p90_ret, line_dash="dot", line_color="#27AE60", line_width=2)
        
        fig_hist_ret.update_layout(
            title=f'Distribution des Rendements à {selected_horizon}',
            xaxis_title="Rendement (%)",
            yaxis_title="Nombre de simulations",
            template="plotly_white",
            height=380,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)"
            ),
            margin=dict(t=70, b=50)
        )
        
        margin_ret = (p90_val - p10_val) * 0.15
        fig_hist_ret.update_xaxes(range=[p10_val - margin_ret, p90_val + margin_ret], ticksuffix="%")
        
        st.plotly_chart(fig_hist_ret, use_container_width=True)
        
        # Explication de la méthodologie
        with st.expander("📚 Méthodologie de la Simulation Monte Carlo"):
            st.markdown(f"""
            ### Comment fonctionne cette simulation ?
            
            La simulation utilise le **Mouvement Brownien Géométrique (GBM)**, le modèle standard 
            en finance pour modéliser l'évolution des prix d'actifs :
            
            $$\\ln\\left(\\frac{{S_{{t+1}}}}{{S_t}}\\right) = \\text{{drift}} + \\sigma \\cdot Z$$
            
            Ce qui équivaut à : $S_{{t+1}} = S_t \\times e^{{\\text{{drift}} + \\sigma \\cdot Z}}$
            
            Où:
            - $S_t$ = Prix au temps t
            - $\\text{{drift}}$ = Rendement log attendu par période, issu de la régression (estimé: **{pente_log_periode*100:.4f}%** par {period_label.lower()})
            - $\\sigma$ = Volatilité des résidus log (estimée: **{sigma_log*100:.4f}%** par {period_label.lower()})
            - $Z$ = Variable aléatoire normale standard $\\mathcal{{N}}(0, 1)$
            
            ### Paramètres utilisés:
            | Paramètre | Valeur | Source |
            |-----------|--------|--------|
            | Prix initial | {current_price:.2f} {currency} | Dernier prix de clôture |
            | Drift (rendement log) | {pente_log_periode*100:.4f}% / {period_label.lower()} | Pente de la régression log-linéaire |
            | Volatilité (σ) | {sigma_log*100:.4f}% / {period_label.lower()} | Écart-type des résidus log |
            | Nombre de simulations | {num_simulations:,} | Paramètre utilisateur |
            | Horizon | {num_periods} {period_label.lower()}s | Paramètre utilisateur |
            
            ### Note technique:
            Le drift utilisé est directement la pente de la régression log-linéaire, qui représente 
            $E[\\ln(S_{{t+1}}/S_t)]$. Dans la théorie GBM, cela correspond à $(\\mu - \\sigma^2/2)$ où $\\mu$ 
            est le rendement instantané. Nous utilisons directement cette valeur observée sans ajustement.
            
            ### Bornes appliquées (affichage uniquement):
            Pour éviter les scénarios extrêmes, les valeurs **affichées** dans les cartes sont plafonnées.
            Les calculs de probabilités utilisent les vraies valeurs de simulation.
            
            | Borne | CAGR | Multiple sur {num_years:.0f} an(s) |
            |-------|------|-----------------------------------|
            | Optimiste (P90) | +50%/an | x{(1.50 ** num_years):.2f} |
            | Pessimiste (P10) | -50%/an | x{(0.50 ** num_years):.2f} |
            
            ### Limites du modèle:
            - Suppose que les rendements futurs suivent la même distribution que les rendements passés
            - Ne prend pas en compte les événements extrêmes (cygnes noirs)
            - La volatilité est supposée constante dans le temps
            - Le modèle GBM suppose des rendements log-normaux (distribution symétrique en log)
            """)

    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement des données ou du téléchargement: {e}")
        st.caption("Vérifiez que le code de l'action (ticker) est correct.")


# Exécuter l'application
if __name__ == "__main__":
    run_app()