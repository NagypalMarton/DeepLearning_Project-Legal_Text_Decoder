import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict


# Configuration
API_URL = os.getenv('API_URL', 'http://localhost:8000')

# Page config
st.set_page_config(
    page_title="Legal Text Decoder",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> Dict:
    """Check API health status"""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "models_available": [], "models_loaded": []}
    except Exception as e:
        return {"status": "offline", "error": str(e), "models_available": [], "models_loaded": []}


def predict_text(text: str, model_type: str) -> Dict:
    """Send prediction request to API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text, "model_type": model_type},
            timeout=30
        )
        
        if response.status_code == 200:
            try:
                return {"success": True, "data": response.json()}
            except Exception:
                return {"success": False, "error": f"Invalid JSON response: {response.text[:200]}"}
        else:
            # Robust error extraction: try JSON, else use raw text
            err_detail = None
            try:
                err_detail = response.json().get("detail")
            except Exception:
                err_detail = None
            return {"success": False, "error": err_detail or response.text or f"HTTP {response.status_code}"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def create_probability_chart(probabilities: Dict[str, float]):
    """Create bar chart for class probabilities"""
    # Sort by probability
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_probs]
    values = [item[1] for item in sorted_probs]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color='lightblue',
            text=[f'{v:.1%}' for v in values],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Osztály valószínűségek",
        xaxis_title="Érthetőségi kategória",
        yaxis_title="Valószínűség",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        height=400
    )
    
    return fig


def main():
    # Header
    st.markdown('<div class="main-header">⚖️ Legal Text Decoder</div>', unsafe_allow_html=True)
    st.markdown("### Jogi szövegek érthetőségének automatikus értékelése AI segítségével")
    
    # Check API status first
    health = check_api_health()
    
    # Use transformer model only
    model_type = "transformer"
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Információ")
        
        if health['status'] == 'offline':
            st.error("🔴 API nem elérhető")
            st.info("Ellenőrizd, hogy a backend fut-e:\n```\npython src/api/app.py\n```")
        elif health['status'] == 'no_models_loaded':
            st.warning("⚠️ Nincs betöltött modell!")
            st.info("Futtasd le először a training pipeline-t!")
        else:
            st.success("✅ API elérhető")
        
        st.markdown("---")
        
        st.markdown("---")
        st.markdown("**Érthetőségi skála:**")
        st.markdown("""
        - **1**: Nagyon nehezen érthető
        - **2**: Nehezen érthető
        - **3**: Közepesen érthető
        - **4**: Könnyen érthető
        - **5**: Nagyon könnyen érthető
        """)
    
    # Check if we should stop early
    if health['status'] == 'offline':
        return
    
    # Stop if no models loaded
    if not health.get('models_loaded'):
        st.warning("⚠️ Nincs betöltött modell. Futtasd le először a training pipeline-t!")
        st.stop()
    
    # Main content
    st.subheader("📝 Jogi szöveg bekezdés")
    text_input = st.text_area(
        "Írd be vagy illeszd be a jogi szöveg egy bekezdését:",
        height=200,
        placeholder="Például: A jelen Általános Szerződési Feltételek (továbbiakban: ÁSZF) tartalmazzák...",
        value=st.session_state.get('example_text', '')
    )
    
    # Clear example text from session state after use
    if 'example_text' in st.session_state:
        del st.session_state['example_text']
    
    # Predict button
    st.markdown("---")
    predict_button = st.button("🔍 Értékelés", type="primary", use_container_width=True)
    
    # Perform prediction
    if predict_button:
        if not text_input or len(text_input.strip()) == 0:
            st.error("❌ Kérlek, adj meg egy szöveget!")
            return
        
        with st.spinner("🔄 Értékelés folyamatban..."):
            result = predict_text(text_input, model_type)
        
        if not result['success']:
            st.error(f"❌ Hiba történt: {result['error']}")
            return
        
        # Display results
        data = result['data']
        
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        
        # Main prediction
        st.markdown("## 📊 Eredmény")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Érthetőségi kategória",
                value=data['prediction']
            )
        
        with col2:
            st.metric(
                label="Bizalmi szint",
                value=f"{data['confidence']:.1%}"
            )       
        st.markdown("---")
        
        # Probability chart
        st.markdown("### 📈 Valószínűség eloszlás")
        fig = create_probability_chart(data['probabilities'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Probability table
        st.markdown("### 📋 Részletes eredmények")
        prob_df = pd.DataFrame([
            {"Kategória": k, "Valószínűség": f"{v:.2%}"}
            for k, v in sorted(data['probabilities'].items(), key=lambda x: x[1], reverse=True)
        ])
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interpretation
        confidence = data['confidence']
        if confidence > 0.8:
            st.success("✅ **Magas bizalmi szint** - A modell magabiztosan osztályozta a szöveget.")
        elif confidence > 0.5:
            st.info("ℹ️ **Közepes bizalmi szint** - Az eredmény valószínűleg helyes, de érdemes óvatosan kezelni.")
        else:
            st.warning("⚠️ **Alacsony bizalmi szint** - A modell bizonytalan az eredményben.")


if __name__ == "__main__":
    main()
