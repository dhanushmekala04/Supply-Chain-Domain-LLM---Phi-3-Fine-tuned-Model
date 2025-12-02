import streamlit as st
from ollama import Client
import time
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title=" AI Text-to-Cypher Generator | Supply Chain Analytics",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://huggingface.co/dhanushmekaka/phi-3-mini-text-to-cypher-supply',
        'Report a bug': 'https://github.com/yourusername/supply-chain-query-generator/issues',
        'About': """
        ## **AI-Powered Text-to-Cypher Generator**
        
        Convert natural language to Neo4j Cypher queries using a fine-tuned Phi-3 model.
        
        **Features:**
        - 95%+ accuracy on supply chain queries
        - Local deployment (privacy-first)
        - 1-2 second response time
        - Zero API costs
        
        Built with ❤️ using Phi-3, QLoRA, and Ollama
        """
    }
)
# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .example-query {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    .example-query:hover {
        background-color: #e9ecef;
        transform: translateX(5px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'total_time' not in st.session_state:
    st.session_state.total_time = 0.0

# Initialize Ollama client
@st.cache_resource
def get_client():
    return Client()

def generate_cypher(query, model_name="phi3-local"):
    """Generate Cypher query from natural language"""
    client = get_client()
    
    prompt = f"""<|user|>
{query}
<|end|>
<|assistant|>
"""
    
    try:
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"], None
    except Exception as e:
        return None, str(e)

def save_to_history(question, query, elapsed_time):
    """Save query to history"""
    st.session_state.history.append({
        'question': question,
        'query': query,
        'time': elapsed_time,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    st.session_state.query_count += 1
    st.session_state.total_time += elapsed_time

def export_history():
    """Export history as JSON"""
    return json.dumps(st.session_state.history, indent=2)

# Header
st.markdown('<p class="main-header">🔗 Supply Chain Text-to-Cypher Generator</p>', unsafe_allow_html=True)
st.markdown("**Convert natural language questions into Neo4j Cypher queries using AI**")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    
    st.markdown("### ℹ️ About This Tool")
    st.info("""
    This AI-powered tool uses a fine-tuned **Phi-3 model** trained on 10,000+ 
    supply chain examples to generate accurate Neo4j Cypher queries.
    """)
    
    st.markdown("### 📊 Model Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Size", "~4GB")
        st.metric("Training Data", "10K")
    with col2:
        st.metric("Accuracy", "95%+")
        st.metric("Avg Speed", "10-20s")
    
    st.markdown("### 🎯 Capabilities")
    st.markdown("""
    ✅ Aggregations (COUNT, SUM, AVG)  
    ✅ Complex filtering (WHERE)  
    ✅ Multi-hop relationships  
    ✅ Sorting & Pagination  
    ✅ Date-based queries  
    ✅ Pattern matching  
    """)
    
    st.markdown("### 📈 Session Stats")
    st.metric("Queries Generated", st.session_state.query_count)
    st.metric("Total Time", f"{st.session_state.total_time:.2f}s")
    if st.session_state.query_count > 0:
        avg_time = st.session_state.total_time / st.session_state.query_count
        st.metric("Avg Query Time", f"{avg_time:.2f}s")
    
    st.markdown("---")
    
    # Model settings
    st.markdown("### ⚙️ Settings")
    model_name = st.text_input("Model Name", value="phi3-local", help="Ollama model name")
    
    # Export history
    if st.session_state.history:
        if st.download_button(
            label="📥 Export History",
            data=export_history(),
            file_name=f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        ):
            st.success("History exported!")
    
    # Clear history
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.query_count = 0
        st.session_state.total_time = 0.0
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; font-size: 0.8rem;'>
        <p>Built with ❤️ using Streamlit</p>
        <p><a href='https://huggingface.co/dhanushmekaka/phi-3-mini-text-to-cypher-supply' target='_blank'>
            🤗 Model on Hugging Face
        </a></p>
    </div>
    """, unsafe_allow_html=True)

# Main content
tab1, tab2, tab3 = st.tabs(["🚀 Query Generator", "💡 Examples", "📜 History"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📝 Your Question")
        
        # Quick example buttons
        st.markdown("**Quick Examples:**")
        example_cols = st.columns(2)
        with example_cols[0]:
            if st.button("🔝 Top Products", use_container_width=True):
                st.session_state.quick_query = "Show top 5 products by total ordered quantity"
        with example_cols[1]:
            if st.button("📦 Low Stock", use_container_width=True):
                st.session_state.quick_query = "Show products with low stock levels"
        
        user_query = st.text_area(
            "Enter your natural language question:",
            value=st.session_state.get('quick_query', ''),
            height=200,
            placeholder="Example: Show top 5 products by total ordered quantity",
            key="query_input",
            help="Type your question in plain English"
        )
        
        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            generate_button = st.button(
                "🚀 Generate Cypher Query", 
                type="primary", 
                use_container_width=True
            )
        with col_btn2:
            clear_button = st.button("🔄 Clear", use_container_width=True)
        
        if clear_button:
            st.session_state.quick_query = ''
            st.rerun()
    
    with col2:
        st.markdown("### ⚡ Generated Query")
        
        if generate_button:
            if not user_query.strip():
                st.warning("⚠️ Please enter a question first!")
            else:
                with st.spinner("🔄 Generating query..."):
                    start_time = time.time()
                    cypher_query, error = generate_cypher(user_query, model_name)
                    elapsed_time = time.time() - start_time
                
                if error:
                    st.error(f"❌ Error: {error}")
                    st.info("💡 Make sure Ollama is running and the model is loaded:\n```bash\nollama run phi3-local\n```")
                elif cypher_query:
                    st.code(cypher_query, language="cypher", line_numbers=True)
                    
                    # Success message with metrics
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.success("✅ Generated!")
                    with col_m2:
                        st.metric("Time", f"{elapsed_time:.2f}s")
                    with col_m3:
                        st.metric("Length", f"{len(cypher_query)} chars")
                    
                    # Action buttons
                    col_a1, col_a2 = st.columns(2)
                    with col_a1:
                        st.download_button(
                            label="📋 Download Query",
                            data=cypher_query,
                            file_name=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.cypher",
                            mime="text/plain",
                            use_container_width=True
                        )
                    with col_a2:
                        if st.button("📌 Save to History", use_container_width=True):
                            save_to_history(user_query, cypher_query, elapsed_time)
                            st.success("Saved to history!")
                    
                    # Auto-save to history
                    save_to_history(user_query, cypher_query, elapsed_time)
        else:
            st.info("👈 Enter a question and click **Generate** to create a Cypher query")

with tab2:
    st.markdown("### 💡 Example Queries")
    st.markdown("Click on any example to load it into the query box")
    
    examples = {
        "📊 Product Analytics": [
            "Show top 5 products by total ordered quantity",
            "Products with low stock levels",
            "Most expensive products in inventory",
            "Products with no orders in last 30 days"
        ],
        "🏭 Supplier Analysis": [
            "Which suppliers provide the most products?",
            "Suppliers located in California",
            "Find suppliers with delivery delays",
            "Suppliers with highest quality ratings"
        ],
        "📦 Order Management": [
            "Find orders placed in the last 30 days",
            "Orders with total amount over $1000",
            "Pending orders by customer",
            "Orders with delayed shipments"
        ],
        "🚚 Shipping & Logistics": [
            "Show shipments using air transport with high costs",
            "Calculate average lead time by transport mode",
            "Find the longest delivery routes",
            "Shipments delayed by more than 5 days"
        ],
        "🏢 Warehouse Operations": [
            "Which warehouses store products from supplier XYZ?",
            "Warehouse capacity utilization",
            "Products stored in multiple warehouses",
            "Warehouse inventory value by location"
        ],
        "🔗 Complex Queries": [
            "Show products ordered by customers with their suppliers and shipping costs",
            "Find supply chain bottlenecks",
            "Complete order-to-delivery tracking",
            "Multi-warehouse product distribution analysis"
        ]
    }
    
    for category, queries in examples.items():
        with st.expander(f"**{category}**", expanded=True):
            for query in queries:
                if st.button(f"📌 {query}", key=f"ex_{query}", use_container_width=True):
                    st.session_state.quick_query = query
                    st.rerun()

with tab3:
    st.markdown("### 📜 Query History")
    
    if not st.session_state.history:
        st.info("📭 No queries generated yet. Start by generating your first query!")
    else:
        # Display history in reverse order (newest first)
        for idx, item in enumerate(reversed(st.session_state.history)):
            actual_idx = len(st.session_state.history) - idx
            
            with st.expander(
                f"**Query #{actual_idx}** - {item['timestamp']} - ⏱️ {item['time']:.2f}s",
                expanded=(idx == 0)
            ):
                st.markdown(f"**Question:**")
                st.info(item['question'])
                
                st.markdown(f"**Generated Query:**")
                st.code(item['query'], language="cypher", line_numbers=True)
                
                col_h1, col_h2, col_h3 = st.columns(3)
                with col_h1:
                    st.download_button(
                        label="💾 Download",
                        data=item['query'],
                        file_name=f"query_{actual_idx}.cypher",
                        mime="text/plain",
                        key=f"download_{actual_idx}",
                        use_container_width=True
                    )
                with col_h2:
                    if st.button("🔄 Reuse Query", key=f"reuse_{actual_idx}", use_container_width=True):
                        st.session_state.quick_query = item['question']
                        st.rerun()
                with col_h3:
                    st.metric("Generation Time", f"{item['time']:.2f}s")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>🔗 Supply Chain Text-to-Cypher Generator</strong></p>
    <p>Powered by <strong>Phi-3</strong> | Fine-tuned with <strong>QLoRA</strong> | Deployed with <strong>Ollama</strong></p>
    <p style='font-size: 0.9rem;'>
        <a href='https://huggingface.co/dhanushmekaka/phi-3-mini-text-to-cypher-supply' target='_blank'>View Model on Hugging Face</a>
    </p>
</div>
""", unsafe_allow_html=True)