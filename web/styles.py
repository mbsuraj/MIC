# Custom CSS for modern bright appearance
MAIN_STYLES = """
<style>
    /* Main app background */
    .stApp {
        background: #0e1117;
        color: #e6e6e6;
    }
    
    /* Main content area */
    .main .block-container {
        background: #161b22;
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
    }
    
    /* Fancy title styling (kept colorful) */
    .main-header {
        text-align: center;
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: none;
        margin-bottom: 1rem;
        font-family: 'Arial Black', sans-serif;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Question containers */
    .question-container {
        background: #1f2933;
        padding: 2.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 1px solid #2d333b;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        color: #e6e6e6;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Metric styling */
    .metric-container {
        background: #1f2933;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        color: #e6e6e6;
    }
</style>
"""